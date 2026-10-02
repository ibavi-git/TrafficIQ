"""Thin job manager and adapter for the existing M4 traffic pipeline."""

from __future__ import annotations

import asyncio
import base64
import logging
import sys
import tempfile
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Set

import cv2
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
AI_CORE = REPO_ROOT / "ai-core"
if str(AI_CORE) not in sys.path:
    sys.path.insert(0, str(AI_CORE))

from detection.classes import (  # noqa: E402
    COCO_VEHICLE_IDS,
    DEFAULT_CONF,
    DEFAULT_DEVICE,
    DEFAULT_MODEL,
    group_for_class,
)
from traffic.analyze import (  # noqa: E402
    TrackInfo,
    calculate_congestion_index,
    calculate_occupancy,
    congestion_label,
    detect_anomalies,
    draw_overlay,
    update_track,
    centroid,
)

LOGGER = logging.getLogger(__name__)
RUNTIME_ROOT = Path(tempfile.gettempdir()) / "TrafficIQ" / "jobs"
QUEUE_SIZE = 4
PREVIEW_EVERY = 5
SLOW_THRESHOLD = 2.0
STOPPED_FRAMES = 30


@dataclass
class TrafficJob:
    job_id: str
    source_path: Path
    output_path: Path
    status: str = "queued"
    latest_message: Dict[str, Any] = field(default_factory=dict)
    subscribers: Set[asyncio.Queue] = field(default_factory=set)
    task: Optional[asyncio.Task] = None


class TrafficJobManager:
    def __init__(self) -> None:
        self.jobs: Dict[str, TrafficJob] = {}

    def create_job(self, suffix: str) -> TrafficJob:
        job_id = uuid.uuid4().hex
        job_dir = RUNTIME_ROOT / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        job = TrafficJob(
            job_id=job_id,
            source_path=job_dir / f"input{suffix}",
            output_path=job_dir / "processed.mp4",
            latest_message={"type": "status", "status": "queued", "progress": 0},
        )
        self.jobs[job_id] = job
        return job

    def start_job(self, job_id: str) -> None:
        job = self.jobs[job_id]
        job.task = asyncio.create_task(self._run_job(job))

    async def subscribe(self, job_id: str) -> Optional[asyncio.Queue]:
        job = self.jobs.get(job_id)
        if job is None:
            return None
        queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_SIZE)
        job.subscribers.add(queue)
        queue.put_nowait(job.latest_message)
        return queue

    def unsubscribe(self, job_id: str, queue: asyncio.Queue) -> None:
        job = self.jobs.get(job_id)
        if job is not None:
            job.subscribers.discard(queue)

    async def publish(self, job: TrafficJob, message: Dict[str, Any]) -> None:
        if message.get("type") == "status":
            job.status = str(message.get("status", job.status))
        job.latest_message = message
        for queue in tuple(job.subscribers):
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                pass

    async def _run_job(self, job: TrafficJob) -> None:
        await self.publish(
            job,
            {"type": "status", "status": "processing", "progress": 0},
        )
        loop = asyncio.get_running_loop()

        def emit_from_worker(message: Dict[str, Any]) -> None:
            asyncio.run_coroutine_threadsafe(self.publish(job, message), loop).result()

        try:
            summary = await asyncio.to_thread(
                process_traffic_video,
                job.source_path,
                job.output_path,
                emit_from_worker,
            )
            await self.publish(
                job,
                {
                    "type": "status",
                    "status": "completed",
                    "progress": 100,
                    **summary,
                },
            )
        except Exception as exc:
            LOGGER.exception("Traffic processing failed for job %s", job.job_id)
            message = (
                "The video could not be processed. Check that it is a valid, "
                "non-empty traffic video and try again."
                if isinstance(exc, InvalidVideoError)
                else "Traffic processing failed. Please try another video."
            )
            await self.publish(
                job,
                {"type": "status", "status": "error", "progress": 0, "message": message},
            )


class InvalidVideoError(ValueError):
    """The uploaded file cannot be decoded as a video."""


def process_traffic_video(
    source_path: Path,
    output_path: Path,
    emit: Any,
) -> Dict[str, Any]:
    """Process every frame using the existing M1-M4 model and analysis logic."""
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise InvalidVideoError("Could not open uploaded video")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0:
        capture.release()
        raise InvalidVideoError("Video has invalid dimensions")

    model = YOLO(str(REPO_ROOT / "yolov8n.pt"))
    writer: Optional[cv2.VideoWriter] = None
    tracks: Dict[int, TrackInfo] = {}
    seen_ids: Set[int] = set()
    peak_vehicles = 0
    processed_frames = 0
    last_metrics: Dict[str, Any] = {}

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            results = model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=DEFAULT_CONF,
                device=DEFAULT_DEVICE,
                classes=list(COCO_VEHICLE_IDS),
                verbose=False,
            )
            if not results:
                raise InvalidVideoError("Video frame could not be analyzed")
            result = results[0]
            boxes = result.boxes
            vehicle_counts: Counter = Counter()
            current_ids: Set[int] = set()
            slow_count = 0

            if boxes is not None and boxes.id is not None:
                for track_id, class_id, xyxy in zip(
                    boxes.id.int().tolist(),
                    boxes.cls.int().tolist(),
                    boxes.xyxy.tolist(),
                ):
                    seen_ids.add(track_id)
                    current_ids.add(track_id)
                    class_name = result.names[class_id]
                    group = group_for_class(class_name) or class_name
                    vehicle_counts[group] += 1
                    track = tracks.setdefault(
                        track_id,
                        TrackInfo(track_id=track_id, group=group),
                    )
                    update_track(track, centroid(xyxy), SLOW_THRESHOLD)
                    if track.speed_pixels < SLOW_THRESHOLD:
                        slow_count += 1

            vehicles = sum(vehicle_counts.values())
            occupancy = calculate_occupancy(boxes, width, height)
            slow_ratio = slow_count / vehicles if vehicles > 0 else 0.0
            congestion_index = calculate_congestion_index(
                vehicles,
                occupancy,
                slow_ratio,
            )
            congestion_status = congestion_label(congestion_index)
            anomalies = detect_anomalies(tracks, STOPPED_FRAMES)
            peak_vehicles = max(peak_vehicles, vehicles)

            annotated = result.plot()
            draw_overlay(
                annotated,
                vehicle_counts,
                congestion_index,
                congestion_status,
                occupancy,
                slow_ratio,
                anomalies,
            )
            if writer is None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError("Could not create processed video")
            writer.write(annotated)

            frame_number = processed_frames + 1
            progress = (
                min(99, int(frame_number * 100 / total_frames))
                if total_frames > 0
                else 0
            )
            timestamp = (frame_number - 1) / fps
            last_metrics = {
                "type": "metrics",
                "frame": frame_number,
                "total_frames": total_frames if total_frames > 0 else None,
                "processed_frames": frame_number,
                "progress": progress,
                "unique_vehicles": len(seen_ids),
                "peak_vehicles": peak_vehicles,
                "vehicles": vehicles,
                "occupancy": round(float(occupancy), 2),
                "congestion_index": round(float(congestion_index), 2),
                "congestion_status": congestion_status,
                "anomalies": len(anomalies),
                "timestamp": round(timestamp, 2),
            }
            if processed_frames % PREVIEW_EVERY == 0:
                encoded, jpeg = cv2.imencode(
                    ".jpg",
                    annotated,
                    [cv2.IMWRITE_JPEG_QUALITY, 78],
                )
                if encoded:
                    last_metrics["frame_jpeg_base64"] = base64.b64encode(
                        jpeg.tobytes()
                    ).decode("ascii")
            emit(last_metrics)
            processed_frames += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    if processed_frames == 0:
        raise InvalidVideoError("Video contains no readable frames")
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError("Processed video was not written")

    return {
        "processed_frames": processed_frames,
        "unique_vehicles": len(seen_ids),
        "output_video_available": True,
        "final_metrics": last_metrics,
    }


job_manager = TrafficJobManager()