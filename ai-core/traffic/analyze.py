"""
M4 — Traffic congestion and anomaly analysis.

TrafficIQ pipeline:

    Video
      ↓
    YOLOv8n
      ↓
    ByteTrack
      ↓
    Persistent vehicle IDs
      ↓
    Traffic statistics
      ↓
    Congestion analysis
      ↓
    Anomaly detection

Usage from repository root:

    python ai-core/traffic/analyze.py --source samples/traffic.mp4

With output:

    python ai-core/traffic/analyze.py \
        --source samples/traffic.mp4 \
        --output outputs/analyzed_traffic.mp4
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
from ultralytics import YOLO


# -------------------------------------------------------------------
# PATH SETUP
# -------------------------------------------------------------------

AI_CORE = Path(__file__).resolve().parents[1]

if str(AI_CORE) not in sys.path:
    sys.path.insert(0, str(AI_CORE))


from detection.classes import (  # noqa: E402
    COCO_VEHICLE_IDS,
    COUNT_GROUPS,
    DEFAULT_CONF,
    DEFAULT_DEVICE,
    DEFAULT_MODEL,
    group_for_class,
)


# -------------------------------------------------------------------
# TYPES
# -------------------------------------------------------------------

Point = Tuple[float, float]


# -------------------------------------------------------------------
# TRACK INFORMATION
# -------------------------------------------------------------------

@dataclass
class TrackInfo:
    """
    Stores the movement history of one tracked vehicle.
    """

    track_id: int

    group: str

    previous_center: Optional[Point] = None

    current_center: Optional[Point] = None

    total_distance: float = 0.0

    speed_pixels: float = 0.0

    frames_stationary: int = 0

    frames_seen: int = 0

    max_speed_pixels: float = 0.0


# -------------------------------------------------------------------
# ARGUMENTS
# -------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TrafficIQ M4 — congestion and anomaly analysis"
    )

    parser.add_argument(
        "--source",
        type=Path,
        default=Path("samples/traffic.mp4"),
        help="Input traffic video",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output annotated video",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="YOLO model name or .pt path",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONF,
        help="YOLO confidence threshold",
    )

    parser.add_argument(
        "--slow-threshold",
        type=float,
        default=2.0,
        help="Movement threshold in pixels/frame below which a vehicle is considered slow",
    )

    parser.add_argument(
        "--stopped-frames",
        type=int,
        default=30,
        help="Number of consecutive slow frames required for stopped-vehicle anomaly",
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save annotated output video",
    )

    return parser.parse_args()


# -------------------------------------------------------------------
# DEFAULT OUTPUT
# -------------------------------------------------------------------

def default_output(source: Path) -> Path:
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir / f"analyzed_{source.stem}.mp4"


# -------------------------------------------------------------------
# GEOMETRY
# -------------------------------------------------------------------

def centroid(xyxy) -> Point:
    """
    Calculate the center point of a bounding box.
    """

    x1, y1, x2, y2 = xyxy

    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
    )


def box_area(xyxy) -> float:
    """
    Calculate bounding-box area.
    """

    x1, y1, x2, y2 = xyxy

    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)

    return width * height


# -------------------------------------------------------------------
# CONGESTION
# -------------------------------------------------------------------

def calculate_occupancy(
    boxes,
    frame_width: int,
    frame_height: int,
) -> float:
    """
    Estimate road occupancy using the total detected vehicle
    bounding-box area.

    Returns:
        occupancy percentage from 0 to 100.
    """

    if boxes is None or len(boxes) == 0:
        return 0.0

    frame_area = float(frame_width * frame_height)

    if frame_area <= 0:
        return 0.0

    total_vehicle_area = 0.0

    for xyxy in boxes.xyxy.tolist():
        total_vehicle_area += box_area(xyxy)

    occupancy = (total_vehicle_area / frame_area) * 100.0

    return min(occupancy, 100.0)


def calculate_density(
    vehicle_count: int,
    frame_width: int,
    frame_height: int,
) -> float:
    """
    Calculate a normalized vehicle density.

    This is a simple frame-level density measure.
    """

    frame_area = frame_width * frame_height

    if frame_area <= 0:
        return 0.0

    # Number of vehicles per megapixel.
    density = vehicle_count / (frame_area / 1_000_000)

    return density


def calculate_congestion_index(
    vehicle_count: int,
    occupancy: float,
    slow_ratio: float,
) -> float:
    """
    Calculate an interpretable congestion index from 0–100.

    Components:

        vehicle count  -> 40%
        occupancy      -> 35%
        slow ratio     -> 25%

    These are initial engineering weights and should be
    calibrated later using real traffic data.
    """

    # Normalize vehicle count.
    #
    # 20 vehicles is treated as a high-density reference point
    # for the initial prototype.
    count_score = min(vehicle_count / 20.0, 1.0) * 100.0

    occupancy_score = min(occupancy / 20.0, 1.0) * 100.0

    slow_score = min(slow_ratio, 1.0) * 100.0

    congestion = (
        0.40 * count_score
        + 0.35 * occupancy_score
        + 0.25 * slow_score
    )

    return min(congestion, 100.0)


def congestion_label(index: float) -> str:
    """
    Convert congestion index to a human-readable category.
    """

    if index < 25:
        return "LOW"

    if index < 50:
        return "MODERATE"

    if index < 75:
        return "HIGH"

    return "SEVERE"


# -------------------------------------------------------------------
# TRACK ANALYSIS
# -------------------------------------------------------------------

def update_track(
    track: TrackInfo,
    center: Point,
    slow_threshold: float,
) -> None:
    """
    Update movement information for a tracked vehicle.
    """

    track.current_center = center
    track.frames_seen += 1

    if track.previous_center is None:
        track.previous_center = center
        track.speed_pixels = 0.0
        return

    previous_x, previous_y = track.previous_center
    current_x, current_y = center

    dx = current_x - previous_x
    dy = current_y - previous_y

    distance = (dx * dx + dy * dy) ** 0.5

    track.speed_pixels = distance

    track.total_distance += distance

    track.max_speed_pixels = max(
        track.max_speed_pixels,
        distance,
    )

    if distance < slow_threshold:
        track.frames_stationary += 1
    else:
        track.frames_stationary = 0

    track.previous_center = center


# -------------------------------------------------------------------
# ANOMALY DETECTION
# -------------------------------------------------------------------

def detect_anomalies(
    tracks: Dict[int, TrackInfo],
    stopped_frames: int,
) -> List[str]:
    """
    Detect basic traffic anomalies.

    Current anomaly:

        stopped vehicle

    Returns a list of human-readable anomaly messages.
    """

    anomalies: List[str] = []

    for track_id, track in tracks.items():

        if track.frames_stationary >= stopped_frames:

            anomalies.append(
                f"Stopped vehicle: ID {track_id}"
            )

    return anomalies


# -------------------------------------------------------------------
# DRAWING
# -------------------------------------------------------------------

def draw_text(
    frame,
    text: str,
    position: Tuple[int, int],
    scale: float = 0.65,
    thickness: int = 2,
) -> None:
    """
    Draw readable text with a black outline.
    """

    x, y = position

    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness + 2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


def draw_overlay(
    frame,
    vehicle_counts: Counter,
    congestion_index: float,
    congestion_status: str,
    occupancy: float,
    slow_ratio: float,
    anomalies: List[str],
) -> None:
    """
    Draw M4 traffic-analysis information on the video.
    """

    total = sum(vehicle_counts.values())

    y = 30

    draw_text(
        frame,
        f"Vehicles: {total}",
        (12, y),
    )

    y += 28

    for group in COUNT_GROUPS:
        draw_text(
            frame,
            f"{group}: {vehicle_counts.get(group, 0)}",
            (12, y),
            scale=0.55,
        )

        y += 23

    y += 5

    draw_text(
        frame,
        f"Occupancy: {occupancy:.1f}%",
        (12, y),
    )

    y += 28

    draw_text(
        frame,
        f"Slow vehicles: {slow_ratio * 100:.1f}%",
        (12, y),
    )

    y += 28

    draw_text(
        frame,
        f"Congestion: {congestion_index:.1f} ({congestion_status})",
        (12, y),
    )

    # ---------------------------------------------------------------
    # ANOMALIES
    # ---------------------------------------------------------------

    if anomalies:

        y += 35

        draw_text(
            frame,
            "ANOMALIES:",
            (12, y),
            scale=0.65,
        )

        y += 25

        # Show only the first 3 anomalies to avoid covering
        # the entire video.
        for anomaly in anomalies[:3]:

            draw_text(
                frame,
                anomaly,
                (12, y),
                scale=0.55,
            )

            y += 23


# -------------------------------------------------------------------
# MAIN VIDEO PROCESSING
# -------------------------------------------------------------------

def main() -> None:

    args = parse_args()

    source: Path = args.source

    # ---------------------------------------------------------------
    # CHECK SOURCE
    # ---------------------------------------------------------------

    if not source.exists():

        raise SystemExit(
            f"Source not found: {source}\n"
            "Use --source <path-to-video>."
        )

    # ---------------------------------------------------------------
    # OUTPUT
    # ---------------------------------------------------------------

    output = None

    if not args.no_save:
        output = args.output or default_output(source)

    # ---------------------------------------------------------------
    # LOAD MODEL
    # ---------------------------------------------------------------

    print()
    print("=" * 60)
    print("TrafficIQ M4 — Traffic Analysis")
    print("=" * 60)
    print()

    print(f"Model: {args.model}")
    print(f"Source: {source}")
    print(f"Device: {DEFAULT_DEVICE}")
    print()

    model = YOLO(args.model)

    # ---------------------------------------------------------------
    # OPEN VIDEO
    # ---------------------------------------------------------------

    cap = cv2.VideoCapture(str(source))

    if not cap.isOpened():

        raise SystemExit(
            f"Could not open video: {source}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    print(f"Resolution: {width} x {height}")
    print(f"FPS: {fps:.2f}")
    print(f"Frames: {total_frames}")
    print()

    # ---------------------------------------------------------------
    # VIDEO WRITER
    # ---------------------------------------------------------------

    writer = None

    # ---------------------------------------------------------------
    # TRACK STORAGE
    # ---------------------------------------------------------------

    tracks: Dict[int, TrackInfo] = {}

    seen_ids = set()

    # ---------------------------------------------------------------
    # M4 STATISTICS
    # ---------------------------------------------------------------

    peak_vehicle_count = 0

    peak_occupancy = 0.0

    peak_congestion = 0.0

    total_anomaly_frames = 0

    processed_frames = 0

    # ---------------------------------------------------------------
    # PROCESS VIDEO
    # ---------------------------------------------------------------

    try:

        while True:

            ok, frame = cap.read()

            if not ok:
                break

            # -------------------------------------------------------
            # YOLO + BYTE TRACK
            # -------------------------------------------------------

            results = model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=args.conf,
                device=DEFAULT_DEVICE,
                classes=list(COCO_VEHICLE_IDS),
                verbose=False,
            )

            result = results[0]

            boxes = result.boxes

            # -------------------------------------------------------
            # CURRENT FRAME DATA
            # -------------------------------------------------------

            vehicle_counts: Counter = Counter()

            current_track_ids = set()

            total_slow = 0

            # -------------------------------------------------------
            # PROCESS TRACKED VEHICLES
            # -------------------------------------------------------

            if (
                boxes is not None
                and boxes.id is not None
            ):

                track_ids = boxes.id.int().tolist()

                class_ids = boxes.cls.int().tolist()

                xyxys = boxes.xyxy.tolist()

                for track_id, cls_id, xyxy in zip(
                    track_ids,
                    class_ids,
                    xyxys,
                ):

                    # ------------------------------------------------
                    # TRACK ID
                    # ------------------------------------------------

                    seen_ids.add(track_id)

                    current_track_ids.add(track_id)

                    # ------------------------------------------------
                    # VEHICLE GROUP
                    # ------------------------------------------------

                    class_name = result.names[cls_id]

                    group = (
                        group_for_class(class_name)
                        or class_name
                    )

                    vehicle_counts[group] += 1

                    # ------------------------------------------------
                    # CREATE TRACK
                    # ------------------------------------------------

                    if track_id not in tracks:

                        tracks[track_id] = TrackInfo(
                            track_id=track_id,
                            group=group,
                        )

                    track = tracks[track_id]

                    # ------------------------------------------------
                    # UPDATE TRACK MOVEMENT
                    # ------------------------------------------------

                    center = centroid(xyxy)

                    update_track(
                        track,
                        center,
                        args.slow_threshold,
                    )

                    # ------------------------------------------------
                    # SLOW VEHICLE
                    # ------------------------------------------------

                    if (
                        track.speed_pixels
                        < args.slow_threshold
                    ):
                        total_slow += 1

            # -------------------------------------------------------
            # FRAME STATISTICS
            # -------------------------------------------------------

            total_vehicles = sum(
                vehicle_counts.values()
            )

            occupancy = calculate_occupancy(
                boxes,
                width,
                height,
            )

            if total_vehicles > 0:

                slow_ratio = (
                    total_slow / total_vehicles
                )

            else:

                slow_ratio = 0.0

            density = calculate_density(
                total_vehicles,
                width,
                height,
            )

            congestion_index = calculate_congestion_index(
                total_vehicles,
                occupancy,
                slow_ratio,
            )

            congestion_status = congestion_label(
                congestion_index
            )

            # -------------------------------------------------------
            # ANOMALIES
            # -------------------------------------------------------

            anomalies = detect_anomalies(
                tracks,
                args.stopped_frames,
            )

            if anomalies:
                total_anomaly_frames += 1

            # -------------------------------------------------------
            # UPDATE PEAK VALUES
            # -------------------------------------------------------

            peak_vehicle_count = max(
                peak_vehicle_count,
                total_vehicles,
            )

            peak_occupancy = max(
                peak_occupancy,
                occupancy,
            )

            peak_congestion = max(
                peak_congestion,
                congestion_index,
            )

            # -------------------------------------------------------
            # DRAW YOLO RESULT
            # -------------------------------------------------------

            plotted = result.plot()

            # -------------------------------------------------------
            # DRAW M4 INFORMATION
            # -------------------------------------------------------

            draw_overlay(
                plotted,
                vehicle_counts,
                congestion_index,
                congestion_status,
                occupancy,
                slow_ratio,
                anomalies,
            )

            # -------------------------------------------------------
            # SAVE FRAME
            # -------------------------------------------------------

            if output is not None:

                if writer is None:

                    h, w = plotted.shape[:2]

                    fourcc = cv2.VideoWriter_fourcc(
                        *"mp4v"
                    )

                    output.parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    writer = cv2.VideoWriter(
                        str(output),
                        fourcc,
                        fps,
                        (w, h),
                    )

                writer.write(plotted)

            processed_frames += 1

            # -------------------------------------------------------
            # PROGRESS
            # -------------------------------------------------------

            if processed_frames % 50 == 0:

                print(
                    f"Processed frames: "
                    f"{processed_frames}"
                )

    finally:

        cap.release()

        if writer is not None:
            writer.release()

    # ----------------------------------------------------------------
    # FINAL SUMMARY
    # ----------------------------------------------------------------

    print()
    print("=" * 60)
    print("M4 TRAFFIC ANALYSIS RESULTS")
    print("=" * 60)

    print(
        f"Processed frames: {processed_frames}"
    )

    print(
        f"Unique vehicles seen: {len(seen_ids)}"
    )

    print(
        f"Peak vehicles in one frame: "
        f"{peak_vehicle_count}"
    )

    print(
        f"Peak occupancy: "
        f"{peak_occupancy:.2f}%"
    )

    print(
        f"Peak congestion index: "
        f"{peak_congestion:.2f}"
    )

    print(
        f"Peak congestion status: "
        f"{congestion_label(peak_congestion)}"
    )

    print(
        f"Frames containing anomalies: "
        f"{total_anomaly_frames}"
    )

    print(
        f"Final density: "
        f"{density:.2f} vehicles/MP"
    )

    if output is not None:

        print(
            f"Saved annotated video → {output}"
        )

    print("=" * 60)
if __name__ == "__main__":
    main()