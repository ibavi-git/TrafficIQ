"""M2 — YOLO nano + ByteTrack on video, persistent per-vehicle IDs (CPU only).

Uses Ultralytics' built-in ByteTrack (`model.track(..., tracker="bytetrack.yaml")`).

Usage (from repo root):
    python ai-core/tracking/track.py --source samples/traffic.mp4
    python ai-core/tracking/track.py --source samples/traffic.mp4 --output outputs/tracked.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Set

import cv2
from ultralytics import YOLO

AI_CORE = Path(__file__).resolve().parents[1]
if str(AI_CORE) not in sys.path:
    sys.path.insert(0, str(AI_CORE))

from detection.classes import (  # noqa: E402
    COCO_VEHICLE_IDS,
    DEFAULT_CONF,
    DEFAULT_DEVICE,
    DEFAULT_MODEL,
    group_for_class,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO + ByteTrack vehicle tracking (CPU)")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("samples/traffic.mp4"),
        help="Video path (default: samples/traffic.mp4)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Annotated output video. Defaults to outputs/tracked_<name>.mp4",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ultralytics model name or .pt path")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="Confidence threshold")
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print unique IDs only; do not write an annotated video",
    )
    return parser.parse_args()


def default_output(source: Path) -> Path:
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"tracked_{source.stem}.mp4"


def overlay_ids(result):
    """Ultralytics plot() already draws boxes + ByteTrack IDs when ids exist."""
    return result.plot()


def main() -> None:
    args = parse_args()
    source: Path = args.source
    if not source.exists():
        raise SystemExit(
            f"Source not found: {source}\n"
            "Pass --source <path-to-video>. Placeholder default: samples/traffic.mp4."
        )

    output = None if args.no_save else (args.output or default_output(source))
    model = YOLO(args.model)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {source}")

    writer = None
    seen_ids: Set[int] = set()
    id_to_group = {}
    frame_index = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

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
            if boxes is not None and boxes.id is not None:
                for track_id, cls_id in zip(boxes.id.int().tolist(), boxes.cls.int().tolist()):
                    seen_ids.add(track_id)
                    name = result.names[cls_id]
                    id_to_group[track_id] = group_for_class(name) or name

            plotted = overlay_ids(result)
            if output is not None:
                if writer is None:
                    h, w = plotted.shape[:2]
                    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    output.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(str(output), fourcc, fps, (w, h))
                writer.write(plotted)
            frame_index += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    grouped_counts = {}
    for group in id_to_group.values():
        grouped_counts[group] = grouped_counts.get(group, 0) + 1

    print(f"Video: {source} ({frame_index} frames)")
    print(f"  unique track IDs: {len(seen_ids)}")
    print(f"  unique vehicles by class: {grouped_counts}")
    if output is not None:
        print(f"Saved annotated video → {output}")


if __name__ == "__main__":
    main()
