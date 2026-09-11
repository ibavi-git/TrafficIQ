"""M3 — Line-crossing vehicle counts from YOLO + ByteTrack (CPU only).

Each persistent track ID is counted at most once, the first time its
bounding-box centroid crosses the counting line.

Usage (from repo root):
    python ai-core/tracking/count.py --source samples/traffic.mp4
    python ai-core/tracking/count.py --source samples/traffic.mp4 --line-norm 0,0.5,1,0.5
    python ai-core/tracking/count.py --source samples/traffic.mp4 --line 100,360,1180,360
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import cv2
from ultralytics import YOLO

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

Point = Tuple[float, float]
Line = Tuple[float, float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Line-crossing vehicle counting (CPU)")
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
        help="Annotated output video. Defaults to outputs/counted_<name>.mp4",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ultralytics model name or .pt path")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="Confidence threshold")
    parser.add_argument(
        "--line",
        default=None,
        help="Counting line in pixels: x1,y1,x2,y2",
    )
    parser.add_argument(
        "--line-norm",
        default="0,0.5,1,0.5",
        help="Counting line in normalized coords (0–1): x1,y1,x2,y2 "
        "(default: horizontal midline)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print counts only; do not write an annotated video",
    )
    return parser.parse_args()


def default_output(source: Path) -> Path:
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"counted_{source.stem}.mp4"


def parse_line(text: str) -> Line:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 4:
        raise SystemExit(f"Line must be x1,y1,x2,y2 — got: {text}")
    try:
        x1, y1, x2, y2 = (float(p) for p in parts)
    except ValueError as exc:
        raise SystemExit(f"Line coordinates must be numbers: {text}") from exc
    return x1, y1, x2, y2


def resolve_line(args: argparse.Namespace, width: int, height: int) -> Line:
    if args.line:
        return parse_line(args.line)
    nx1, ny1, nx2, ny2 = parse_line(args.line_norm)
    return nx1 * width, ny1 * height, nx2 * width, ny2 * height


def centroid(xyxy) -> Point:
    x1, y1, x2, y2 = xyxy
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _side(px: float, py: float, line: Line) -> float:
    x1, y1, x2, y2 = line
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


def crossed_segment(prev: Point, curr: Point, line: Line) -> bool:
    """True if the movement segment intersects the finite counting-line segment."""
    s1 = _side(prev[0], prev[1], line)
    s2 = _side(curr[0], curr[1], line)
    if s1 == 0.0 and s2 == 0.0:
        return False
    if s1 * s2 > 0:
        return False
    t1 = _side(line[0], line[1], (prev[0], prev[1], curr[0], curr[1]))
    t2 = _side(line[2], line[3], (prev[0], prev[1], curr[0], curr[1]))
    return t1 * t2 <= 0


def empty_counts() -> Dict[str, int]:
    return {name: 0 for name in COUNT_GROUPS}


def draw_overlay(frame, line: Line, counts: Dict[str, int], unique_ids: int):
    x1, y1, x2, y2 = map(int, line)
    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        frame,
        "count line",
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    total = sum(counts.values())
    lines = [f"{name}: {counts[name]}" for name in COUNT_GROUPS]
    lines.append(f"total: {total}   unique IDs seen: {unique_ids}")
    y = 28
    for text in lines:
        cv2.putText(
            frame,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        y += 26
    return frame


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

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    line: Optional[Line] = None
    writer = None
    prev_centroid: Dict[int, Point] = {}
    counted: Set[int] = set()
    counts = empty_counts()
    seen_ids: Set[int] = set()
    frame_index = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if line is None:
                h, w = frame.shape[:2]
                width, height = w, h
                line = resolve_line(args, width, height)

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
                for track_id, cls_id, xyxy in zip(
                    boxes.id.int().tolist(),
                    boxes.cls.int().tolist(),
                    boxes.xyxy.tolist(),
                ):
                    seen_ids.add(track_id)
                    group = group_for_class(result.names[cls_id])
                    center = centroid(xyxy)
                    prev = prev_centroid.get(track_id)
                    if (
                        prev is not None
                        and group in counts
                        and track_id not in counted
                        and crossed_segment(prev, center, line)
                    ):
                        counted.add(track_id)
                        counts[group] += 1
                    prev_centroid[track_id] = center

            plotted = result.plot()
            draw_overlay(plotted, line, counts, len(seen_ids))
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

    print(f"Video: {source} ({frame_index} frames)")
    print(f"  counting line (pixels): {tuple(round(v, 1) for v in (line or ()))}")
    print(f"  unique track IDs seen: {len(seen_ids)}")
    print(f"  line-crossing counts: {counts}")
    print(f"  total crossed: {sum(counts.values())}")
    if output is not None:
        print(f"Saved annotated video → {output}")


if __name__ == "__main__":
    main()
