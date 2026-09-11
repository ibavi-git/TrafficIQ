"""M1 — YOLO nano vehicle detection on an image or video (CPU only).

Usage (from repo root):
    python ai-core/detection/detect.py --source samples/traffic.jpg
    python ai-core/detection/detect.py --source samples/traffic.mp4 --output outputs/detected.mp4
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import List, Tuple

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

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO nano vehicle detection (CPU)")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("samples/traffic.mp4"),
        help="Image or video path (default: samples/traffic.mp4)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Annotated output path. Defaults to outputs/detected_<name>.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ultralytics model name or .pt path")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="Confidence threshold")
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print counts only; do not write an annotated file",
    )
    return parser.parse_args()


def default_output(source: Path) -> Path:
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = source.suffix if source.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS else ".mp4"
    return out_dir / f"detected_{source.stem}{suffix}"


def names_from_result(result) -> Tuple[List[str], List[float]]:
    names: List[str] = []
    confs: List[float] = []
    boxes = result.boxes
    if boxes is None or boxes.cls is None:
        return names, confs
    id_to_name = result.names
    for cls_id, conf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
        names.append(id_to_name[int(cls_id)])
        confs.append(float(conf))
    return names, confs


def print_summary(title: str, names: List[str], confs: List[float]) -> None:
    print(title)
    if not names:
        print("  (no vehicle detections)")
        return
    grouped = Counter(group_for_class(n) or n for n in names)
    print("  per-class counts:", dict(grouped))
    pairs = ", ".join(f"{n} {c:.2f}" for n, c in zip(names, confs))
    print(f"  detections ({len(names)}): {pairs}")


def annotate_and_maybe_write(result, save_path: Path | None, is_image: bool) -> None:
    if save_path is None:
        return
    frame = result.plot()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if is_image:
        cv2.imwrite(str(save_path), frame)


def run_image(model: YOLO, source: Path, conf: float, output: Path | None) -> None:
    results = model.predict(
        source=str(source),
        conf=conf,
        device=DEFAULT_DEVICE,
        classes=list(COCO_VEHICLE_IDS),
        verbose=False,
    )
    result = results[0]
    names, confs = names_from_result(result)
    print_summary(f"Image: {source}", names, confs)
    if output is not None:
        annotate_and_maybe_write(result, output, is_image=True)
        print(f"Saved annotated image → {output}")


def run_video(model: YOLO, source: Path, conf: float, output: Path | None) -> None:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {source}")

    writer = None
    frame_index = 0
    last_names: List[str] = []
    last_confs: List[float] = []
    peak_grouped: Counter[str] = Counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            results = model.predict(
                source=frame,
                conf=conf,
                device=DEFAULT_DEVICE,
                classes=list(COCO_VEHICLE_IDS),
                verbose=False,
            )
            result = results[0]
            names, confs = names_from_result(result)
            last_names, last_confs = names, confs
            grouped = Counter(group_for_class(n) or n for n in names)
            for key, value in grouped.items():
                peak_grouped[key] = max(peak_grouped[key], value)

            plotted = result.plot()
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
    print("  peak per-class counts in a single frame:", dict(peak_grouped))
    print_summary("  last frame", last_names, last_confs)
    if output is not None:
        print(f"Saved annotated video → {output}")


def main() -> None:
    args = parse_args()
    source: Path = args.source
    if not source.exists():
        raise SystemExit(
            f"Source not found: {source}\n"
            "Pass --source <path-to-image-or-video>. "
            "A placeholder default is samples/traffic.mp4."
        )

    output = None if args.no_save else (args.output or default_output(source))
    model = YOLO(args.model)

    ext = source.suffix.lower()
    if ext in IMAGE_EXTS:
        run_image(model, source, args.conf, output)
    elif ext in VIDEO_EXTS:
        run_video(model, source, args.conf, output)
    else:
        raise SystemExit(f"Unsupported file type: {ext}")


if __name__ == "__main__":
    main()
