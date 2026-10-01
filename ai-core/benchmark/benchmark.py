"""M5 - CPU performance benchmark for the TrafficIQ YOLO + ByteTrack baseline."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import psutil
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
AI_CORE = REPO_ROOT / "ai-core"
if str(AI_CORE) not in sys.path:
    sys.path.insert(0, str(AI_CORE))

from detection.classes import (  # noqa: E402
    COCO_VEHICLE_IDS,
    COUNT_GROUPS,
    DEFAULT_CONF,
    DEFAULT_MODEL,
    group_for_class,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TrafficIQ M5 CPU performance benchmark"
    )
    parser.add_argument("--source", type=Path, required=True, help="Input video path")
    parser.add_argument("--output", type=Path, default=None, help="Benchmark JSON path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="YOLO model name or .pt path")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process (default: complete video)",
    )
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be greater than zero")
    return args


class ResourceSampler:
    def __init__(self) -> None:
        self.process = psutil.Process(os.getpid())
        self.logical_cpus = psutil.cpu_count(logical=True) or 1
        self.initial_ram_mb = self.process.memory_info().rss / (1024 * 1024)
        self.peak_ram_mb = self.initial_ram_mb
        self.cpu_percent_samples: list[float] = []
        self.process.cpu_percent(interval=None)

    def reset_cpu_baseline(self) -> None:
        self.process.cpu_percent(interval=None)

    def sample(self) -> None:
        process_cpu = self.process.cpu_percent(interval=None)
        self.cpu_percent_samples.append(process_cpu / self.logical_cpus)
        ram_mb = self.process.memory_info().rss / (1024 * 1024)
        self.peak_ram_mb = max(self.peak_ram_mb, ram_mb)

    def results(self) -> dict[str, float]:
        cpu_average = (
            sum(self.cpu_percent_samples) / len(self.cpu_percent_samples)
            if self.cpu_percent_samples
            else 0.0
        )
        cpu_peak = max(self.cpu_percent_samples, default=0.0)
        return {
            "cpu_average_percent": cpu_average,
            "cpu_peak_percent": cpu_peak,
            "ram_initial_mb": self.initial_ram_mb,
            "ram_peak_mb": self.peak_ram_mb,
        }


def cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return str(name).strip()
        except OSError:
            pass
    return platform.processor() or platform.uname().processor or platform.machine()


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source
    if not source.is_file():
        raise SystemExit(f"Source video not found: {source}")

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {source}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / input_fps if input_fps > 0 else 0.0

    sampler = ResourceSampler()
    model_path = Path(args.model)
    if not model_path.is_file() and (REPO_ROOT / model_path).is_file():
        model_path = REPO_ROOT / model_path

    model_load_started = time.perf_counter()
    model = YOLO(str(model_path))
    model_loading_time = time.perf_counter() - model_load_started
    sampler.reset_cpu_baseline()

    seen_vehicles: dict[int, str] = {}
    peak_vehicles = 0
    frame_fps: list[float] = []
    processed_frames = 0
    max_frames = args.max_frames
    processing_started = time.perf_counter()

    try:
        while max_frames is None or processed_frames < max_frames:
            frame_started = time.perf_counter()
            ok, frame = capture.read()
            if not ok:
                break

            results = model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=DEFAULT_CONF,
                device="cpu",
                classes=list(COCO_VEHICLE_IDS),
                verbose=False,
            )
            result = results[0]
            boxes = result.boxes
            active_ids: list[int] = []
            if boxes is not None and boxes.id is not None:
                track_ids = boxes.id.int().tolist()
                class_ids = boxes.cls.int().tolist()
                active_ids = track_ids
                for track_id, class_id in zip(track_ids, class_ids):
                    class_name = result.names[class_id]
                    group = group_for_class(class_name)
                    if group is not None:
                        seen_vehicles[track_id] = group

            peak_vehicles = max(peak_vehicles, len(active_ids))
            processed_frames += 1
            frame_elapsed = time.perf_counter() - frame_started
            if frame_elapsed > 0:
                frame_fps.append(1.0 / frame_elapsed)
            sampler.sample()
    finally:
        capture.release()

    total_processing_time = time.perf_counter() - processing_started
    average_fps = (
        processed_frames / total_processing_time if total_processing_time > 0 else 0.0
    )
    average_ms_per_frame = (
        total_processing_time * 1000 / processed_frames if processed_frames else 0.0
    )
    grouped_vehicles = Counter(seen_vehicles.values())
    traffic = {
        "unique_vehicles": len(seen_vehicles),
        "peak_vehicles": peak_vehicles,
        **{group: grouped_vehicles.get(group, 0) for group in COUNT_GROUPS},
    }

    return {
        "project": "TrafficIQ",
        "stage": "M5",
        "model": args.model,
        "device": "cpu",
        "hardware": {
            "cpu_name": cpu_name(),
            "logical_cpu_count": os.cpu_count() or 1,
            "total_ram_gb": psutil.virtual_memory().total / (1024 ** 3),
        },
        "video": {
            "source": str(source),
            "width": width,
            "height": height,
            "fps": input_fps,
            "frames": total_frames,
            "duration_seconds": duration,
        },
        "performance": {
            "model_loading_time_seconds": model_loading_time,
            "total_processing_time_seconds": total_processing_time,
            "processed_frames": processed_frames,
            "average_fps": average_fps,
            "average_ms_per_frame": average_ms_per_frame,
            "minimum_fps": min(frame_fps, default=0.0),
            "maximum_fps": max(frame_fps, default=0.0),
        },
        "resources": sampler.results(),
        "traffic": traffic,
    }


def print_report(results: dict[str, Any], output: Path) -> None:
    hardware = results["hardware"]
    video = results["video"]
    performance = results["performance"]
    resources = results["resources"]
    traffic = results["traffic"]

    print("=" * 60)
    print("TrafficIQ M5 - Performance Benchmark")
    print("=" * 60)
    print(f"\nModel: {results['model']}\nDevice: CPU")
    print("\nHARDWARE\n--------")
    print(f"CPU: {hardware['cpu_name']}")
    print(f"Logical CPUs: {hardware['logical_cpu_count']}")
    print(f"RAM: {hardware['total_ram_gb']:.2f} GB")
    print("\nVIDEO\n-----")
    print(f"Resolution: {video['width']} x {video['height']}")
    print(f"Input FPS: {video['fps']:.2f}")
    print(f"Frames: {video['frames']} ({video['duration_seconds']:.2f} sec)")
    print("\nPERFORMANCE\n-----------")
    print(f"Processed frames: {performance['processed_frames']}")
    print(f"Model loading time: {performance['model_loading_time_seconds']:.2f} sec")
    print(f"Total processing time: {performance['total_processing_time_seconds']:.2f} sec")
    print(f"Average FPS: {performance['average_fps']:.2f} FPS")
    print(f"Average frame time: {performance['average_ms_per_frame']:.2f} ms")
    print(
        f"Per-frame FPS range: {performance['minimum_fps']:.2f} - "
        f"{performance['maximum_fps']:.2f} FPS"
    )
    print("\nRESOURCES\n---------")
    print(f"Average CPU: {resources['cpu_average_percent']:.2f} %")
    print(f"Peak CPU: {resources['cpu_peak_percent']:.2f} %")
    print(f"Initial RAM: {resources['ram_initial_mb']:.2f} MB")
    print(f"Peak RAM: {resources['ram_peak_mb']:.2f} MB")
    print("\nTRAFFIC\n-------")
    print(f"Unique vehicles: {traffic['unique_vehicles']}")
    print(f"Peak vehicles/frame: {traffic['peak_vehicles']}")
    print(f"Cars: {traffic['car']}")
    print(f"Bikes: {traffic['bike']}")
    print(f"Buses: {traffic['bus']}")
    print(f"Trucks: {traffic['truck']}")
    print(f"\nBenchmark saved to:\n{output}")
    print("=" * 60)


def main() -> None:
    args = parse_args()
    output = args.output or REPO_ROOT / "outputs" / "benchmark" / "m5_baseline.json"
    results = run_benchmark(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as result_file:
        json.dump(results, result_file, indent=4)
        result_file.write("\n")
    print_report(results, output)


if __name__ == "__main__":
    main()