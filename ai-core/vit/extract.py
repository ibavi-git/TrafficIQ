"""
M6 - Vision Transformer (ViT) scene-level feature extraction for TrafficIQ.

This module extracts global visual context representations from sampled
video frames using a pretrained Vision Transformer (google/vit-base-patch16-224)
on CPU, preparing embeddings for downstream YOLO + ViT fusion (M7).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Suppress extraneous warning banners
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()

# Optional RAM tracking if psutil is available
try:
    import psutil
except ImportError:
    psutil = None

REPO_ROOT = Path(__file__).resolve().parents[2]
AI_CORE = REPO_ROOT / "ai-core"
if str(AI_CORE) not in sys.path:
    sys.path.insert(0, str(AI_CORE))

DEFAULT_MODEL = "google/vit-base-patch16-224"
DEFAULT_SAMPLE_EVERY = 10
DEFAULT_BATCH_SIZE = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TrafficIQ M6 — Vision Transformer Feature Extraction"
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to input video file (e.g. samples/traffic2.mp4)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Pretrained ViT model identifier (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=DEFAULT_SAMPLE_EVERY,
        help=f"Sample every N-th frame (default: {DEFAULT_SAMPLE_EVERY})",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of video frames to consider (default: complete video)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for ViT inference (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save outputs (default: outputs/vit)",
    )

    args = parser.parse_args()

    if args.sample_every <= 0:
        parser.error("--sample-every must be greater than zero")
    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be greater than zero")

    return args


def extract_vit_features(
    source_path: Path,
    model_name: str = DEFAULT_MODEL,
    sample_every: int = DEFAULT_SAMPLE_EVERY,
    max_frames: Optional[int] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Extract ViT CLS token embeddings for sampled frames of a video."""
    source = Path(source_path)
    if not source.is_file() and (REPO_ROOT / source).is_file():
        source = REPO_ROOT / source

    if not source.is_file():
        raise SystemExit(f"Source video not found: {source}")

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise SystemExit(f"Could not open video file: {source}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if output_dir is None:
        out_dir = REPO_ROOT / "outputs" / "vit"
    else:
        out_dir = Path(output_dir)
        if not out_dir.is_absolute():
            out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ViT model and processor loading
    device = torch.device("cpu")
    model_load_started = time.perf_counter()
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    model_load_time = time.perf_counter() - model_load_started

    hidden_size = int(model.config.hidden_size)

    # Frame extraction and batch inference
    sampled_indices: List[int] = []
    sampled_timestamps: List[float] = []
    batch_rgb_frames: List[np.ndarray] = []
    all_embeddings: List[np.ndarray] = []
    total_inference_time = 0.0

    process = psutil.Process(os.getpid()) if psutil is not None else None
    peak_ram_mb = (
        process.memory_info().rss / (1024 * 1024) if process is not None else None
    )

    def _process_batch(frames: List[np.ndarray]) -> None:
        nonlocal total_inference_time, peak_ram_mb
        if not frames:
            return
        t0 = time.perf_counter()
        inputs = image_processor(images=frames, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            # CLS token representation from last_hidden_state[:, 0, :]
            cls_repr = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        total_inference_time += time.perf_counter() - t0
        all_embeddings.append(cls_repr)

        if process is not None:
            current_ram = process.memory_info().rss / (1024 * 1024)
            if peak_ram_mb is None or current_ram > peak_ram_mb:
                peak_ram_mb = current_ram

    frame_idx = 0
    try:
        while max_frames is None or frame_idx < max_frames:
            ok, frame_bgr = capture.read()
            if not ok:
                break

            if frame_idx % sample_every == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                batch_rgb_frames.append(frame_rgb)
                sampled_indices.append(frame_idx)
                timestamp = frame_idx / fps if fps > 0 else 0.0
                sampled_timestamps.append(timestamp)

                if len(batch_rgb_frames) >= batch_size:
                    _process_batch(batch_rgb_frames)
                    batch_rgb_frames = []

            frame_idx += 1

        # Process any remaining frames in last partial batch
        if batch_rgb_frames:
            _process_batch(batch_rgb_frames)
            batch_rgb_frames = []
    finally:
        capture.release()

    if all_embeddings:
        embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    else:
        embeddings = np.empty((0, hidden_size), dtype=np.float32)

    frame_indices_arr = np.array(sampled_indices, dtype=np.int32)
    timestamps_arr = np.array(sampled_timestamps, dtype=np.float64)

    sampled_count = len(sampled_indices)
    average_ms_per_sample = (
        (total_inference_time * 1000.0) / sampled_count if sampled_count > 0 else 0.0
    )
    samples_per_second = (
        sampled_count / total_inference_time if total_inference_time > 0 else 0.0
    )

    # Save NPZ embeddings
    npz_filename = f"{source.stem}_embeddings.npz"
    npz_path = out_dir / npz_filename
    np.savez_compressed(
        npz_path,
        frame_indices=frame_indices_arr,
        timestamps=timestamps_arr,
        embeddings=embeddings,
    )

    # Relative path for results JSON
    rel_embedding_file = f"outputs/vit/{npz_filename}"
    rel_results_file = "outputs/vit/m6_vit_results.json"
    results_path = out_dir / "m6_vit_results.json"

    # Format JSON with exact schema specified in M6 requirements
    formatted_json = f"""{{
    "project": "TrafficIQ",
    "stage": "M6",
    "model": "{model_name}",
    "device": "cpu",
    "video": {{
        "width": {width},
        "height": {height},
        "fps": {round(fps, 2)},
        "frames": {total_frames}
    }},
    "sampling": {{
        "sample_every": {sample_every},
        "sampled_frames": {sampled_count}
    }},
    "model": {{
        "name": "{model_name}",
        "hidden_size": {hidden_size},
        "embedding_dimension": {hidden_size}
    }},
    "performance": {{
        "model_load_time_seconds": {round(model_load_time, 4)},
        "inference_time_seconds": {round(total_inference_time, 4)},
        "average_ms_per_sample": {round(average_ms_per_sample, 2)},
        "samples_per_second": {round(samples_per_second, 2)}{f',\n        "peak_ram_mb": {round(peak_ram_mb, 2)}' if peak_ram_mb is not None else ''}
    }},
    "outputs": {{
        "embedding_file": "{rel_embedding_file}"
    }}
}}
"""

    with results_path.open("w", encoding="utf-8") as f:
        f.write(formatted_json)

    return {
        "model": model_name,
        "device": "CPU",
        "video": {
            "width": width,
            "height": height,
            "fps": fps,
            "frames": total_frames,
        },
        "sampling": {
            "sample_every": sample_every,
            "sampled_frames": sampled_count,
        },
        "model_info": {
            "hidden_size": hidden_size,
            "embedding_dimension": hidden_size,
        },
        "performance": {
            "model_load_time_seconds": model_load_time,
            "inference_time_seconds": total_inference_time,
            "average_ms_per_sample": average_ms_per_sample,
            "samples_per_second": samples_per_second,
        },
        "outputs": {
            "embedding_file": rel_embedding_file,
            "results_file": rel_results_file,
        },
        "embeddings": embeddings,
        "frame_indices": frame_indices_arr,
        "timestamps": timestamps_arr,
    }


def print_report(results: Dict[str, Any]) -> None:
    video = results["video"]
    sampling = results["sampling"]
    model_info = results["model_info"]
    perf = results["performance"]
    outputs = results["outputs"]

    print("=" * 60)
    print("TrafficIQ M6 — Vision Transformer")
    print("=" * 60)
    print(f"\nModel: {results['model']}")
    print(f"Device: {results['device']}")
    print("\nVIDEO")
    print("-----")
    print(f"Resolution: {video['width']} x {video['height']}")
    print(f"FPS: {video['fps']:.2f}")
    print(f"Frames: {video['frames']}")
    print("\nSAMPLING")
    print("--------")
    print(f"Sample every: {sampling['sample_every']} frames")
    print(f"Sampled frames: {sampling['sampled_frames']}")
    print("\nMODEL")
    print("-----")
    print(f"Embedding dimension: {model_info['embedding_dimension']}")
    print("\nPERFORMANCE")
    print("-----------")
    print(f"Model load time: {perf['model_load_time_seconds']:.2f} sec")
    print(f"Inference time: {perf['inference_time_seconds']:.2f} sec")
    print(f"Average time/sample: {perf['average_ms_per_sample']:.2f} ms")
    print(f"Samples/second: {perf['samples_per_second']:.2f} FPS")
    print("\nOUTPUT")
    print("------")
    print("Embeddings:")
    print(outputs["embedding_file"])
    print("\nResults:")
    print(outputs["results_file"])
    print("\n" + "=" * 60)


def main() -> None:
    args = parse_args()
    results = extract_vit_features(
        source_path=args.source,
        model_name=args.model,
        sample_every=args.sample_every,
        max_frames=args.max_frames,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )
    print_report(results)


if __name__ == "__main__":
    main()
