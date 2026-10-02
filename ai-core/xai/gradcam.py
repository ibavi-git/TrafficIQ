"""M8 Grad-CAM visualizations for the pretrained M6 ViT representation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "google/vit-base-patch16-224"
METHOD_NOTE = (
    "This visualization explains the pretrained ViT representation and is not a "
    "validated traffic-congestion classifier explanation."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TrafficIQ M8 - ViT representation Grad-CAM visualization"
    )
    parser.add_argument("--source", type=Path, required=True, help="Input video path")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Pretrained ViT model identifier (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--sample-every", type=int, default=20, help="Sample every N-th frame"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=20,
        help="Maximum raw video frames to consider",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: outputs/xai)",
    )
    args = parser.parse_args()
    if args.sample_every <= 0:
        parser.error("--sample-every must be greater than zero")
    if args.max_frames <= 0:
        parser.error("--max-frames must be greater than zero")
    return args


def _resolve_target_layer(model: torch.nn.Module) -> Tuple[torch.nn.Module, str]:
    """Find the final block's pre-attention normalization across HF ViT layouts."""
    if hasattr(model, "layers"):
        blocks = model.layers
        prefix = "layers"
    elif hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        blocks = model.encoder.layer
        prefix = "encoder.layer"
    elif hasattr(model, "vit") and hasattr(model.vit, "encoder"):
        blocks = model.vit.encoder.layer
        prefix = "vit.encoder.layer"
    else:
        raise RuntimeError("Could not locate ViT transformer blocks in AutoModel")

    block_index = len(blocks) - 1
    block = blocks[block_index]
    for name in ("layernorm_before", "layernorm1", "norm1"):
        if hasattr(block, name):
            return getattr(block, name), f"{prefix}.{block_index}.{name}"
    raise RuntimeError("Could not locate pre-attention normalization in final ViT block")


def _gradcam_for_frame(
    frame_rgb: np.ndarray,
    image_processor: Any,
    model: torch.nn.Module,
    target_layer: torch.nn.Module,
    patch_size: int,
) -> Tuple[np.ndarray, int, Tuple[int, int]]:
    captured: Dict[str, torch.Tensor] = {}

    def capture_activations(
        _module: torch.nn.Module, _inputs: Tuple[torch.Tensor, ...], output: Any
    ) -> None:
        activation = output[0] if isinstance(output, tuple) else output
        captured["activations"] = activation

        def capture_gradients(gradient: torch.Tensor) -> torch.Tensor:
            captured["gradients"] = gradient
            return gradient

        activation.register_hook(capture_gradients)

    handle = target_layer.register_forward_hook(capture_activations)
    try:
        inputs = image_processor(images=frame_rgb, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(torch.device("cpu"))
        model.zero_grad(set_to_none=True)
        outputs = model(pixel_values=pixel_values)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        target_component = int(cls_embedding.detach().abs().argmax(dim=-1).item())
        cls_embedding[0, target_component].backward()

        activations = captured["activations"][:, 1:, :]
        gradients = captured["gradients"][:, 1:, :]
        token_count = int(activations.shape[1])
        grid_size = int(round(token_count**0.5))
        if grid_size * grid_size != token_count:
            raise RuntimeError(f"Patch token count is not a square grid: {token_count}")

        channel_weights = gradients.mean(dim=1, keepdim=True)
        patch_map = torch.relu((channel_weights * activations).sum(dim=-1))
        patch_map = patch_map.reshape(grid_size, grid_size)
        heatmap = patch_map.detach().cpu().numpy().astype(np.float32)
        if not np.isfinite(heatmap).all():
            raise RuntimeError("Grad-CAM heatmap contains NaN or Inf values")

        heatmap_min = float(heatmap.min())
        heatmap_max = float(heatmap.max())
        if heatmap_max > heatmap_min:
            heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
        else:
            heatmap = np.zeros_like(heatmap)

        input_height, input_width = pixel_values.shape[-2:]
        if grid_size != input_height // patch_size or grid_size != input_width // patch_size:
            raise RuntimeError(
                "ViT patch grid does not match processor input size and patch size"
            )
        return heatmap, target_component, (input_height, input_width)
    finally:
        handle.remove()


def generate_gradcam(
    source_path: Path,
    model_name: str = DEFAULT_MODEL,
    sample_every: int = 20,
    max_frames: int = 20,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    source = Path(source_path)
    if not source.is_file() and (REPO_ROOT / source).is_file():
        source = REPO_ROOT / source
    if not source.is_file():
        raise SystemExit(f"Source video not found: {source}")
    if sample_every <= 0 or max_frames <= 0:
        raise ValueError("sample_every and max_frames must be greater than zero")

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise SystemExit(f"Could not open video file: {source}")

    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if output_dir is None:
        out_dir = REPO_ROOT / "outputs" / "xai"
    else:
        out_dir = Path(output_dir)
        if not out_dir.is_absolute():
            out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / "m8_gradcam_preview.mp4"
    results_path = out_dir / "m8_gradcam_results.json"

    device = torch.device("cpu")
    model_load_started = time.perf_counter()
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    model_load_time = time.perf_counter() - model_load_started
    target_layer, target_layer_name = _resolve_target_layer(model)
    patch_size = int(model.config.patch_size)

    sampled_indices: List[int] = []
    sampled_timestamps: List[float] = []
    processed_indices: List[int] = []
    processed_timestamps: List[float] = []
    target_components: List[int] = []
    errors: List[str] = []
    input_resolution: Optional[Tuple[int, int]] = None
    heatmap_dimensions: Optional[Tuple[int, int]] = None
    writer: Optional[cv2.VideoWriter] = None
    frame_idx = 0
    processing_started = time.perf_counter()

    try:
        while frame_idx < max_frames:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            if frame_idx % sample_every == 0:
                timestamp = frame_idx / fps if fps > 0 else 0.0
                sampled_indices.append(frame_idx)
                sampled_timestamps.append(timestamp)
                try:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    heatmap, target_component, resolution = _gradcam_for_frame(
                        frame_rgb,
                        image_processor,
                        model,
                        target_layer,
                        patch_size,
                    )
                    heatmap_dimensions = tuple(int(value) for value in heatmap.shape)
                    input_resolution = resolution
                    heatmap_u8 = np.uint8(np.clip(heatmap * 255.0, 0, 255))
                    resized_heatmap = cv2.resize(
                        heatmap_u8,
                        (frame_width, frame_height),
                        interpolation=cv2.INTER_CUBIC,
                    )
                    color_heatmap = cv2.applyColorMap(resized_heatmap, cv2.COLORMAP_JET)
                    overlay = cv2.addWeighted(frame_bgr, 0.58, color_heatmap, 0.42, 0)
                    combined = np.hstack((frame_bgr, overlay))
                    cv2.rectangle(combined, (0, 0), (combined.shape[1], 34), (24, 28, 30), -1)
                    cv2.putText(
                        combined,
                        "M8 ViT Representation Grad-CAM",
                        (12, 23),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.62,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        combined,
                        "Original",
                        (12, 58),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        combined,
                        "Representation sensitivity",
                        (frame_width + 12, 58),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

                    if writer is None:
                        output_fps = fps if np.isfinite(fps) and fps > 0 else 1.0
                        writer = cv2.VideoWriter(
                            str(video_path),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            output_fps,
                            (combined.shape[1], combined.shape[0]),
                        )
                        if not writer.isOpened():
                            raise RuntimeError(f"Could not create output video: {video_path}")
                    writer.write(combined)
                    processed_indices.append(frame_idx)
                    processed_timestamps.append(timestamp)
                    target_components.append(target_component)
                except Exception as exc:
                    errors.append(f"Frame {frame_idx}: {type(exc).__name__}: {exc}")
            frame_idx += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    processing_time = time.perf_counter() - processing_started
    if not processed_indices:
        raise RuntimeError(
            "No sampled frames were successfully processed. "
            + (errors[0] if errors else "The video contained no readable frames.")
        )

    results: Dict[str, Any] = {
        "project": "TrafficIQ",
        "stage": "M8",
        "method": "ViT Grad-CAM over final-block pre-attention patch activations",
        "target": "strongest-magnitude component of the CLS representation",
        "model": model_name,
        "target_layer": target_layer_name,
        "device": "cpu",
        "video": {
            "source": str(source),
            "frame_width": frame_width,
            "frame_height": frame_height,
            "fps": fps,
            "total_frames": total_video_frames,
        },
        "input_resolution": list(input_resolution) if input_resolution else None,
        "original_frame_resolution": [frame_height, frame_width],
        "sampling": {
            "sample_every": sample_every,
            "max_frames_considered": max_frames,
            "sampled_frame_indices": sampled_indices,
            "timestamps_seconds": sampled_timestamps,
        },
        "heatmap_dimensions": list(heatmap_dimensions) if heatmap_dimensions else None,
        "successfully_processed_frames": len(processed_indices),
        "processed_frame_indices": processed_indices,
        "processed_timestamps_seconds": processed_timestamps,
        "selected_cls_components": target_components,
        "performance": {
            "model_load_time_seconds": round(model_load_time, 4),
            "processing_time_seconds": round(processing_time, 4),
        },
        "outputs": {"video": str(video_path), "results": str(results_path)},
        "methodological_note": METHOD_NOTE,
        "errors": errors,
    }
    with results_path.open("w", encoding="utf-8") as result_file:
        json.dump(results, result_file, indent=2)
        result_file.write("\n")
    return results


def main() -> None:
    args = parse_args()
    results = generate_gradcam(
        source_path=args.source,
        model_name=args.model,
        sample_every=args.sample_every,
        max_frames=args.max_frames,
        output_dir=args.output_dir,
    )
    print(f"Model: {results['model']}")
    print(f"Target layer: {results['target_layer']}")
    print(f"Frames processed: {results['successfully_processed_frames']}")
    print(f"Heatmap dimensions: {results['heatmap_dimensions']}")
    print(f"Processing time: {results['performance']['processing_time_seconds']:.2f} sec")
    print(f"Video: {results['outputs']['video']}")
    print(f"Results: {results['outputs']['results']}")
    if results["errors"]:
        print(f"Warnings: {len(results['errors'])} frame(s) failed; see results JSON.")


if __name__ == "__main__":
    main()