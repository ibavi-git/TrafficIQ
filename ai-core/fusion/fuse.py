"""
M7 - YOLO + ViT Feature Fusion Layer for TrafficIQ.

This module combines:
1. Object-level traffic features from YOLOv8n + ByteTrack (counts, density, occupancy, speed, congestion)
2. Global scene-level visual context embeddings from Vision Transformer (M6)

The output is a fused multimodal representation suitable for downstream traffic
intelligence tasks (e.g. M10 traffic demand prediction, M11 adaptive signal control).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Suppress external warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
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
from traffic.analyze import (  # noqa: E402
    calculate_congestion_index,
    calculate_density,
    calculate_occupancy,
    centroid,
)

DEFAULT_SAMPLE_EVERY = 10
DEFAULT_SLOW_THRESHOLD = 2.0

YOLO_FEATURE_NAMES: List[str] = [
    "total_vehicle_count",
    "car_count",
    "bike_count",
    "bus_count",
    "truck_count",
    "vehicle_density",
    "occupancy",
    "slow_vehicle_ratio",
    "congestion_index",
    "avg_center_x",
    "avg_center_y",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TrafficIQ M7 — YOLO + ViT Feature Fusion"
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to input video file (e.g. samples/traffic2.mp4)",
    )
    parser.add_argument(
        "--vit-embeddings",
        type=Path,
        default=None,
        help="Path to precomputed M6 ViT embeddings NPZ file (default: reuse outputs/vit/<source>_embeddings.npz)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"YOLO model name or .pt path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONF,
        help=f"YOLO confidence threshold (default: {DEFAULT_CONF})",
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
        help="Maximum video frames to process (default: complete video)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size if generating ViT embeddings (default: 4)",
    )
    parser.add_argument(
        "--slow-threshold",
        type=float,
        default=DEFAULT_SLOW_THRESHOLD,
        help=f"Movement threshold in pixels/frame for slow vehicle anomaly (default: {DEFAULT_SLOW_THRESHOLD})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save fusion outputs (default: outputs/fusion)",
    )

    args = parser.parse_args()

    if args.sample_every <= 0:
        parser.error("--sample-every must be greater than zero")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be greater than zero")

    return args


def extract_yolo_features(
    source_path: Path,
    model_path: str = DEFAULT_MODEL,
    conf: float = DEFAULT_CONF,
    sample_every: int = DEFAULT_SAMPLE_EVERY,
    max_frames: Optional[int] = None,
    slow_threshold: float = DEFAULT_SLOW_THRESHOLD,
) -> Tuple[Dict[int, np.ndarray], Dict[int, float], Dict[str, Any]]:
    """
    Extract normalized YOLO traffic feature vectors for sampled video frames.

    Returns:
        (yolo_features_dict, timestamps_dict, video_info)
    """
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

    model_file = Path(model_path)
    if not model_file.is_file() and (REPO_ROOT / model_file).is_file():
        model_file = REPO_ROOT / model_file
    model = YOLO(str(model_file))

    yolo_features_map: Dict[int, np.ndarray] = {}
    timestamps_map: Dict[int, float] = {}
    previous_centers: Dict[int, Tuple[float, float, int]] = {}

    frame_idx = 0
    try:
        while max_frames is None or frame_idx < max_frames:
            ok, frame = capture.read()
            if not ok:
                break

            if frame_idx % sample_every == 0:
                results = model.track(
                    source=frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    conf=conf,
                    device="cpu",
                    classes=list(COCO_VEHICLE_IDS),
                    verbose=False,
                )
                result = results[0]
                boxes = result.boxes

                # Vehicle class counts
                group_counts: Dict[str, int] = {g: 0 for g in COUNT_GROUPS}
                active_count = len(boxes) if boxes is not None else 0
                centers_in_frame: List[Tuple[float, float]] = []
                slow_count = 0

                if boxes is not None and boxes.cls is not None:
                    class_ids = boxes.cls.int().tolist()
                    track_ids = (
                        boxes.id.int().tolist() if boxes.id is not None else []
                    )
                    xyxy_list = boxes.xyxy.tolist()

                    for i, cid in enumerate(class_ids):
                        cname = result.names[cid]
                        grp = group_for_class(cname)
                        if grp in group_counts:
                            group_counts[grp] += 1

                        cx, cy = centroid(xyxy_list[i])
                        centers_in_frame.append((cx, cy))

                        # Speed tracking if ID is active
                        if i < len(track_ids):
                            tid = track_ids[i]
                            if tid in previous_centers:
                                prev_cx, prev_cy, prev_frame = previous_centers[tid]
                                dt = frame_idx - prev_frame
                                if dt > 0:
                                    displacement = float(
                                        np.hypot(cx - prev_cx, cy - prev_cy)
                                    )
                                    speed = displacement / dt
                                    if speed < slow_threshold:
                                        slow_count += 1
                            previous_centers[tid] = (cx, cy, frame_idx)

                # Metric calculations (reusing M4 functions)
                occupancy = calculate_occupancy(boxes, width, height)
                density = calculate_density(active_count, width, height)
                slow_ratio = (
                    slow_count / active_count if active_count > 0 else 0.0
                )
                congestion_idx = calculate_congestion_index(
                    active_count, occupancy, slow_ratio
                )

                # Normalized average vehicle position
                if centers_in_frame and width > 0 and height > 0:
                    avg_cx = sum(c[0] for c in centers_in_frame) / len(centers_in_frame)
                    avg_cy = sum(c[1] for c in centers_in_frame) / len(centers_in_frame)
                    avg_cx_norm = float(np.clip(avg_cx / width, 0.0, 1.0))
                    avg_cy_norm = float(np.clip(avg_cy / height, 0.0, 1.0))
                else:
                    avg_cx_norm = 0.5
                    avg_cy_norm = 0.5

                # Compact normalized YOLO feature vector
                feature_vec = np.array(
                    [
                        min(active_count / 50.0, 1.0),
                        min(group_counts["car"] / 50.0, 1.0),
                        min(group_counts["bike"] / 50.0, 1.0),
                        min(group_counts["bus"] / 50.0, 1.0),
                        min(group_counts["truck"] / 50.0, 1.0),
                        min(density / 25.0, 1.0),
                        occupancy / 100.0,
                        slow_ratio,
                        congestion_idx / 100.0,
                        avg_cx_norm,
                        avg_cy_norm,
                    ],
                    dtype=np.float32,
                )

                timestamp = frame_idx / fps if fps > 0 else 0.0
                yolo_features_map[frame_idx] = feature_vec
                timestamps_map[frame_idx] = timestamp

            frame_idx += 1
    finally:
        capture.release()

    video_info = {
        "source": str(source),
        "stem": source.stem,
        "width": width,
        "height": height,
        "fps": fps,
        "frames": total_frames,
    }

    return yolo_features_map, timestamps_map, video_info


def load_or_generate_vit_embeddings(
    source_path: Path,
    vit_path: Optional[Path] = None,
    sample_every: int = DEFAULT_SAMPLE_EVERY,
    max_frames: Optional[int] = None,
    batch_size: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Path]:
    """
    Load precomputed M6 ViT embeddings or generate them if missing.

    Returns:
        (vit_frame_indices, vit_timestamps, vit_embeddings, resolved_path)
    """
    source = Path(source_path)
    resolved_vit_path: Optional[Path] = None

    if vit_path is not None:
        p = Path(vit_path)
        if not p.is_file() and (REPO_ROOT / p).is_file():
            p = REPO_ROOT / p
        if not p.is_file():
            raise SystemExit(f"Specified ViT embeddings file not found: {vit_path}")
        resolved_vit_path = p
    else:
        # Check standard M6 location
        default_path = REPO_ROOT / "outputs" / "vit" / f"{source.stem}_embeddings.npz"
        if default_path.is_file():
            resolved_vit_path = default_path
        else:
            print(
                f"Notice: Precomputed ViT embeddings not found at {default_path}."
            )
            print("Generating M6 ViT embeddings now...")
            from vit.extract import extract_vit_features

            extract_vit_features(
                source_path=source,
                sample_every=sample_every,
                max_frames=max_frames,
                batch_size=batch_size,
            )
            resolved_vit_path = default_path

    vit_data = np.load(resolved_vit_path)
    frame_indices = vit_data["frame_indices"]
    timestamps = vit_data["timestamps"]
    embeddings = vit_data["embeddings"]

    return frame_indices, timestamps, embeddings, resolved_vit_path


def fuse_features(
    source_path: Path,
    vit_embeddings_path: Optional[Path] = None,
    model_path: str = DEFAULT_MODEL,
    conf: float = DEFAULT_CONF,
    sample_every: int = DEFAULT_SAMPLE_EVERY,
    max_frames: Optional[int] = None,
    batch_size: int = 4,
    slow_threshold: float = DEFAULT_SLOW_THRESHOLD,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Perform feature-level fusion of YOLO traffic features and ViT scene embeddings."""
    source = Path(source_path)
    if not source.is_file() and (REPO_ROOT / source).is_file():
        source = REPO_ROOT / source

    # 1. Load or extract ViT embeddings
    (
        vit_indices,
        vit_timestamps,
        vit_embeddings,
        used_vit_path,
    ) = load_or_generate_vit_embeddings(
        source_path=source,
        vit_path=vit_embeddings_path,
        sample_every=sample_every,
        max_frames=max_frames,
        batch_size=batch_size,
    )

    # 2. Extract YOLO features for the sampled video frames
    (
        yolo_map,
        yolo_time_map,
        video_info,
    ) = extract_yolo_features(
        source_path=source,
        model_path=model_path,
        conf=conf,
        sample_every=sample_every,
        max_frames=max_frames,
        slow_threshold=slow_threshold,
    )

    # 3. Align frames strictly by frame index
    vit_map: Dict[int, Tuple[np.ndarray, float]] = {}
    for i, idx in enumerate(vit_indices):
        vit_map[int(idx)] = (vit_embeddings[i], float(vit_timestamps[i]))

    yolo_frame_keys = set(yolo_map.keys())
    vit_frame_keys = set(vit_map.keys())

    matched_indices = sorted(list(yolo_frame_keys & vit_frame_keys))
    unmatched_yolo = sorted(list(yolo_frame_keys - vit_frame_keys))
    unmatched_vit = sorted(list(vit_frame_keys - yolo_frame_keys))

    if not matched_indices:
        raise ValueError(
            "No matching frames found between YOLO detections and ViT embeddings."
        )

    aligned_indices = np.array(matched_indices, dtype=np.int32)
    aligned_timestamps = np.array(
        [vit_map[idx][1] for idx in matched_indices], dtype=np.float64
    )
    aligned_yolo = np.array(
        [yolo_map[idx] for idx in matched_indices], dtype=np.float32
    )
    aligned_vit = np.array(
        [vit_map[idx][0] for idx in matched_indices], dtype=np.float32
    )

    # 4. Feature normalization and concatenation
    # Normalize YOLO features (L2 norm per vector with eps for stability)
    yolo_norm = aligned_yolo / (
        np.linalg.norm(aligned_yolo, axis=-1, keepdims=True) + 1e-8
    )

    # Normalize ViT embeddings (L2 norm per vector with eps for stability)
    vit_norm = aligned_vit / (
        np.linalg.norm(aligned_vit, axis=-1, keepdims=True) + 1e-8
    )

    # Concatenate normalized representations: [YOLO_norm || ViT_norm]
    fused_features = np.concatenate([yolo_norm, vit_norm], axis=-1).astype(
        np.float32
    )

    # Determine feature dimensions dynamically
    yolo_dim = int(aligned_yolo.shape[1])
    vit_dim = int(aligned_vit.shape[1])
    fused_dim = int(fused_features.shape[1])

    # 5. Output file setup
    if output_dir is None:
        out_dir = REPO_ROOT / "outputs" / "fusion"
    else:
        out_dir = Path(output_dir)
        if not out_dir.is_absolute():
            out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_filename = f"{source.stem}_fused_features.npz"
    npz_path = out_dir / npz_filename
    json_path = out_dir / "m7_fusion_results.json"

    rel_npz_path = f"outputs/fusion/{npz_filename}"
    rel_json_path = "outputs/fusion/m7_fusion_results.json"

    # Save NPZ file
    np.savez_compressed(
        npz_path,
        frame_indices=aligned_indices,
        timestamps=aligned_timestamps,
        yolo_features=aligned_yolo,
        vit_embeddings=aligned_vit,
        fused_features=fused_features,
    )

    # Save JSON results adhering to M7 specification
    results_json = {
        "project": "TrafficIQ",
        "stage": "M7",
        "fusion_method": "feature_concatenation",
        "video": {
            "source": str(source),
            "width": video_info["width"],
            "height": video_info["height"],
            "fps": round(video_info["fps"], 2),
            "frames": video_info["frames"],
        },
        "alignment": {
            "matched_frames": len(matched_indices),
            "unmatched_yolo_frames": len(unmatched_yolo),
            "unmatched_vit_frames": len(unmatched_vit),
        },
        "features": {
            "yolo_feature_dimension": yolo_dim,
            "vit_feature_dimension": vit_dim,
            "fused_feature_dimension": fused_dim,
            "yolo_feature_names": YOLO_FEATURE_NAMES,
        },
        "outputs": {
            "feature_file": rel_npz_path,
        },
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=4)
        f.write("\n")

    return {
        "video": video_info,
        "alignment": {
            "yolo_frames": len(yolo_map),
            "vit_frames": len(vit_map),
            "matched_frames": len(matched_indices),
        },
        "features": {
            "yolo_feature_dimension": yolo_dim,
            "vit_feature_dimension": vit_dim,
            "fused_feature_dimension": fused_dim,
        },
        "outputs": {
            "npz_file": rel_npz_path,
            "json_file": rel_json_path,
        },
        "fused_features": fused_features,
        "yolo_features": aligned_yolo,
        "vit_embeddings": aligned_vit,
        "frame_indices": aligned_indices,
        "timestamps": aligned_timestamps,
    }


def print_report(results: Dict[str, Any]) -> None:
    video = results["video"]
    align = results["alignment"]
    feats = results["features"]
    outs = results["outputs"]

    print("=" * 60)
    print("TrafficIQ M7 — YOLO + ViT Feature Fusion")
    print("=" * 60)
    print("\nVIDEO")
    print("-----")
    print(f"Resolution: {video['width']} x {video['height']}")
    print(f"Frames: {video['frames']}")
    print("\nALIGNMENT")
    print("---------")
    print(f"YOLO frames: {align['yolo_frames']}")
    print(f"ViT frames: {align['vit_frames']}")
    print(f"Matched frames: {align['matched_frames']}")
    print("\nFEATURES")
    print("--------")
    print(f"YOLO feature dimension: {feats['yolo_feature_dimension']}")
    print(f"ViT feature dimension: {feats['vit_feature_dimension']}")
    print(f"Fused feature dimension: {feats['fused_feature_dimension']}")
    print("\nFUSION")
    print("------")
    print("Method: Feature Concatenation")
    print("\nOUTPUT")
    print("------")
    print(outs["npz_file"])
    print(outs["json_file"])
    print("\n" + "=" * 60)


def main() -> None:
    args = parse_args()
    results = fuse_features(
        source_path=args.source,
        vit_embeddings_path=args.vit_embeddings,
        model_path=args.model,
        conf=args.conf,
        sample_every=args.sample_every,
        max_frames=args.max_frames,
        batch_size=args.batch_size,
        slow_threshold=args.slow_threshold,
        output_dir=args.output_dir,
    )
    print_report(results)


if __name__ == "__main__":
    main()
