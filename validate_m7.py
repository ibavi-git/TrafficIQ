"""Validation script for TrafficIQ M7 YOLO + ViT Feature Fusion."""

from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parent
NPZ_PATH = REPO_ROOT / "outputs" / "fusion" / "traffic2_fused_features.npz"
JSON_PATH = REPO_ROOT / "outputs" / "fusion" / "m7_fusion_results.json"


def validate_m7() -> bool:
    print("=" * 60)
    print("TrafficIQ M7 Validation")
    print("=" * 60)

    # 1. NPZ exists
    assert NPZ_PATH.is_file(), f"NPZ file not found at {NPZ_PATH}"
    print(f"[PASS] 1. NPZ file exists: {NPZ_PATH}")

    # 2. JSON exists
    assert JSON_PATH.is_file(), f"JSON file not found at {JSON_PATH}"
    print(f"[PASS] 2. JSON file exists: {JSON_PATH}")

    # Load outputs
    npz_data = np.load(NPZ_PATH)
    frame_indices = npz_data["frame_indices"]
    timestamps = npz_data["timestamps"]
    yolo_features = npz_data["yolo_features"]
    vit_embeddings = npz_data["vit_embeddings"]
    fused_features = npz_data["fused_features"]

    with JSON_PATH.open("r", encoding="utf-8") as f:
        results = json.load(f)

    n_samples = len(frame_indices)

    # 9. At least one matched frame exists
    assert n_samples > 0, "No matched frames found (n_samples == 0)"
    print(f"[PASS] 9. At least one matched frame exists: {n_samples} matched frames")

    # 3. frame_indices length matches all feature arrays
    assert yolo_features.shape[0] == n_samples, "yolo_features rows != frame_indices length"
    assert vit_embeddings.shape[0] == n_samples, "vit_embeddings rows != frame_indices length"
    assert fused_features.shape[0] == n_samples, "fused_features rows != frame_indices length"
    print(f"[PASS] 3. frame_indices length ({n_samples}) matches all feature arrays rows")

    # 4. timestamps length matches all feature arrays
    assert len(timestamps) == n_samples, "timestamps length != frame_indices length"
    print(f"[PASS] 4. timestamps length ({len(timestamps)}) matches all feature arrays")

    # 5. YOLO and ViT frame indices are aligned
    matched_frames_json = results["alignment"]["matched_frames"]
    assert matched_frames_json == n_samples, f"JSON matched_frames ({matched_frames_json}) != NPZ ({n_samples})"
    # Check strictly increasing frame indices
    assert np.all(np.diff(frame_indices) > 0), "frame_indices are not strictly monotonically increasing"
    print(f"[PASS] 5. YOLO and ViT frame indices are aligned ({n_samples} frames strictly aligned)")

    # 6. fused dimension = YOLO dimension + ViT dimension
    yolo_dim = yolo_features.shape[1]
    vit_dim = vit_embeddings.shape[1]
    fused_dim = fused_features.shape[1]
    assert fused_dim == yolo_dim + vit_dim, (
        f"Fused dimension ({fused_dim}) != YOLO dim ({yolo_dim}) + ViT dim ({vit_dim})"
    )
    print(f"[PASS] 6. fused dimension ({fused_dim}) = YOLO dim ({yolo_dim}) + ViT dim ({vit_dim})")

    # 7. No NaN values
    assert not np.isnan(yolo_features).any(), "NaN found in yolo_features"
    assert not np.isnan(vit_embeddings).any(), "NaN found in vit_embeddings"
    assert not np.isnan(fused_features).any(), "NaN found in fused_features"
    print("[PASS] 7. No NaN values across any feature arrays")

    # 8. No infinite values
    assert not np.isinf(yolo_features).any(), "Inf found in yolo_features"
    assert not np.isinf(vit_embeddings).any(), "Inf found in vit_embeddings"
    assert not np.isinf(fused_features).any(), "Inf found in fused_features"
    print("[PASS] 8. No infinite values across any feature arrays")

    print("\nArray Shapes:")
    print(f"YOLO shape: {yolo_features.shape}")
    print(f"ViT shape: {vit_embeddings.shape}")
    print(f"Fused shape: {fused_features.shape}")

    print("\nSummary:")
    print(f"  Frame indices range: [{frame_indices[0]}, {frame_indices[-1]}]")
    print(f"  Timestamp range: [{timestamps[0]:.2f}s, {timestamps[-1]:.2f}s]")
    print(f"  YOLO features: {results['features']['yolo_feature_names']}")
    print("=" * 60)
    print("All M7 validation checks PASSED successfully.")
    print("=" * 60)
    return True


if __name__ == "__main__":
    validate_m7()
