"""Validation script for TrafficIQ M6 Vision Transformer feature extraction."""

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
NPZ_PATH = REPO_ROOT / "outputs" / "vit" / "traffic2_embeddings.npz"
JSON_PATH = REPO_ROOT / "outputs" / "vit" / "m6_vit_results.json"


def validate_m6() -> bool:
    print("=" * 60)
    print("TrafficIQ M6 Validation")
    print("=" * 60)

    # 1. NPZ exists
    assert NPZ_PATH.is_file(), f"NPZ file not found at {NPZ_PATH}"
    print(f"[PASS] NPZ file exists: {NPZ_PATH}")

    # 2. JSON exists
    assert JSON_PATH.is_file(), f"JSON file not found at {JSON_PATH}"
    print(f"[PASS] JSON file exists: {JSON_PATH}")

    # Load outputs
    npz_data = np.load(NPZ_PATH)
    frame_indices = npz_data["frame_indices"]
    timestamps = npz_data["timestamps"]
    embeddings = npz_data["embeddings"]

    with JSON_PATH.open("r", encoding="utf-8") as f:
        results = json.load(f)

    # 3. Embeddings are 2D
    assert embeddings.ndim == 2, f"Embeddings are not 2D: shape={embeddings.shape}"
    print(f"[PASS] Embeddings are 2D array: shape={embeddings.shape}")

    # 4. Number of rows equals sampled frames
    sampled_frames = results["sampling"]["sampled_frames"]
    assert embeddings.shape[0] == sampled_frames, (
        f"Row count ({embeddings.shape[0]}) does not match "
        f"sampled_frames in JSON ({sampled_frames})"
    )
    print(f"[PASS] Number of rows ({embeddings.shape[0]}) equals sampled frames ({sampled_frames})")

    # 5. Embedding dimension equals model hidden size
    model_field = results["model"]
    hidden_size = model_field["hidden_size"] if isinstance(model_field, dict) else 768
    assert embeddings.shape[1] == hidden_size, (
        f"Embedding dimension ({embeddings.shape[1]}) does not match "
        f"hidden_size ({hidden_size})"
    )
    print(f"[PASS] Embedding dimension ({embeddings.shape[1]}) equals model hidden size ({hidden_size})")

    # 6. frame_indices length matches embeddings rows
    assert len(frame_indices) == embeddings.shape[0], (
        f"frame_indices length ({len(frame_indices)}) != rows ({embeddings.shape[0]})"
    )
    print(f"[PASS] frame_indices length ({len(frame_indices)}) matches embeddings rows")

    # 7. timestamps length matches embeddings rows
    assert len(timestamps) == embeddings.shape[0], (
        f"timestamps length ({len(timestamps)}) != rows ({embeddings.shape[0]})"
    )
    print(f"[PASS] timestamps length ({len(timestamps)}) matches embeddings rows")

    print("\nSummary:")
    print(f"  Sampled frames: {embeddings.shape[0]}")
    print(f"  Embedding shape: {embeddings.shape}")
    print(f"  Data type: {embeddings.dtype}")
    print(f"  Frame index range: [{frame_indices[0]}, {frame_indices[-1]}]")
    print(f"  Timestamp range: [{timestamps[0]:.2f}s, {timestamps[-1]:.2f}s]")
    print("=" * 60)
    print("All M6 validation checks PASSED successfully.")
    print("=" * 60)
    return True


if __name__ == "__main__":
    validate_m6()
