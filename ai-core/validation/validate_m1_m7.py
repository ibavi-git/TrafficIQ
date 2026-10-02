"""
Read-Only Structural and Data Integrity Validation Script for TrafficIQ M6 and M7.

Validates:
- M6: frame_indices, timestamps, embeddings existence, frame counts, embedding dim (768), no NaN, no Inf.
- M7: frame_indices, timestamps, yolo_features, vit_embeddings, fused_features existence, length consistency,
      frame alignment, dynamic YOLO dimension detection, ViT dim (768), fused dim = YOLO dim + ViT dim,
      no NaN, no Inf, matched frame counts.
- M7 JSON metadata vs. actual NPZ array values.
- M6 vs M7 cross-stage feature consistency.

NOTE: This script performs read-only structural validation and does NOT claim model accuracy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parents[2]

# Default paths
DEFAULT_M6_NPZ = REPO_ROOT / "outputs" / "vit" / "traffic2_embeddings.npz"
DEFAULT_M6_JSON = REPO_ROOT / "outputs" / "vit" / "m6_vit_results.json"
DEFAULT_M7_NPZ = REPO_ROOT / "outputs" / "fusion" / "traffic2_fused_features.npz"
DEFAULT_M7_JSON = REPO_ROOT / "outputs" / "fusion" / "m7_fusion_results.json"


class ValidationTracker:
    def __init__(self) -> None:
        self.results: List[Tuple[str, str, bool, str]] = []

    def check(self, stage: str, description: str, condition: bool, details: str = "") -> bool:
        status_str = "PASS" if condition else "FAIL"
        self.results.append((stage, description, condition, details))
        msg = f"[{status_str}] [{stage}] {description}"
        if details:
            msg += f" ({details})"
        print(msg)
        return condition

    @property
    def all_passed(self) -> bool:
        return all(cond for _, _, cond, _ in self.results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TrafficIQ Read-Only M6 and M7 Structural Validation"
    )
    parser.add_argument("--m6-npz", type=Path, default=DEFAULT_M6_NPZ, help="Path to M6 embeddings NPZ")
    parser.add_argument("--m6-json", type=Path, default=DEFAULT_M6_JSON, help="Path to M6 results JSON")
    parser.add_argument("--m7-npz", type=Path, default=DEFAULT_M7_NPZ, help="Path to M7 fused features NPZ")
    parser.add_argument("--m7-json", type=Path, default=DEFAULT_M7_JSON, help="Path to M7 results JSON")
    return parser.parse_args()


def validate_m6_and_m7(
    m6_npz_path: Path,
    m6_json_path: Path,
    m7_npz_path: Path,
    m7_json_path: Path,
) -> bool:
    tracker = ValidationTracker()

    print("=" * 75)
    print("TrafficIQ M6 & M7 Read-Only Integrity Validation")
    print("=" * 75)
    print("Notice: Validating data structures and numeric consistency. No claim of model accuracy.\n")

    # ---------------------------------------------------------
    # File Existence Checks
    # ---------------------------------------------------------
    print("--- 1. File Existence Checks ---")
    tracker.check("M6", "Embeddings NPZ exists", m6_npz_path.is_file(), str(m6_npz_path))
    tracker.check("M6", "Results JSON exists", m6_json_path.is_file(), str(m6_json_path))
    tracker.check("M7", "Fused Features NPZ exists", m7_npz_path.is_file(), str(m7_npz_path))
    tracker.check("M7", "Results JSON exists", m7_json_path.is_file(), str(m7_json_path))

    if not (m6_npz_path.is_file() and m6_json_path.is_file() and m7_npz_path.is_file() and m7_json_path.is_file()):
        print("\n[ERROR] Required output files are missing. Cannot proceed with data validation.")
        return False

    # Load outputs (Read-Only)
    m6_npz = np.load(m6_npz_path)
    with m6_json_path.open("r", encoding="utf-8") as f:
        m6_json = json.load(f)

    m7_npz = np.load(m7_npz_path)
    with m7_json_path.open("r", encoding="utf-8") as f:
        m7_json = json.load(f)

    # ---------------------------------------------------------
    # M6 Validation
    # ---------------------------------------------------------
    print("\n--- 2. M6 Vision Transformer Embedding Checks ---")
    
    # Check keys exist
    m6_has_indices = "frame_indices" in m6_npz
    m6_has_timestamps = "timestamps" in m6_npz
    m6_has_embeddings = "embeddings" in m6_npz

    tracker.check("M6", "frame_indices array exists", m6_has_indices)
    tracker.check("M6", "timestamps array exists", m6_has_timestamps)
    tracker.check("M6", "embeddings array exists", m6_has_embeddings)

    if m6_has_indices and m6_has_timestamps and m6_has_embeddings:
        m6_indices = m6_npz["frame_indices"]
        m6_timestamps = m6_npz["timestamps"]
        m6_embeddings = m6_npz["embeddings"]

        n_m6_frames = len(m6_indices)
        m6_json_sampled = m6_json.get("sampling", {}).get("sampled_frames", None)

        # Number of frames check
        frames_valid = (n_m6_frames > 0) and (len(m6_timestamps) == n_m6_frames) and (m6_embeddings.shape[0] == n_m6_frames)
        if m6_json_sampled is not None:
            frames_valid = frames_valid and (n_m6_frames == m6_json_sampled)
        tracker.check("M6", "Number of frames consistent across arrays & JSON", frames_valid, f"count={n_m6_frames}")

        # Embedding dimension = 768
        m6_dim = m6_embeddings.shape[1] if m6_embeddings.ndim == 2 else -1
        m6_dim_expected = 768
        m6_json_dim = m6_json.get("model", {}).get("hidden_size", 768) if isinstance(m6_json.get("model"), dict) else 768
        tracker.check(
            "M6",
            "Embedding dimension = 768",
            m6_dim == m6_dim_expected == m6_json_dim,
            f"actual={m6_dim}, expected={m6_dim_expected}"
        )

        # No NaN checks
        m6_has_nan = bool(np.isnan(m6_embeddings).any() or np.isnan(m6_timestamps).any() or np.isnan(m6_indices).any())
        tracker.check("M6", "No NaN values in embeddings, timestamps, or indices", not m6_has_nan)

        # No Inf checks
        m6_has_inf = bool(np.isinf(m6_embeddings).any() or np.isinf(m6_timestamps).any() or np.isinf(m6_indices).any())
        tracker.check("M6", "No Inf values in embeddings, timestamps, or indices", not m6_has_inf)
    else:
        m6_indices = np.array([])
        m6_timestamps = np.array([])
        m6_embeddings = np.array([])

    # ---------------------------------------------------------
    # M7 Validation
    # ---------------------------------------------------------
    print("\n--- 3. M7 YOLO + ViT Feature Fusion Checks ---")

    # Check keys exist
    m7_has_indices = "frame_indices" in m7_npz
    m7_has_timestamps = "timestamps" in m7_npz
    m7_has_yolo = "yolo_features" in m7_npz
    m7_has_vit = "vit_embeddings" in m7_npz
    m7_has_fused = "fused_features" in m7_npz

    tracker.check("M7", "frame_indices array exists", m7_has_indices)
    tracker.check("M7", "timestamps array exists", m7_has_timestamps)
    tracker.check("M7", "yolo_features array exists", m7_has_yolo)
    tracker.check("M7", "vit_embeddings array exists", m7_has_vit)
    tracker.check("M7", "fused_features array exists", m7_has_fused)

    if m7_has_indices and m7_has_timestamps and m7_has_yolo and m7_has_vit and m7_has_fused:
        m7_indices = m7_npz["frame_indices"]
        m7_timestamps = m7_npz["timestamps"]
        m7_yolo = m7_npz["yolo_features"]
        m7_vit = m7_npz["vit_embeddings"]
        m7_fused = m7_npz["fused_features"]

        n_m7_samples = len(m7_indices)

        # Matched frame count > 0
        tracker.check("M7", "Matched frame count > 0", n_m7_samples > 0, f"count={n_m7_samples}")

        # All lengths match
        lengths_match = (
            len(m7_timestamps) == n_m7_samples
            and m7_yolo.shape[0] == n_m7_samples
            and m7_vit.shape[0] == n_m7_samples
            and m7_fused.shape[0] == n_m7_samples
        )
        tracker.check(
            "M7",
            "All array lengths match frame_indices count",
            lengths_match,
            f"frame_indices={n_m7_samples}, timestamps={len(m7_timestamps)}, yolo_rows={m7_yolo.shape[0]}, vit_rows={m7_vit.shape[0]}, fused_rows={m7_fused.shape[0]}"
        )

        # Frame indices are aligned (strictly monotonically increasing)
        is_strictly_increasing = bool(np.all(np.diff(m7_indices) > 0)) if n_m7_samples > 1 else True
        tracker.check("M7", "Frame indices are strictly monotonically increasing and aligned", is_strictly_increasing)

        # YOLO dimension detected dynamically
        yolo_dim = m7_yolo.shape[1] if m7_yolo.ndim == 2 else 0
        tracker.check("M7", "YOLO dimension is detected dynamically", yolo_dim > 0, f"detected_dim={yolo_dim}")

        # ViT dimension = 768
        vit_dim = m7_vit.shape[1] if m7_vit.ndim == 2 else 0
        tracker.check("M7", "ViT dimension = 768", vit_dim == 768, f"detected_dim={vit_dim}")

        # Fused dimension = YOLO dimension + ViT dimension
        fused_dim = m7_fused.shape[1] if m7_fused.ndim == 2 else 0
        expected_fused_dim = yolo_dim + vit_dim
        tracker.check(
            "M7",
            "Fused dimension = YOLO dimension + ViT dimension",
            fused_dim == expected_fused_dim,
            f"fused_dim={fused_dim}, yolo_dim={yolo_dim}, vit_dim={vit_dim}, expected={expected_fused_dim}"
        )

        # No NaN checks
        m7_has_nan = bool(
            np.isnan(m7_indices).any()
            or np.isnan(m7_timestamps).any()
            or np.isnan(m7_yolo).any()
            or np.isnan(m7_vit).any()
            or np.isnan(m7_fused).any()
        )
        tracker.check("M7", "No NaN values in any M7 array", not m7_has_nan)

        # No Inf checks
        m7_has_inf = bool(
            np.isinf(m7_indices).any()
            or np.isinf(m7_timestamps).any()
            or np.isinf(m7_yolo).any()
            or np.isinf(m7_vit).any()
            or np.isinf(m7_fused).any()
        )
        tracker.check("M7", "No Inf values in any M7 array", not m7_has_inf)

    # ---------------------------------------------------------
    # M7 JSON Metadata vs. Actual NPZ Values
    # ---------------------------------------------------------
    print("\n--- 4. M7 JSON Metadata vs Actual NPZ Checks ---")
    json_alignment = m7_json.get("alignment", {})
    json_features = m7_json.get("features", {})
    json_outputs = m7_json.get("outputs", {})

    json_matched_frames = json_alignment.get("matched_frames")
    tracker.check(
        "M7-JSON",
        "JSON alignment.matched_frames matches actual NPZ length",
        json_matched_frames == n_m7_samples,
        f"json={json_matched_frames}, npz={n_m7_samples}"
    )

    json_yolo_dim = json_features.get("yolo_feature_dimension")
    tracker.check(
        "M7-JSON",
        "JSON features.yolo_feature_dimension matches NPZ YOLO dimension",
        json_yolo_dim == yolo_dim,
        f"json={json_yolo_dim}, npz={yolo_dim}"
    )

    json_vit_dim = json_features.get("vit_feature_dimension")
    tracker.check(
        "M7-JSON",
        "JSON features.vit_feature_dimension matches NPZ ViT dimension",
        json_vit_dim == vit_dim,
        f"json={json_vit_dim}, npz={vit_dim}"
    )

    json_fused_dim = json_features.get("fused_feature_dimension")
    tracker.check(
        "M7-JSON",
        "JSON features.fused_feature_dimension matches NPZ fused dimension",
        json_fused_dim == fused_dim,
        f"json={json_fused_dim}, npz={fused_dim}"
    )

    json_feature_file = json_outputs.get("feature_file", "")
    tracker.check(
        "M7-JSON",
        "JSON outputs.feature_file corresponds to NPZ filename",
        Path(json_feature_file).name == m7_npz_path.name,
        f"json='{json_feature_file}', actual='{m7_npz_path.name}'"
    )

    # Cross-Stage ViT consistency check between M6 and M7
    if m6_has_embeddings and m7_has_vit and len(m6_indices) == len(m7_indices):
        frames_equal = bool(np.array_equal(m6_indices, m7_indices))
        vit_equal = bool(np.allclose(m6_embeddings, m7_vit, atol=1e-6))
        timestamps_equal = bool(np.allclose(m6_timestamps, m7_timestamps, atol=1e-5))

        tracker.check(
            "CROSS-STAGE",
            "M7 aligned frame indices match M6 frame indices",
            frames_equal,
            f"count={len(m7_indices)}"
        )
        tracker.check(
            "CROSS-STAGE",
            "M7 vit_embeddings match M6 raw embeddings numerically",
            vit_equal
        )
        tracker.check(
            "CROSS-STAGE",
            "M7 timestamps match M6 timestamps numerically",
            timestamps_equal
        )

    # ---------------------------------------------------------
    # Final Summary
    # ---------------------------------------------------------
    print("\n" + "=" * 75)
    total_checks = len(tracker.results)
    passed_checks = sum(1 for _, _, passed, _ in tracker.results if passed)
    failed_checks = total_checks - passed_checks

    print(f"Validation Summary: {passed_checks}/{total_checks} checks PASSED ({failed_checks} FAILED)")
    print("=" * 75)
    if tracker.all_passed:
        print("RESULT: ALL M6 AND M7 INTEGRITY CHECKS PASSED.")
    else:
        print("RESULT: INTEGRITY CHECKS FAILED.")
    print("=" * 75)

    return tracker.all_passed


def main() -> None:
    args = parse_args()
    success = validate_m6_and_m7(
        m6_npz_path=args.m6_npz,
        m6_json_path=args.m6_json,
        m7_npz_path=args.m7_npz,
        m7_json_path=args.m7_json,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
