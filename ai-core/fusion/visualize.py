"""
M7 Visual Validation - TrafficIQ.

Generates a comprehensive visual validation figure and report for M7 YOLO + ViT Feature Fusion:
1. Temporal traffic metrics profile (vehicle count, occupancy, congestion index)
2. Normalized YOLO feature heatmap across sampled frames
3. ViT temporal cosine similarity matrix
4. 2D PCA representation of the fused multimodal traffic state
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Non-interactive backend for headless environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parents[2]
AI_CORE = REPO_ROOT / "ai-core"
if str(AI_CORE) not in sys.path:
    sys.path.insert(0, str(AI_CORE))

DEFAULT_NPZ = REPO_ROOT / "outputs" / "fusion" / "traffic2_fused_features.npz"
DEFAULT_JSON = REPO_ROOT / "outputs" / "fusion" / "m7_fusion_results.json"
DEFAULT_OUTPUT_IMG = REPO_ROOT / "outputs" / "fusion" / "m7_visual_validation.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TrafficIQ M7 Visual Validation Dashboard"
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_NPZ,
        help="Path to fused features NPZ file",
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        default=DEFAULT_JSON,
        help="Path to M7 results JSON file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_IMG,
        help="Path to save the generated visualization PNG",
    )
    return parser.parse_args()


def compute_pca_2d(matrix: np.ndarray) -> np.ndarray:
    """Compute 2D PCA coordinates using SVD without external sklearn dependency."""
    centered = matrix - np.mean(matrix, axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return np.dot(centered, vt[:2].T)


def generate_visual_validation(
    features_npz_path: Path,
    results_json_path: Path,
    output_image_path: Path,
) -> Path:
    if not features_npz_path.is_file():
        raise SystemExit(f"Features file not found: {features_npz_path}")
    if not results_json_path.is_file():
        raise SystemExit(f"Results JSON not found: {results_json_path}")

    # Load data
    npz_data = np.load(features_npz_path)
    frame_indices = npz_data["frame_indices"]
    timestamps = npz_data["timestamps"]
    yolo_features = npz_data["yolo_features"]  # (N, 11)
    vit_embeddings = npz_data["vit_embeddings"]  # (N, 768)
    fused_features = npz_data["fused_features"]  # (N, 779)

    with results_json_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    feature_names = meta.get("features", {}).get("yolo_feature_names", [
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
    ])

    n_samples = len(frame_indices)

    # 1. Unscale relevant YOLO metrics for human-interpretable display
    # Feature 0: total_count / 50 -> * 50
    # Feature 6: occupancy / 100 -> * 100
    # Feature 8: congestion_index / 100 -> * 100
    total_counts = yolo_features[:, 0] * 50.0
    occupancy_pct = yolo_features[:, 6] * 100.0
    congestion_pct = yolo_features[:, 8] * 100.0

    # 2. Compute ViT Cosine Similarity Matrix
    vit_norms = np.linalg.norm(vit_embeddings, axis=1, keepdims=True) + 1e-8
    vit_unit = vit_embeddings / vit_norms
    cos_sim_matrix = np.dot(vit_unit, vit_unit.T)  # (N, N)

    # Consecutive frame similarity (temporal consistency)
    consecutive_sims = [cos_sim_matrix[i, i + 1] for i in range(n_samples - 1)]
    mean_temp_sim = float(np.mean(consecutive_sims)) if consecutive_sims else 1.0

    # 3. PCA on Fused Features
    fused_pca = compute_pca_2d(fused_features)

    # Setup 4-panel figure
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig = plt.figure(figsize=(16, 12), dpi=150)
    fig.patch.set_facecolor("#fcfcfc")

    # Title Banner
    fig.suptitle(
        f"TrafficIQ M7 — YOLO + ViT Feature Fusion Validation\n"
        f"Video: {meta.get('video', {}).get('source', 'traffic2.mp4')} | "
        f"Matched Frames: {n_samples} | Fused Dimensions: {fused_features.shape[1]} (11 YOLO + 768 ViT)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # Subplot 1: Time-series of traffic dynamics
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(timestamps, total_counts, color="#2b5c8f", linewidth=2.2, marker="o", markersize=3.5, label="Vehicle Count (YOLO)")
    ax1.plot(timestamps, occupancy_pct, color="#e67e22", linewidth=2.0, linestyle="--", label="Occupancy % (M4)")
    ax1.plot(timestamps, congestion_pct, color="#c0392b", linewidth=2.2, label="Congestion Index % (M4)")
    ax1.set_title("1. Temporal Traffic Dynamics (Sampled Frames)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Video Timestamp (seconds)", fontsize=10)
    ax1.set_ylabel("Metric Value / Percentage", fontsize=10)
    ax1.legend(loc="upper right", frameon=True, framealpha=0.9)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Subplot 2: YOLO Normalized Feature Heatmap
    ax2 = fig.add_subplot(2, 2, 2)
    im2 = ax2.imshow(
        yolo_features.T,
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
    )
    ax2.set_title("2. Normalized YOLO Traffic Feature Matrix (11 × N)", fontsize=11, fontweight="bold")
    ax2.set_yticks(np.arange(len(feature_names)))
    ax2.set_yticklabels(feature_names, fontsize=8.5)
    # Subsampled x-ticks
    tick_step = max(1, n_samples // 6)
    ax2.set_xticks(np.arange(0, n_samples, tick_step))
    ax2.set_xticklabels([f"{timestamps[i]:.1f}s\n(f#{frame_indices[i]})" for i in range(0, n_samples, tick_step)], fontsize=8)
    ax2.set_xlabel("Sampled Frame / Timestamp", fontsize=10)
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.03)
    cbar2.set_label("Normalized Feature Value [0, 1]", fontsize=9)

    # Subplot 3: ViT Scene Cosine Similarity Matrix
    ax3 = fig.add_subplot(2, 2, 3)
    im3 = ax3.imshow(
        cos_sim_matrix,
        aspect="equal",
        cmap="mako" if "mako" in plt.colormaps() else "plasma",
        vmin=float(np.min(cos_sim_matrix)),
        vmax=1.0,
    )
    ax3.set_title(
        f"3. ViT Scene Cosine Similarity Matrix (Avg Consecutive = {mean_temp_sim:.3f})",
        fontsize=11,
        fontweight="bold",
    )
    ax3.set_xticks(np.arange(0, n_samples, tick_step))
    ax3.set_xticklabels([f"{frame_indices[i]}" for i in range(0, n_samples, tick_step)], fontsize=8)
    ax3.set_yticks(np.arange(0, n_samples, tick_step))
    ax3.set_yticklabels([f"{frame_indices[i]}" for i in range(0, n_samples, tick_step)], fontsize=8)
    ax3.set_xlabel("Sampled Frame Index", fontsize=10)
    ax3.set_ylabel("Sampled Frame Index", fontsize=10)
    cbar3 = fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.03)
    cbar3.set_label("Cosine Similarity", fontsize=9)

    # Subplot 4: Fused Representation 2D Projection (PCA)
    ax4 = fig.add_subplot(2, 2, 4)
    scatter = ax4.scatter(
        fused_pca[:, 0],
        fused_pca[:, 1],
        c=congestion_pct,
        cmap="coolwarm",
        s=60,
        edgecolor="black",
        linewidth=0.7,
        zorder=3,
    )
    # Draw trajectory line over time
    ax4.plot(fused_pca[:, 0], fused_pca[:, 1], color="#7f8c8d", linestyle="-", linewidth=1.0, alpha=0.5, zorder=2)
    # Annotate start and end frames
    ax4.annotate(f"Start (0s)", (fused_pca[0, 0], fused_pca[0, 1]), fontsize=8.5, fontweight="bold", textcoords="offset points", xytext=(8, -5))
    ax4.annotate(f"End ({timestamps[-1]:.1f}s)", (fused_pca[-1, 0], fused_pca[-1, 1]), fontsize=8.5, fontweight="bold", textcoords="offset points", xytext=(8, 5))
    ax4.set_title("4. Fused Multimodal Feature Manifold (PCA 2D Projection)", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Principal Component 1", fontsize=10)
    ax4.set_ylabel("Principal Component 2", fontsize=10)
    ax4.grid(True, linestyle=":", alpha=0.6)
    cbar4 = fig.colorbar(scatter, ax=ax4, fraction=0.046, pad=0.03)
    cbar4.set_label("Congestion Index (%)", fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_image_path, bbox_inches="tight", dpi=150)
    plt.close()

    return output_image_path


def main() -> None:
    args = parse_args()
    img_path = generate_visual_validation(
        features_npz_path=args.features,
        results_json_path=args.results_json,
        output_image_path=args.output,
    )

    # Read data for terminal report
    npz_data = np.load(args.features)
    timestamps = npz_data["timestamps"]
    yolo_features = npz_data["yolo_features"]
    vit_embeddings = npz_data["vit_embeddings"]
    fused_features = npz_data["fused_features"]

    counts = yolo_features[:, 0] * 50.0
    congestion = yolo_features[:, 8] * 100.0

    vit_unit = vit_embeddings / (np.linalg.norm(vit_embeddings, axis=1, keepdims=True) + 1e-8)
    sims = [float(np.dot(vit_unit[i], vit_unit[i + 1])) for i in range(len(timestamps) - 1)]
    mean_sim = float(np.mean(sims)) if sims else 1.0

    print("=" * 60)
    print("TrafficIQ M7 — Visual Validation Report")
    print("=" * 60)
    print(f"\nFigure saved to: {img_path}")
    print("\nMETRICS SUMMARY")
    print("---------------")
    print(f"Sampled Frames: {len(timestamps)}")
    print(f"Timestamp Duration: {timestamps[0]:.2f}s to {timestamps[-1]:.2f}s")
    print(f"Vehicle Count Range: {int(np.min(counts))} – {int(np.max(counts))} vehicles (Mean: {np.mean(counts):.1f})")
    print(f"Congestion Index Range: {np.min(congestion):.1f}% – {np.max(congestion):.1f}% (Mean: {np.mean(congestion):.1f}%)")
    print(f"ViT Temporal Coherence (Avg Consecutive Cosine Similarity): {mean_sim:.4f}")
    print(f"Fused Feature Shape: {fused_features.shape}")
    print("=" * 60)


if __name__ == "__main__":
    main()
