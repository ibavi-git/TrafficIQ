# TrafficIQ progress (M1–M9)

## Complete

- **M1 — YOLO vehicle detection:** `ai-core/detection/detect.py`
- **M2 — ByteTrack tracking:** `ai-core/tracking/track.py`
- **M3 — Line-crossing counts:** `ai-core/tracking/count.py`
- **M4 — Congestion and stopped-vehicle anomaly analysis:** `ai-core/traffic/analyze.py`
- **M5 — CPU performance baseline:** `ai-core/benchmark/benchmark.py`
- **M6 — Vision Transformer:** `ai-core/vit/extract.py`
- **M7 — YOLO + ViT feature fusion:** `ai-core/fusion/fuse.py`

Run the complete M5 baseline from the repository root:

```bash
python ai-core/benchmark/benchmark.py --source samples/traffic2.mp4
```

Use `--max-frames 200` for a partial run. The JSON report defaults to `outputs/benchmark/m5_baseline.json`. Model loading is reported separately from processing FPS.

Run the M6 Vision Transformer feature extraction:

```bash
python ai-core/vit/extract.py --source samples/traffic2.mp4
```

Use `--sample-every 10` and `--batch-size 4`. Embeddings are saved to `outputs/vit/traffic2_embeddings.npz` and metrics to `outputs/vit/m6_vit_results.json`.

Run the M7 YOLO + ViT feature fusion:

```bash
python ai-core/fusion/fuse.py --source samples/traffic2.mp4 --sample-every 10
```

Fused features are saved to `outputs/fusion/traffic2_fused_features.npz` and metrics to `outputs/fusion/m7_fusion_results.json`.

## Not started

- **M8:** Grad-CAM / explainability
- **M9:** CPU optimization

The benchmark reports performance and traffic counts; it does not evaluate accuracy or claim real-world speed or density.
