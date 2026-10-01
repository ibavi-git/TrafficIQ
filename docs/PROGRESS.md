# TrafficIQ progress (M1–M9)

## Complete

- **M1 — YOLO vehicle detection:** `ai-core/detection/detect.py`
- **M2 — ByteTrack tracking:** `ai-core/tracking/track.py`
- **M3 — Line-crossing counts:** `ai-core/tracking/count.py`
- **M4 — Congestion and stopped-vehicle anomaly analysis:** `ai-core/traffic/analyze.py`
- **M5 — CPU performance baseline:** `ai-core/benchmark/benchmark.py`

Run the complete M5 baseline from the repository root:

```bash
python ai-core/benchmark/benchmark.py --source samples/traffic2.mp4
```

Use `--max-frames 200` for a partial run. The JSON report defaults to `outputs/benchmark/m5_baseline.json`. Model loading is reported separately from processing FPS.

## Not started

- **M6:** Vision Transformer
- **M7:** YOLO + ViT fusion
- **M8:** Grad-CAM / explainability
- **M9:** CPU optimization

The benchmark reports performance and traffic counts; it does not evaluate accuracy or claim real-world speed or density.
