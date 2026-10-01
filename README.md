# 🚦 TrafficIQ: Vision Transformer and YOLO-Based Explainable Intelligent Traffic Monitoring for Smart Cities

> An explainable, context-aware traffic monitoring system that detects, tracks, and counts vehicles, estimates congestion and anomalies, and runs efficiently on CPU/edge hardware — served through a live web dashboard.

---

## 1. Project Summary

Traditional traffic-monitoring CV systems are fast detectors but poor explainers: they output boxes and counts, not *why* a scene is congested or anomalous, and most are tuned for GPU throughput rather than affordable edge deployment.

**TrafficIQ** combines:

- **YOLO** for fast, local object detection (what vehicles, where)
- **A lightweight Vision Transformer (ViT)** for global scene context (how the traffic scene behaves as a whole)
- **Grad-CAM / attention-based explainability** so every prediction ships with a visual justification
- **CPU-first optimization** (ONNX Runtime / OpenVINO, model quantization, nano-scale backbones) so it can run on a laptop or a low-cost edge box, not just a GPU server

The end deliverable is a working full-stack product — not just a notebook: a FastAPI inference backend and a React dashboard showing live counts, congestion level, anomaly flags, and Grad-CAM overlays.

---

## 2. System Architecture

```
                 CAMERA / VIDEO / IMAGE
                          │
                          ▼
                 ┌──────────────────┐
                 │ OpenCV Preprocess│
                 └────────┬─────────┘
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
           ┌──────┐               ┌───────┐
           │ YOLO │               │  ViT  │
           │(local│               │(global│
           │boxes)│               │context│
           └──┬───┘               └───┬───┘
              │                       │
              └───────────┬───────────┘
                          ▼
                  Feature / Decision Fusion
                          │
                          ▼
                   Vehicle Tracking (ByteTrack)
                          │
                          ▼
                   Vehicle Counting (line/region)
                          │
                          ▼
               Congestion Scoring + Anomaly Detection
                          │
                          ▼
                Grad-CAM / Attention Explainability
                          │
                          ▼
               Model Optimization (ONNX / quantized)
                          │
                          ▼
                     CPU / Edge Inference
                          │
                          ▼
                        FastAPI
                          │
                          ▼
                    React Dashboard
```

---

## 3. Product Output

The dashboard surfaces, per video/stream:

| Metric | Example |
|---|---|
| Vehicles | 47 |
| Cars / Bikes / Buses / Trucks | 31 / 9 / 4 / 3 |
| Congestion | HIGH |
| FPS | 17.2 |
| Anomaly | Detected (stationary vehicle, 12s) |

Plus a three-panel visual: **original frame → detections/tracks → Grad-CAM explanation overlay**.

---

## 4. Tech Stack

| Layer              | Choice                                      |
|--------------------|---------------------------------------------|
| Language           | Python                                      |
| Computer vision    | OpenCV                                      |
| Detection          | Ultralytics YOLO (nano)                     |
| Tracking           | ByteTrack                                   |
| Transformer        | PyTorch + Hugging Face (MobileViT / DeiT)   |
| Explainability     | pytorch-grad-cam                            |
| Optimization       | ONNX Runtime                                |
| Backend            | FastAPI                                     |
| Frontend           | React + Vite                                |
| Charts             | Recharts                                    |
| Database           | PostgreSQL                                  |
| Container          | Docker                                      |
| Data labeling      | Roboflow / CVAT                             |
| Experiment tracking| MLflow                                      |
| Deployment         | Vercel (frontend) + Render/Railway (backend)|

---

## 5. Build Roadmap (AI Core → Product)

### AI Core
1. **Dataset** — start with a public set (BDD100K, UA-DETRAC, VisDrone, or COCO for initial YOLO testing); add a small custom-labeled set later if time allows.
2. **YOLO vehicle detection** — Ultralytics YOLO nano, class + confidence + bounding box.
3. **Video processing** — OpenCV frame-by-frame pipeline; first demo target is `traffic.mp4 → traffic_detected.mp4`.
4. **Object tracking** — ByteTrack assigns persistent IDs across frames.
5. **Vehicle counting** — start with line-crossing counters, upgrade to region- or lane-wise counting later.
6. **Congestion estimation** — start with a simple count-based Low/Medium/High bucket; upgrade to a weighted score (count + occupancy + speed + queue length) once you have data to calibrate thresholds.
7. **Anomaly detection** — start rule-based (stationary > N seconds, wrong-direction, sudden stop); explore Isolation Forest / One-Class SVM as a stretch goal.
8. **Vision Transformer** — add a lightweight ViT (MobileViT recommended for CPU) as a *scene context classifier* running alongside YOLO.
9. **Feature fusion** — start with the simplest approach: ViT produces a scene-context label/vector that augments YOLO's output, rather than a full joint feature-fusion architecture.
10. **Grad-CAM explainability** — attach Grad-CAM to the YOLO backbone first (CNN-friendly); decide on an attention/attribution method for the ViT branch once the hybrid architecture is fixed.
11. **Optimization** — baseline FPS/latency/RAM/model-size first, then reduce input resolution, switch to a smaller model, quantize (FP32→INT8), and/or export to ONNX Runtime.
12. **Edge deployment** — validate on a laptop first (this is your real baseline device); a Mini PC or Intel + OpenVINO setup is a good stretch target; Raspberry Pi/Jetson are optional extensions, not requirements.

### Product
13. **Backend** — FastAPI exposing `POST /detect`, `POST /upload-video`, `GET /traffic-status`, `GET /analytics`.
14. **Frontend** — React + Vite dashboard: live counts, congestion badge, FPS, video + Grad-CAM panels, analytics charts.
15. **Docker** — containerize frontend, backend, and AI engine separately.
16. **Deployment** — cloud prototype (Vercel + Render/Railway) for demo purposes; note that CPU-heavy inference on free tiers may not hit real-time FPS, so the edge-device path is the "real" deployment story.

---

## 6. Milestones (near-term)

- **M1 — YOLO vehicle detection:** complete
- **M2 — ByteTrack vehicle tracking:** complete
- **M3 — Line-crossing vehicle counting:** complete
- **M4 — Congestion and stopped-vehicle anomaly analysis:** complete
- **M5 — CPU performance baseline:** implemented
- **M6 — Vision Transformer:** not started
- **M7 — YOLO + ViT fusion:** not started
- **M8 — Grad-CAM / explainability:** not started
- **M9 — CPU optimization:** not started

M5 measures the existing YOLOv8n + ByteTrack + vehicle-filtering pipeline. It is a performance baseline, not an accuracy evaluation.

### Run M1–M5 (CPU)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python ai-core/detection/detect.py --source samples/YOUR_CLIP.jpg
python ai-core/tracking/track.py --source samples/YOUR_CLIP.mp4
python ai-core/tracking/count.py --source samples/YOUR_CLIP.mp4 --line-norm 0,0.5,1,0.5
python ai-core/traffic/analyze.py --source samples/YOUR_CLIP.mp4
python ai-core/benchmark/benchmark.py --source samples/traffic2.mp4
```

Use `--max-frames 200` to benchmark only part of a video. Results default to `outputs/benchmark/m5_baseline.json`; generated outputs and sample media remain gitignored. Track status in `docs/PROGRESS.md`.

---

## 7. Evaluation Plan

| Category | Metrics |
|---|---|
| Detection | mAP@50, mAP@50:95, Precision, Recall |
| Speed | FPS, latency/frame |
| Resources | RAM, CPU utilization, model size |
| Explainability | Localization accuracy, faithfulness, human interpretability |

**Core comparison table for the final report:**

| Model | mAP | FPS | CPU-only | Explainable |
|---|---|---|---|---|
| YOLO baseline | — | — | ✅ | ❌ |
| YOLO + ViT | — | — | ✅ | ❌ |
| YOLO + ViT + Grad-CAM | — | — | ✅ | ✅ |
| Optimized Hybrid | — | — | ✅ | ✅ |

---

## 8. Suggested Repository Structure

```
traffic-sense/
├── ai-core/
│   ├── data/                 # dataset scripts, download/prep
│   ├── detection/             # YOLO training/inference
│   ├── tracking/               # ByteTrack integration
│   ├── vit/                    # ViT scene-context model
│   ├── fusion/                 # YOLO+ViT decision fusion
│   ├── explainability/         # Grad-CAM / attention maps
│   ├── optimization/           # ONNX export, quantization scripts
│   └── evaluation/             # mAP, FPS, resource benchmarks
├── backend/                    # FastAPI app
├── frontend/                   # React + Vite dashboard
├── docker/                     # Dockerfiles + docker-compose
├── docs/                       # FAER proposal, report, diagrams
└── README.md
```

