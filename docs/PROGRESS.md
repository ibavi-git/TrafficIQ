# TrafficIQ progress (M1–M3)

Update the status line and notes as you finish each milestone. Later milestones (ViT, fusion, Grad-CAM, FastAPI, React) stay out of scope until M1–M3 are stable.

## M1 — YOLO vehicle detection

- **Status:** scaffolded (not verified on your data yet)
- **Script:** `ai-core/detection/detect.py`
- **Done when:** an image (and optionally a video) produces per-class vehicle boxes + confidences, and an annotated file is saved.
- **Command:**
  ```bash
  python ai-core/detection/detect.py --source samples/YOUR_IMAGE_OR_VIDEO
  ```
- **Notes:**
  - Model default: `yolov8n.pt` on **CPU**
  - Classes: COCO `bicycle`, `car`, `motorcycle`, `bus`, `truck` (`bicycle` + `motorcycle` → `bike`)
  - Swap the source path; do not assume `samples/traffic.mp4` exists until you add a clip.

## M2 — YOLO + ByteTrack on video

- **Status:** scaffolded (not verified on your data yet)
- **Script:** `ai-core/tracking/track.py`
- **Done when:** a video is written with persistent IDs on boxes, and the console reports unique IDs.
- **Command:**
  ```bash
  python ai-core/tracking/track.py --source samples/YOUR_VIDEO.mp4
  ```
- **Notes:**
  - Uses Ultralytics `model.track(..., tracker="bytetrack.yaml", persist=True)`
  - Same vehicle class filter as M1

## M3 — Line-crossing counts by class

- **Status:** scaffolded (not verified on your data yet)
- **Script:** `ai-core/tracking/count.py`
- **Done when:** per-class `car` / `bike` / `bus` / `truck` counts match vehicles that crossed a visible line (each track ID counted once).
- **Command:**
  ```bash
  python ai-core/tracking/count.py --source samples/YOUR_VIDEO.mp4 --line-norm 0,0.5,1,0.5
  ```
- **Notes:**
  - Default line: horizontal midline (`--line-norm 0,0.5,1,0.5`)
  - Pixel line: `--line x1,y1,x2,y2` (vertical example: `--line 640,0,640,720`)
  - Tune the line to the road in *your* clip; midline is only a placeholder.

## Not started (later sessions)

- ViT scene context (`ai-core/vit/`)
- YOLO + ViT fusion (`ai-core/fusion/`)
- Grad-CAM (`ai-core/explainability/`)
- Optimization / ONNX (`ai-core/optimization/`)
- FPS / mAP eval scripts (`ai-core/evaluation/`)
- FastAPI (`backend/`)
- React dashboard (`frontend/`)
- Docker (`docker/`)
