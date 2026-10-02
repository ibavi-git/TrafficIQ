"""FastAPI application for the TrafficIQ MVP."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend.api.upload import router as upload_router
from backend.api.websocket import router as websocket_router
from backend.api.xai import router as xai_router

REPO_ROOT = Path(__file__).resolve().parents[1]
XAI_PREVIEW = REPO_ROOT / "outputs" / "xai" / "m8_gradcam_preview.mp4"

app = FastAPI(title="TrafficIQ API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
app.include_router(upload_router)
app.include_router(websocket_router)
app.include_router(xai_router)


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/xai/preview")
async def xai_preview() -> FileResponse:
    if not XAI_PREVIEW.is_file():
        raise HTTPException(status_code=404, detail="The M8 Grad-CAM preview is unavailable.")
    return FileResponse(XAI_PREVIEW, media_type="video/mp4")