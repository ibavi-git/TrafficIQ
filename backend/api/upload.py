"""Video upload endpoint for TrafficIQ analysis jobs."""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.services.traffic_processor import job_manager

router = APIRouter(prefix="/api", tags=["upload"])
SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov"}
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
CHUNK_BYTES = 1024 * 1024


@router.post("/upload")
async def upload_video(file: UploadFile = File(...)) -> dict[str, str]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Choose a video file to upload.")

    filename = file.filename.replace("\\", "/").rsplit("/", 1)[-1]
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail="Unsupported video format. Upload an MP4, AVI, or MOV file.",
        )

    job = job_manager.create_job(suffix, filename)
    size = 0
    try:
        with job.source_path.open("wb") as destination:
            while chunk := await file.read(CHUNK_BYTES):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    job.source_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail="Video is too large. The upload limit is 500 MB.",
                    )
                destination.write(chunk)
    finally:
        await file.close()

    if size == 0:
        job.source_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="The uploaded video is empty.")

    job_manager.start_job(job.job_id)
    return {"job_id": job.job_id, "status": "queued"}