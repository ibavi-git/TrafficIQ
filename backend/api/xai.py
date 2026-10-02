"""On-demand M8 generation and job-specific result routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.services.traffic_processor import (
    XaiJobNotFoundError,
    XaiJobNotReadyError,
    job_manager,
)

router = APIRouter(prefix="/api/jobs", tags=["xai"])


@router.post("/{job_id}/xai")
async def generate_job_xai(job_id: str) -> dict[str, str | int]:
    try:
        return job_manager.request_xai(job_id)
    except XaiJobNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Analysis job not found.") from exc
    except XaiJobNotReadyError as exc:
        raise HTTPException(
            status_code=409,
            detail="Traffic analysis must complete before generating Explainable AI.",
        ) from exc


def _result_path(job_id: str, kind: str):
    job = job_manager.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Analysis job not found.")
    if job.xai_status != "completed":
        raise HTTPException(status_code=404, detail="Explainable AI result is not ready.")
    path = job.xai_video_path if kind == "video" else job.xai_metadata_path
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="Explainable AI result is unavailable.")
    return path


@router.get("/{job_id}/xai/video")
async def get_job_xai_video(job_id: str) -> FileResponse:
    path = _result_path(job_id, "video")
    return FileResponse(path, media_type="video/mp4", filename=path.name)


@router.get("/{job_id}/xai/metadata")
async def get_job_xai_metadata(job_id: str) -> FileResponse:
    path = _result_path(job_id, "metadata")
    return FileResponse(path, media_type="application/json", filename=path.name)