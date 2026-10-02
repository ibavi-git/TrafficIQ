"""WebSocket endpoint for live TrafficIQ job updates."""

from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.services.traffic_processor import job_manager

router = APIRouter(tags=["websocket"])


@router.websocket("/ws/{job_id}")
async def job_updates(websocket: WebSocket, job_id: str) -> None:
    await websocket.accept()
    queue = await job_manager.subscribe(job_id)
    if queue is None:
        await websocket.send_json(
            {"type": "status", "status": "error", "message": "Analysis job not found."}
        )
        await websocket.close(code=4404)
        return

    try:
        while True:
            message = await queue.get()
            await websocket.send_json(message)
            if message.get("type") == "status" and message.get("status") in {
                "completed",
                "error",
            }:
                break
    except WebSocketDisconnect:
        pass
    finally:
        job_manager.unsubscribe(job_id, queue)