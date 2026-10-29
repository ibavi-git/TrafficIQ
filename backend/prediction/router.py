from fastapi import APIRouter, HTTPException

from backend.prediction.schemas import (
    TrafficPredictionRequest,
    TrafficPredictionResponse,
)
from backend.prediction.service import predict_traffic

router = APIRouter(
    prefix="/api/predictions",
    tags=["Traffic Prediction"],
)


@router.post(
    "/traffic-volume",
    response_model=TrafficPredictionResponse,
)
def predict_traffic_volume(
    request: TrafficPredictionRequest,
):
    try:
        prediction = predict_traffic(request.model_dump())

        return TrafficPredictionResponse(
            predicted_traffic_volume=round(prediction, 2),
            prediction_horizon="next hour",
            model_name="RandomForestRegressor",
        )

    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc