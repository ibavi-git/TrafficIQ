from pydantic import BaseModel, Field


class TrafficPredictionRequest(BaseModel):
    temp: float
    rain_1h: float = Field(ge=0)
    snow_1h: float = Field(ge=0)
    clouds_all: float = Field(ge=0, le=100)

    hour: int = Field(ge=0, le=23)
    dayofweek: int = Field(ge=0, le=6)
    month: int = Field(ge=1, le=12)
    is_weekend: int = Field(ge=0, le=1)

    lag_1: float = Field(ge=0)
    lag_2: float = Field(ge=0)
    lag_3: float = Field(ge=0)
    lag_24: float = Field(ge=0)


class TrafficPredictionResponse(BaseModel):
    predicted_traffic_volume: float
    prediction_horizon: str
    model_name: str