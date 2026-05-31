# src/api/schemas.py
from pydantic import BaseModel, Field


CLASS_NAMES = [
    "clear",
    "fog",
    "fog_rain",
    "night",
    "night_fog",
    "night_rain",
    "night_snow",
    "rain",
    "snow",
]


class PredictionResponse(BaseModel):
    class_name: str   = Field(..., alias="class",        description="Предсказанный класс погоды")
    confidence: float = Field(...,                        description="Уверенность модели (0–1)")
    probabilities: dict[str, float] = Field(...,          description="Вероятности по всем 9 классам")
    model:      str   = Field(...,                        description="Используемая модель")
    inference_ms: float = Field(...,                      description="Время инференса, мс (0 если из кэша)")
    cached:     bool  = Field(False,                      description="True если результат из кэша")

    model_config = {"populate_by_name": True}


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    cache_stats: dict[str, int]


class ErrorResponse(BaseModel):
    detail: str