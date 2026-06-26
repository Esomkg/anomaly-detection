from pydantic import BaseModel, Field


class MetricsInput(BaseModel):
    cpu_pct: float = Field(..., ge=0, le=100)
    mem_pct: float = Field(..., ge=0, le=100)
    disk_io: float = Field(..., ge=0)
    latency_ms: float = Field(..., ge=0)
    error_rate: float = Field(..., ge=0)


class BatchMetricsInput(BaseModel):
    instances: list[MetricsInput]


class PredictionOutput(BaseModel):
    is_anomaly: bool
    anomaly_score: float
    threshold: float


class BatchPredictionOutput(BaseModel):
    predictions: list[PredictionOutput]


class HealthOutput(BaseModel):
    status: str
    model_loaded: bool
