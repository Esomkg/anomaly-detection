import json
import time
from contextlib import asynccontextmanager

import joblib
import pandas as pd
import yaml
from fastapi import FastAPI, Response

from src.api.schemas import (
    BatchMetricsInput,
    BatchPredictionOutput,
    HealthOutput,
    MetricsInput,
    PredictionOutput,
)
from src.features.builder import build_features
from src.metrics.collector import (
    metrics_response,
    model_loaded,
    predict_anomalies,
    predict_latency,
    predict_non_anomalies,
    predict_requests,
)
from src.models.autoencoder import LSTMAutoencoder, compute_anomaly_scores
from src.models.ensemble import EnsembleDetector
from src.models.isolation import IsolationForestModel

MODEL_CONFIG = None
AUTOENCODER = None
ISOLATION_FOREST = None
ENSEMBLE = None
METADATA = None


def load_models(config_path: str = "configs/config.yaml"):
    global AUTOENCODER, ISOLATION_FOREST, ENSEMBLE, MODEL_CONFIG, METADATA

    with open(config_path) as f:
        MODEL_CONFIG = yaml.safe_load(f)

    with open("models/metadata.json") as f:
        METADATA = json.load(f)

    AUTOENCODER = LSTMAutoencoder.load_from_checkpoint(
        "models/autoencoder.ckpt",
        input_size=METADATA["input_size"],
        hidden_size=METADATA["ae_hidden_size"],
        num_layers=METADATA["ae_num_layers"],
        learning_rate=METADATA["ae_learning_rate"],
    )
    AUTOENCODER.eval()

    iso_data = joblib.load("models/isolation_forest.pkl")
    ISOLATION_FOREST = IsolationForestModel(MODEL_CONFIG)
    ISOLATION_FOREST.model = iso_data["model"]
    ISOLATION_FOREST.scaler = iso_data["scaler"]

    ENSEMBLE = EnsembleDetector(MODEL_CONFIG)
    ENSEMBLE.threshold = METADATA["ensemble_threshold"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    is_loaded = all([AUTOENCODER is not None, ISOLATION_FOREST is not None])
    model_loaded.set(1 if is_loaded else 0)
    yield


app = FastAPI(title="Anomaly Detection API", version="0.1.0", lifespan=lifespan)


def _predict_single(data: dict) -> tuple[bool, float]:
    df = pd.DataFrame([{**data, "timestamp": pd.Timestamp.now()}])
    feature_df = build_features(df, MODEL_CONFIG, dropna=False)
    feature_cols = METADATA["feature_cols"]
    for col in feature_cols:
        if col not in feature_df.columns:
            feature_df[col] = 0.0
    X = feature_df[feature_cols].values

    scores_ae = compute_anomaly_scores(
        AUTOENCODER, X, MODEL_CONFIG["model"]["autoencoder"]["sequence_length"]
    )
    scores_if = ISOLATION_FOREST.predict(X)
    predictions, combined = ENSEMBLE.predict(scores_ae, scores_if)
    return bool(predictions[-1]), float(combined[-1])


@app.get("/health", response_model=HealthOutput)
def health():
    return HealthOutput(
        status="ok",
        model_loaded=all([AUTOENCODER is not None, ISOLATION_FOREST is not None]),
    )


@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: MetricsInput):
    t_start = time.perf_counter()
    is_anomaly, score = _predict_single(input_data.model_dump())
    elapsed = time.perf_counter() - t_start

    predict_requests.labels(endpoint="/predict").inc()
    predict_latency.labels(endpoint="/predict").observe(elapsed)
    if is_anomaly:
        predict_anomalies.labels(endpoint="/predict").inc()
    else:
        predict_non_anomalies.labels(endpoint="/predict").inc()

    return PredictionOutput(
        is_anomaly=is_anomaly,
        anomaly_score=round(score, 4),
        threshold=round(ENSEMBLE.threshold, 4),
    )


@app.post("/predict_batch", response_model=BatchPredictionOutput)
def predict_batch(input_data: BatchMetricsInput):
    t_start = time.perf_counter()
    results = []
    anomaly_count = 0

    for instance in input_data.instances:
        is_anomaly, score = _predict_single(instance.model_dump())
        results.append(
            PredictionOutput(
                is_anomaly=is_anomaly,
                anomaly_score=round(score, 4),
                threshold=round(ENSEMBLE.threshold, 4),
            )
        )
        if is_anomaly:
            anomaly_count += 1

    elapsed = time.perf_counter() - t_start
    n = len(input_data.instances)

    predict_requests.labels(endpoint="/predict_batch").inc()
    predict_latency.labels(endpoint="/predict_batch").observe(elapsed)
    predict_anomalies.labels(endpoint="/predict_batch").inc(anomaly_count)
    predict_non_anomalies.labels(endpoint="/predict_batch").inc(n - anomaly_count)

    return BatchPredictionOutput(predictions=results)


@app.get("/metrics")
def metrics():
    return Response(content=metrics_response(), media_type="text/plain")
