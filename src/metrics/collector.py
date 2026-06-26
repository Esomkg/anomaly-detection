from prometheus_client import Counter, Gauge, Histogram, generate_latest

predict_requests = Counter(
    "anomaly_predict_requests_total",
    "Total predict requests",
    ["endpoint"],
)

predict_anomalies = Counter(
    "anomaly_predict_anomalies_total",
    "Total anomaly predictions",
    ["endpoint"],
)

predict_non_anomalies = Counter(
    "anomaly_predict_non_anomalies_total",
    "Total non-anomaly predictions",
    ["endpoint"],
)

predict_latency = Histogram(
    "anomaly_predict_latency_seconds",
    "Predict request latency in seconds",
    ["endpoint"],
)

model_loaded = Gauge(
    "anomaly_model_loaded",
    "Whether the model is loaded (1) or not (0)",
)

anomaly_score = Gauge(
    "anomaly_score_latest",
    "Most recent anomaly score",
)


def metrics_response() -> bytes:
    return generate_latest()
