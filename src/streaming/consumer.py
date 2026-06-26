import json

import yaml
from kafka import KafkaConsumer

from src.api.main import _predict_single, load_models


class MetricsConsumer:
    def __init__(
        self, config_path: str, bootstrap_servers: str = "localhost:9092", topic: str = "metrics"
    ):
        self.topic = topic
        self.config_path = config_path
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        load_models(config_path)

        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            auto_offset_reset="latest",
            enable_auto_commit=True,
        )

        self.alerts = []

    def run(self, max_records: int | None = None):
        print(f"Consuming from topic '{self.topic}'...")
        count = 0

        for message in self.consumer:
            record = message.value
            metrics = record["metrics"]
            true_label = record.get("label")

            is_anomaly, score = _predict_single(metrics)

            if is_anomaly:
                alert = {
                    "timestamp": record["timestamp"],
                    "metrics": metrics,
                    "anomaly_score": round(score, 4),
                    "predicted": 1,
                }
                if true_label is not None:
                    alert["true_label"] = true_label
                    alert["correct"] = bool(true_label == 1)
                self.alerts.append(alert)
                print(f"ALERT: {alert}")

            count += 1
            if max_records and count >= max_records:
                break

        print(f"Processed {count} records, {len(self.alerts)} alerts")
        return self.alerts
