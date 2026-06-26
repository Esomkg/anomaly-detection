import json
import time

import yaml
from kafka import KafkaProducer

from src.data.generator import generate_metrics


class MetricsProducer:
    def __init__(
        self, config_path: str, bootstrap_servers: str = "localhost:9092", topic: str = "metrics"
    ):
        self.topic = topic
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
        )
        self.data = generate_metrics(self.config)
        self.idx = 0

    def send_next(self, delay: float = 0.01):
        if self.idx >= len(self.data):
            return False

        row = self.data.iloc[self.idx]
        record = {
            "timestamp": str(row["timestamp"]),
            "metrics": {
                "cpu_pct": float(row["cpu_pct"]),
                "mem_pct": float(row["mem_pct"]),
                "disk_io": float(row["disk_io"]),
                "latency_ms": float(row["latency_ms"]),
                "error_rate": float(row["error_rate"]),
            },
            "label": int(row["is_anomaly"]),
        }

        self.producer.send(self.topic, value=record)
        self.idx += 1
        time.sleep(delay)
        return True

    def run(self, delay: float = 0.01):
        print(f"Producing to topic '{self.topic}'...")
        while self.send_next(delay):
            if self.idx % 100 == 0:
                print(f"Sent {self.idx} records")
        print(f"Done. Sent {self.idx} total records")

    def close(self):
        self.producer.close()
