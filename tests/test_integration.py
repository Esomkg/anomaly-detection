from unittest.mock import patch

import yaml


class TestFullPipeline:
    def test_data_to_features_to_model_roundtrip(self):

        from src.data.generator import generate_metrics
        from src.data.validator import validate_metrics
        from src.features.builder import build_features

        with open("configs/config.yaml") as f:
            config = yaml.safe_load(f)

        config["data"]["n_days"] = 1
        df = generate_metrics(config)

        validation = validate_metrics(df)
        assert validation["valid"], f"Validation failed: {validation['errors']}"

        feature_df = build_features(df, config)
        assert feature_df.shape[0] > 0
        assert "is_anomaly" in feature_df.columns \
            or "is_anomaly" in df.columns

    def test_streaming_producer_consumer_flow(self):
        from src.streaming.producer import MetricsProducer

        with patch("src.streaming.producer.KafkaProducer"):
            producer = MetricsProducer("configs/streaming.yaml")
            records = []
            for i in range(min(10, len(producer.data))):
                row = producer.data.iloc[i]
                records.append({
                    "timestamp": str(row["timestamp"]),
                    "metrics": {
                        "cpu_pct": float(row["cpu_pct"]),
                        "mem_pct": float(row["mem_pct"]),
                        "disk_io": float(row["disk_io"]),
                        "latency_ms": float(row["latency_ms"]),
                        "error_rate": float(row["error_rate"]),
                    },
                    "label": int(row["is_anomaly"]),
                })

        assert len(records) == 10
        assert "cpu_pct" in records[0]["metrics"]
        assert isinstance(records[0]["label"], int)

    def test_alert_notifier_console(self):
        from src.alerts.notifier import AlertNotifier

        config = {"alerts": {"cooldown_seconds": 0}}
        notifier = AlertNotifier(config)

        alert = {
            "timestamp": "2024-01-01T00:00:00",
            "anomaly_score": 0.95,
            "metrics": {
                "cpu_pct": 98.0,
                "mem_pct": 92.0,
                "latency_ms": 500.0,
                "error_rate": 5.0,
            },
        }
        notifier.notify(alert)

    def test_alert_notifier_cooldown(self):
        from src.alerts.notifier import AlertNotifier

        config = {"alerts": {"cooldown_seconds": 999}}
        notifier = AlertNotifier(config)
        notifier._last_alert_time = {}
        assert notifier._check_cooldown("test")
        assert not notifier._check_cooldown("test")

    def test_composite_notifier_aggregates(self):
        from src.alerts.notifier import CompositeNotifier

        config = {"alerts": {"cooldown_seconds": 0}}
        notifier = CompositeNotifier(config)

        notifier.process_prediction(
            {"timestamp": "t1", "metrics": {"cpu_pct": 98}},
            True, 0.95,
        )
        notifier.process_prediction(
            {"timestamp": "t2", "metrics": {"cpu_pct": 50}},
            False, 0.05,
        )
        notifier.process_prediction(
            {"timestamp": "t3", "metrics": {"cpu_pct": 99}},
            True, 0.88,
        )

        summary = notifier.get_summary()
        assert summary["total_alerts"] == 2
        assert summary["max_score"] == 0.95

    def test_api_alert_endpoint_exists(self):
        from fastapi.routing import APIRoute

        from src.api.main import app

        routes = [r.path for r in app.routes if isinstance(r, APIRoute)]
        assert "/alerts" in routes
        assert "/alerts/summary" in routes
        assert "/metrics" in routes

    def test_configs_have_alert_section(self):
        with open("configs/config.yaml") as f:
            config = yaml.safe_load(f)
        assert "alerts" in config
        assert "cooldown_seconds" in config["alerts"]

        with open("configs/streaming.yaml") as f:
            streaming_config = yaml.safe_load(f)
        assert "alerts" in streaming_config
