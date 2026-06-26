from unittest.mock import MagicMock, patch


class TestMetricsProducer:
    def test_generates_data_from_config(self):
        from src.streaming.producer import MetricsProducer

        with patch("src.streaming.producer.KafkaProducer"):
            producer = MetricsProducer("configs/streaming.yaml")
            assert producer.topic == "metrics"
            assert len(producer.data) > 0
            assert "cpu_pct" in producer.data.columns

    def test_send_next_formats_record_correctly(self):
        from src.streaming.producer import MetricsProducer

        with patch("src.streaming.producer.KafkaProducer") as mock_kp:
            producer = MetricsProducer("configs/streaming.yaml")
            result = producer.send_next(delay=0)

            assert result is True
            mock_kp.return_value.send.assert_called_once()
            call_args = mock_kp.return_value.send.call_args
            assert call_args is not None

            record = call_args[1]["value"]
            assert "timestamp" in record
            assert "metrics" in record
            assert "label" in record
            assert "cpu_pct" in record["metrics"]
            assert isinstance(record["metrics"]["cpu_pct"], float)

    def test_send_next_returns_false_at_end(self):
        from src.streaming.producer import MetricsProducer

        with patch("src.streaming.producer.KafkaProducer"):
            producer = MetricsProducer("configs/streaming.yaml")
            producer.idx = len(producer.data)
            result = producer.send_next(delay=0)
            assert result is False

    def test_close_calls_producer_close(self):
        from src.streaming.producer import MetricsProducer

        with patch("src.streaming.producer.KafkaProducer") as mock_kp:
            producer = MetricsProducer("configs/streaming.yaml")
            producer.close()
            mock_kp.return_value.close.assert_called_once()


class TestMetricsConsumer:
    def test_init_loads_models_and_creates_consumer(self):
        from src.streaming.consumer import MetricsConsumer

        with patch("src.streaming.consumer.KafkaConsumer") as mock_kc:
            with patch("src.streaming.consumer.load_models") as mock_load:
                consumer = MetricsConsumer("configs/streaming.yaml")
                assert consumer.topic == "metrics"
                mock_load.assert_called_once_with("configs/streaming.yaml")
                mock_kc.assert_called_once()

    def test_process_record_handles_alert(self):
        from src.streaming.consumer import MetricsConsumer

        with patch("src.streaming.consumer.KafkaConsumer"):
            with patch("src.streaming.consumer.load_models"):
                consumer = MetricsConsumer("configs/streaming.yaml")
                consumer.alerts = []

                record = {
                    "timestamp": "2024-01-01T00:00:00",
                    "metrics": {
                        "cpu_pct": 98.0,
                        "mem_pct": 95.0,
                        "disk_io": 80.0,
                        "latency_ms": 500.0,
                        "error_rate": 8.0,
                    },
                    "label": 1,
                }

                with patch("src.streaming.consumer._predict_single") as mock_pred:
                    mock_pred.return_value = (True, 0.95)
                    msg = MagicMock()
                    msg.value = record
                    consumer.consumer = [msg]

                    alerts = consumer.run(max_records=1)

                    assert len(alerts) == 1
                    assert alerts[0]["predicted"] == 1
                    assert alerts[0]["correct"] is True
                    assert alerts[0]["anomaly_score"] == 0.95

    def test_process_record_no_alert(self):
        from src.streaming.consumer import MetricsConsumer

        with patch("src.streaming.consumer.KafkaConsumer"):
            with patch("src.streaming.consumer.load_models"):
                consumer = MetricsConsumer("configs/streaming.yaml")
                consumer.alerts = []

                record = {
                    "timestamp": "2024-01-01T00:00:00",
                    "metrics": {
                        "cpu_pct": 50.0,
                        "mem_pct": 60.0,
                        "disk_io": 30.0,
                        "latency_ms": 100.0,
                        "error_rate": 0.5,
                    },
                    "label": 0,
                }

                with patch("src.streaming.consumer._predict_single") as mock_pred:
                    mock_pred.return_value = (False, 0.05)
                    msg = MagicMock()
                    msg.value = record
                    consumer.consumer = [msg]

                    alerts = consumer.run(max_records=1)

                    assert len(alerts) == 0

    def test_alerts_collected(self):
        from src.streaming.consumer import MetricsConsumer

        with patch("src.streaming.consumer.KafkaConsumer"):
            with patch("src.streaming.consumer.load_models"):
                consumer = MetricsConsumer("configs/streaming.yaml")

                alert1 = {"timestamp": "t1", "predicted": 1, "correct": True}
                alert2 = {"timestamp": "t2", "predicted": 1, "correct": False}
                consumer.alerts = [alert1, alert2]

                assert len(consumer.alerts) == 2
                assert consumer.alerts[0]["correct"] is True
                assert consumer.alerts[1]["correct"] is False


class TestRunStreamingCLI:
    def test_imports(self):
        import run_streaming

        assert run_streaming is not None

    def test_producer_import(self):
        from src.streaming.producer import MetricsProducer

        assert MetricsProducer is not None

    def test_consumer_import(self):
        from src.streaming.consumer import MetricsConsumer

        assert MetricsConsumer is not None
