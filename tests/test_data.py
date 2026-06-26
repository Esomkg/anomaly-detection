import pandas as pd
import pytest
import yaml

from src.data.generator import generate_metrics
from src.data.validator import validate_metrics


@pytest.fixture
def config():
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["data"]["n_days"] = 1
    cfg["data"]["anomaly_rate"] = 0.1
    return cfg


def test_generate_metrics_returns_dataframe(config):
    df = generate_metrics(config)
    assert isinstance(df, pd.DataFrame)


def test_generate_metrics_has_required_columns(config):
    df = generate_metrics(config)
    required = [
        "timestamp",
        "cpu_pct",
        "mem_pct",
        "disk_io",
        "latency_ms",
        "error_rate",
        "is_anomaly",
    ]
    for col in required:
        assert col in df.columns, f"Missing column: {col}"


def test_generate_metrics_correct_length(config):
    df = generate_metrics(config)
    expected = int(1 * 24 * 60 / config["data"]["freq_minutes"])
    assert len(df) == expected


def test_generate_metrics_has_anomalies(config):
    df = generate_metrics(config)
    assert df["is_anomaly"].sum() > 0


def test_generate_metrics_values_in_range(config):
    df = generate_metrics(config)
    assert df["cpu_pct"].between(0, 100).all()
    assert df["mem_pct"].between(0, 100).all()
    assert df["latency_ms"].ge(0).all()
    assert df["error_rate"].ge(0).all()


def test_validate_metrics_valid(config):
    df = generate_metrics(config)
    result = validate_metrics(df)
    assert result["valid"]


def test_validate_metrics_detects_missing_column():
    df = pd.DataFrame({"cpu_pct": [1.0]})
    result = validate_metrics(df)
    assert not result["valid"]


def test_generate_metrics_reproducible(config):
    df1 = generate_metrics(config)
    df2 = generate_metrics(config)
    pd.testing.assert_frame_equal(df1, df2)
