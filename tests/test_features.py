import numpy as np
import pandas as pd
import pytest
import yaml

from src.features.builder import build_features


@pytest.fixture
def config():
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["data"]["n_days"] = 1
    return cfg


@pytest.fixture
def sample_df(config):
    from src.data.generator import generate_metrics

    return generate_metrics(config)


def test_build_features_returns_dataframe(sample_df, config):
    result = build_features(sample_df, config)
    assert isinstance(result, pd.DataFrame)


def test_build_features_adds_window_columns(sample_df, config):
    result = build_features(sample_df, config)
    for metric in config["data"]["metrics"]:
        for w in config["features"]["window_sizes"]:
            for stat in ["mean", "std", "min", "max"]:
                col = f"{metric}_{stat}_{w}"
                assert col in result.columns, f"Missing: {col}"


def test_build_features_adds_lag_columns(sample_df, config):
    result = build_features(sample_df, config)
    for metric in config["data"]["metrics"]:
        for lag in config["features"]["lags"]:
            col = f"{metric}_lag_{lag}"
            assert col in result.columns, f"Missing: {col}"


def test_build_features_adds_rate_of_change(sample_df, config):
    result = build_features(sample_df, config)
    for metric in config["data"]["metrics"]:
        col = f"{metric}_roc"
        assert col in result.columns, f"Missing: {col}"


def test_build_features_adds_time_features(sample_df, config):
    result = build_features(sample_df, config)
    time_features = ["hour", "day_of_week", "is_weekend", "hour_sin", "hour_cos"]
    for feat in time_features:
        assert feat in result.columns, f"Missing: {feat}"


def test_build_features_removes_nan(sample_df, config):
    result = build_features(sample_df, config)
    assert result.isnull().sum().sum() == 0, "NaN values present after feature building"


def test_build_features_keeps_label(sample_df, config):
    result = build_features(sample_df, config)
    assert "is_anomaly" in result.columns


def test_build_features_hour_encoding(sample_df, config):
    result = build_features(sample_df, config)
    assert np.isclose((result["hour_sin"] ** 2 + result["hour_cos"] ** 2).mean(), 1.0, atol=1e-6)
