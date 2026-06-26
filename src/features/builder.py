import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame, config: dict, dropna: bool = True) -> pd.DataFrame:
    fc = config["features"]
    result = df.copy()
    metrics = config["data"]["metrics"]

    if "timestamp" in result.columns:
        result = result.set_index("timestamp")

    _add_window_features(result, metrics, fc["window_sizes"])
    _add_lag_features(result, metrics, fc["lags"])
    _add_rate_of_change(result, metrics)

    if fc["include_time_features"]:
        _add_time_features(result)

    if dropna:
        result = result.dropna()
    else:
        result = result.fillna(0)

    return result


def _add_window_features(df: pd.DataFrame, metrics: list[str], window_sizes: list[int]):
    for col in metrics:
        if col not in df.columns:
            continue
        for w in window_sizes:
            if len(df) < w:
                continue
            df[f"{col}_mean_{w}"] = df[col].rolling(window=w, min_periods=1).mean()
            df[f"{col}_std_{w}"] = df[col].rolling(window=w, min_periods=1).std()
            df[f"{col}_min_{w}"] = df[col].rolling(window=w, min_periods=1).min()
            df[f"{col}_max_{w}"] = df[col].rolling(window=w, min_periods=1).max()


def _add_lag_features(df: pd.DataFrame, metrics: list[str], lags: list[int]):
    for col in metrics:
        if col not in df.columns:
            continue
        for lag in lags:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)


def _add_rate_of_change(df: pd.DataFrame, metrics: list[str]):
    for col in metrics:
        if col not in df.columns:
            continue
        df[f"{col}_roc"] = df[col].pct_change().replace([np.inf, -np.inf], 0).fillna(0)


def _add_time_features(df: pd.DataFrame):
    if not isinstance(df.index, pd.DatetimeIndex):
        return
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
