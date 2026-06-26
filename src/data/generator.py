from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def generate_metrics(config: dict) -> pd.DataFrame:
    seed = config["data"]["seed"]
    rng = np.random.default_rng(seed)

    n_days = config["data"]["n_days"]
    freq_minutes = config["data"]["freq_minutes"]
    anomaly_rate = config["data"]["anomaly_rate"]

    n_points = int(n_days * 24 * 60 / freq_minutes)
    timestamps = [
        datetime(2024, 1, 1) + timedelta(minutes=i * freq_minutes) for i in range(n_points)
    ]

    t = np.arange(n_points, dtype=np.float64)
    period = 24 * 60 // freq_minutes

    cpu_pct = _seasonal_series(t, period, base=50, amplitude=20, noise=5, rng=rng)
    mem_pct = _seasonal_series(t, period, base=65, amplitude=10, noise=3, trend=0.005, rng=rng)
    disk_io = _seasonal_series(t, period, base=30, amplitude=15, noise=8, rng=rng)
    latency_ms = _seasonal_series(t, period, base=100, amplitude=40, noise=15, rng=rng)
    error_rate = _seasonal_series(t, period, base=0.5, amplitude=0.3, noise=0.2, rng=rng)
    error_rate = np.clip(error_rate, 0, None)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "cpu_pct": cpu_pct,
            "mem_pct": mem_pct,
            "disk_io": disk_io,
            "latency_ms": latency_ms,
            "error_rate": error_rate,
            "is_anomaly": 0,
        }
    )

    df = _inject_anomalies(df, anomaly_rate, rng)
    return df


def _seasonal_series(
    t: np.ndarray,
    period: int,
    base: float,
    amplitude: float,
    noise: float,
    rng: np.random.Generator,
    trend: float = 0.0,
) -> np.ndarray:
    seasonal = amplitude * np.sin(2 * np.pi * t / period)
    trend_component = trend * t
    noise_component = rng.normal(0, noise, size=len(t))
    values = base + seasonal + trend_component + noise_component
    return np.clip(values, 0, 100)


def _inject_anomalies(df: pd.DataFrame, rate: float, rng: np.random.Generator) -> pd.DataFrame:
    n_anomalies = max(1, int(len(df) * rate))
    anomaly_indices = rng.choice(len(df), size=n_anomalies, replace=False)
    metrics = ["cpu_pct", "mem_pct", "disk_io", "latency_ms", "error_rate"]
    n_cols = len(metrics)

    for idx in anomaly_indices:
        col = metrics[rng.integers(0, n_cols)]
        anomaly_type = rng.choice(["spike", "level_shift", "flatline"], p=[0.5, 0.3, 0.2])

        if anomaly_type == "spike":
            multiplier = rng.uniform(2.0, 4.0)
            df.loc[idx, col] = df.loc[idx, col] * multiplier
            df.loc[idx, col] = min(df.loc[idx, col], 100)

        elif anomaly_type == "level_shift":
            shift = rng.uniform(20, 50) * rng.choice([-1, 1])
            shift_len = rng.integers(6, 24)
            end = min(idx + shift_len, len(df) - 1)
            df.loc[idx:end, col] = df.loc[idx:end, col] + shift
            df.loc[idx:end, col] = df.loc[idx:end, col].clip(0, 100)

        elif anomaly_type == "flatline":
            flat_len = rng.integers(6, 18)
            end = min(idx + flat_len, len(df) - 1)
            constant_val = rng.uniform(df[col].quantile(0.1), df[col].quantile(0.3))
            df.loc[idx:end, col] = constant_val

        df.loc[idx, "is_anomaly"] = 1
        if anomaly_type in ("level_shift", "flatline"):
            df.loc[idx + 1 : end, "is_anomaly"] = 1

    return df
