import numpy as np
import pandas as pd


def validate_metrics(df: pd.DataFrame) -> dict:
    errors = []
    warnings = []

    required_cols = ["timestamp", "cpu_pct", "mem_pct", "disk_io", "latency_ms", "error_rate"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        errors.append(f"Missing columns: {missing}")

    if "timestamp" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            warnings.append("timestamp column is not datetime type")
        if df["timestamp"].isnull().any():
            errors.append("timestamp contains null values")
        if not df["timestamp"].is_monotonic_increasing:
            warnings.append("timestamp is not monotonically increasing")

    numeric_cols = [c for c in required_cols if c != "timestamp" and c in df.columns]
    for col in numeric_cols:
        if df[col].isnull().any():
            errors.append(f"{col} contains null values")
        if not np.issubdtype(df[col].dtype, np.number):
            errors.append(f"{col} is not numeric")

    for col in ["cpu_pct", "mem_pct"]:
        if col in df.columns:
            out_of_range = ((df[col] < 0) | (df[col] > 100)).sum()
            if out_of_range > 0:
                warnings.append(f"{col}: {out_of_range} values outside [0, 100]")

    if "latency_ms" in df.columns:
        neg_latency = (df["latency_ms"] < 0).sum()
        if neg_latency > 0:
            errors.append(f"latency_ms: {neg_latency} negative values")

    if "error_rate" in df.columns:
        neg_errors = (df["error_rate"] < 0).sum()
        if neg_errors > 0:
            errors.append(f"error_rate: {neg_errors} negative values")

    if "is_anomaly" in df.columns:
        anomaly_pct = df["is_anomaly"].mean() * 100
        if anomaly_pct == 0:
            warnings.append("no anomalies detected in data")
        elif anomaly_pct > 20:
            warnings.append(f"high anomaly rate: {anomaly_pct:.1f}%")

    if "disk_io" in df.columns:
        neg_io = (df["disk_io"] < 0).sum()
        if neg_io > 0:
            errors.append(f"disk_io: {neg_io} negative values")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "n_rows": len(df),
        "n_columns": len(df.columns),
    }
