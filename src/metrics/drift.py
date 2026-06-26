from collections import deque
from typing import Any

import numpy as np


class DriftDetector:
    def __init__(self, window_size: int = 1000, threshold: float = 0.3):
        self.window_size = window_size
        self.threshold = threshold
        self.reference_stats: dict[str, dict[str, float]] = {}
        self.recent_values: dict[str, deque[float]] = {}
        self.drift_detected: dict[str, bool] = {}
        self.total_predictions = 0

    def fit_reference(self, feature_df, feature_cols: list[str]) -> None:
        for col in feature_cols:
            if col not in feature_df.columns:
                continue
            values = feature_df[col].dropna().values
            if len(values) < 2:
                continue
            self.reference_stats[col] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
            self.recent_values[col] = deque(maxlen=self.window_size)

    def update(self, metrics: dict[str, float]) -> dict[str, Any]:
        self.total_predictions += 1

        for key, value in metrics.items():
            if key not in self.recent_values:
                self.recent_values[key] = deque(maxlen=self.window_size)
            self.recent_values[key].append(float(value))

        if self.total_predictions < self.window_size or not self.reference_stats:
            return {"drift_detected": False, "total_predictions": self.total_predictions}

        drift_scores = {}
        for col, ref in self.reference_stats.items():
            if col not in self.recent_values or len(self.recent_values[col]) < 10:
                continue

            recent = np.array(self.recent_values[col])
            recent_mean = float(np.mean(recent))
            ref_mean = ref["mean"]

            if abs(ref_mean) < 1e-10:
                drift_scores[col] = 0.0
            else:
                drift_scores[col] = abs(recent_mean - ref_mean) / (abs(ref_mean) + 1e-10)

        max_drift = max(drift_scores.values()) if drift_scores else 0.0
        drift_triggered = max_drift > self.threshold

        return {
            "drift_detected": drift_triggered,
            "max_drift_score": round(max_drift, 4),
            "drift_scores": {k: round(v, 4) for k, v in drift_scores.items()},
            "total_predictions": self.total_predictions,
        }
