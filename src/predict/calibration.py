from __future__ import annotations

import numpy as np


class SigmoidCalibrator:
    """Small NumPy-only Platt-style calibrator for scalar probabilities."""

    def __init__(self, intercept: float, coefficient: float, x_mean: float, x_scale: float) -> None:
        self.intercept = float(intercept)
        self.coefficient = float(coefficient)
        self.x_mean = float(x_mean)
        self.x_scale = float(x_scale) if float(x_scale) > 0 else 1.0

    def _predict_positive(self, values: np.ndarray) -> np.ndarray:
        x = (np.asarray(values, dtype=float).reshape(-1) - self.x_mean) / self.x_scale
        z = np.clip(self.intercept + self.coefficient * x, -35.0, 35.0)
        return 1.0 / (1.0 + np.exp(-z))

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        positive = self._predict_positive(values)
        return np.column_stack([1.0 - positive, positive])
