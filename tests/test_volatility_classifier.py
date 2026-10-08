from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.volatility_classifier import build_labels, evaluate_model, select_threshold_from_sweep


class _ProbabilityModel:
    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities, dtype=float)

    def predict_proba(self, _x):
        return np.column_stack([1.0 - self.probabilities, self.probabilities])


def test_build_labels_uses_error_percentile_not_environment_alone():
    df = pd.DataFrame(
        {
            "abs_margin_error": [1.0, 2.0, 8.0, 13.0, 21.0],
            "log_loss_per_game": [0.10, 0.20, 0.70, 1.30, 2.10],
            "indoor_game": [False, False, False, False, False],
            "wind_mph": [25.0, 25.0, 25.0, 25.0, 25.0],
            "qb_uncertain": [True, True, True, True, True],
            "home_rest": [5.0, 5.0, 5.0, 5.0, 5.0],
            "away_rest": [5.0, 5.0, 5.0, 5.0, 5.0],
        }
    )

    labels, stable = build_labels(df, percentile=0.6, use_logloss=True, use_margin=True)

    assert labels.tolist() == [0, 0, 1, 1, 1]
    assert stable.tolist() == [1, 1, 0, 0, 0]


def test_build_labels_respects_selected_error_metrics():
    df = pd.DataFrame(
        {
            "abs_margin_error": [100.0, 1.0, 2.0, 3.0],
            "log_loss_per_game": [0.10, 0.20, 0.30, 4.00],
        }
    )

    margin_labels, _ = build_labels(df, percentile=0.75, use_logloss=False, use_margin=True)
    logloss_labels, _ = build_labels(df, percentile=0.75, use_logloss=True, use_margin=False)

    assert margin_labels.tolist() == [1, 0, 0, 1]
    assert logloss_labels.tolist() == [0, 0, 1, 1]


def test_evaluate_model_can_select_threshold_below_half():
    model = _ProbabilityModel([0.20, 0.25, 0.31, 0.36, 0.39, 0.43])
    y_test = np.array([0, 0, 0, 1, 1, 1])

    metrics, _prob, pred, threshold = evaluate_model(
        model,
        np.zeros((len(y_test), 1)),
        y_test,
        threshold_metric="balanced",
    )

    assert threshold < 0.5
    assert metrics["pred_positive_rate"] > 0.0
    assert metrics["recall"] > 0.0
    assert pred.tolist().count(1) > 0


def test_select_threshold_from_sweep_respects_coverage_band():
    sweep = [
        {
            "threshold": 0.80,
            "f1": 0.91,
            "balanced_accuracy": 0.91,
            "precision": 1.0,
            "recall": 0.20,
            "pred_positive_rate": 0.05,
        },
        {
            "threshold": 0.50,
            "f1": 0.84,
            "balanced_accuracy": 0.88,
            "precision": 0.85,
            "recall": 0.70,
            "pred_positive_rate": 0.35,
        },
    ]

    selected = select_threshold_from_sweep(
        sweep,
        threshold_metric="balanced",
        min_pred_positive_rate=0.20,
        max_pred_positive_rate=0.65,
        target_pred_positive_rate=0.35,
        target_pred_positive_tolerance=0.20,
    )

    assert selected["threshold"] == 0.50
