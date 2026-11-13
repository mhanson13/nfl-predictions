from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.calibration_selection import apply_calibrator, fit_calibrator, select_calibration_method


def _history(distort: bool, seed: int = 4) -> tuple[pd.Series, pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    n_per, seasons = 400, [2019, 2020, 2021, 2022, 2023]
    true_p = rng.uniform(0.1, 0.9, size=n_per * len(seasons))
    y = (rng.uniform(size=true_p.size) < true_p).astype(int)
    pred = 0.5 + (true_p - 0.5) * 0.4 if distort else true_p  # distorted = badly under-confident
    return pd.Series(pred), pd.Series(y), pd.Series(np.repeat(seasons, n_per))


def test_well_calibrated_probabilities_keep_identity():
    preds, actual, seasons = _history(distort=False)
    selection = select_calibration_method(preds, actual, seasons, season_window=2)
    assert selection.method == "none"
    assert selection.eval_seasons == [2020, 2021, 2022, 2023]


def test_under_confident_probabilities_get_a_calibrator():
    preds, actual, seasons = _history(distort=True)
    selection = select_calibration_method(preds, actual, seasons, season_window=2)
    assert selection.method in {"sigmoid", "isotonic"}
    assert selection.scores[selection.method]["brier"] < selection.scores["none"]["brier"]


def test_identity_calibrator_is_a_no_op():
    preds = pd.Series([0.2, 0.5, 0.8])
    assert fit_calibrator("none", preds, pd.Series([0, 1, 1])) is None
    np.testing.assert_allclose(apply_calibrator(None, preds), preds)
