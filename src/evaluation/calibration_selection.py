"""Out-of-sample selection of the win-probability calibration layer.

Live prediction applies the calibrator to the raw model probability *before*
market caps and volatility shrinkage, so candidates are fit and scored on raw
probabilities. Each candidate is judged walk-forward: fit on the
``season_window`` seasons before a season, scored on that season. A calibrator
is only kept if it beats leaving the probabilities alone ("none") out of
sample; an in-sample isotonic fit always looks better and is not evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss

CANDIDATE_METHODS = ("none", "sigmoid", "isotonic")
DEFAULT_MIN_IMPROVEMENT = 0.0005  # Brier a calibrator must gain over "none" out of sample


@dataclass
class CalibrationSelection:
    method: str
    scores: dict[str, dict[str, float]] = field(default_factory=dict)
    eval_seasons: list[int] = field(default_factory=list)
    n_games: int = 0


def fit_calibrator(method: str, preds: pd.Series, actual: pd.Series) -> Optional[Any]:
    """Fit one candidate on raw probabilities; ``None`` for the identity ("none")."""
    if method == "none":
        return None
    if method == "sigmoid":
        from src.evaluation.calibrate_winprob import _fit_sigmoid_calibrator

        calibrator, _ = _fit_sigmoid_calibrator(preds, actual)
        return calibrator
    if method == "isotonic":
        return IsotonicRegression(out_of_bounds="clip").fit(preds.to_numpy(float), actual.to_numpy(float))
    raise ValueError(f"Unknown calibration method: {method}")


def apply_calibrator(calibrator: Optional[Any], preds: pd.Series) -> np.ndarray:
    values = preds.to_numpy(float)
    if calibrator is None:
        return values
    if hasattr(calibrator, "predict_proba"):
        return np.asarray(calibrator.predict_proba(values.reshape(-1, 1)))[:, 1]
    return np.asarray(calibrator.predict(values), dtype=float)


def select_calibration_method(
    preds: pd.Series,
    actual: pd.Series,
    seasons: pd.Series,
    *,
    season_window: int,
    methods: Sequence[str] = CANDIDATE_METHODS,
    min_improvement: float = DEFAULT_MIN_IMPROVEMENT,
    min_train_games: int = 150,
) -> CalibrationSelection:
    """Pick the calibration method with the best walk-forward (next-season) Brier."""
    frame = pd.DataFrame({"p": preds.astype(float), "y": actual.astype(float), "season": seasons.astype(int)}).dropna()
    ordered = sorted(frame["season"].unique())
    pooled: dict[str, list[np.ndarray]] = {m: [] for m in methods}
    ys: list[np.ndarray] = []
    eval_seasons: list[int] = []
    for season in ordered:
        train = frame[(frame["season"] < season) & (frame["season"] >= season - season_window)]
        test = frame[frame["season"] == season]
        if len(train) < min_train_games or test.empty or train["y"].nunique() < 2:
            continue
        eval_seasons.append(int(season))
        ys.append(test["y"].to_numpy())
        for method in methods:
            calibrator = fit_calibrator(method, train["p"], train["y"])
            pooled[method].append(apply_calibrator(calibrator, test["p"]))
    if not ys:
        return CalibrationSelection(method="none", eval_seasons=[], n_games=0)
    y = np.concatenate(ys)
    scores: dict[str, dict[str, float]] = {}
    for method in methods:
        p = np.clip(np.concatenate(pooled[method]), 1e-6, 1 - 1e-6)
        scores[method] = {"brier": float(brier_score_loss(y, p)), "logloss": float(log_loss(y, p, labels=[0, 1]))}
    baseline = scores.get("none", {}).get("brier", np.inf)
    best = min(scores, key=lambda m: scores[m]["brier"])
    if best != "none" and baseline - scores[best]["brier"] < min_improvement:
        best = "none"
    return CalibrationSelection(method=best, scores=scores, eval_seasons=eval_seasons, n_games=int(len(y)))
