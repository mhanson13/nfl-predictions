# Copyright (c) 2025 Matt Hanson
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Time-ordered (walk-forward) hyperparameter search for the team models.

Each validation season is scored by a model trained only on earlier seasons,
mirroring how `src.predict.predict_history` evaluates and how production is
used. Trials are built through `src.models.win_model`, so a saved config means
the same thing to tuning, production training and walk-forward history.

Saved configs carry ``tuning_scheme="walk_forward"`` plus a feature fingerprint;
`tools.run_pipeline` only auto-applies configs with that marker, which retires
older random-split configs tuned against earlier (leaky) feature sets.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, mean_squared_error, roc_auc_score

from src.models import train as base_train
from src.models import win_model
from src.models.feature_columns import make_feature_diffs, select_feature_columns

if TYPE_CHECKING:  # pragma: no cover - typing only
    from optuna.trial import Trial

TUNING_SCHEME = "walk_forward"
TUNING_SCHEMA_VERSION = 2
DEFAULT_N_VALIDATION_SEASONS = 4
CLASSIFIER_METRICS = ("logloss", "brier", "auc", "auc_brier")
REGRESSOR_METRICS = ("mae", "rmse")


@dataclass
class WalkForwardData:
    X: np.ndarray
    y: np.ndarray
    seasons: np.ndarray
    feature_names: list[str]
    validation_seasons: list[int]
    market_prob: Optional[np.ndarray] = None
    folds: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    market_anchor: bool = False


def feature_fingerprint(feature_names: Iterable[str]) -> str:
    """Short stable hash of the model feature set (order-insensitive)."""
    payload = "\n".join(sorted(str(c) for c in feature_names)).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:10]


def default_validation_seasons(seasons: Iterable[int], n_seasons: int = DEFAULT_N_VALIDATION_SEASONS) -> list[int]:
    """Most recent ``n_seasons`` complete seasons, excluding the latest (possibly in-progress) one.

    At least one earlier season is always kept for training the first fold.
    """
    ordered = sorted({int(s) for s in seasons})
    if len(ordered) < 3:
        raise ValueError("Walk-forward tuning needs at least three seasons of completed games.")
    candidates = ordered[1:-1]
    return candidates[-int(n_seasons):]


def prepare_walk_forward_data(
    df: pd.DataFrame,
    *,
    target: str,
    validation_seasons: Optional[list[int]] = None,
    n_validation_seasons: int = DEFAULT_N_VALIDATION_SEASONS,
    include_bdl_features: bool = False,
    market_anchor: bool = False,
) -> WalkForwardData:
    """Build the feature matrix and season-blocked walk-forward folds for ``target``.

    With ``market_anchor`` the no-vig anchor columns are appended to the matrix
    so every trial is scored as a residual-over-market model.
    """

    target_col = "home_win" if target == "win_prob" else "home_margin"
    feat_df = make_feature_diffs(df)
    feat_df = feat_df.loc[:, ~pd.Index(feat_df.columns).duplicated()]
    feature_names = [c for c in select_feature_columns(feat_df, include_bdl=include_bdl_features) if c in feat_df.columns]
    if not feature_names:
        raise ValueError("No usable features discovered for walk-forward tuning.")
    if market_anchor:
        anchor_columns = win_model.win_feature_columns if target == "win_prob" else win_model.spread_feature_columns
        feature_names = anchor_columns(feature_names, {"market_anchor": True}, feat_df.columns)
    frame = base_train._require_target(feat_df, target_col)
    frame = frame[pd.to_numeric(frame["season"], errors="coerce").notna()].copy()
    seasons = pd.to_numeric(frame["season"], errors="coerce").astype(int).to_numpy()
    if validation_seasons is None:
        validation_seasons = default_validation_seasons(seasons, n_validation_seasons)
    validation_seasons = sorted(int(s) for s in validation_seasons)

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for season in validation_seasons:
        train_idx = np.flatnonzero(seasons < season)
        val_idx = np.flatnonzero(seasons == season)
        if len(train_idx) == 0 or len(val_idx) == 0:
            raise ValueError(f"Validation season {season} has no prior-season training rows or no rows to score.")
        folds.append((train_idx, val_idx))

    X = frame[feature_names].astype(float).to_numpy()
    if target == "win_prob":
        y = frame["home_win"].astype(int).to_numpy()
    else:
        y = frame["home_margin"].astype(float).to_numpy()
    market_prob = None
    if {"sched_implied_prob_home", "sched_implied_prob_away"}.issubset(frame.columns):
        home = pd.to_numeric(frame["sched_implied_prob_home"], errors="coerce")
        away = pd.to_numeric(frame["sched_implied_prob_away"], errors="coerce")
        market_prob = (home / (home + away)).to_numpy(dtype=float)
    return WalkForwardData(
        X=X,
        y=y,
        seasons=seasons,
        feature_names=feature_names,
        validation_seasons=validation_seasons,
        market_prob=market_prob,
        folds=folds,
        market_anchor=bool(market_anchor),
    )


def suggest_win_params(trial: "Trial", model_types: list[str]) -> dict[str, object]:
    """Bounded, low-capacity search space sized for ~2-3k games and ~180 features."""
    model_type = trial.suggest_categorical("model_type", model_types)
    if model_type == "xgb":
        return {
            "model_type": "xgb",
            "xgb_n_estimators": trial.suggest_int("xgb_n_estimators", 100, 1000, step=25),
            "xgb_learning_rate": trial.suggest_float("xgb_learning_rate", 0.005, 0.05, log=True),
            "xgb_max_depth": trial.suggest_int("xgb_max_depth", 2, 4),
            "xgb_min_child_weight": trial.suggest_float("xgb_min_child_weight", 10.0, 80.0, log=True),
            "xgb_subsample": trial.suggest_float("xgb_subsample", 0.5, 0.9),
            "xgb_colsample_bytree": trial.suggest_float("xgb_colsample_bytree", 0.3, 0.8),
            "xgb_reg_lambda": trial.suggest_float("xgb_reg_lambda", 1.0, 100.0, log=True),
            "xgb_reg_alpha": trial.suggest_float("xgb_reg_alpha", 1e-3, 10.0, log=True),
            "xgb_gamma": trial.suggest_float("xgb_gamma", 0.0, 5.0),
        }
    return {
        "model_type": "hist",
        "hgb_max_iter": trial.suggest_int("hgb_max_iter", 50, 600, step=25),
        "hgb_learning_rate": trial.suggest_float("hgb_learning_rate", 0.01, 0.1, log=True),
        "hgb_max_depth": trial.suggest_int("hgb_max_depth", 2, 4),
        "hgb_max_leaf_nodes": trial.suggest_int("hgb_max_leaf_nodes", 4, 15),
        "hgb_min_samples_leaf": trial.suggest_int("hgb_min_samples_leaf", 30, 150),
        "hgb_l2_regularization": trial.suggest_float("hgb_l2_regularization", 0.1, 50.0, log=True),
    }


def suggest_spread_params(trial: "Trial", model_types: list[str]) -> dict[str, object]:
    model_type = trial.suggest_categorical("model_type", model_types)
    if model_type == "xgb":
        return {
            "model_type": "xgb",
            "xgb_n_estimators": trial.suggest_int("xgb_n_estimators", 100, 1000, step=25),
            "xgb_learning_rate": trial.suggest_float("xgb_learning_rate", 0.005, 0.05, log=True),
            "xgb_max_depth": trial.suggest_int("xgb_max_depth", 2, 4),
            "xgb_min_child_weight": trial.suggest_float("xgb_min_child_weight", 10.0, 80.0, log=True),
            "xgb_subsample": trial.suggest_float("xgb_subsample", 0.5, 0.9),
            "xgb_colsample_bytree": trial.suggest_float("xgb_colsample_bytree", 0.3, 0.8),
            "xgb_reg_lambda": trial.suggest_float("xgb_reg_lambda", 1.0, 100.0, log=True),
            "xgb_reg_alpha": trial.suggest_float("xgb_reg_alpha", 1e-3, 10.0, log=True),
            "xgb_gamma": trial.suggest_float("xgb_gamma", 0.0, 5.0),
        }
    if model_type == "gbr":
        return {
            "model_type": "gbr",
            "gbr_n_estimators": trial.suggest_int("gbr_n_estimators", 100, 800, step=25),
            "gbr_learning_rate": trial.suggest_float("gbr_learning_rate", 0.005, 0.05, log=True),
            "gbr_max_depth": trial.suggest_int("gbr_max_depth", 2, 4),
            "gbr_min_samples_leaf": trial.suggest_int("gbr_min_samples_leaf", 30, 150),
            "gbr_subsample": trial.suggest_float("gbr_subsample", 0.5, 0.9),
            "gbr_alpha": trial.suggest_float("gbr_alpha", 0.85, 0.99),
        }
    return {
        "model_type": "hist",
        "hgb_max_iter": trial.suggest_int("hgb_max_iter", 50, 600, step=25),
        "hgb_learning_rate": trial.suggest_float("hgb_learning_rate", 0.01, 0.1, log=True),
        "hgb_max_depth": trial.suggest_int("hgb_max_depth", 2, 4),
        "hgb_max_leaf_nodes": trial.suggest_int("hgb_max_leaf_nodes", 4, 15),
        "hgb_min_samples_leaf": trial.suggest_int("hgb_min_samples_leaf", 30, 150),
        "hgb_l2_regularization": trial.suggest_float("hgb_l2_regularization", 0.1, 50.0, log=True),
    }


def score_win_params(
    params: dict[str, object],
    data: WalkForwardData,
    *,
    use_gpu: bool = False,
) -> dict[str, float]:
    """Pooled out-of-fold metrics for one win-probability config across all walk-forward folds."""
    y_parts: list[np.ndarray] = []
    p_parts: list[np.ndarray] = []
    m_parts: list[np.ndarray] = []
    if data.market_anchor:
        params = {**params, "market_anchor": True}
    for fold_id, (train_idx, val_idx) in enumerate(data.folds):
        model, _ = win_model.fit_win_classifier(
            data.X[train_idx],
            data.y[train_idx],
            params,
            use_gpu=use_gpu,
            context=f"[tune_wf][win][fold={fold_id}]",
            feature_names=data.feature_names,
        )
        p_parts.append(np.clip(model.predict_proba(data.X[val_idx])[:, 1], 1e-6, 1 - 1e-6))
        y_parts.append(data.y[val_idx])
        if data.market_prob is not None:
            m_parts.append(data.market_prob[val_idx])
    y = np.concatenate(y_parts)
    p = np.concatenate(p_parts)
    out = {
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)),
        "n_games": int(len(y)),
    }
    out["auc_brier"] = (1.0 - out["auc"]) + out["brier"]
    if m_parts:
        market = np.concatenate(m_parts)
        mask = np.isfinite(market)
        if mask.any():
            mk = np.clip(market[mask], 1e-6, 1 - 1e-6)
            out["market_logloss"] = float(log_loss(y[mask], mk, labels=[0, 1]))
            out["market_brier"] = float(brier_score_loss(y[mask], mk))
            out["market_auc"] = float(roc_auc_score(y[mask], mk))
            out["model_brier_same_rows"] = float(brier_score_loss(y[mask], p[mask]))
    return out


def score_spread_params(
    params: dict[str, object],
    data: WalkForwardData,
    *,
    use_gpu: bool = False,
) -> dict[str, float]:
    """Pooled out-of-fold MAE/RMSE for one spread config (with the production bias correction)."""
    if data.market_anchor:
        params = {**params, "market_anchor": True}
    y_parts: list[np.ndarray] = []
    p_parts: list[np.ndarray] = []
    for fold_id, (train_idx, val_idx) in enumerate(data.folds):
        imputer = base_train._make_median_imputer()
        X_train = imputer.fit_transform(data.X[train_idx])
        X_val = imputer.transform(data.X[val_idx])
        model, _ = win_model.fit_spread_regressor(
            X_train,
            data.y[train_idx],
            params,
            use_gpu=use_gpu,
            context=f"[tune_wf][spread][fold={fold_id}]",
            feature_names=data.feature_names,
        )
        bias = float((model.predict(X_train) - data.y[train_idx]).mean())
        p_parts.append(model.predict(X_val) - bias)
        y_parts.append(data.y[val_idx])
    y = np.concatenate(y_parts)
    p = np.concatenate(p_parts)
    return {
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(mean_squared_error(y, p) ** 0.5),
        "n_games": int(len(y)),
    }


def win_objective(trial: "Trial", data: WalkForwardData, metric: str, model_types: list[str], use_gpu: bool) -> float:
    params = suggest_win_params(trial, model_types)
    scores = score_win_params(params, data, use_gpu=use_gpu)
    for key, value in scores.items():
        trial.set_user_attr(key, value)
    if metric == "auc":
        return 1.0 - scores["auc"]
    return float(scores[metric])


def spread_objective(trial: "Trial", data: WalkForwardData, metric: str, model_types: list[str], use_gpu: bool) -> float:
    params = suggest_spread_params(trial, model_types)
    scores = score_spread_params(params, data, use_gpu=use_gpu)
    for key, value in scores.items():
        trial.set_user_attr(key, value)
    return float(scores[metric])


def config_from_trial_params(trial_params: dict[str, object]) -> dict[str, object]:
    """Optuna's stored params already use the config key format consumed by win_model."""
    return dict(trial_params)


def best_config_payload(
    study,
    data: WalkForwardData,
    *,
    target: str,
    metric: str,
    include_bdl_features: bool,
) -> dict[str, object]:
    best = study.best_trial
    best_params = config_from_trial_params(best.params)
    if data.market_anchor:
        best_params["market_anchor"] = True
    return {
        "best_value": study.best_value,
        "best_params": best_params,
        "user_attrs": dict(best.user_attrs),
        "tuning_scheme": TUNING_SCHEME,
        "schema_version": TUNING_SCHEMA_VERSION,
        "target": target,
        "metric": metric,
        "validation_seasons": list(data.validation_seasons),
        "n_features": len(data.feature_names),
        "feature_fingerprint": feature_fingerprint(data.feature_names),
        "include_bdl_features": bool(include_bdl_features),
        "market_anchor": bool(data.market_anchor),
        "n_trials": len(study.trials),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def is_walk_forward_config(payload: object) -> bool:
    """Return whether a loaded tuning JSON payload came from walk-forward tuning."""
    return isinstance(payload, dict) and payload.get("tuning_scheme") == TUNING_SCHEME
