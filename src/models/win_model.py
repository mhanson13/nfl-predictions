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

"""Shared team-model fit/predict recipes.

Production training (`src.models.train`) and walk-forward history scoring
(`src.predict.predict_history`) both build their win-probability and spread
models through this module, so the model that is evaluated is the same recipe
as the model that is shipped.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models import train as train_module

# Fallback defaults, used only when no walk-forward tuning config is supplied.
# They are deliberately low-capacity: ~2.5k games and ~180 features overfit
# badly at the previous defaults (depth 6, 800-1000 trees, lr 0.05), which
# scored walk-forward Brier 0.265 / spread MAE 10.51 on 2019-2022 folds versus
# 0.215 / 10.02 for the values below.
DEFAULT_XGB_WIN_PARAMS: dict[str, object] = {
    "n_estimators": 250,
    "learning_rate": 0.01,
    "max_depth": 2,
    "min_child_weight": 30.0,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "random_state": 42,
    "eval_metric": "logloss",
    "verbosity": 0,
}

DEFAULT_HGB_WIN_PARAMS: dict[str, object] = {
    "random_state": 42,
    "max_iter": 300,
    "max_depth": 2,
    "max_leaf_nodes": 4,
    "learning_rate": 0.01,
    "l2_regularization": 1.0,
    "min_samples_leaf": 40,
    "max_bins": 255,
}

DEFAULT_XGB_SPREAD_PARAMS: dict[str, object] = {
    "n_estimators": 400,
    "learning_rate": 0.01,
    "max_depth": 2,
    "min_child_weight": 20.0,
    "subsample": 0.7,
    "colsample_bytree": 0.6,
    "reg_lambda": 5.0,
    "random_state": 42,
    "objective": "reg:squarederror",
    "verbosity": 0,
}

DEFAULT_HGB_SPREAD_PARAMS: dict[str, object] = {
    "loss": "squared_error",
    "max_iter": 300,
    "learning_rate": 0.02,
    "max_depth": 2,
    "max_leaf_nodes": 4,
    "min_samples_leaf": 40,
    "l2_regularization": 1.0,
    "max_bins": 255,
    "random_state": 42,
}

DEFAULT_GBR_SPREAD_PARAMS: dict[str, object] = {
    "loss": "huber",
    "alpha": 0.9,
    "learning_rate": 0.02,
    "n_estimators": 300,
    "max_depth": 2,
    "min_samples_leaf": 40,
    "max_features": "sqrt",
    "subsample": 0.8,
    "random_state": 42,
}

LOGISTIC_CANDIDATES: tuple[str, ...] = (
    "spread_line",
    "home_moneyline",
    "away_moneyline",
    "total_line",
    "inj_listed_total_diff",
    "inj_out_diff",
    "roster_injured_pct_diff",
    "team_news_injury_kw7_diff",
    "news_injury_kw7_diff",
)

_HIST_CLASSIFIER_CHOICES = {"hist", "hgb", "gb_hist"}
_HIST_REGRESSOR_CHOICES = {"hist", "hgb", "gb_hist", "huber"}
_GBR_REGRESSOR_CHOICES = {"gb", "gbr", "gbdt"}

# Market anchoring: the classifier starts every game at the no-vig moneyline
# log-odds (XGBoost ``base_margin``) and the trees learn only the residual.
MARKET_ANCHOR_HOME_COL = "sched_implied_prob_home"
MARKET_ANCHOR_AWAY_COL = "sched_implied_prob_away"
MARKET_ANCHOR_COLUMNS: tuple[str, str] = (MARKET_ANCHOR_HOME_COL, MARKET_ANCHOR_AWAY_COL)
MARKET_PROB_CLIP = (0.01, 0.99)
# Current-game line columns. Lagged variants (prev/rolling/delta/pct_change)
# describe earlier games and stay available to the residual trees.
_CURRENT_MARKET_PATTERN = re.compile(
    r"^(sched_(implied_prob|moneyline|spread_close|spread_close_abs)_(home|away|diff)"
    r"|spread_line|home_moneyline|away_moneyline|home_spread_odds|away_spread_odds)$"
)


def is_current_market_column(name: str) -> bool:
    """True for this game's own betting-line columns (the anchor and its re-expressions)."""
    return bool(_CURRENT_MARKET_PATTERN.match(str(name)))


def uses_market_anchor(param_config: Optional[dict[str, object]]) -> bool:
    return bool(param_config) and bool(param_config.get("market_anchor", False))


def win_feature_columns(
    feature_cols: Iterable[str],
    param_config: Optional[dict[str, object]],
    available_columns: Iterable[str],
) -> list[str]:
    """Model input columns for the win classifier (adds the anchor columns when anchoring)."""
    available = set(available_columns)
    cols = [c for c in feature_cols if c in available]
    if uses_market_anchor(param_config):
        missing = [c for c in MARKET_ANCHOR_COLUMNS if c not in available]
        if missing:
            raise ValueError(f"market_anchor requires columns {missing}, which are not in the feature frame.")
        cols.extend(c for c in MARKET_ANCHOR_COLUMNS if c not in cols)
    return cols


def no_vig_market_prob(home: np.ndarray, away: np.ndarray) -> np.ndarray:
    """No-vig home probability from implied probabilities; NaN where either side is missing."""
    home = np.asarray(home, dtype=float)
    away = np.asarray(away, dtype=float)
    valid = np.isfinite(home) & np.isfinite(away) & (home > 0) & (away > 0)
    out = np.full(home.shape, np.nan, dtype=float)
    out[valid] = home[valid] / (home[valid] + away[valid])
    return out


def _logit(p: np.ndarray) -> np.ndarray:
    return np.log(p / (1.0 - p))


class MarketAnchoredClassifier(BaseEstimator, ClassifierMixin):
    """XGBoost classifier that models the residual over the no-vig market.

    ``X`` holds every column in ``feature_names`` (including the two anchor
    columns), so callers keep using plain ``fit(X, y)`` / ``predict_proba(X)``.
    Games without a usable line start from the training home-win base rate.
    """

    def __init__(self, estimator=None, feature_names=None, use_gpu: bool = False, context: str = "[win_prob]"):
        self.estimator = estimator
        self.feature_names = feature_names
        self.use_gpu = use_gpu
        self.context = context

    def _layout(self) -> None:
        names = list(self.feature_names or [])
        missing = [c for c in MARKET_ANCHOR_COLUMNS if c not in names]
        if missing:
            raise ValueError(f"MarketAnchoredClassifier needs anchor columns {missing} in feature_names.")
        self.home_idx_ = names.index(MARKET_ANCHOR_HOME_COL)
        self.away_idx_ = names.index(MARKET_ANCHOR_AWAY_COL)
        self.residual_idx_ = np.array([i for i, n in enumerate(names) if not is_current_market_column(n)], dtype=int)
        self.residual_features_ = [names[i] for i in self.residual_idx_]
        self.n_features_in_ = len(names)

    def _check_width(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} columns ({MARKET_ANCHOR_COLUMNS} included); got {X.shape}.")
        return X

    def market_probability(self, X: np.ndarray) -> np.ndarray:
        """No-vig market probability per row (NaN where the line is missing)."""
        X = self._check_width(X)
        return no_vig_market_prob(X[:, self.home_idx_], X[:, self.away_idx_])

    def offset(self, X: np.ndarray) -> np.ndarray:
        market = self.market_probability(X)
        out = np.full(market.shape, self.prior_logit_, dtype=float)
        has_line = np.isfinite(market)
        out[has_line] = _logit(np.clip(market[has_line], *MARKET_PROB_CLIP))
        return out

    def fit(self, X, y):
        self._layout()
        X = self._check_width(X)
        y = np.asarray(y, dtype=int)
        base_rate = float(np.clip(y.mean(), *MARKET_PROB_CLIP)) if len(y) else 0.5
        self.prior_logit_ = float(_logit(np.asarray(base_rate)))
        offset = self.offset(X)
        X_res = X[:, self.residual_idx_]
        model = clone(self.estimator)
        try:
            model.fit(X_res, y, base_margin=offset)
            self.gpu_used_ = bool(self.use_gpu)
        except Exception as ex:  # noqa: B902 - mirror train._fit_xgb_with_fallback
            message = str(ex).lower()
            if not (self.use_gpu and ("gpu" in message or "cuda" in message)):
                raise
            print(f"{self.context} GPU unavailable ({ex}); retrying on CPU hist.")
            fallback_cfg = train_module.get_xgb_tree_config(False)
            if "device" in model.get_params():
                fallback_cfg["device"] = "cpu"
            model.set_params(**fallback_cfg)
            model.fit(X_res, y, base_margin=offset)
            self.gpu_used_ = False
        self.estimator_ = model
        self.classes_ = np.array([0, 1])
        self.market_coverage_ = float(np.isfinite(self.market_probability(X)).mean()) if len(X) else 0.0
        return self

    def decision_function(self, X) -> np.ndarray:
        X = self._check_width(X)
        return np.asarray(
            self.estimator_.predict(X[:, self.residual_idx_], base_margin=self.offset(X), output_margin=True),
            dtype=float,
        )

    def predict_proba(self, X) -> np.ndarray:
        p = 1.0 / (1.0 + np.exp(-self.decision_function(X)))
        return np.column_stack([1.0 - p, p])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def get_booster(self):
        """Booster of the residual trees (inputs are ``residual_features_``, not ``feature_names``)."""
        return self.estimator_.get_booster()

    @property
    def feature_importances_(self) -> np.ndarray:
        """Importances over all input columns (anchor/current-line columns are 0)."""
        full = np.zeros(self.n_features_in_, dtype=float)
        full[self.residual_idx_] = np.asarray(self.estimator_.feature_importances_, dtype=float)
        return full


SPREAD_ANCHOR_COL = "sched_spread_close_home"  # expected home margin implied by the closing line
SPREAD_ANCHOR_COLUMNS: tuple[str] = (SPREAD_ANCHOR_COL,)


def spread_feature_columns(
    feature_cols: Iterable[str],
    param_config: Optional[dict[str, object]],
    available_columns: Iterable[str],
) -> list[str]:
    """Model input columns for the spread regressor (adds the closing spread when anchoring)."""
    available = set(available_columns)
    cols = [c for c in feature_cols if c in available]
    if uses_market_anchor(param_config):
        if SPREAD_ANCHOR_COL not in available:
            raise ValueError(f"market_anchor requires column {SPREAD_ANCHOR_COL!r}, which is not in the feature frame.")
        if SPREAD_ANCHOR_COL not in cols:
            cols.append(SPREAD_ANCHOR_COL)
    return cols


class MarketAnchoredRegressor(BaseEstimator, RegressorMixin):
    """Margin regressor that predicts the closing spread plus a learned residual.

    The wrapped regressor is fit on ``home_margin - spread`` using only
    non-line features; games without a line start from the training mean margin.
    """

    def __init__(self, estimator=None, feature_names=None, use_gpu: bool = False, context: str = "[spread]"):
        self.estimator = estimator
        self.feature_names = feature_names
        self.use_gpu = use_gpu
        self.context = context

    def _layout(self) -> None:
        names = list(self.feature_names or [])
        if SPREAD_ANCHOR_COL not in names:
            raise ValueError(f"MarketAnchoredRegressor needs {SPREAD_ANCHOR_COL!r} in feature_names.")
        self.anchor_idx_ = names.index(SPREAD_ANCHOR_COL)
        self.residual_idx_ = np.array([i for i, n in enumerate(names) if not is_current_market_column(n)], dtype=int)
        self.residual_features_ = [names[i] for i in self.residual_idx_]
        self.n_features_in_ = len(names)

    def _check_width(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} columns ({SPREAD_ANCHOR_COL} included); got {X.shape}.")
        return X

    def offset(self, X) -> np.ndarray:
        line = self._check_width(X)[:, self.anchor_idx_]
        return np.where(np.isfinite(line), line, self.prior_margin_)

    def fit(self, X, y):
        self._layout()
        X = self._check_width(X)
        y = np.asarray(y, dtype=float)
        self.prior_margin_ = float(y.mean()) if len(y) else 0.0
        model = clone(self.estimator)
        residual = y - self.offset(X)
        X_res = X[:, self.residual_idx_]
        if hasattr(model, "get_booster"):
            self.gpu_used_ = train_module._fit_xgb_with_fallback(model, X_res, residual, self.use_gpu, self.context)
        else:
            model.fit(X_res, residual)
            self.gpu_used_ = False
        self.estimator_ = model
        return self

    def predict(self, X) -> np.ndarray:
        X = self._check_width(X)
        return self.offset(X) + np.asarray(self.estimator_.predict(X[:, self.residual_idx_]), dtype=float)

    def get_booster(self):
        return self.estimator_.get_booster()

    @property
    def feature_importances_(self) -> np.ndarray:
        full = np.zeros(self.n_features_in_, dtype=float)
        full[self.residual_idx_] = np.asarray(self.estimator_.feature_importances_, dtype=float)
        return full


def _model_choice(param_config: Optional[dict[str, object]]) -> str:
    if not param_config or "model_type" not in param_config:
        return ""
    return str(param_config["model_type"]).strip().lower()


def resolve_win_model_kind(param_config: Optional[dict[str, object]]) -> str:
    """Return ``"xgb"`` or ``"hgb"`` for the win-probability classifier."""
    if _model_choice(param_config) in _HIST_CLASSIFIER_CHOICES:
        return "hgb"
    return "xgb" if train_module.XGB_AVAILABLE else "hgb"


def resolve_spread_model_kind(param_config: Optional[dict[str, object]]) -> str:
    """Return ``"xgb"``, ``"hgb"`` or ``"gbr"`` for the spread regressor."""
    choice = _model_choice(param_config)
    if choice in _HIST_REGRESSOR_CHOICES:
        return "hgb"
    if choice in _GBR_REGRESSOR_CHOICES:
        return "gbr"
    return "xgb" if train_module.XGB_AVAILABLE else "gbr"


def build_win_classifier(
    param_config: Optional[dict[str, object]] = None,
    *,
    use_gpu: bool = False,
):
    """Construct (unfitted) the win-probability classifier and return ``(estimator, kind)``."""
    kind = resolve_win_model_kind(param_config)
    if kind == "xgb":
        params = dict(DEFAULT_XGB_WIN_PARAMS)
        if param_config:
            params.update(train_module._apply_prefixed_params(param_config, "xgb_"))
        params.update(train_module.get_xgb_tree_config(use_gpu))
        return train_module.XGBClassifier(**params), kind
    params = dict(DEFAULT_HGB_WIN_PARAMS)
    if param_config:
        params.update(train_module._apply_prefixed_params(param_config, "hgb_"))
    return HistGradientBoostingClassifier(**params), kind


def fit_win_classifier(
    X: np.ndarray,
    y: np.ndarray,
    param_config: Optional[dict[str, object]] = None,
    *,
    use_gpu: bool = False,
    context: str = "[win_prob]",
    feature_names: Optional[Iterable[str]] = None,
):
    """Fit the win-probability classifier. Returns ``(model, label)``.

    ``feature_names`` (the column order of ``X``) is required when the config
    sets ``market_anchor``; see `win_feature_columns`.
    """
    if use_gpu and not train_module.XGB_AVAILABLE:
        raise RuntimeError("GPU training requested but XGBoost is not installed.")
    model, kind = build_win_classifier(param_config, use_gpu=use_gpu)
    if uses_market_anchor(param_config):
        if kind != "xgb":
            raise ValueError("market_anchor requires the XGBoost win model (base_margin support).")
        if feature_names is None:
            raise ValueError("market_anchor requires feature_names describing the columns of X.")
        anchored = MarketAnchoredClassifier(model, list(feature_names), use_gpu=use_gpu, context=context).fit(X, y)
        return anchored, ("xgb_anchored_gpu" if anchored.gpu_used_ else "xgb_anchored")
    if kind == "xgb":
        gpu_used = train_module._fit_xgb_with_fallback(model, X, y, use_gpu, f"{context}[xgb]")
        return model, ("xgb_gpu" if gpu_used else "xgb")
    model.fit(X, y)
    return model, "hgb"


def build_spread_regressor(
    param_config: Optional[dict[str, object]] = None,
    *,
    use_gpu: bool = False,
):
    """Construct (unfitted) the spread regressor and return ``(estimator, kind)``."""
    kind = resolve_spread_model_kind(param_config)
    if kind == "xgb":
        params = dict(DEFAULT_XGB_SPREAD_PARAMS)
        if param_config:
            params.update(train_module._apply_prefixed_params(param_config, "xgb_"))
        params.update(train_module.get_xgb_tree_config(use_gpu))
        return train_module.XGBRegressor(**params), kind
    if kind == "hgb":
        params = dict(DEFAULT_HGB_SPREAD_PARAMS)
        if param_config:
            params.update(train_module._apply_prefixed_params(param_config, "hgb_"))
        return HistGradientBoostingRegressor(**params), kind
    params = dict(DEFAULT_GBR_SPREAD_PARAMS)
    if param_config:
        params.update(train_module._apply_prefixed_params(param_config, "gbr_"))
    return GradientBoostingRegressor(**params), kind


def fit_spread_regressor(
    X: np.ndarray,
    y: np.ndarray,
    param_config: Optional[dict[str, object]] = None,
    *,
    use_gpu: bool = False,
    context: str = "[spread]",
    feature_names: Optional[Iterable[str]] = None,
):
    """Fit the spread regressor on already-imputed features. Returns ``(model, label)``.

    ``feature_names`` (the column order of ``X``) is required when the config
    sets ``market_anchor``; see `spread_feature_columns`.
    """
    if use_gpu and not train_module.XGB_AVAILABLE:
        raise RuntimeError("GPU training requested but XGBoost is not installed.")
    model, kind = build_spread_regressor(param_config, use_gpu=use_gpu)
    if uses_market_anchor(param_config):
        if feature_names is None:
            raise ValueError("market_anchor requires feature_names describing the columns of X.")
        anchored = MarketAnchoredRegressor(model, list(feature_names), use_gpu=use_gpu, context=context).fit(X, y)
        label = f"{kind}_anchored" + ("_gpu" if anchored.gpu_used_ else "")
        return anchored, label
    if kind == "xgb":
        gpu_used = train_module._fit_xgb_with_fallback(model, X, y, use_gpu, f"{context}[xgb]")
        return model, ("xgb_gpu" if gpu_used else "xgb")
    model.fit(X, y)
    return model, kind


def build_logistic_model() -> Pipeline:
    """Logistic companion model used by the optional win-probability blend."""
    return Pipeline(
        steps=[
            ("imputer", train_module._make_median_imputer()),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def logistic_feature_columns(columns: Iterable[str]) -> list[str]:
    available = set(columns)
    return [c for c in LOGISTIC_CANDIDATES if c in available]


def prepare_matrix(
    df: pd.DataFrame,
    columns: Iterable[str],
    ranges: Optional[dict] = None,
) -> pd.DataFrame:
    """Select and float-cast the exact model columns (missing columns become 0.0)."""
    X = df.reindex(columns=list(columns), fill_value=0.0).astype(float)
    if ranges:
        for col, bounds in ranges.items():
            if col in X.columns and isinstance(bounds, (tuple, list)) and len(bounds) == 2:
                lo, hi = bounds
                X[col] = X[col].clip(lo, hi)
    return X


def fit_win_bundle(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    param_config: Optional[dict[str, object]] = None,
    use_gpu: bool = False,
    skip_logit: bool = True,
    context: str = "[win_prob]",
) -> dict[str, object]:
    """Fit the win-probability model on every row of ``train_df``.

    The logistic companion is fit when requested, but its blend weight stays 1.0
    (GB only) unless a caller explicitly tunes it on held-out data.
    """
    use_cols = win_feature_columns(feature_cols, param_config, train_df.columns)
    X = prepare_matrix(train_df, use_cols)
    y = train_df["home_win"].astype(int).to_numpy()
    model, label = fit_win_classifier(
        X.values, y, param_config, use_gpu=use_gpu, context=context, feature_names=use_cols
    )

    logistic_model = None
    logistic_features: list[str] = []
    if not skip_logit:
        logistic_features = logistic_feature_columns(train_df.columns)
        if logistic_features:
            logistic_model = build_logistic_model()
            logistic_model.fit(train_df[logistic_features].astype(float).values, y)

    return {
        "model": model,
        "features": use_cols,
        "logistic_model": logistic_model,
        "logistic_features": logistic_features,
        "ensemble_weight": 1.0,
        "model_type": label,
    }


def predict_win_bundle(bundle: dict[str, object], score_df: pd.DataFrame) -> np.ndarray:
    """Return home-win probabilities (GB, optionally blended with the logistic companion)."""
    features = list(bundle["features"])
    X = prepare_matrix(score_df, features, bundle.get("feature_ranges") or None)
    proba = bundle["model"].predict_proba(X.values)[:, 1]
    logistic_model = bundle.get("logistic_model")
    logistic_features = list(bundle.get("logistic_features") or [])
    if logistic_model is not None and logistic_features:
        log_proba = logistic_model.predict_proba(score_df[logistic_features].astype(float).values)[:, 1]
        weight = float(bundle.get("ensemble_weight", 1.0))
        proba = weight * proba + (1.0 - weight) * log_proba
    return np.asarray(proba, dtype=float)


def fit_spread_bundle(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    param_config: Optional[dict[str, object]] = None,
    use_gpu: bool = False,
    context: str = "[spread]",
) -> dict[str, object]:
    """Fit the spread (home margin) model on every row of ``train_df``."""
    use_cols = spread_feature_columns(feature_cols, param_config, train_df.columns)
    X_raw = prepare_matrix(train_df, use_cols)
    y = train_df["home_margin"].astype(float).to_numpy()
    imputer = train_module._make_median_imputer()
    X = imputer.fit_transform(X_raw.values)
    model, label = fit_spread_regressor(
        X, y, param_config, use_gpu=use_gpu, context=context, feature_names=use_cols
    )
    train_pred = model.predict(X)
    return {
        "model": model,
        "features": use_cols,
        "imputer": imputer,
        "bias_correction": float((train_pred - y).mean()),
        "model_type": label,
    }


def predict_spread_bundle(bundle: dict[str, object], score_df: pd.DataFrame) -> np.ndarray:
    features = list(bundle["features"])
    imputer = bundle.get("imputer")
    X_raw = prepare_matrix(score_df, features)
    X = imputer.transform(X_raw.values) if imputer is not None else X_raw.values
    return np.asarray(bundle["model"].predict(X), dtype=float) - float(bundle.get("bias_correction", 0.0))
