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

"""Model training CLI for win probability and spread regressions."""

from __future__ import annotations
import argparse

import json

from pathlib import Path

import joblib

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    mean_absolute_error,
    r2_score,
    brier_score_loss,
    log_loss,
)
from sklearn.ensemble import (

    HistGradientBoostingClassifier,

    HistGradientBoostingRegressor,

    GradientBoostingRegressor,

)

from sklearn.impute import SimpleImputer

from sklearn.calibration import CalibratedClassifierCV

try:

    from sklearn.calibration import FrozenEstimator



    HAS_FROZEN_ESTIMATOR = True

except ImportError:  # pragma: no cover - older scikit-learn

    HAS_FROZEN_ESTIMATOR = False

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler



def _make_median_imputer() -> SimpleImputer:
    """Return a median imputer that preserves all-null feature columns when supported."""
    try:
        return SimpleImputer(strategy="median", keep_empty_features=True)
    except TypeError:  # pragma: no cover - older scikit-learn
        return SimpleImputer(strategy="median")


try:

    from xgboost import XGBClassifier, XGBRegressor

    from xgboost.core import XGBoostError

    from xgboost import __version__ as XGB_VERSION



    XGB_AVAILABLE = True

except ImportError:

    XGB_AVAILABLE = False

    XGBoostError = Exception

    XGB_VERSION = "0.0.0"

from src.models.feature_columns import (

    LEAKAGE_PATTERNS,

    make_feature_diffs,

    select_feature_columns,

)

from src.utils.io import PROC_DIR, MODELS_DIR, read_df
from src.utils.logging import configure as configure_logging
from src.utils.week_filter import filter_before_week

EVAL_DIR = MODELS_DIR.parent / "predictions" / "evaluation"


def report_feature_quality(
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    context: str = "",
    warn_null_threshold: float = 0.20,
    warn_var_threshold: float = 1e-9,
) -> pd.DataFrame:
    """
    Compute per-feature quality diagnostics and log warnings for problematic columns.

    Checks null rate, variance, and observation count for each feature. Any column
    with null_rate > warn_null_threshold or variance < warn_var_threshold is logged
    as a WARNING. The summary is also written to
    predictions/evaluation/feature_quality.csv (appended with run timestamp).

    Parameters
    ----------
    feature_df : pd.DataFrame
        The full feature DataFrame (before train/test split).
    feature_cols : list[str]
        The subset of columns actually used as model inputs.
    context : str
        Label for the calling model (e.g. "win_prob", "spread") — used in log output.
    warn_null_threshold : float
        Fraction of NaNs above which a WARNING is emitted (default 0.20).
    warn_var_threshold : float
        Variance below which a WARNING is emitted (near-constant feature, default 1e-9).

    Returns
    -------
    pd.DataFrame
        One row per feature with columns: feature, null_rate, variance, n_obs, min, max, mean.
    """
    present_cols = [c for c in feature_cols if c in feature_df.columns]
    if not present_cols:
        return pd.DataFrame()

    sub = feature_df[present_cols]
    rows = []
    warnings_emitted: list[str] = []

    for col in present_cols:
        series = pd.to_numeric(sub[col], errors="coerce")
        n_total = len(series)
        n_null = int(series.isna().sum())
        null_rate = n_null / n_total if n_total > 0 else 1.0
        n_obs = n_total - n_null
        variance = float(series.var()) if n_obs > 1 else 0.0
        col_min = float(series.min()) if n_obs > 0 else float("nan")
        col_max = float(series.max()) if n_obs > 0 else float("nan")
        col_mean = float(series.mean()) if n_obs > 0 else float("nan")

        flag = ""
        if null_rate > warn_null_threshold:
            flag += " HIGH_NULL"
            warnings_emitted.append(f"  {col}: null_rate={null_rate:.1%}")
        if variance < warn_var_threshold:
            flag += " NEAR_CONSTANT"
            warnings_emitted.append(f"  {col}: variance={variance:.2e}")

        rows.append({
            "feature": col,
            "null_rate": round(null_rate, 4),
            "variance": variance,
            "n_obs": n_obs,
            "min": col_min,
            "max": col_max,
            "mean": col_mean,
            "flags": flag.strip(),
        })

    summary = pd.DataFrame(rows)
    prefix = f"[{context}]" if context else "[feature_quality]"
    print(f"{prefix} feature quality: {len(present_cols)} features  "
          f"null>20%: {(summary['null_rate'] > warn_null_threshold).sum()}  "
          f"near-constant: {(summary['variance'] < warn_var_threshold).sum()}")
    if warnings_emitted:
        import logging
        logger = logging.getLogger(__name__)
        for msg in warnings_emitted:
            logger.warning("%s WARNING%s", prefix, msg)

    # Append to predictions/evaluation/feature_quality.csv
    try:
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        quality_path = EVAL_DIR / "feature_quality.csv"
        summary_out = summary.copy()
        summary_out.insert(0, "context", context)
        summary_out.insert(1, "run_timestamp", pd.Timestamp.utcnow().isoformat())
        write_header = not quality_path.exists()
        summary_out.to_csv(quality_path, mode="a", header=write_header, index=False)
    except Exception:
        pass  # Non-fatal: diagnostics should never block training

    return summary


# Private aliases used by tune.py for backwards compatibility
_make_feature_diffs = make_feature_diffs
_select_feature_columns = select_feature_columns


def _parse_version_tuple(version: str) -> tuple[int, ...]:
    parts: list[int] = []

    for chunk in version.split("."):

        digits = ""

        for ch in chunk:

            if ch.isdigit():

                digits += ch

            else:

                break

        if digits:

            parts.append(int(digits))

        else:

            parts.append(0)

    return tuple(parts) if parts else (0, 0, 0)





XGB_SUPPORTS_DEVICE_PARAM = _parse_version_tuple(XGB_VERSION) >= (2, 0, 0)





def get_xgb_tree_config(use_gpu: bool) -> dict[str, object]:

    """

    Base XGBoost tree configuration that adapts to GPU availability and

    XGBoost version (device parameter introduced in 2.0).

    """

    cfg: dict[str, object] = {

        "tree_method": "hist",

        "predictor": "auto",

        "n_jobs": -1,

    }

    if not use_gpu:

        return cfg



    cfg["n_jobs"] = 0

    if XGB_SUPPORTS_DEVICE_PARAM:

        cfg["device"] = "cuda"

    else:

        cfg["tree_method"] = "gpu_hist"

        cfg["predictor"] = "gpu_predictor"

    return cfg





def _apply_prefixed_params(params: dict[str, object], prefix: str) -> dict[str, object]:

    """Strip a prefix (e.g., xgb_/hgb_) from parameter keys for estimator kwargs."""

    result: dict[str, object] = {}

    for key, value in params.items():

        if key.startswith(prefix):

            result[key[len(prefix) :]] = value

    return result





def load_param_config(path: str | None) -> tuple[dict[str, object], Path] | tuple[None, None]:

    if not path:

        return None, None

    cfg_path = Path(path)

    if not cfg_path.exists():

        print(f"[train] Param config {cfg_path} not found; using defaults.")

        return None, None

    try:

        data = json.loads(cfg_path.read_text(encoding="utf-8"))

    except Exception as exc:  # pragma: no cover - invalid json

        print(f"[train] Failed to read param config {cfg_path}: {exc}. Using defaults.")

        return None, None

    params = data.get("best_params") if isinstance(data, dict) else None

    if not params:

        params = data

    if not isinstance(params, dict):

        print(f"[train] Param config {cfg_path} malformed; using defaults.")

        return None, None

    return params, cfg_path





def _fit_xgb_with_fallback(model, X_train, y_train, use_gpu: bool, context: str) -> bool:

    """

    Fit an XGBoost model, retrying on CPU if GPU tree methods are unsupported.



    Returns True when GPU training succeeded, False otherwise.

    """

    if not use_gpu:

        model.fit(X_train, y_train)

        return False



    try:

        model.fit(X_train, y_train)

        return True

    except Exception as ex:  # noqa: B902

        gpu_error = isinstance(ex, XGBoostError) if XGB_AVAILABLE else False

        message = str(ex).lower()

        if gpu_error and ("gpu" in message or "cuda" in message):

            print(f"{context} GPU unavailable ({ex}); retrying on CPU hist.")

            fallback_cfg = get_xgb_tree_config(False)

            current_params = getattr(model, "get_params", lambda: {})()

            if isinstance(current_params, dict) and "device" in current_params:

                fallback_cfg["device"] = "cpu"

            model.set_params(**fallback_cfg)

            model.fit(X_train, y_train)

            return False

        raise



def _require_target(df: pd.DataFrame, col: str) -> pd.DataFrame:

    if col not in df.columns:

        raise ValueError(f"Target column '{col}' not found in features. "

                         f"Make sure build_features created it (home_score/away_score → home_margin/home_win).")

    out = df.dropna(subset=[col]).copy()

    if out.empty:

        raise ValueError(f"Target column '{col}' has no valid rows after dropping NaNs.")

    return out





def train_win_prob(
    df: pd.DataFrame,
    calibrate: bool = False,
    use_gpu: bool = False,
    param_config: dict[str, object] | None = None,
    config_source: Path | None = None,
    skip_logit: bool = False,
    cv_folds: int = 3,
) -> None:
    """
    Train the win-probability classification stack and persist models to disk.

    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix from `matchup_features.parquet`.
    calibrate : bool
        Whether to fit an isotonic calibration layer (requires holdout set).
    use_gpu : bool
        Use XGBoost GPU estimators when available.
    param_config : dict[str, object], optional
        Tuned hyperparameters loaded from JSON.
    config_source : Path, optional
        Path of the hyperparameter file (logged for traceability).
    skip_logit : bool
        Skip the auxiliary logistic regression model when requested.
    cv_folds : int
        Number of CV folds for evaluation metrics. Use 1 to disable CV and
        use a single 80/20 split instead.
    """
    feat_df = make_feature_diffs(df)

    feat_cols = select_feature_columns(feat_df)

    if not feat_cols:

        raise ValueError("No usable *_diff features discovered. "

                         "Check that your matchup_features has *_home/_away numeric columns (not scores).")



    tgt_df = _require_target(feat_df, "home_win")



    # Deduplicate columns to avoid duplicated-name expansion during selection

    tgt_df_nodup = tgt_df.loc[:, ~pd.Index(tgt_df.columns).duplicated()]

    use_cols = [c for c in feat_cols if c in tgt_df_nodup.columns]

    if not use_cols:

        raise ValueError("No usable features after deduplication. Check feature creation and names.")

    X_full = tgt_df_nodup[use_cols].astype(float).values

    y_full = np.asarray(tgt_df["home_win"].astype(int).values, dtype=int)



    # Split indices: train/cal/test for better calibration stability

    indices = np.arange(len(y_full))

    idx_tmp, idx_test, y_tmp, y_test = train_test_split(

        indices, y_full, test_size=0.20, random_state=42, stratify=y_full

    )

    idx_train, idx_cal, y_train, y_cal = train_test_split(

        idx_tmp, y_tmp, test_size=0.25, random_state=42, stratify=y_tmp

    )  # 60/20/20



    X_train = X_full[idx_train]
    X_cal = X_full[idx_cal]
    X_test = X_full[idx_test]

    if use_gpu and not XGB_AVAILABLE:
        raise RuntimeError("GPU training requested but XGBoost is not installed.")

    model_choice = None
    if param_config and "model_type" in param_config:
        model_choice = str(param_config["model_type"]).lower()
    if param_config and config_source:
        print(f"[win_prob] applying tuned params from {config_source} (model_type={model_choice})")

    use_xgb = XGB_AVAILABLE
    if model_choice in {"hist", "hgb", "gb_hist"}:
        use_xgb = False
    elif model_choice in {"xgb", "xgb_gpu", "xgb_cpu"}:
        use_xgb = XGB_AVAILABLE

    # --- k-fold cross-validation for evaluation metrics ---
    if cv_folds > 1:
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=False)
        cv_aucs, cv_briers, cv_loglosses, cv_accs = [], [], [], []
        train_pool_idx = np.concatenate([idx_train, idx_cal])
        for fold_idx, (fold_train_idx, fold_val_idx) in enumerate(
            skf.split(X_full[train_pool_idx], y_full[train_pool_idx])
        ):
            X_ftrain = X_full[train_pool_idx[fold_train_idx]]
            y_ftrain = y_full[train_pool_idx[fold_train_idx]]
            X_fval = X_full[train_pool_idx[fold_val_idx]]
            y_fval = y_full[train_pool_idx[fold_val_idx]]
            if use_xgb:
                fold_xgb_params: dict[str, object] = {
                    "n_estimators": 800, "learning_rate": 0.05, "max_depth": 6,
                    "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0,
                    "reg_alpha": 0.0, "random_state": 42, "eval_metric": "logloss",
                    "verbosity": 0, "use_label_encoder": False,
                }
                if param_config:
                    fold_xgb_params.update(_apply_prefixed_params(param_config, "xgb_"))
                fold_xgb_params.update(get_xgb_tree_config(use_gpu))
                fold_model = XGBClassifier(**fold_xgb_params)
                _fit_xgb_with_fallback(fold_model, X_ftrain, y_ftrain, use_gpu, f"[win_prob][cv_fold={fold_idx}][xgb]")
            else:
                fold_hgb_params: dict[str, object] = {
                    "random_state": 42, "max_leaf_nodes": 31, "learning_rate": 0.05,
                    "l2_regularization": 0.1, "min_samples_leaf": 20, "max_bins": 255,
                }
                if param_config:
                    fold_hgb_params.update(_apply_prefixed_params(param_config, "hgb_"))
                fold_model = HistGradientBoostingClassifier(**fold_hgb_params)
                fold_model.fit(X_ftrain, y_ftrain)
            fold_proba = np.clip(fold_model.predict_proba(X_fval)[:, 1], 1e-6, 1 - 1e-6)
            cv_aucs.append(roc_auc_score(y_fval, fold_proba))
            cv_briers.append(brier_score_loss(y_fval, fold_proba))
            cv_loglosses.append(log_loss(y_fval, fold_proba))
            cv_accs.append(accuracy_score(y_fval, (fold_proba >= 0.5).astype(int)))
        print(
            f"[win_prob][cv={cv_folds}] "
            f"AUC={np.mean(cv_aucs):.4f}±{np.std(cv_aucs):.4f}  "
            f"Brier={np.mean(cv_briers):.4f}±{np.std(cv_briers):.4f}  "
            f"LogLoss={np.mean(cv_loglosses):.4f}±{np.std(cv_loglosses):.4f}  "
            f"ACC={np.mean(cv_accs):.4f}±{np.std(cv_accs):.4f}"
        )

    gpu_used = False

    if use_xgb:

        xgb_params: dict[str, object] = {

            "n_estimators": 800,

            "learning_rate": 0.05,

            "max_depth": 6,

            "subsample": 0.8,

            "colsample_bytree": 0.8,

            "reg_lambda": 1.0,

            "reg_alpha": 0.0,

            "random_state": 42,

            "eval_metric": "logloss",

            "verbosity": 0,

            "use_label_encoder": False,

        }

        if param_config:

            xgb_params.update(_apply_prefixed_params(param_config, "xgb_"))

        xgb_params.update(get_xgb_tree_config(use_gpu))

        base = XGBClassifier(**xgb_params)

        gpu_used = _fit_xgb_with_fallback(base, X_train, y_train, use_gpu, "[win_prob][xgb]")

        model_label = "xgb_gpu" if gpu_used else "xgb"

    else:

        hgb_params: dict[str, object] = {

            "random_state": 42,

            "max_leaf_nodes": 31,

            "learning_rate": 0.05,

            "l2_regularization": 0.1,

            "min_samples_leaf": 20,

            "max_bins": 255,

        }

        if param_config:

            hgb_params.update(_apply_prefixed_params(param_config, "hgb_"))

        base = HistGradientBoostingClassifier(**hgb_params)

        base.fit(X_train, y_train)

        model_label = "hgb"

    if calibrate:

        if HAS_FROZEN_ESTIMATOR:

            frozen = FrozenEstimator(base)

            model = CalibratedClassifierCV(frozen, method="isotonic", ensemble="auto")

        else:

            model = CalibratedClassifierCV(base, cv="prefit", method="isotonic")

        model.fit(X_cal, y_cal)

    else:

        model = base



    proba_test = model.predict_proba(X_test)[:, 1]

    proba_cal = model.predict_proba(X_cal)[:, 1]

    pred = (proba_test >= 0.5).astype(int)



    auc = roc_auc_score(y_test, proba_test)

    acc = accuracy_score(y_test, pred)

    # Diagnostics: class balance and probability spread

    pos_rate = float(y_full.mean()) if len(y_full) else float('nan')

    pmin, pmax = float(proba_test.min()) if len(proba_test) else float('nan'), float(proba_test.max()) if len(proba_test) else float('nan')

    print(f"[win_prob][{model_label}] feats={len(feat_cols)}  AUC={auc:.3f}  ACC={acc:.3f}  n_test={len(y_test)}  pos_rate={pos_rate:.3f}  proba[min,max]=[{pmin:.3f},{pmax:.3f}]")



    # Logistic companion model on select features

    logistic_candidates = [

        "spread_line",

        "home_moneyline",

        "away_moneyline",

        "total_line",

        "inj_listed_total_diff",

        "inj_out_diff",

        "roster_injured_pct_diff",

        "team_news_injury_kw7_diff",

        "news_injury_kw7_diff",

    ]

    log_feat_cols = [c for c in logistic_candidates if c in tgt_df_nodup.columns]

    logistic_model = None

    proba_test_log = None

    proba_cal_log = None

    if skip_logit:

        print("[win_prob][logit] skipped via --skip-logit")

    elif log_feat_cols:

        X_log_full = tgt_df_nodup[log_feat_cols].astype(float).values

        X_log_train = X_log_full[idx_train]

        X_log_cal = X_log_full[idx_cal]

        X_log_test = X_log_full[idx_test]



        logistic_model = Pipeline(

            steps=[

                ("imputer", _make_median_imputer()),


    report_feature_quality(tgt_df_nodup, use_cols, context="win_prob")


                ("scaler", StandardScaler()),

                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),

            ]

        )

        logistic_model.fit(X_log_train, y_train)

        proba_test_log = logistic_model.predict_proba(X_log_test)[:, 1]

        proba_cal_log = logistic_model.predict_proba(X_log_cal)[:, 1]



        log_auc = roc_auc_score(y_test, proba_test_log)

        log_acc = accuracy_score(y_test, (proba_test_log >= 0.5).astype(int))

        print(f"[win_prob][logit] feats={len(log_feat_cols)}  AUC={log_auc:.3f}  ACC={log_acc:.3f}")

    else:

        print("[win_prob][logit] skipped (no logistic features present)")



    # Blend on calibration fold

    ensemble_weight = 1.0

    if proba_cal_log is not None:

        best_w = 1.0

        best_brier = float("inf")

        for w in np.linspace(0.0, 1.0, 21):

            blended = w * proba_cal + (1.0 - w) * proba_cal_log

            brier = np.mean((blended - y_cal) ** 2)

            if brier < best_brier:

                best_brier = brier

                best_w = float(w)

        ensemble_weight = best_w

        proba_test_ensemble = ensemble_weight * proba_test + (1.0 - ensemble_weight) * proba_test_log

        ens_auc = roc_auc_score(y_test, proba_test_ensemble)

        ens_acc = accuracy_score(y_test, (proba_test_ensemble >= 0.5).astype(int))

        print(f"[win_prob][blend] weight_gb={ensemble_weight:.2f}  AUC={ens_auc:.3f}  ACC={ens_acc:.3f}  Brier(cal)={best_brier:.4f}")

    else:

        proba_test_ensemble = proba_test

    print(f"[win_prob][blend] logistic unavailable, using {model_label} only")



    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.DataFrame(X_train, columns=use_cols)

    feat_ranges = {c: (float(train_df[c].min()), float(train_df[c].max())) for c in use_cols}

    joblib.dump(

        {

            "model": model,

            "features": use_cols,

            "calibrated": calibrate,

            "feature_ranges": feat_ranges,

            "logistic_model": logistic_model,

            "logistic_features": log_feat_cols,

            "ensemble_weight": float(ensemble_weight),

            "model_type": model_label,

        },

        MODELS_DIR / "winprob_gb.pkl",

    )





def train_spread(
    df: pd.DataFrame,
    use_gpu: bool = False,
    param_config: dict[str, object] | None = None,
    config_source: Path | None = None,
    cv_folds: int = 3,
) -> None:
    """
    Train regression models for point-spread prediction and quantile ranges.

    Parameters mirror `train_win_prob`, but target is `home_margin` and models
    include quantile regressors for volatility estimates.

    cv_folds : int
        Number of CV folds for evaluation metrics. Use 1 to disable CV and
        use a single 75/25 split instead.
    """
    feat_df = make_feature_diffs(df)

    feat_cols = select_feature_columns(feat_df)

    if not feat_cols:

        raise ValueError("No usable *_diff features discovered for spread.")



    tgt_df = _require_target(feat_df, "home_margin")



    tgt_df_nodup = tgt_df.loc[:, ~pd.Index(tgt_df.columns).duplicated()]

    use_cols = [c for c in feat_cols if c in tgt_df_nodup.columns]

    if not use_cols:

        raise ValueError("No usable features after deduplication for spread.")

    X = tgt_df_nodup[use_cols].astype(float).values
    report_feature_quality(tgt_df_nodup, use_cols, context="spread")

    y = tgt_df["home_margin"].astype(float).values



    X_train, X_test, y_train, y_test = train_test_split(

        X, y, test_size=0.25, random_state=42

    )



    if use_gpu and not XGB_AVAILABLE:

        raise RuntimeError("GPU training requested but XGBoost is not installed.")



    model_choice_raw = None

    if param_config and "model_type" in param_config:

        model_choice_raw = str(param_config["model_type"]).strip().lower()



    if model_choice_raw in {"hist", "hgb", "gb_hist", "huber"}:

        model_choice = "hist"

    elif model_choice_raw in {"gb", "gbr", "gbdt"}:

        model_choice = "gbr"

    elif model_choice_raw in {"xgb", "xgboost"}:

        model_choice = "xgb"

    else:

        model_choice = model_choice_raw



    if param_config and config_source:

        print(

            f"[spread] applying tuned params from {config_source} "

            f"(model_type={model_choice or 'auto'}; raw={model_choice_raw})"

        )



    use_hist = model_choice == "hist"
    use_gbr = model_choice == "gbr"
    use_xgb = XGB_AVAILABLE and model_choice in {None, "xgb"}
    if model_choice not in {"hist", "gbr", "xgb", None} and param_config:
        print(f"[spread] warning: unknown model_type '{model_choice_raw}', defaulting to XGBoost.")
        use_xgb = XGB_AVAILABLE

    # --- k-fold cross-validation for evaluation metrics ---
    if cv_folds > 1:
        kf = KFold(n_splits=cv_folds, shuffle=False)
        cv_maes, cv_r2s = [], []
        cv_imputer = _make_median_imputer()
        for fold_idx, (fold_train_idx, fold_val_idx) in enumerate(kf.split(X)):
            X_ftrain_raw = X[fold_train_idx]
            y_ftrain = y[fold_train_idx]
            X_fval_raw = X[fold_val_idx]
            y_fval = y[fold_val_idx]
            X_ftrain_imp = cv_imputer.fit_transform(X_ftrain_raw)
            X_fval_imp = cv_imputer.transform(X_fval_raw)
            if use_xgb:
                fold_xgb_params: dict[str, object] = {
                    "n_estimators": 1000, "learning_rate": 0.05, "max_depth": 6,
                    "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0,
                    "random_state": 42, "objective": "reg:squarederror", "verbosity": 0,
                }
                if param_config:
                    fold_xgb_params.update(_apply_prefixed_params(param_config, "xgb_"))
                fold_xgb_params.update(get_xgb_tree_config(use_gpu))
                fold_model = XGBRegressor(**fold_xgb_params)
                _fit_xgb_with_fallback(fold_model, X_ftrain_imp, y_ftrain, use_gpu, f"[spread][cv_fold={fold_idx}][xgb]")
            elif use_hist:
                fold_hgb_params: dict[str, object] = {
                    "loss": "squared_error", "learning_rate": 0.05, "max_depth": None,
                    "max_leaf_nodes": 31, "min_samples_leaf": 20,
                    "l2_regularization": 0.1, "max_bins": 255, "random_state": 42,
                }
                if param_config:
                    fold_hgb_params.update(_apply_prefixed_params(param_config, "hgb_"))
                fold_model = HistGradientBoostingRegressor(**fold_hgb_params)
                fold_model.fit(X_ftrain_imp, y_ftrain)
            else:
                fold_gbr_params: dict[str, object] = {
                    "loss": "huber", "alpha": 0.9, "learning_rate": 0.05,
                    "n_estimators": 600, "max_depth": 4, "min_samples_leaf": 20,
                    "max_features": "sqrt", "subsample": 0.8, "random_state": 42,
                }
                if param_config:
                    fold_gbr_params.update(_apply_prefixed_params(param_config, "gbr_"))
                fold_model = GradientBoostingRegressor(**fold_gbr_params)
                fold_model.fit(X_ftrain_imp, y_ftrain)
            fold_pred = fold_model.predict(X_fval_imp)
            cv_maes.append(mean_absolute_error(y_fval, fold_pred))
            cv_r2s.append(r2_score(y_fval, fold_pred))
        print(
            f"[spread][cv={cv_folds}] "
            f"MAE={np.mean(cv_maes):.4f}±{np.std(cv_maes):.4f}  "
            f"R2={np.mean(cv_r2s):.4f}±{np.std(cv_r2s):.4f}"
        )

    imputer = _make_median_imputer()
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)


    gpu_used = False

    bias_correction = 0.0

    if use_xgb:

        xgb_params: dict[str, object] = {

            "n_estimators": 1000,

            "learning_rate": 0.05,

            "max_depth": 6,

            "subsample": 0.8,

            "colsample_bytree": 0.8,

            "reg_lambda": 1.0,

            "random_state": 42,

            "objective": "reg:squarederror",

            "verbosity": 0,

        }

        if param_config:

            xgb_params.update(_apply_prefixed_params(param_config, "xgb_"))

        xgb_params.update(get_xgb_tree_config(use_gpu))

        model = XGBRegressor(**xgb_params)

        gpu_used = _fit_xgb_with_fallback(model, X_train_imp, y_train, use_gpu, "[spread][xgb]")

        model_label = "xgb_gpu" if gpu_used else "xgb"

    elif use_hist:

        hgb_params: dict[str, object] = {

            "loss": "squared_error",

            "learning_rate": 0.05,

            "max_depth": None,

            "max_leaf_nodes": 31,

            "min_samples_leaf": 20,

            "l2_regularization": 0.1,

            "max_bins": 255,

            "random_state": 42,

        }

        if param_config:

            hgb_params.update(_apply_prefixed_params(param_config, "hgb_"))

        model = HistGradientBoostingRegressor(**hgb_params)

        model.fit(X_train_imp, y_train)

        model_label = "hgb"

    else:

        gbr_params: dict[str, object] = {

            "loss": "huber",

            "alpha": 0.9,

            "learning_rate": 0.05,

            "n_estimators": 600,

            "max_depth": 4,

            "min_samples_leaf": 20,

            "max_features": "sqrt",

            "subsample": 0.8,

            "random_state": 42,

        }

        if param_config:

            gbr_params.update(_apply_prefixed_params(param_config, "gbr_"))

        model = GradientBoostingRegressor(**gbr_params)

        model.fit(X_train_imp, y_train)

        model_label = "gbr"



    train_pred = model.predict(X_train_imp)

    bias_correction = float((train_pred - y_train).mean())

    if bias_correction:

        pred = model.predict(X_test_imp) - bias_correction

    else:

        pred = model.predict(X_test_imp)



    mae = mean_absolute_error(y_test, pred)

    r2 = r2_score(y_test, pred)

    print(

        f"[spread][{model_label}] feats={len(feat_cols)}  MAE={mae:.2f}  "

        f"R2={r2:.3f}  bias={bias_correction:.3f}  n_test={len(y_test)}"

    )



    quantile_models: dict[float, GradientBoostingRegressor] = {}

    for q in (0.2, 0.8):

        q_model = GradientBoostingRegressor(

            loss="quantile",

            alpha=q,

            learning_rate=0.05,

            n_estimators=500,

            max_depth=3,

            min_samples_leaf=20,

            max_features="sqrt",

            subsample=0.85,

            random_state=42,

        )

        q_model.fit(X_train_imp, y_train)

        quantile_models[q] = q_model



    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(

        {

            "model": model,

            "features": use_cols,

            "imputer": imputer,

            "quantile_models": quantile_models,

            "model_type": model_label,

            "bias_correction": bias_correction,

        },

        MODELS_DIR / "spread_gb.pkl",

    )





def main():
    """Parse CLI args, load features, and train the requested model target."""
    ap = argparse.ArgumentParser()

    ap.add_argument("--target", choices=["win_prob", "spread"], required=True)

    ap.add_argument("--calibrate", action="store_true", help="Calibrate win probability with isotonic regression (win_prob only)")

    ap.add_argument("--debug", action="store_true", help="Enable verbose debug output")

    ap.add_argument(

        "--train-start-year",

        type=int,

        default=None,

        help="If provided, restrict training data to seasons >= this year.",

    )

    ap.add_argument(

        "--use-gpu",

        action="store_true",

        help="Use GPU-accelerated models (requires XGBoost).",

    )

    ap.add_argument(

        "--param-config",

        type=str,

        default=None,

        help="Optional JSON file containing tuned hyperparameters (from src.models.tune).",

    )

    ap.add_argument(
        "--skip-logit",
        action="store_true",
        help="Skip the logistic companion model when training win probability.",
    )
    ap.add_argument(
        "--cv-folds",
        type=int,
        default=3,
        help="Number of CV folds for evaluation. Use 1 to disable CV and use a single split.",
    )
    ap.add_argument("--exclude-from-season", type=int, default=None, help="Exclude this season/week and later rows from training.")
    ap.add_argument("--exclude-from-week", type=int, default=None, help="Exclude this season/week and later rows from training.")
    args = ap.parse_args()


    configure_logging(args.debug)



    feats_path = PROC_DIR / "matchup_features.parquet"

    if not feats_path.exists():

        raise FileNotFoundError(f"Expected features at {feats_path}. Run build_features first.")

    df = read_df(feats_path)

    if args.train_start_year is not None and "season" in df.columns:

        df = df[df["season"] >= args.train_start_year].copy()

    before_cutoff = len(df)
    df = filter_before_week(df, args.exclude_from_season, args.exclude_from_week)
    if args.exclude_from_season is not None and args.exclude_from_week is not None and len(df) != before_cutoff:
        print(
            f"[train] excluded {before_cutoff - len(df)} rows at/after "
            f"{args.exclude_from_season} Week {args.exclude_from_week}"
        )



    param_cfg, cfg_path = load_param_config(args.param_config)



    if args.target == "win_prob":
        train_win_prob(
            df,
            calibrate=args.calibrate,
            use_gpu=args.use_gpu,
            param_config=param_cfg,
            config_source=cfg_path,
            skip_logit=args.skip_logit,
            cv_folds=args.cv_folds,
        )
    else:
        train_spread(
            df,
            use_gpu=args.use_gpu,
            param_config=param_cfg,
            config_source=cfg_path,
            cv_folds=args.cv_folds,
        )




if __name__ == "__main__":

    main()

