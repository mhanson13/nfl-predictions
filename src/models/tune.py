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

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import optuna
import pandas as pd
from optuna.trial import Trial
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

from src.utils.io import PROC_DIR, read_df
from src.utils.week_filter import filter_before_week
from src.models import train as base_train
from src.models.train import (
    _make_feature_diffs,
    _select_feature_columns,
    _require_target,
    _fit_xgb_with_fallback,
)

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:  # pragma: no cover - optional dependency
    XGBClassifier = None
    XGBRegressor = None


@dataclass
class DatasetSplit:
    X_train: np.ndarray
    X_valid: np.ndarray
    y_train: np.ndarray
    y_valid: np.ndarray
    feature_names: list[str]


CLASSIFIER_METRICS = ("logloss", "brier", "auc", "accuracy")
REGRESSOR_METRICS = ("mae", "rmse")
ALLOWED_MODEL_TYPES: set[str] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hyperparameter tuning with Optuna.")
    parser.add_argument("--target", choices=["win_prob", "spread"], required=True)
    parser.add_argument("--train-start-year", type=int, default=None, help="Restrict training rows to seasons >= this year.")
    parser.add_argument("--use-gpu", action="store_true", help="Request GPU-accelerated XGBoost (falls back to CPU if unavailable).")
    parser.add_argument("--metric", type=str, help="Optimization metric (default depends on target).")
    parser.add_argument("--trials", type=int, default=40, help="Number of Optuna trials.")
    parser.add_argument("--timeout", type=float, default=None, help="Time budget in seconds for the study.")
    parser.add_argument("--study-name", type=str, default=None, help="Optional Optuna study name.")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URI (e.g., sqlite:///optuna.db).")
    parser.add_argument("--load-if-exists", action="store_true", help="Reuse an existing study if one is found in storage.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed used for data splits and estimators.")
    parser.add_argument("--save-best", type=Path, default=None, help="Optional path to persist the best params as JSON.")
    parser.add_argument(
        "--model-types",
        nargs="+",
        help="Restrict tuning to specific model types (e.g., hist xgb gbr).",
    )
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging.")
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=3,
        help="Number of CV folds for Optuna objective scoring. Use 1 to disable CV and use a single split.",
    )
    parser.add_argument("--exclude-from-season", type=int, default=None, help="Exclude this season/week and later rows from tuning.")
    parser.add_argument("--exclude-from-week", type=int, default=None, help="Exclude this season/week and later rows from tuning.")
    return parser.parse_args()


def load_features(
    train_start_year: Optional[int],
    exclude_from_season: Optional[int] = None,
    exclude_from_week: Optional[int] = None,
) -> pd.DataFrame:
    feats_path = PROC_DIR / "matchup_features.parquet"
    if not feats_path.exists():
        raise FileNotFoundError(f"Expected features at {feats_path}. Run build_features first.")
    df = read_df(feats_path)
    if train_start_year is not None and "season" in df.columns:
        df = df[df["season"] >= train_start_year].copy()
    before_cutoff = len(df)
    df = filter_before_week(df, exclude_from_season, exclude_from_week)
    if exclude_from_season is not None and exclude_from_week is not None and len(df) != before_cutoff:
        print(
            f"[tune] excluded {before_cutoff - len(df)} rows at/after "
            f"{exclude_from_season} Week {exclude_from_week}"
        )
    if df.empty:
        raise ValueError("Feature frame is empty after filtering; nothing to tune.")
    return df


def _deduplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~pd.Index(df.columns).duplicated()]


def prepare_win_prob_data(df: pd.DataFrame, random_state: int) -> DatasetSplit:
    feat_df = _make_feature_diffs(df)
    feat_cols = _select_feature_columns(feat_df)
    if not feat_cols:
        raise ValueError("No usable *_diff features discovered for win probability tuning.")

    tgt_df = _require_target(feat_df, "home_win")
    tgt_df = _deduplicate_columns(tgt_df)
    use_cols = [c for c in feat_cols if c in tgt_df.columns]
    if not use_cols:
        raise ValueError("No usable features remain after deduplication.")

    X = tgt_df[use_cols].astype(float).values
    y = np.asarray(tgt_df["home_win"].astype(int).values, dtype=int)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )
    return DatasetSplit(X_train, X_valid, y_train, y_valid, use_cols)


def prepare_spread_data(df: pd.DataFrame, random_state: int) -> DatasetSplit:
    feat_df = _make_feature_diffs(df)
    feat_cols = _select_feature_columns(feat_df)
    if not feat_cols:
        raise ValueError("No usable *_diff features discovered for spread tuning.")

    tgt_df = _require_target(feat_df, "home_margin")
    tgt_df = _deduplicate_columns(tgt_df)
    use_cols = [c for c in feat_cols if c in tgt_df.columns]
    if not use_cols:
        raise ValueError("No usable spread features remain after deduplication.")

    X = tgt_df[use_cols].astype(float).values
    y = tgt_df["home_margin"].astype(float).values

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
    )
    return DatasetSplit(X_train, X_valid, y_train, y_valid, use_cols)


def available_classifier_models() -> list[str]:
    models = ["hist"]
    if XGBClassifier is not None and getattr(base_train, "XGB_AVAILABLE", False):
        models.append("xgb")
    if ALLOWED_MODEL_TYPES is not None:
        models = [m for m in models if m in ALLOWED_MODEL_TYPES]
    return models


def available_regressor_models() -> list[str]:
    models = ["hist", "gbr"]
    if XGBRegressor is not None and getattr(base_train, "XGB_AVAILABLE", False):
        models.append("xgb")
    if ALLOWED_MODEL_TYPES is not None:
        models = [m for m in models if m in ALLOWED_MODEL_TYPES]
    return models


def win_prob_objective(
    trial: Trial,
    data: DatasetSplit,
    metric: str,
    use_gpu: bool,
    random_state: int,
    cv_folds: int = 3,
) -> float:
    model_choice = trial.suggest_categorical("model_type", available_classifier_models())

    if model_choice == "xgb":
        booster_params = {
            "n_estimators": trial.suggest_int("xgb_n_estimators", 250, 1600),
            "learning_rate": trial.suggest_float("xgb_learning_rate", 0.005, 0.3, log=True),
            "max_depth": trial.suggest_int("xgb_max_depth", 3, 10),
            "min_child_weight": trial.suggest_float("xgb_min_child_weight", 0.5, 12.0),
            "subsample": trial.suggest_float("xgb_subsample", 0.4, 1.0),
            "colsample_bytree": trial.suggest_float("xgb_colsample_bytree", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("xgb_reg_lambda", 0.05, 20.0, log=True),
            "reg_alpha": trial.suggest_float("xgb_reg_alpha", 1e-9, 2.0, log=True),
            "gamma": trial.suggest_float("xgb_gamma", 0.0, 6.0),
            "random_state": random_state,
            "eval_metric": "logloss",
            "use_label_encoder": False,
        }
        booster_params.update(base_train.get_xgb_tree_config(use_gpu))
        model = XGBClassifier(**booster_params)
        gpu_used = _fit_xgb_with_fallback(model, data.X_train, data.y_train, use_gpu, "[optuna][win_prob][xgb]")
        trial.set_user_attr("used_gpu", gpu_used)
    else:
        model = HistGradientBoostingClassifier(
            learning_rate=trial.suggest_float("hgb_learning_rate", 0.01, 0.25),
            max_depth=trial.suggest_categorical("hgb_max_depth", [None, 3, 4, 5, 6, 7, 8]),
            max_leaf_nodes=trial.suggest_int("hgb_max_leaf_nodes", 16, 96),
            min_samples_leaf=trial.suggest_int("hgb_min_samples_leaf", 5, 80),
            l2_regularization=trial.suggest_float("hgb_l2_regularization", 1e-5, 2.0, log=True),
            max_bins=trial.suggest_int("hgb_max_bins", 64, 255),
            random_state=random_state,
        )
        model.fit(data.X_train, data.y_train)
        trial.set_user_attr("used_gpu", False)

    if cv_folds > 1:
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=False)
        fold_loglosses, fold_briers, fold_aucs, fold_accs = [], [], [], []
        for fold_train_idx, fold_val_idx in skf.split(data.X_train, data.y_train):
            X_ft = data.X_train[fold_train_idx]
            y_ft = data.y_train[fold_train_idx]
            X_fv = data.X_train[fold_val_idx]
            y_fv = data.y_train[fold_val_idx]
            fold_model_clone = type(model)(**model.get_params())
            if model_choice == "xgb":
                _fit_xgb_with_fallback(fold_model_clone, X_ft, y_ft, use_gpu, "[optuna][win_prob][cv][xgb]")
            else:
                fold_model_clone.fit(X_ft, y_ft)
            fp = np.clip(fold_model_clone.predict_proba(X_fv)[:, 1], 1e-6, 1 - 1e-6)
            fold_loglosses.append(log_loss(y_fv, fp))
            fold_briers.append(brier_score_loss(y_fv, fp))
            fold_aucs.append(roc_auc_score(y_fv, fp))
            fold_accs.append(accuracy_score(y_fv, (fp >= 0.5).astype(int)))
        logloss = float(np.mean(fold_loglosses))
        brier = float(np.mean(fold_briers))
        auc = float(np.mean(fold_aucs))
        acc = float(np.mean(fold_accs))
    else:
        proba_valid = np.clip(model.predict_proba(data.X_valid)[:, 1], 1e-6, 1 - 1e-6)
        logloss = log_loss(data.y_valid, proba_valid)
        brier = brier_score_loss(data.y_valid, proba_valid)
        auc = roc_auc_score(data.y_valid, proba_valid)
        acc = accuracy_score(data.y_valid, (proba_valid >= 0.5).astype(int))

    trial.set_user_attr("logloss", logloss)
    trial.set_user_attr("brier", brier)
    trial.set_user_attr("auc", auc)
    trial.set_user_attr("accuracy", acc)
    trial.set_user_attr("model_choice", model_choice)

    if metric == "logloss":
        return logloss
    if metric == "brier":
        return brier
    if metric == "accuracy":
        return 1.0 - acc
    if metric == "auc":
        return 1.0 - auc
    if metric == "auc_brier":
        return (1.0 - auc) + brier
    raise ValueError(f"Unknown metric: {metric}")


def spread_objective(
    trial: Trial,
    data: DatasetSplit,
    metric: str,
    use_gpu: bool,
    random_state: int,
    cv_folds: int = 3,
) -> float:
    model_choice = trial.suggest_categorical("model_type", available_regressor_models())
    imputer = base_train._make_median_imputer()
    X_train_imp = imputer.fit_transform(data.X_train)
    X_valid_imp = imputer.transform(data.X_valid)

    if model_choice == "xgb":
        booster_params = {
            "n_estimators": trial.suggest_int("xgb_n_estimators", 300, 2000),
            "learning_rate": trial.suggest_float("xgb_learning_rate", 0.005, 0.3, log=True),
            "max_depth": trial.suggest_int("xgb_max_depth", 3, 10),
            "min_child_weight": trial.suggest_float("xgb_min_child_weight", 0.5, 15.0),
            "subsample": trial.suggest_float("xgb_subsample", 0.4, 1.0),
            "colsample_bytree": trial.suggest_float("xgb_colsample_bytree", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("xgb_reg_lambda", 0.05, 20.0, log=True),
            "reg_alpha": trial.suggest_float("xgb_reg_alpha", 1e-9, 2.0, log=True),
            "gamma": trial.suggest_float("xgb_gamma", 0.0, 6.0),
            "random_state": random_state,
            "verbosity": 0,
        }
        booster_params.update(base_train.get_xgb_tree_config(use_gpu))
        model = XGBRegressor(**booster_params)
        gpu_used = _fit_xgb_with_fallback(model, X_train_imp, data.y_train, use_gpu, "[optuna][spread][xgb]")
        trial.set_user_attr("used_gpu", gpu_used)
    elif model_choice == "hist":
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=trial.suggest_float("hgb_learning_rate", 0.01, 0.25),
            max_depth=trial.suggest_categorical("hgb_max_depth", [None, 3, 4, 5, 6, 7, 8]),
            max_leaf_nodes=trial.suggest_int("hgb_max_leaf_nodes", 16, 96),
            min_samples_leaf=trial.suggest_int("hgb_min_samples_leaf", 5, 80),
            l2_regularization=trial.suggest_float("hgb_l2_regularization", 1e-5, 2.0, log=True),
            max_bins=trial.suggest_int("hgb_max_bins", 64, 255),
            random_state=random_state,
        )
        model.fit(X_train_imp, data.y_train)
        trial.set_user_attr("used_gpu", False)
    else:
        model = GradientBoostingRegressor(
            loss="huber",
            alpha=trial.suggest_float("gbr_alpha", 0.85, 0.99),
            learning_rate=trial.suggest_float("gbr_learning_rate", 0.01, 0.2),
            n_estimators=trial.suggest_int("gbr_n_estimators", 300, 1600),
            max_depth=trial.suggest_int("gbr_max_depth", 2, 6),
            min_samples_leaf=trial.suggest_int("gbr_min_samples_leaf", 5, 80),
            max_features=trial.suggest_categorical("gbr_max_features", [None, "sqrt", "log2"]),
            subsample=trial.suggest_float("gbr_subsample", 0.5, 1.0),
            random_state=random_state,
        )
        model.fit(X_train_imp, data.y_train)
        trial.set_user_attr("used_gpu", False)

    if cv_folds > 1:
        kf = KFold(n_splits=cv_folds, shuffle=False)
        fold_maes, fold_rmses = [], []
        for fold_train_idx, fold_val_idx in kf.split(data.X_train):
            X_ft_raw = data.X_train[fold_train_idx]
            y_ft = data.y_train[fold_train_idx]
            X_fv_raw = data.X_train[fold_val_idx]
            y_fv = data.y_train[fold_val_idx]
            fold_imp = base_train._make_median_imputer()
            X_ft_imp = fold_imp.fit_transform(X_ft_raw)
            X_fv_imp = fold_imp.transform(X_fv_raw)
            fold_model_clone = type(model)(**model.get_params())
            if model_choice == "xgb":
                _fit_xgb_with_fallback(fold_model_clone, X_ft_imp, y_ft, use_gpu, "[optuna][spread][cv][xgb]")
            else:
                fold_model_clone.fit(X_ft_imp, y_ft)
            fold_pred = fold_model_clone.predict(X_fv_imp)
            fold_maes.append(mean_absolute_error(y_fv, fold_pred))
            fold_mse = mean_squared_error(y_fv, fold_pred)
            fold_rmses.append(float(fold_mse) ** 0.5)
        mae = float(np.mean(fold_maes))
        rmse = float(np.mean(fold_rmses))
    else:
        preds = model.predict(X_valid_imp)
        mae = mean_absolute_error(data.y_valid, preds)
        mse_val = mean_squared_error(data.y_valid, preds)
        rmse = float(mse_val) ** 0.5

    trial.set_user_attr("mae", mae)
    trial.set_user_attr("rmse", rmse)
    trial.set_user_attr("model_choice", model_choice)

    if metric == "mae":
        return mae
    if metric == "rmse":
        return rmse
    raise ValueError(f"Unknown metric: {metric}")


def run_study(args: argparse.Namespace) -> optuna.study.Study:
    df = load_features(args.train_start_year, args.exclude_from_season, args.exclude_from_week)
    metric = args.metric or ("logloss" if args.target == "win_prob" else "mae")
    cv_folds = getattr(args, "cv_folds", 3)

    if args.target == "win_prob":
        if metric not in CLASSIFIER_METRICS + ("auc_brier",):
            raise ValueError(
                f"Metric '{metric}' not supported for win_prob. Options: {CLASSIFIER_METRICS + ('auc_brier',)}."
            )
        data = prepare_win_prob_data(df, args.random_state)
        objective = lambda trial: win_prob_objective(trial, data, metric, args.use_gpu, args.random_state, cv_folds)
        direction = "minimize"
    else:
        if metric not in REGRESSOR_METRICS:
            raise ValueError(f"Metric '{metric}' not supported for spread. Options: {REGRESSOR_METRICS}.")
        data = prepare_spread_data(df, args.random_state)
        objective = lambda trial: spread_objective(trial, data, metric, args.use_gpu, args.random_state, cv_folds)
        direction = "minimize"

    sampler = optuna.samplers.TPESampler(seed=args.random_state)
    study = optuna.create_study(
        sampler=sampler,
        direction=direction,
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=args.load_if_exists,
    )
    study.optimize(objective, n_trials=args.trials, timeout=args.timeout)
    return study


def save_best(study: optuna.study.Study, path: Path) -> None:
    payload: Dict[str, Any] = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "user_attrs": study.best_trial.user_attrs,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[optuna] Best params saved to {path}")


def print_summary(study: optuna.study.Study) -> None:
    best = study.best_trial
    print("\n[optuna] Study summary")
    print(f"  Trials: {len(study.trials)} (completed)")
    print(f"  Best value: {best.value:.6f}")
    print(f"  Best params:")
    for key, value in best.params.items():
        print(f"    - {key}: {value}")
    if best.user_attrs:
        print("  Metrics/User attrs:")
        for key, value in best.user_attrs.items():
            print(f"    - {key}: {value}")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    logging.getLogger("urllib3").setLevel(logging.DEBUG if args.debug else logging.INFO)
    global ALLOWED_MODEL_TYPES
    if args.model_types:
        ALLOWED_MODEL_TYPES = {m.lower() for m in args.model_types}
    if args.use_gpu and not getattr(base_train, "XGB_AVAILABLE", False):
        print("[optuna] GPU requested but XGBoost is not available; proceeding with CPU models.")
        args.use_gpu = False

    study = run_study(args)
    print_summary(study)
    if args.save_best:
        save_best(study, args.save_best)


if __name__ == "__main__":
    main()
