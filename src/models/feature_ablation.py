# Copyright (c) 2025 Matt Hanson
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature-family ablation diagnostics for team prediction models."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold

from src.models.feature_columns import make_feature_diffs, is_bdl_feature_column, select_feature_columns
from src.models import train as train_module
from src.models.train import (
    XGB_AVAILABLE,
    _apply_prefixed_params,
    _fit_xgb_with_fallback,
    _make_median_imputer,
    get_xgb_tree_config,
    load_param_config,
)
from src.utils.io import PROC_DIR, read_df
from src.utils.logging_config import setup_logging
from src.utils.week_filter import filter_before_week

DEFAULT_FEATURES_PATH = PROC_DIR / "matchup_features.parquet"
DEFAULT_OUTPUT_PATH = Path("predictions/evaluation/feature_family_ablation.csv")
def is_bdl_feature(column: str) -> bool:
    """Return whether a selected model feature belongs to the BallDontLie family."""
    return is_bdl_feature_column(column)


def is_sparse_bdl_sack_rate_feature(column: str) -> bool:
    """Return whether a BallDontLie feature is a known sparse sack-rate derivative."""
    name = str(column).lower()
    return is_bdl_feature_column(name) and "sack_rate" in name


def build_feature_sets(feature_cols: Iterable[str]) -> dict[str, list[str]]:
    """Build ordered feature sets used by the ablation report."""
    cols = list(feature_cols)
    no_bdl = [col for col in cols if not is_bdl_feature(col)]
    no_sparse_sack_rate = [col for col in cols if not is_sparse_bdl_sack_rate_feature(col)]
    return {
        "baseline_no_bdl": no_bdl,
        "baseline_plus_bdl": cols,
        "baseline_plus_bdl_no_sparse_sack_rate": no_sparse_sack_rate,
    }


def _load_filtered_features(
    path: Path,
    *,
    train_start_year: int | None,
    exclude_from_season: int | None,
    exclude_from_week: int | None,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Expected features at {path}. Run build_features first.")
    frame = read_df(path)
    if train_start_year is not None and "season" in frame.columns:
        frame = frame[frame["season"] >= train_start_year].copy()
    frame = filter_before_week(frame, exclude_from_season, exclude_from_week)
    return frame


def _safe_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if len(arr) else float("nan")


def _safe_std(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.std()) if len(arr) else float("nan")


def _classification_model(
    *,
    use_gpu: bool,
    param_config: dict[str, object] | None,
    n_estimators: int,
):
    if XGB_AVAILABLE:
        params: dict[str, object] = {
            "n_estimators": n_estimators,
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
            params.update(_apply_prefixed_params(param_config, "xgb_"))
        params.update(get_xgb_tree_config(use_gpu))
        params["n_estimators"] = int(params.get("n_estimators", n_estimators))
        return train_module.XGBClassifier(**params), "xgb"

    if use_gpu:
        raise RuntimeError("GPU ablation requested but XGBoost is not installed.")

    params: dict[str, object] = {
        "random_state": 42,
        "max_leaf_nodes": 31,
        "learning_rate": 0.05,
        "l2_regularization": 0.1,
        "min_samples_leaf": 20,
        "max_bins": 255,
    }
    if param_config:
        params.update(_apply_prefixed_params(param_config, "hgb_"))
    return HistGradientBoostingClassifier(**params), "hgb"


def _spread_model(
    *,
    use_gpu: bool,
    param_config: dict[str, object] | None,
    n_estimators: int,
):
    if XGB_AVAILABLE:
        params: dict[str, object] = {
            "n_estimators": n_estimators,
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
            params.update(_apply_prefixed_params(param_config, "xgb_"))
        params.update(get_xgb_tree_config(use_gpu))
        params["n_estimators"] = int(params.get("n_estimators", n_estimators))
        return train_module.XGBRegressor(**params), "xgb"

    if use_gpu:
        raise RuntimeError("GPU ablation requested but XGBoost is not installed.")

    params = {
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
        params.update(_apply_prefixed_params(param_config, "hgb_"))
    return HistGradientBoostingRegressor(**params), "hgb"


def _evaluate_win_prob(
    frame: pd.DataFrame,
    feature_cols: list[str],
    *,
    cv_folds: int,
    use_gpu: bool,
    param_config: dict[str, object] | None,
    n_estimators: int,
) -> dict[str, float | str | int]:
    target_frame = frame.dropna(subset=["home_win"]).copy()
    if target_frame.empty or not feature_cols:
        raise ValueError("No win-probability rows/features available for ablation.")

    X_raw = target_frame[feature_cols].astype(float).to_numpy()
    y = target_frame["home_win"].astype(int).to_numpy()
    folds = max(2, int(cv_folds))
    splitter = StratifiedKFold(n_splits=folds, shuffle=False)
    aucs: list[float] = []
    briers: list[float] = []
    loglosses: list[float] = []
    accs: list[float] = []
    backend = "unknown"

    for train_idx, val_idx in splitter.split(X_raw, y):
        imputer = _make_median_imputer()
        X_train = imputer.fit_transform(X_raw[train_idx])
        X_val = imputer.transform(X_raw[val_idx])
        model, backend = _classification_model(
            use_gpu=use_gpu,
            param_config=param_config,
            n_estimators=n_estimators,
        )
        _fit_xgb_with_fallback(model, X_train, y[train_idx], use_gpu, "[feature_ablation][win_prob][xgb]")
        proba = np.clip(model.predict_proba(X_val)[:, 1], 1e-6, 1 - 1e-6)
        aucs.append(roc_auc_score(y[val_idx], proba))
        briers.append(brier_score_loss(y[val_idx], proba))
        loglosses.append(log_loss(y[val_idx], proba))
        accs.append(accuracy_score(y[val_idx], (proba >= 0.5).astype(int)))

    return {
        "model_backend": backend,
        "n_rows": int(len(y)),
        "auc_mean": _safe_mean(aucs),
        "auc_std": _safe_std(aucs),
        "brier_mean": _safe_mean(briers),
        "brier_std": _safe_std(briers),
        "logloss_mean": _safe_mean(loglosses),
        "logloss_std": _safe_std(loglosses),
        "accuracy_mean": _safe_mean(accs),
        "accuracy_std": _safe_std(accs),
        "primary_metric": "auc",
        "primary_value": _safe_mean(aucs),
    }


def _evaluate_spread(
    frame: pd.DataFrame,
    feature_cols: list[str],
    *,
    cv_folds: int,
    use_gpu: bool,
    param_config: dict[str, object] | None,
    n_estimators: int,
) -> dict[str, float | str | int]:
    target_frame = frame.dropna(subset=["home_margin"]).copy()
    if target_frame.empty or not feature_cols:
        raise ValueError("No spread rows/features available for ablation.")

    X_raw = target_frame[feature_cols].astype(float).to_numpy()
    y = target_frame["home_margin"].astype(float).to_numpy()
    folds = max(2, int(cv_folds))
    splitter = KFold(n_splits=folds, shuffle=False)
    maes: list[float] = []
    rmses: list[float] = []
    r2s: list[float] = []
    backend = "unknown"

    for train_idx, val_idx in splitter.split(X_raw):
        imputer = _make_median_imputer()
        X_train = imputer.fit_transform(X_raw[train_idx])
        X_val = imputer.transform(X_raw[val_idx])
        model, backend = _spread_model(
            use_gpu=use_gpu,
            param_config=param_config,
            n_estimators=n_estimators,
        )
        _fit_xgb_with_fallback(model, X_train, y[train_idx], use_gpu, "[feature_ablation][spread][xgb]")
        pred = model.predict(X_val)
        maes.append(mean_absolute_error(y[val_idx], pred))
        rmses.append(float(np.sqrt(np.mean((np.asarray(y[val_idx]) - np.asarray(pred)) ** 2))))
        r2s.append(r2_score(y[val_idx], pred))

    return {
        "model_backend": backend,
        "n_rows": int(len(y)),
        "mae_mean": _safe_mean(maes),
        "mae_std": _safe_std(maes),
        "rmse_mean": _safe_mean(rmses),
        "rmse_std": _safe_std(rmses),
        "r2_mean": _safe_mean(r2s),
        "r2_std": _safe_std(r2s),
        "primary_metric": "mae",
        "primary_value": _safe_mean(maes),
    }


def build_ablation_report(
    raw_features: pd.DataFrame,
    *,
    cv_folds: int = 3,
    use_gpu: bool = False,
    winprob_param_config: dict[str, object] | None = None,
    spread_param_config: dict[str, object] | None = None,
    winprob_estimators: int = 300,
    spread_estimators: int = 400,
) -> pd.DataFrame:
    """Evaluate selected feature-family variants on identical CV folds."""
    feature_frame = make_feature_diffs(raw_features)
    feature_frame = feature_frame.loc[:, ~pd.Index(feature_frame.columns).duplicated()]
    selected_features = [
        col for col in select_feature_columns(feature_frame, include_bdl=True) if col in feature_frame.columns
    ]
    if not selected_features:
        raise ValueError("No usable features available for ablation.")

    generated_at = pd.Timestamp.utcnow().isoformat()
    rows: list[dict[str, object]] = []
    feature_sets = build_feature_sets(selected_features)

    for feature_set_name, feature_cols in feature_sets.items():
        if not feature_cols:
            continue
        bdl_count = sum(is_bdl_feature(col) for col in feature_cols)
        sparse_bdl_count = sum(is_sparse_bdl_sack_rate_feature(col) for col in feature_cols)
        for target in ("win_prob", "spread"):
            if target == "win_prob":
                metrics = _evaluate_win_prob(
                    feature_frame,
                    feature_cols,
                    cv_folds=cv_folds,
                    use_gpu=use_gpu,
                    param_config=winprob_param_config,
                    n_estimators=winprob_estimators,
                )
            else:
                metrics = _evaluate_spread(
                    feature_frame,
                    feature_cols,
                    cv_folds=cv_folds,
                    use_gpu=use_gpu,
                    param_config=spread_param_config,
                    n_estimators=spread_estimators,
                )
            rows.append(
                {
                    "generated_at": generated_at,
                    "target": target,
                    "feature_set": feature_set_name,
                    "cv_folds": int(cv_folds),
                    "feature_count": len(feature_cols),
                    "total_selected_feature_count": len(selected_features),
                    "bdl_feature_count": bdl_count,
                    "sparse_bdl_sack_rate_feature_count": sparse_bdl_count,
                    **metrics,
                }
            )

    report = pd.DataFrame(rows)
    if report.empty:
        return report

    report["primary_delta_vs_baseline"] = np.nan
    metric_columns = [
        "auc_mean",
        "brier_mean",
        "logloss_mean",
        "accuracy_mean",
        "mae_mean",
        "rmse_mean",
        "r2_mean",
    ]
    for col in metric_columns:
        if col not in report.columns:
            report[col] = np.nan
        report[f"{col}_delta_vs_baseline"] = np.nan

    for target, group in report.groupby("target"):
        baseline = group[group["feature_set"].eq("baseline_no_bdl")]
        if baseline.empty:
            continue
        baseline_row = baseline.iloc[0]
        target_mask = report["target"].eq(target)
        for col in metric_columns:
            base_value = baseline_row.get(col)
            if pd.notna(base_value):
                report.loc[target_mask, f"{col}_delta_vs_baseline"] = report.loc[target_mask, col] - float(base_value)
        primary = str(baseline_row.get("primary_metric", ""))
        primary_col = f"{primary}_mean"
        if primary_col in report.columns and pd.notna(baseline_row.get(primary_col)):
            report.loc[target_mask, "primary_delta_vs_baseline"] = (
                report.loc[target_mask, primary_col] - float(baseline_row[primary_col])
            )

    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare model performance with and without selected feature families.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--features-path", type=Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--train-start-year", type=int, default=None)
    parser.add_argument("--exclude-from-season", type=int, default=None)
    parser.add_argument("--exclude-from-week", type=int, default=None)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--winprob-param-config", type=Path, default=None)
    parser.add_argument("--spread-param-config", type=Path, default=None)
    parser.add_argument("--winprob-estimators", type=int, default=300)
    parser.add_argument("--spread-estimators", type=int, default=400)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    setup_logging("nfl_predictions", level="DEBUG" if args.debug else "INFO")
    winprob_params, _winprob_cfg_path = load_param_config(
        str(args.winprob_param_config) if args.winprob_param_config else None
    )
    spread_params, _spread_cfg_path = load_param_config(
        str(args.spread_param_config) if args.spread_param_config else None
    )
    features = _load_filtered_features(
        args.features_path,
        train_start_year=args.train_start_year,
        exclude_from_season=args.exclude_from_season,
        exclude_from_week=args.exclude_from_week,
    )
    report = build_ablation_report(
        features,
        cv_folds=args.cv_folds,
        use_gpu=args.use_gpu,
        winprob_param_config=winprob_params,
        spread_param_config=spread_params,
        winprob_estimators=args.winprob_estimators,
        spread_estimators=args.spread_estimators,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output, index=False)
    print(f"[feature_ablation] wrote {len(report)} rows -> {args.output}")

    for row in report.to_dict("records"):
        target = row.get("target")
        feature_set = row.get("feature_set")
        primary = row.get("primary_metric")
        value = float(row.get("primary_value", np.nan))
        delta = float(row.get("primary_delta_vs_baseline", np.nan))
        print(
            f"[feature_ablation] {target} {feature_set}: "
            f"{primary}={value:.4f} delta_vs_baseline={delta:+.4f} "
            f"features={row.get('feature_count')} bdl={row.get('bdl_feature_count')}"
        )


if __name__ == "__main__":
    main()
