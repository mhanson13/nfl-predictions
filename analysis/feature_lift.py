from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

from src.models.train import (
    _make_feature_diffs,
    _require_target,
    _select_feature_columns,
)


RANDOM_STATE = 42
TRAIN_START_YEAR = 2016
ANALYSIS_DIR = Path("analysis")
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

MATCHUP_PATH = Path("data/processed/matchup_features.parquet")


def _filter_features(features: Iterable[str], excluded_prefixes: Iterable[str]) -> list[str]:
    excluded_prefixes = tuple(excluded_prefixes)
    return [c for c in features if not c.startswith(excluded_prefixes)]


def prepare_win_prob(df: pd.DataFrame, drop_prefixes: Iterable[str] | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    feat_df = _make_feature_diffs(df)
    feat_cols = _select_feature_columns(feat_df)

    tgt_df = _require_target(feat_df, "home_win")
    tgt_df = tgt_df.loc[:, ~pd.Index(tgt_df.columns).duplicated()]
    use_cols = [c for c in feat_cols if c in tgt_df.columns]
    if drop_prefixes:
        use_cols = _filter_features(use_cols, drop_prefixes)
    X = tgt_df[use_cols].astype(float).values
    y = tgt_df["home_win"].astype(int).values
    return X, y, use_cols


def prepare_spread(df: pd.DataFrame, drop_prefixes: Iterable[str] | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    feat_df = _make_feature_diffs(df)
    feat_cols = _select_feature_columns(feat_df)

    tgt_df = _require_target(feat_df, "home_margin")
    tgt_df = tgt_df.loc[:, ~pd.Index(tgt_df.columns).duplicated()]
    use_cols = [c for c in feat_cols if c in tgt_df.columns]
    if drop_prefixes:
        use_cols = _filter_features(use_cols, drop_prefixes)
    X = tgt_df[use_cols].astype(float).values
    y = tgt_df["home_margin"].astype(float).values
    return X, y, use_cols


def train_xgb_classifier(X: np.ndarray, y: np.ndarray) -> tuple[XGBClassifier, dict[str, float]]:
    idx_tmp, idx_test, y_tmp, y_test = train_test_split(
        np.arange(len(y)),
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    idx_train, idx_val, y_train, y_val = train_test_split(
        idx_tmp,
        y_tmp,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_tmp,
    )  # 60/20/20

    model = XGBClassifier(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    model.fit(X[idx_train], y[idx_train])

    proba = model.predict_proba(X[idx_test])[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y[idx_test], pred)),
        "auc": float(roc_auc_score(y[idx_test], proba)),
        "brier": float(brier_score_loss(y[idx_test], proba)),
        "logloss": float(log_loss(y[idx_test], proba)),
    }
    return model, metrics, X[idx_test], y[idx_test]


def train_xgb_regressor(X: np.ndarray, y: np.ndarray) -> tuple[XGBRegressor, dict[str, float]]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    model = XGBRegressor(
        n_estimators=650,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = float(mean_squared_error(y_test, preds))
    metrics = {
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(mse ** 0.5),
    }
    return model, metrics, X_test, y_test


def permutation_scores(model, X_test, y_test, feature_names: list[str], *, n_repeats: int = 5) -> pd.DataFrame:
    perm = permutation_importance(model, X_test, y_test, n_repeats=n_repeats, random_state=RANDOM_STATE, n_jobs=1)
    df = pd.DataFrame(
        {
            "feature": feature_names,
            "permutation_importance": perm.importances_mean,
            "permutation_std": perm.importances_std,
        }
    )
    return df.sort_values("permutation_importance", ascending=False)


def compute_feature_importance(
    model,
    feature_names: list[str],
    perm_df: pd.DataFrame,
) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        gain = model.feature_importances_
        gain_df = pd.DataFrame({"feature": feature_names, "model_importance": gain})
    else:
        gain_df = pd.DataFrame({"feature": feature_names, "model_importance": np.nan})
    merged = gain_df.merge(perm_df, on="feature", how="left")
    return merged.sort_values(["permutation_importance", "model_importance"], ascending=False)


def run_analysis() -> None:
    if not MATCHUP_PATH.exists():
        raise FileNotFoundError(f"{MATCHUP_PATH} not found. Run build_features first.")

    try:
        df = pd.read_parquet(MATCHUP_PATH)
    except Exception:
        df = pd.read_parquet(MATCHUP_PATH, engine="fastparquet")
    df = df[df["season"] >= TRAIN_START_YEAR].copy()

    excluded_prefixes = ("sched_", "inj_qb_")

    # --- Win probability ---
    X_full, y_full, features_full = prepare_win_prob(df, drop_prefixes=None)
    X_base, y_base, features_base = prepare_win_prob(df, drop_prefixes=excluded_prefixes)

    model_full, metrics_full, X_full_test, y_full_test = train_xgb_classifier(X_full, y_full)
    perm_full = permutation_scores(model_full, X_full_test, y_full_test, features_full)
    importances_full = compute_feature_importance(model_full, features_full, perm_full)
    importances_full.to_csv(ANALYSIS_DIR / "feature_importance_winprob_full.csv", index=False)

    model_base, metrics_base, X_base_test, y_base_test = train_xgb_classifier(X_base, y_base)
    perm_base = permutation_scores(model_base, X_base_test, y_base_test, features_base)
    importances_base = compute_feature_importance(model_base, features_base, perm_base)
    importances_base.to_csv(ANALYSIS_DIR / "feature_importance_winprob_baseline.csv", index=False)

    # --- Spread ---
    Xs_full, ys_full, spread_features_full = prepare_spread(df, drop_prefixes=None)
    Xs_base, ys_base, spread_features_base = prepare_spread(df, drop_prefixes=excluded_prefixes)

    spread_model_full, spread_metrics_full, Xs_full_test, ys_full_test = train_xgb_regressor(Xs_full, ys_full)
    spread_perm_full = permutation_scores(spread_model_full, Xs_full_test, ys_full_test, spread_features_full)
    spread_importances_full = compute_feature_importance(spread_model_full, spread_features_full, spread_perm_full)
    spread_importances_full.to_csv(ANALYSIS_DIR / "feature_importance_spread_full.csv", index=False)

    spread_model_base, spread_metrics_base, Xs_base_test, ys_base_test = train_xgb_regressor(Xs_base, ys_base)
    spread_perm_base = permutation_scores(spread_model_base, Xs_base_test, ys_base_test, spread_features_base)
    spread_importances_base = compute_feature_importance(spread_model_base, spread_features_base, spread_perm_base)
    spread_importances_base.to_csv(ANALYSIS_DIR / "feature_importance_spread_baseline.csv", index=False)

    summary = {
        "win_prob_enhanced": metrics_full,
        "win_prob_baseline": metrics_base,
        "spread_enhanced": spread_metrics_full,
        "spread_baseline": spread_metrics_base,
    }
    summary_path = ANALYSIS_DIR / "feature_lift_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=== Win Probability Metrics ===")
    print("Enhanced:", metrics_full)
    print("Baseline:", metrics_base)
    print("\n=== Spread Metrics ===")
    print("Enhanced:", spread_metrics_full)
    print("Baseline:", spread_metrics_base)
    print(f"\nArtifacts written to {ANALYSIS_DIR.resolve()}")


if __name__ == "__main__":
    run_analysis()
