from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from xgboost import XGBClassifier, XGBRegressor

from src.models.train import (
    _make_feature_diffs,
    _require_target,
    _select_feature_columns,
)

RANDOM_STATE = 42
TRAIN_END_SEASON = 2023
TEST_START_SEASON = 2024

MATCHUP_PATH = Path("data/processed/matchup_features.parquet")
OUTPUT_DIR = Path("analysis/forward_validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_matchup() -> pd.DataFrame:
    if not MATCHUP_PATH.exists():
        raise FileNotFoundError(f"{MATCHUP_PATH} not found. Run build_features first.")
    try:
        df = pd.read_parquet(MATCHUP_PATH)
    except Exception:
        df = pd.read_parquet(MATCHUP_PATH, engine="fastparquet")
    needed_cols = {"season", "week", "game_id", "home_team", "away_team"}
    missing = needed_cols - set(df.columns)
    if missing:
        raise ValueError(f"Matchup features missing columns: {missing}")
    return df


def build_design(df: pd.DataFrame, target_col: str) -> Tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    feat_df = _make_feature_diffs(df)
    feat_cols = _select_feature_columns(feat_df)
    tgt_df = _require_target(feat_df, target_col)

    # capture metadata
    meta_cols = ["season", "week", "game_id", "home_team", "away_team"]
    meta = tgt_df.loc[:, [c for c in meta_cols if c in tgt_df.columns]].copy()

    tgt_df = tgt_df.loc[:, ~pd.Index(tgt_df.columns).duplicated()]
    use_cols = [c for c in feat_cols if c in tgt_df.columns]
    X = tgt_df[use_cols].astype(float).values
    if target_col == "home_win":
        y = tgt_df[target_col].astype(int).values
    else:
        y = tgt_df[target_col].astype(float).values
    return X, y, use_cols, meta


def split_indices(meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    train_mask = meta["season"] <= TRAIN_END_SEASON
    test_mask = meta["season"] >= TEST_START_SEASON
    return np.where(train_mask)[0], np.where(test_mask)[0]


def forward_validate_win_prob() -> dict[str, float]:
    df = load_matchup()
    X, y, features, meta = build_design(df, "home_win")
    train_idx, test_idx = split_indices(meta)
    if len(test_idx) == 0:
        raise ValueError("No test samples found for win probability forward validation.")

    model = XGBClassifier(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    model.fit(X[train_idx], y[train_idx])

    proba = model.predict_proba(X[test_idx])[:, 1]
    pred = (proba >= 0.5).astype(int)
    y_true = y[test_idx]

    metrics = {
        "samples": int(len(test_idx)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "auc": float(roc_auc_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
        "logloss": float(log_loss(y_true, proba)),
    }

    out = meta.loc[test_idx, ["season", "week", "game_id", "home_team", "away_team"]].copy()
    out["actual_home_win"] = y_true
    out["pred_home_win_prob"] = proba
    out["pred_home_win"] = pred
    out.to_csv(OUTPUT_DIR / "winprob_forward_predictions.csv", index=False)

    return metrics


def forward_validate_spread() -> dict[str, float]:
    df = load_matchup()
    X, y, features, meta = build_design(df, "home_margin")
    train_idx, test_idx = split_indices(meta)
    if len(test_idx) == 0:
        raise ValueError("No test samples found for spread forward validation.")

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
    model.fit(X[train_idx], y[train_idx])
    preds = model.predict(X[test_idx])
    y_true = y[test_idx]

    metrics = {
        "samples": int(len(test_idx)),
        "mae": float(mean_absolute_error(y_true, preds)),
        "rmse": float(mean_squared_error(y_true, preds) ** 0.5),
    }

    out = meta.loc[test_idx, ["season", "week", "game_id", "home_team", "away_team"]].copy()
    out["actual_home_margin"] = y_true
    out["pred_home_margin"] = preds
    out.to_csv(OUTPUT_DIR / "spread_forward_predictions.csv", index=False)

    return metrics


def main() -> None:
    win_metrics = forward_validate_win_prob()
    spread_metrics = forward_validate_spread()

    summary = {
        "train_end_season": TRAIN_END_SEASON,
        "test_start_season": TEST_START_SEASON,
        "win_prob": win_metrics,
        "spread": spread_metrics,
    }
    (OUTPUT_DIR / "forward_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=== Forward Validation Metrics (Win Probability) ===")
    print(win_metrics)
    print("\n=== Forward Validation Metrics (Spread) ===")
    print(spread_metrics)
    print(f"\nArtifacts written to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
