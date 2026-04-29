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

"""
Walk-Forward Validation Framework

Implements rolling window validation to test model performance on truly unseen data.
Instead of a single train/test split, this trains on N years and tests on the next season,
then rolls forward through multiple test periods to measure consistency and detect overfitting.

Example:
    Train 2016-2020 → Test 2021
    Train 2017-2021 → Test 2022
    Train 2018-2022 → Test 2023
    Train 2019-2023 → Test 2024
    Train 2020-2024 → Test 2025

This provides multiple independent test periods to assess variance and stability.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

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

from src.models.feature_columns import (
    make_feature_diffs,
    select_feature_columns,
)

RANDOM_STATE = 42
MATCHUP_PATH = Path("data/processed/matchup_features.parquet")
OUTPUT_DIR = Path("analysis/walk_forward_validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ValidationWindow:
    """Represents a single train/test split in the walk-forward process."""
    train_start: int
    train_end: int
    test_season: int
    
    @property
    def name(self) -> str:
        return f"train_{self.train_start}_{self.train_end}_test_{self.test_season}"
    
    @property
    def train_years(self) -> int:
        return self.train_end - self.train_start + 1


@dataclass
class WindowMetrics:
    """Metrics for a single validation window."""
    window: ValidationWindow
    n_train: int
    n_test: int
    
    # Win probability metrics
    accuracy: float
    auc: float
    brier: float
    logloss: float
    
    # Spread metrics
    mae: float
    rmse: float


def load_matchup() -> pd.DataFrame:
    """Load matchup features with error handling for PyArrow issues."""
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


def generate_windows(
    start_season: int,
    end_season: int,
    train_window_years: int,
    min_test_season: int | None = None,
) -> List[ValidationWindow]:
    """
    Generate rolling validation windows.
    
    Args:
        start_season: First season available for training
        end_season: Last season available (may be used for testing)
        train_window_years: Number of years to include in each training window
        min_test_season: Minimum season to use for testing (default: start_season + train_window_years)
    
    Returns:
        List of ValidationWindow objects
    """
    if min_test_season is None:
        min_test_season = start_season + train_window_years
    
    windows = []
    test_season = min_test_season
    
    while test_season <= end_season:
        train_end = test_season - 1
        train_start = train_end - train_window_years + 1
        
        if train_start >= start_season:
            windows.append(ValidationWindow(
                train_start=train_start,
                train_end=train_end,
                test_season=test_season,
            ))
        
        test_season += 1
    
    return windows


def build_design(
    df: pd.DataFrame,
    target_col: str,
    train_seasons: List[int],
    test_seasons: List[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    """
    Build feature matrix and split into train/test based on seasons.
    
    Returns:
        X_train, y_train, X_test, y_test, feature_names, test_metadata
    """
    feat_df = make_feature_diffs(df)
    feat_cols = select_feature_columns(feat_df)
    
    # Require target column
    if target_col not in feat_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in features")
    tgt_df = feat_df[feat_df[target_col].notna()].copy()
    
    # Capture metadata for test set
    meta_cols = ["season", "week", "game_id", "home_team", "away_team"]
    meta = tgt_df.loc[:, [c for c in meta_cols if c in tgt_df.columns]].copy()
    
    # Remove duplicate columns
    tgt_df = tgt_df.loc[:, ~pd.Index(tgt_df.columns).duplicated()]
    use_cols = [c for c in feat_cols if c in tgt_df.columns]
    
    # Split by season
    train_mask = tgt_df["season"].isin(train_seasons)
    test_mask = tgt_df["season"].isin(test_seasons)
    
    X_train = tgt_df.loc[train_mask, use_cols].astype(float).values
    X_test = tgt_df.loc[test_mask, use_cols].astype(float).values
    
    if target_col == "home_win":
        y_train = tgt_df.loc[train_mask, target_col].astype(int).values
        y_test = tgt_df.loc[test_mask, target_col].astype(int).values
    else:
        y_train = tgt_df.loc[train_mask, target_col].astype(float).values
        y_test = tgt_df.loc[test_mask, target_col].astype(float).values
    
    test_meta = meta.loc[test_mask].copy()
    
    return X_train, y_train, X_test, y_test, use_cols, test_meta


def validate_winprob_window(
    df: pd.DataFrame,
    window: ValidationWindow,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Train and evaluate win probability model for a single window.
    
    Returns:
        Dictionary of metrics
    """
    train_seasons = list(range(window.train_start, window.train_end + 1))
    test_seasons = [window.test_season]
    
    X_train, y_train, X_test, y_test, features, test_meta = build_design(
        df, "home_win", train_seasons, test_seasons
    )
    
    if len(X_test) == 0:
        raise ValueError(f"No test samples found for {window.name}")
    
    if verbose:
        print(f"  Training on {len(X_train)} games ({window.train_start}-{window.train_end})")
        print(f"  Testing on {len(X_test)} games ({window.test_season})")
    
    # Train model
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
    model.fit(X_train, y_train)
    
    # Predict
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    
    # Calculate metrics
    metrics = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "auc": float(roc_auc_score(y_test, proba)),
        "brier": float(brier_score_loss(y_test, proba)),
        "logloss": float(log_loss(y_test, proba)),
    }
    
    # Save predictions
    out = test_meta.copy()
    out["actual_home_win"] = y_test
    out["pred_home_win_prob"] = proba
    out["pred_home_win"] = pred
    out.to_csv(OUTPUT_DIR / f"winprob_{window.name}.csv", index=False)
    
    return metrics


def validate_spread_window(
    df: pd.DataFrame,
    window: ValidationWindow,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Train and evaluate spread model for a single window.
    
    Returns:
        Dictionary of metrics
    """
    train_seasons = list(range(window.train_start, window.train_end + 1))
    test_seasons = [window.test_season]
    
    X_train, y_train, X_test, y_test, features, test_meta = build_design(
        df, "home_margin", train_seasons, test_seasons
    )
    
    if len(X_test) == 0:
        raise ValueError(f"No test samples found for {window.name}")
    
    if verbose:
        print(f"  Training on {len(X_train)} games ({window.train_start}-{window.train_end})")
        print(f"  Testing on {len(X_test)} games ({window.test_season})")
    
    # Train model
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
    
    # Predict
    preds = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(mean_squared_error(y_test, preds) ** 0.5),
    }
    
    # Save predictions
    out = test_meta.copy()
    out["actual_home_margin"] = y_test
    out["pred_home_margin"] = preds
    out.to_csv(OUTPUT_DIR / f"spread_{window.name}.csv", index=False)
    
    return metrics


def aggregate_metrics(
    window_metrics: List[Dict[str, Any]],
    metric_names: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metrics across windows.
    
    Returns:
        Dictionary with mean, std, min, max for each metric
    """
    aggregated = {}
    
    for metric in metric_names:
        values = [m[metric] for m in window_metrics if metric in m]
        if values:
            aggregated[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "n_windows": len(values),
            }
    
    return aggregated


def run_walk_forward_validation(
    start_season: int = 2016,
    end_season: int = 2025,
    train_window_years: int = 5,
    min_test_season: int | None = None,
    verbose: bool = True,
) -> None:
    """
    Run complete walk-forward validation.
    
    Args:
        start_season: First season available for training
        end_season: Last season available
        train_window_years: Number of years in each training window
        min_test_season: Minimum season to test (default: start_season + train_window_years)
        verbose: Print progress
    """
    if verbose:
        print("=" * 80)
        print("Walk-Forward Validation Framework")
        print("=" * 80)
    
    # Load data
    df = load_matchup()
    if verbose:
        print(f"\nLoaded {len(df)} games from {df['season'].min()} to {df['season'].max()}")
    
    # Generate windows
    windows = generate_windows(start_season, end_season, train_window_years, min_test_season)
    if verbose:
        print(f"\nGenerated {len(windows)} validation windows:")
        for w in windows:
            print(f"  - Train {w.train_start}-{w.train_end} ({w.train_years} years) → Test {w.test_season}")
    
    # Validate win probability
    if verbose:
        print("\n" + "=" * 80)
        print("Win Probability Validation")
        print("=" * 80)
    
    winprob_results: List[Dict[str, Any]] = []
    for i, window in enumerate(windows, 1):
        if verbose:
            print(f"\n[{i}/{len(windows)}] {window.name}")
        
        metrics = validate_winprob_window(df, window, verbose=verbose)
        result: Dict[str, Any] = dict(metrics)
        result["window"] = window.name
        result["train_start"] = window.train_start
        result["train_end"] = window.train_end
        result["test_season"] = window.test_season
        winprob_results.append(result)
        
        if verbose:
            print(f"  Accuracy: {metrics['accuracy']:.3f}")
            print(f"  AUC: {metrics['auc']:.3f}")
            print(f"  Brier: {metrics['brier']:.3f}")
            print(f"  LogLoss: {metrics['logloss']:.3f}")
    
    # Validate spread
    if verbose:
        print("\n" + "=" * 80)
        print("Spread Validation")
        print("=" * 80)
    
    spread_results: List[Dict[str, Any]] = []
    for i, window in enumerate(windows, 1):
        if verbose:
            print(f"\n[{i}/{len(windows)}] {window.name}")
        
        metrics = validate_spread_window(df, window, verbose=verbose)
        result: Dict[str, Any] = dict(metrics)
        result["window"] = window.name
        result["train_start"] = window.train_start
        result["train_end"] = window.train_end
        result["test_season"] = window.test_season
        spread_results.append(result)
        
        if verbose:
            print(f"  MAE: {metrics['mae']:.2f}")
            print(f"  RMSE: {metrics['rmse']:.2f}")
    
    # Save detailed results
    winprob_df = pd.DataFrame(winprob_results)
    spread_df = pd.DataFrame(spread_results)
    
    winprob_df.to_csv(OUTPUT_DIR / "winprob_by_window.csv", index=False)
    spread_df.to_csv(OUTPUT_DIR / "spread_by_window.csv", index=False)
    
    # Aggregate metrics
    winprob_agg = aggregate_metrics(
        winprob_results,
        ["accuracy", "auc", "brier", "logloss", "n_train", "n_test"]
    )
    spread_agg = aggregate_metrics(
        spread_results,
        ["mae", "rmse", "n_train", "n_test"]
    )
    
    # Create summary
    summary = {
        "config": {
            "start_season": start_season,
            "end_season": end_season,
            "train_window_years": train_window_years,
            "n_windows": len(windows),
        },
        "win_probability": {
            "aggregated": winprob_agg,
            "by_window": winprob_results,
        },
        "spread": {
            "aggregated": spread_agg,
            "by_window": spread_results,
        },
    }
    
    # Save summary
    summary_path = OUTPUT_DIR / "walk_forward_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    if verbose:
        print("\n" + "=" * 80)
        print("Summary Statistics")
        print("=" * 80)
        
        print("\nWin Probability (across all test windows):")
        for metric in ["accuracy", "auc", "brier", "logloss"]:
            if metric in winprob_agg:
                stats = winprob_agg[metric]
                print(f"  {metric.upper():10s}: {stats['mean']:.3f} ± {stats['std']:.3f} "
                      f"(range: {stats['min']:.3f} - {stats['max']:.3f})")
        
        print("\nSpread (across all test windows):")
        for metric in ["mae", "rmse"]:
            if metric in spread_agg:
                stats = spread_agg[metric]
                print(f"  {metric.upper():10s}: {stats['mean']:.2f} ± {stats['std']:.2f} "
                      f"(range: {stats['min']:.2f} - {stats['max']:.2f})")
        
        print(f"\nResults saved to {OUTPUT_DIR.resolve()}")
        print(f"  - winprob_by_window.csv: Per-window win probability metrics")
        print(f"  - spread_by_window.csv: Per-window spread metrics")
        print(f"  - walk_forward_summary.json: Complete summary with aggregated stats")
        print(f"  - winprob_train_*_test_*.csv: Individual window predictions")
        print(f"  - spread_train_*_test_*.csv: Individual window predictions")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run walk-forward validation with rolling windows"
    )
    parser.add_argument(
        "--start-season",
        type=int,
        default=2016,
        help="First season available for training (default: 2016)",
    )
    parser.add_argument(
        "--end-season",
        type=int,
        default=2025,
        help="Last season available (default: 2025)",
    )
    parser.add_argument(
        "--train-window-years",
        type=int,
        default=5,
        help="Number of years in each training window (default: 5)",
    )
    parser.add_argument(
        "--min-test-season",
        type=int,
        default=None,
        help="Minimum season to use for testing (default: start_season + train_window_years)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    
    args = parser.parse_args()
    
    run_walk_forward_validation(
        start_season=args.start_season,
        end_season=args.end_season,
        train_window_years=args.train_window_years,
        min_test_season=args.min_test_season,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()

# Made with Bob
