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
Live Prediction Tracking System

Logs predictions before games with timestamps, fetches actual results after games complete,
and tracks performance metrics over time. Provides full audit trail for validation.

Key Features:
- Lock predictions before kickoff (no retroactive changes)
- Fetch actuals from matchup_features.parquet after games
- Calculate weekly accuracy, Brier score, calibration
- Generate weekly validation reports
- Track prediction versions across model updates

Usage:
    # Lock predictions before games (Thursday)
    python -m analysis.live_tracking --lock-predictions --season 2025 --week 10
    
    # Fetch actuals after games (Tuesday)
    python -m analysis.live_tracking --fetch-actuals --season 2025 --week 10
    
    # Generate weekly report
    python -m analysis.live_tracking --generate-report --season 2025 --week 10
    
    # Full weekly cycle
    python -m analysis.live_tracking --weekly-cycle --season 2025 --week 10
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

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

# Paths
PREDICTIONS_DIR = Path("predictions")
PREDICTIONS_LOG_DIR = Path("predictions_log")
MATCHUP_FEATURES_PATH = Path("data/processed/matchup_features.parquet")

# Create log directory structure
PREDICTIONS_LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_season_dir(season: int) -> Path:
    """Get or create season directory in predictions_log."""
    season_dir = PREDICTIONS_LOG_DIR / str(season)
    season_dir.mkdir(parents=True, exist_ok=True)
    return season_dir


def get_prediction_filename(season: int, week: int) -> str:
    """Generate standardized prediction filename."""
    return f"week_{week:02d}_predictions.csv"


def get_actuals_filename(season: int, week: int) -> str:
    """Generate standardized actuals filename."""
    return f"week_{week:02d}_actuals.csv"


def get_metrics_filename(season: int, week: int) -> str:
    """Generate standardized metrics filename."""
    return f"week_{week:02d}_metrics.json"


def lock_predictions(season: int, week: int, source_file: Optional[Path] = None) -> Path:
    """
    Lock predictions for a specific week before games start.
    
    Args:
        season: NFL season year
        week: Week number
        source_file: Path to predictions file (default: predictions/predictions.csv)
    
    Returns:
        Path to locked predictions file
    """
    if source_file is None:
        source_file = PREDICTIONS_DIR / "predictions.csv"
    
    if not source_file.exists():
        raise FileNotFoundError(f"Source predictions file not found: {source_file}")
    
    # Load predictions
    df = pd.read_csv(source_file)
    
    # Filter to specific season/week if columns exist
    if "season" in df.columns and "week" in df.columns:
        df = df[(df["season"] == season) & (df["week"] == week)].copy()
    
    if df.empty:
        raise ValueError(f"No predictions found for season {season}, week {week}")
    
    # Add metadata
    df["locked_at"] = datetime.now(timezone.utc).isoformat()
    df["lock_season"] = season
    df["lock_week"] = week
    
    # Save to predictions_log
    season_dir = get_season_dir(season)
    output_file = season_dir / get_prediction_filename(season, week)
    
    # Check if already locked
    if output_file.exists():
        print(f"⚠️  Predictions already locked for {season} Week {week}")
        print(f"   Existing file: {output_file}")
        response = input("   Overwrite? (yes/no): ").strip().lower()
        if response != "yes":
            print("   Aborted. Keeping existing locked predictions.")
            return output_file
    
    df.to_csv(output_file, index=False)
    print(f"✓ Locked {len(df)} predictions for {season} Week {week}")
    print(f"  File: {output_file}")
    print(f"  Timestamp: {df['locked_at'].iloc[0]}")
    
    return output_file


def fetch_actuals(season: int, week: int) -> Path:
    """
    Fetch actual game results from matchup_features.parquet.
    
    Args:
        season: NFL season year
        week: Week number
    
    Returns:
        Path to actuals file
    """
    if not MATCHUP_FEATURES_PATH.exists():
        raise FileNotFoundError(f"Matchup features not found: {MATCHUP_FEATURES_PATH}")
    
    # Load matchup features
    try:
        df = pd.read_parquet(MATCHUP_FEATURES_PATH)
    except Exception:
        df = pd.read_parquet(MATCHUP_FEATURES_PATH, engine="fastparquet")
    
    # Filter to specific season/week
    actuals = df[(df["season"] == season) & (df["week"] == week)].copy()
    
    if actuals.empty:
        raise ValueError(f"No actual results found for season {season}, week {week}")
    
    # Select relevant columns
    cols = [
        "season", "week", "game_id", "home_team", "away_team",
        "home_score", "away_score", "home_margin"
    ]
    actuals = actuals[[c for c in cols if c in actuals.columns]].copy()
    
    # Add metadata
    actuals["fetched_at"] = datetime.now(timezone.utc).isoformat()
    
    # Calculate actual home win
    if "home_margin" in actuals.columns:
        actuals["actual_home_win"] = (actuals["home_margin"] > 0).astype(int)
    
    # Save to predictions_log
    season_dir = get_season_dir(season)
    output_file = season_dir / get_actuals_filename(season, week)
    
    actuals.to_csv(output_file, index=False)
    print(f"✓ Fetched {len(actuals)} actual results for {season} Week {week}")
    print(f"  File: {output_file}")
    print(f"  Timestamp: {actuals['fetched_at'].iloc[0]}")
    
    return output_file


def calculate_metrics(season: int, week: int) -> Dict[str, Any]:
    """
    Calculate performance metrics by comparing predictions to actuals.
    
    Args:
        season: NFL season year
        week: Week number
    
    Returns:
        Dictionary of metrics
    """
    season_dir = get_season_dir(season)
    pred_file = season_dir / get_prediction_filename(season, week)
    actuals_file = season_dir / get_actuals_filename(season, week)
    
    if not pred_file.exists():
        raise FileNotFoundError(f"Locked predictions not found: {pred_file}")
    if not actuals_file.exists():
        raise FileNotFoundError(f"Actuals not found: {actuals_file}")
    
    # Load data
    preds = pd.read_csv(pred_file)
    actuals = pd.read_csv(actuals_file)
    
    # Merge on game_id
    merged = preds.merge(
        actuals,
        on=["game_id"],
        how="inner",
        suffixes=("_pred", "_actual")
    )
    
    if merged.empty:
        raise ValueError(f"No matching games found between predictions and actuals")
    
    # Calculate metrics
    metrics: Dict[str, Any] = {
        "season": season,
        "week": week,
        "n_games": len(merged),
        "calculated_at": datetime.now(timezone.utc).isoformat(),
    }
    
    # Win probability metrics
    if "home_win_prob" in merged.columns and "actual_home_win" in merged.columns:
        y_true = merged["actual_home_win"].values
        y_prob = merged["home_win_prob"].clip(0.0, 1.0).values
        y_pred = (y_prob >= 0.5).astype(int)
        
        metrics["win_probability"] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else None,
            "brier": float(brier_score_loss(y_true, y_prob)),
            "logloss": float(log_loss(y_true, y_prob)),
        }
    
    # Spread metrics
    if "pred_home_margin" in merged.columns and "home_margin" in merged.columns:
        y_true_margin = merged["home_margin"].values
        y_pred_margin = merged["pred_home_margin"].values
        
        metrics["spread"] = {
            "mae": float(mean_absolute_error(y_true_margin, y_pred_margin)),
            "rmse": float(np.sqrt(mean_squared_error(y_true_margin, y_pred_margin))),
        }
    
    # Save metrics
    metrics_file = season_dir / get_metrics_filename(season, week)
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✓ Calculated metrics for {season} Week {week}")
    print(f"  File: {metrics_file}")
    
    if "win_probability" in metrics:
        wp = metrics["win_probability"]
        print(f"  Win Probability:")
        print(f"    Accuracy: {wp['accuracy']:.3f}")
        if wp['auc'] is not None:
            print(f"    AUC: {wp['auc']:.3f}")
        print(f"    Brier: {wp['brier']:.3f}")
        print(f"    LogLoss: {wp['logloss']:.3f}")
    
    if "spread" in metrics:
        sp = metrics["spread"]
        print(f"  Spread:")
        print(f"    MAE: {sp['mae']:.2f}")
        print(f"    RMSE: {sp['rmse']:.2f}")
    
    return metrics


def generate_report(season: int, week: int) -> Path:
    """
    Generate a comprehensive weekly validation report.
    
    Args:
        season: NFL season year
        week: Week number
    
    Returns:
        Path to report file
    """
    season_dir = get_season_dir(season)
    metrics_file = season_dir / get_metrics_filename(season, week)
    
    if not metrics_file.exists():
        raise FileNotFoundError(f"Metrics not found: {metrics_file}. Run calculate_metrics first.")
    
    with open(metrics_file, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    
    # Generate markdown report
    report_lines = [
        f"# Weekly Validation Report",
        f"",
        f"**Season:** {season}  ",
        f"**Week:** {week}  ",
        f"**Games:** {metrics['n_games']}  ",
        f"**Generated:** {metrics['calculated_at']}  ",
        f"",
        f"---",
        f"",
    ]
    
    if "win_probability" in metrics:
        wp = metrics["win_probability"]
        report_lines.extend([
            f"## Win Probability Performance",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Accuracy | {wp['accuracy']:.3f} |",
        ])
        if wp['auc'] is not None:
            report_lines.append(f"| AUC | {wp['auc']:.3f} |")
        report_lines.extend([
            f"| Brier Score | {wp['brier']:.3f} |",
            f"| Log Loss | {wp['logloss']:.3f} |",
            f"",
        ])
    
    if "spread" in metrics:
        sp = metrics["spread"]
        report_lines.extend([
            f"## Spread Performance",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| MAE (points) | {sp['mae']:.2f} |",
            f"| RMSE (points) | {sp['rmse']:.2f} |",
            f"",
        ])
    
    report_lines.extend([
        f"---",
        f"",
        f"## Files",
        f"",
        f"- **Predictions:** `{get_prediction_filename(season, week)}`",
        f"- **Actuals:** `{get_actuals_filename(season, week)}`",
        f"- **Metrics:** `{get_metrics_filename(season, week)}`",
        f"",
    ])
    
    report_content = "\n".join(report_lines)
    report_file = season_dir / f"week_{week:02d}_report.md"
    
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_content)
    
    print(f"✓ Generated report for {season} Week {week}")
    print(f"  File: {report_file}")
    
    return report_file


def weekly_cycle(season: int, week: int, lock_source: Optional[Path] = None) -> None:
    """
    Run complete weekly validation cycle: lock → fetch → calculate → report.
    
    Args:
        season: NFL season year
        week: Week number
        lock_source: Path to predictions file for locking
    """
    print("=" * 80)
    print(f"Weekly Validation Cycle: {season} Week {week}")
    print("=" * 80)
    
    try:
        # Step 1: Lock predictions
        print("\n[1/4] Locking predictions...")
        lock_predictions(season, week, lock_source)
        
        # Step 2: Fetch actuals
        print("\n[2/4] Fetching actuals...")
        fetch_actuals(season, week)
        
        # Step 3: Calculate metrics
        print("\n[3/4] Calculating metrics...")
        calculate_metrics(season, week)
        
        # Step 4: Generate report
        print("\n[4/4] Generating report...")
        generate_report(season, week)
        
        print("\n" + "=" * 80)
        print("✓ Weekly validation cycle complete!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n✗ Error during weekly cycle: {e}")
        raise


def list_tracked_weeks(season: Optional[int] = None) -> pd.DataFrame:
    """
    List all tracked weeks with their status.
    
    Args:
        season: Optional season filter
    
    Returns:
        DataFrame with tracking status
    """
    records = []
    
    if season is not None:
        season_dirs = [get_season_dir(season)]
    else:
        season_dirs = [d for d in PREDICTIONS_LOG_DIR.iterdir() if d.is_dir() and d.name.isdigit()]
    
    for season_dir in sorted(season_dirs):
        season_num = int(season_dir.name)
        
        # Find all prediction files
        for pred_file in sorted(season_dir.glob("week_*_predictions.csv")):
            week_str = pred_file.stem.split("_")[1]
            week_num = int(week_str)
            
            actuals_file = season_dir / get_actuals_filename(season_num, week_num)
            metrics_file = season_dir / get_metrics_filename(season_num, week_num)
            report_file = season_dir / f"week_{week_num:02d}_report.md"
            
            records.append({
                "season": season_num,
                "week": week_num,
                "predictions_locked": pred_file.exists(),
                "actuals_fetched": actuals_file.exists(),
                "metrics_calculated": metrics_file.exists(),
                "report_generated": report_file.exists(),
            })
    
    if not records:
        return pd.DataFrame(columns=["season", "week", "predictions_locked", "actuals_fetched", "metrics_calculated", "report_generated"])
    
    return pd.DataFrame(records)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Live prediction tracking system"
    )
    parser.add_argument(
        "--season",
        type=int,
        required=False,
        help="NFL season year",
    )
    parser.add_argument(
        "--week",
        type=int,
        required=False,
        help="Week number",
    )
    parser.add_argument(
        "--lock-predictions",
        action="store_true",
        help="Lock predictions before games",
    )
    parser.add_argument(
        "--fetch-actuals",
        action="store_true",
        help="Fetch actual results after games",
    )
    parser.add_argument(
        "--calculate-metrics",
        action="store_true",
        help="Calculate performance metrics",
    )
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Generate weekly validation report",
    )
    parser.add_argument(
        "--weekly-cycle",
        action="store_true",
        help="Run complete weekly cycle (lock → fetch → calculate → report)",
    )
    parser.add_argument(
        "--list-tracked",
        action="store_true",
        help="List all tracked weeks",
    )
    parser.add_argument(
        "--source-file",
        type=Path,
        default=None,
        help="Source predictions file (default: predictions/predictions.csv)",
    )
    
    args = parser.parse_args()
    
    # List tracked weeks
    if args.list_tracked:
        df = list_tracked_weeks(args.season)
        if df.empty:
            print("No tracked weeks found.")
        else:
            print("\nTracked Weeks:")
            print(df.to_string(index=False))
        return
    
    # Require season and week for other operations
    if args.season is None or args.week is None:
        parser.error("--season and --week are required (except for --list-tracked)")
    
    # Run requested operations
    if args.weekly_cycle:
        weekly_cycle(args.season, args.week, args.source_file)
    else:
        if args.lock_predictions:
            lock_predictions(args.season, args.week, args.source_file)
        if args.fetch_actuals:
            fetch_actuals(args.season, args.week)
        if args.calculate_metrics:
            calculate_metrics(args.season, args.week)
        if args.generate_report:
            generate_report(args.season, args.week)


if __name__ == "__main__":
    main()

# Made with Bob
