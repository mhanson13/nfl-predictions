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
Closing Line Value (CLV) Tracker

Calculates and tracks Closing Line Value - the gold standard metric for measuring
true edge against the betting market. CLV measures how often the model's implied
probability beats the closing odds.

Key Concepts:
- CLV = model_prob - closing_odds_implied_prob
- Positive CLV = Model found value vs market
- Consistent positive CLV = True edge (even if win rate < 100%)
- CLV > 52.4% break-even threshold = Profitable long-term

Usage:
    # Calculate CLV for a specific week
    python -m analysis.clv_tracker --season 2025 --week 10
    
    # Calculate CLV for entire season
    python -m analysis.clv_tracker --season 2025
    
    # Generate CLV summary report
    python -m analysis.clv_tracker --season 2025 --generate-report
    
    # Track CLV over time
    python -m analysis.clv_tracker --track-history
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Paths
PREDICTIONS_LOG_DIR = Path("predictions_log")
MATCHUP_FEATURES_PATH = Path("data/processed/matchup_features.parquet")
CLV_OUTPUT_DIR = Path("analysis/clv_tracking")
CLV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def american_to_implied_prob(odds: float) -> float:
    """
    Convert American odds to implied probability.
    
    Args:
        odds: American odds (e.g., -150, +200)
    
    Returns:
        Implied probability (0-1)
    """
    if pd.isna(odds):
        return np.nan
    odds = float(odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    if odds < 0:
        return -odds / (-odds + 100.0)
    return 0.5


def calculate_clv(
    model_prob: float,
    closing_odds: float,
    bet_type: str = "moneyline"
) -> float:
    """
    Calculate Closing Line Value.
    
    Args:
        model_prob: Model's probability (0-1)
        closing_odds: Closing odds (American format)
        bet_type: Type of bet (moneyline, spread, total)
    
    Returns:
        CLV (positive = model found value)
    """
    if pd.isna(model_prob) or pd.isna(closing_odds):
        return np.nan
    
    closing_prob = american_to_implied_prob(closing_odds)
    clv = model_prob - closing_prob
    
    return clv


def load_predictions_with_odds(season: int, week: Optional[int] = None) -> pd.DataFrame:
    """
    Load predictions and merge with closing odds from matchup features.
    
    Args:
        season: NFL season
        week: Optional week number (if None, load all weeks)
    
    Returns:
        DataFrame with predictions and closing odds
    """
    # Load predictions from predictions_log
    season_dir = PREDICTIONS_LOG_DIR / str(season)
    if not season_dir.exists():
        raise FileNotFoundError(f"No predictions found for season {season}")
    
    # Find prediction files
    if week is not None:
        pred_files = [season_dir / f"week_{week:02d}_predictions.csv"]
    else:
        pred_files = sorted(season_dir.glob("week_*_predictions.csv"))
    
    if not pred_files:
        raise FileNotFoundError(f"No prediction files found for season {season}")
    
    # Load all predictions
    pred_dfs = []
    for pred_file in pred_files:
        if pred_file.exists():
            df = pd.read_csv(pred_file)
            pred_dfs.append(df)
    
    if not pred_dfs:
        raise ValueError(f"No predictions loaded for season {season}")
    
    preds = pd.concat(pred_dfs, ignore_index=True)
    
    # Load matchup features for closing odds
    if not MATCHUP_FEATURES_PATH.exists():
        raise FileNotFoundError(f"Matchup features not found: {MATCHUP_FEATURES_PATH}")
    
    try:
        matchup = pd.read_parquet(MATCHUP_FEATURES_PATH)
    except Exception:
        matchup = pd.read_parquet(MATCHUP_FEATURES_PATH, engine="fastparquet")
    
    # Filter to season
    matchup = matchup[matchup["season"] == season].copy()
    
    # Select odds columns
    odds_cols = [
        "game_id", "season", "week",
        "home_moneyline", "away_moneyline",
        "spread_line", "home_spread_odds", "away_spread_odds",
        "total_line", "over_odds", "under_odds",
        "home_margin"  # For actual results
    ]
    odds_df = matchup[[c for c in odds_cols if c in matchup.columns]].copy()
    
    # Merge predictions with odds
    merged = preds.merge(odds_df, on="game_id", how="left", suffixes=("", "_odds"))
    
    return merged


def calculate_moneyline_clv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate CLV for moneyline bets.
    
    Args:
        df: DataFrame with predictions and closing odds
    
    Returns:
        DataFrame with CLV calculations
    """
    results = []
    
    for _, row in df.iterrows():
        game_id = row["game_id"]
        
        # Home team CLV
        if "home_win_prob" in row and "home_moneyline" in row:
            home_clv = calculate_clv(
                row["home_win_prob"],
                row["home_moneyline"],
                "moneyline"
            )
            
            results.append({
                "game_id": game_id,
                "season": row.get("season"),
                "week": row.get("week"),
                "bet_type": "moneyline",
                "side": "home",
                "team": row.get("home_team"),
                "model_prob": row["home_win_prob"],
                "closing_odds": row["home_moneyline"],
                "closing_prob": american_to_implied_prob(row["home_moneyline"]),
                "clv": home_clv,
                "actual_margin": row.get("home_margin"),
                "actual_win": 1 if row.get("home_margin", 0) > 0 else 0,
            })
        
        # Away team CLV
        if "home_win_prob" in row and "away_moneyline" in row:
            away_prob = 1.0 - row["home_win_prob"]
            away_clv = calculate_clv(
                away_prob,
                row["away_moneyline"],
                "moneyline"
            )
            
            results.append({
                "game_id": game_id,
                "season": row.get("season"),
                "week": row.get("week"),
                "bet_type": "moneyline",
                "side": "away",
                "team": row.get("away_team"),
                "model_prob": away_prob,
                "closing_odds": row["away_moneyline"],
                "closing_prob": american_to_implied_prob(row["away_moneyline"]),
                "clv": away_clv,
                "actual_margin": -row.get("home_margin", 0) if pd.notna(row.get("home_margin")) else np.nan,
                "actual_win": 1 if row.get("home_margin", 0) < 0 else 0,
            })
    
    return pd.DataFrame(results)


def calculate_spread_clv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate CLV for spread bets.
    
    Args:
        df: DataFrame with predictions and closing odds
    
    Returns:
        DataFrame with CLV calculations
    """
    results = []
    
    for _, row in df.iterrows():
        game_id = row["game_id"]
        
        if "pred_home_margin" not in row or "spread_line" not in row:
            continue
        
        # Calculate cover probabilities
        # Simplified: use normal distribution assumption
        pred_margin = row["pred_home_margin"]
        spread = row["spread_line"]
        
        # Home cover probability (simplified)
        home_cover_edge = pred_margin + spread
        home_cover_prob = 0.5 + (home_cover_edge / 28.0)  # Rough approximation
        home_cover_prob = np.clip(home_cover_prob, 0.01, 0.99)
        
        # Home spread CLV
        if "home_spread_odds" in row:
            home_clv = calculate_clv(
                home_cover_prob,
                row["home_spread_odds"],
                "spread"
            )
            
            actual_cover = None
            if pd.notna(row.get("home_margin")):
                actual_result = row["home_margin"] + spread
                actual_cover = 1 if actual_result > 0 else 0
            
            results.append({
                "game_id": game_id,
                "season": row.get("season"),
                "week": row.get("week"),
                "bet_type": "spread",
                "side": "home",
                "team": row.get("home_team"),
                "spread_line": spread,
                "model_prob": home_cover_prob,
                "closing_odds": row["home_spread_odds"],
                "closing_prob": american_to_implied_prob(row["home_spread_odds"]),
                "clv": home_clv,
                "actual_cover": actual_cover,
            })
        
        # Away spread CLV
        if "away_spread_odds" in row:
            away_cover_prob = 1.0 - home_cover_prob
            away_clv = calculate_clv(
                away_cover_prob,
                row["away_spread_odds"],
                "spread"
            )
            
            actual_cover = None
            if pd.notna(row.get("home_margin")):
                actual_result = row["home_margin"] + spread
                actual_cover = 1 if actual_result < 0 else 0
            
            results.append({
                "game_id": game_id,
                "season": row.get("season"),
                "week": row.get("week"),
                "bet_type": "spread",
                "side": "away",
                "team": row.get("away_team"),
                "spread_line": -spread,
                "model_prob": away_cover_prob,
                "closing_odds": row["away_spread_odds"],
                "closing_prob": american_to_implied_prob(row["away_spread_odds"]),
                "clv": away_clv,
                "actual_cover": actual_cover,
            })
    
    return pd.DataFrame(results)


def analyze_clv(clv_df: pd.DataFrame, thresholds: List[float] = [0.0, 0.02, 0.05]) -> Dict[str, Any]:
    """
    Analyze CLV distribution and hit rates.
    
    Args:
        clv_df: DataFrame with CLV calculations
        thresholds: CLV thresholds to analyze
    
    Returns:
        Dictionary of analysis results
    """
    analysis = {
        "total_opportunities": len(clv_df),
        "mean_clv": float(clv_df["clv"].mean()),
        "median_clv": float(clv_df["clv"].median()),
        "std_clv": float(clv_df["clv"].std()),
        "positive_clv_rate": float((clv_df["clv"] > 0).mean()),
        "thresholds": {},
    }
    
    for threshold in thresholds:
        filtered = clv_df[clv_df["clv"] > threshold]
        
        if len(filtered) == 0:
            analysis["thresholds"][threshold] = {
                "n_opportunities": 0,
                "mean_clv": np.nan,
                "hit_rate": np.nan,
            }
            continue
        
        # Calculate hit rate
        if "actual_win" in filtered.columns:
            hit_rate = filtered["actual_win"].mean()
        elif "actual_cover" in filtered.columns:
            hit_rate = filtered["actual_cover"].mean()
        else:
            hit_rate = np.nan
        
        analysis["thresholds"][threshold] = {
            "n_opportunities": len(filtered),
            "mean_clv": float(filtered["clv"].mean()),
            "hit_rate": float(hit_rate) if not np.isnan(hit_rate) else None,
        }
    
    return analysis


def generate_clv_report(season: int, week: Optional[int] = None) -> Path:
    """
    Generate comprehensive CLV report.
    
    Args:
        season: NFL season
        week: Optional week number
    
    Returns:
        Path to report file
    """
    # Load data
    df = load_predictions_with_odds(season, week)
    
    # Calculate CLV for different bet types
    ml_clv = calculate_moneyline_clv(df)
    spread_clv = calculate_spread_clv(df)
    
    # Analyze
    ml_analysis = analyze_clv(ml_clv) if not ml_clv.empty else {}
    spread_analysis = analyze_clv(spread_clv) if not spread_clv.empty else {}
    
    # Save detailed CLV data
    if week is not None:
        ml_file = CLV_OUTPUT_DIR / f"{season}_week_{week:02d}_moneyline_clv.csv"
        spread_file = CLV_OUTPUT_DIR / f"{season}_week_{week:02d}_spread_clv.csv"
        report_file = CLV_OUTPUT_DIR / f"{season}_week_{week:02d}_clv_report.md"
    else:
        ml_file = CLV_OUTPUT_DIR / f"{season}_moneyline_clv.csv"
        spread_file = CLV_OUTPUT_DIR / f"{season}_spread_clv.csv"
        report_file = CLV_OUTPUT_DIR / f"{season}_clv_report.md"
    
    ml_clv.to_csv(ml_file, index=False)
    spread_clv.to_csv(spread_file, index=False)
    
    # Generate markdown report
    period = f"Week {week}" if week else "Full Season"
    report_lines = [
        f"# Closing Line Value (CLV) Report",
        f"",
        f"**Season:** {season}  ",
        f"**Period:** {period}  ",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}  ",
        f"",
        f"---",
        f"",
        f"## What is CLV?",
        f"",
        f"Closing Line Value (CLV) measures how often the model's implied probability beats the closing odds. ",
        f"It's the gold standard metric for measuring true edge against the betting market.",
        f"",
        f"- **Positive CLV** = Model found value vs market",
        f"- **Consistent positive CLV** = True edge (even if win rate < 100%)",
        f"- **CLV > 52.4% break-even** = Profitable long-term",
        f"",
        f"---",
        f"",
    ]
    
    if ml_analysis:
        report_lines.extend([
            f"## Moneyline CLV",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Opportunities | {ml_analysis['total_opportunities']} |",
            f"| Mean CLV | {ml_analysis['mean_clv']:.4f} |",
            f"| Median CLV | {ml_analysis['median_clv']:.4f} |",
            f"| Std Dev | {ml_analysis['std_clv']:.4f} |",
            f"| Positive CLV Rate | {ml_analysis['positive_clv_rate']:.1%} |",
            f"",
            f"### By CLV Threshold",
            f"",
            f"| Threshold | Opportunities | Mean CLV | Hit Rate |",
            f"|-----------|---------------|----------|----------|",
        ])
        
        for threshold, stats in ml_analysis["thresholds"].items():
            hit_rate_str = f"{stats['hit_rate']:.1%}" if stats['hit_rate'] is not None else "N/A"
            report_lines.append(
                f"| >{threshold:.2f} | {stats['n_opportunities']} | {stats['mean_clv']:.4f} | {hit_rate_str} |"
            )
        
        report_lines.append("")
    
    if spread_analysis:
        report_lines.extend([
            f"## Spread CLV",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Opportunities | {spread_analysis['total_opportunities']} |",
            f"| Mean CLV | {spread_analysis['mean_clv']:.4f} |",
            f"| Median CLV | {spread_analysis['median_clv']:.4f} |",
            f"| Std Dev | {spread_analysis['std_clv']:.4f} |",
            f"| Positive CLV Rate | {spread_analysis['positive_clv_rate']:.1%} |",
            f"",
            f"### By CLV Threshold",
            f"",
            f"| Threshold | Opportunities | Mean CLV | Hit Rate |",
            f"|-----------|---------------|----------|----------|",
        ])
        
        for threshold, stats in spread_analysis["thresholds"].items():
            hit_rate_str = f"{stats['hit_rate']:.1%}" if stats['hit_rate'] is not None else "N/A"
            report_lines.append(
                f"| >{threshold:.2f} | {stats['n_opportunities']} | {stats['mean_clv']:.4f} | {hit_rate_str} |"
            )
        
        report_lines.append("")
    
    report_lines.extend([
        f"---",
        f"",
        f"## Files",
        f"",
        f"- **Moneyline CLV:** `{ml_file.name}`",
        f"- **Spread CLV:** `{spread_file.name}`",
        f"",
    ])
    
    report_content = "\n".join(report_lines)
    
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_content)
    
    print(f"✓ Generated CLV report for {season} {period}")
    print(f"  Report: {report_file}")
    print(f"  Moneyline CLV: {ml_file}")
    print(f"  Spread CLV: {spread_file}")
    
    if ml_analysis:
        print(f"\n  Moneyline Summary:")
        print(f"    Mean CLV: {ml_analysis['mean_clv']:.4f}")
        print(f"    Positive CLV Rate: {ml_analysis['positive_clv_rate']:.1%}")
    
    if spread_analysis:
        print(f"\n  Spread Summary:")
        print(f"    Mean CLV: {spread_analysis['mean_clv']:.4f}")
        print(f"    Positive CLV Rate: {spread_analysis['positive_clv_rate']:.1%}")
    
    return report_file


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Closing Line Value (CLV) tracker"
    )
    parser.add_argument(
        "--season",
        type=int,
        required=True,
        help="NFL season year",
    )
    parser.add_argument(
        "--week",
        type=int,
        default=None,
        help="Week number (optional, if not provided analyzes full season)",
    )
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Generate CLV report",
    )
    
    args = parser.parse_args()
    
    if args.generate_report:
        generate_clv_report(args.season, args.week)
    else:
        # Default: generate report
        generate_clv_report(args.season, args.week)


if __name__ == "__main__":
    main()

# Made with Bob
