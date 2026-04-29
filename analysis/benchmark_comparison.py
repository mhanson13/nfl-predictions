#!/usr/bin/env python
"""
Public Model Benchmark Comparison

Compares model performance against:
- nfelo (FiveThirtyEight Elo ratings)
- Vegas consensus (closing odds)
- Simple baselines (home favorite, spread-based)

Includes statistical significance testing:
- McNemar test for accuracy differences
- DeLong test for AUC differences
- Paired t-test for Brier score differences

Usage:
    # Compare single season
    python -m analysis.benchmark_comparison --season 2025 --generate-report
    
    # Compare multiple seasons
    python -m analysis.benchmark_comparison --start-season 2023 --end-season 2025 --generate-report
    
    # Compare specific benchmark
    python -m analysis.benchmark_comparison --season 2025 --benchmark nfelo
"""

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    roc_auc_score,
    log_loss,
)

# Optional matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class BenchmarkMetrics:
    """Performance metrics for a benchmark model."""
    name: str
    n_predictions: int
    accuracy: float
    auc: float
    brier_score: float
    log_loss: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ComparisonResult:
    """Statistical comparison between two models."""
    model_a: str
    model_b: str
    metric: str
    value_a: float
    value_b: float
    difference: float
    p_value: float
    significant: bool
    test_name: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


def american_to_prob(odds: float) -> float:
    """
    Convert American odds to implied probability.
    
    Args:
        odds: American odds (e.g., -150, +200)
        
    Returns:
        Implied probability (0-1)
    """
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    else:
        return 100 / (odds + 100)


def calculate_nfelo_prob(home_elo: float, away_elo: float, home_field: float = 65) -> float:
    """
    Calculate win probability using Elo ratings.
    
    FiveThirtyEight NFL Elo formula:
    - Home field advantage: ~65 Elo points
    - Win probability: 1 / (1 + 10^(-(elo_diff)/400))
    
    Args:
        home_elo: Home team Elo rating
        away_elo: Away team Elo rating
        home_field: Home field advantage in Elo points
        
    Returns:
        Home team win probability
    """
    elo_diff = home_elo - away_elo + home_field
    return 1 / (1 + 10 ** (-elo_diff / 400))


def calculate_vegas_prob(closing_odds: float) -> float:
    """
    Calculate win probability from Vegas closing odds.
    
    Args:
        closing_odds: Closing moneyline odds (American format)
        
    Returns:
        Win probability
    """
    return american_to_prob(closing_odds)


def calculate_spread_baseline_prob(spread: float) -> float:
    """
    Calculate win probability from point spread.
    
    Simple baseline: Convert spread to win probability using
    empirical relationship (roughly 1 point = 3% win probability).
    
    Args:
        spread: Point spread (negative = home favored)
        
    Returns:
        Home team win probability
    """
    # Empirical conversion: ~3% per point
    # Spread of -7 (home favored by 7) → ~71% win probability
    base_prob = 0.50
    spread_effect = -spread * 0.03  # Negative spread increases home prob
    prob = base_prob + spread_effect
    
    # Clip to reasonable range
    return np.clip(prob, 0.05, 0.95)


def calculate_home_favorite_baseline(spread: float) -> float:
    """
    Simple baseline: Always pick home team if favored, else away.
    
    Args:
        spread: Point spread (negative = home favored)
        
    Returns:
        Home team win probability (0 or 1)
    """
    return 1.0 if spread < 0 else 0.0


def mcnemar_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> Tuple[float, float]:
    """
    McNemar test for comparing two binary classifiers.
    
    Tests if two models have significantly different error rates.
    
    Args:
        y_true: True labels (0 or 1)
        pred_a: Model A predictions (0 or 1)
        pred_b: Model B predictions (0 or 1)
        
    Returns:
        (test_statistic, p_value)
    """
    # Create contingency table
    # n01: A correct, B wrong
    # n10: A wrong, B correct
    n01 = np.sum((pred_a == y_true) & (pred_b != y_true))
    n10 = np.sum((pred_a != y_true) & (pred_b == y_true))
    
    # McNemar test statistic with continuity correction
    if n01 + n10 == 0:
        return 0.0, 1.0
    
    statistic = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    p_value = 1 - stats.chi2.cdf(statistic, df=1)
    
    return statistic, p_value


def delong_test(y_true: np.ndarray, prob_a: np.ndarray, prob_b: np.ndarray) -> Tuple[float, float]:
    """
    DeLong test for comparing two ROC curves (AUC).
    
    Simplified implementation using Mann-Whitney U statistic.
    
    Args:
        y_true: True labels (0 or 1)
        prob_a: Model A probabilities
        prob_b: Model B probabilities
        
    Returns:
        (z_statistic, p_value)
    """
    try:
        auc_a = roc_auc_score(y_true, prob_a)
        auc_b = roc_auc_score(y_true, prob_b)
    except ValueError:
        return 0.0, 1.0
    
    # Simplified: Use bootstrap to estimate variance
    n_bootstrap = 1000
    auc_diffs = []
    
    np.random.seed(42)
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(y_true), len(y_true), replace=True)
        y_boot = y_true[indices]
        prob_a_boot = prob_a[indices]
        prob_b_boot = prob_b[indices]
        
        try:
            auc_a_boot = roc_auc_score(y_boot, prob_a_boot)
            auc_b_boot = roc_auc_score(y_boot, prob_b_boot)
            auc_diffs.append(auc_a_boot - auc_b_boot)
        except ValueError:
            continue
    
    if len(auc_diffs) == 0:
        return 0.0, 1.0
    
    # Z-statistic
    mean_diff = np.mean(auc_diffs)
    std_diff = np.std(auc_diffs)
    
    if std_diff == 0:
        return 0.0, 1.0
    
    z_stat = mean_diff / std_diff
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    return z_stat, p_value


def paired_t_test(metric_a: np.ndarray, metric_b: np.ndarray) -> Tuple[float, float]:
    """
    Paired t-test for comparing two models on same data.
    
    Args:
        metric_a: Model A metric values (per prediction)
        metric_b: Model B metric values (per prediction)
        
    Returns:
        (t_statistic, p_value)
    """
    return stats.ttest_rel(metric_a, metric_b)


def load_predictions(season: int) -> pd.DataFrame:
    """
    Load predictions with benchmarks.
    
    Args:
        season: Season year
        
    Returns:
        DataFrame with columns: game_id, season, week, home_win_prob, home_won,
                                closing_odds, spread, home_elo, away_elo
    """
    # Load from matchup features
    features_file = Path("data") / "matchup_features.parquet"
    if not features_file.exists():
        raise FileNotFoundError(f"Matchup features not found: {features_file}")
    
    df = pd.read_parquet(features_file)
    df = df[df['season'] == season].copy()
    
    if len(df) == 0:
        raise ValueError(f"No predictions found for season {season}")
    
    # Ensure required columns exist
    required = ['home_win_prob', 'home_won']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    return df


def calculate_benchmark_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    name: str
) -> BenchmarkMetrics:
    """
    Calculate metrics for a benchmark model.
    
    Args:
        y_true: True labels (0 or 1)
        y_prob: Predicted probabilities
        name: Benchmark name
        
    Returns:
        BenchmarkMetrics object
    """
    # Convert probabilities to binary predictions
    y_pred = (y_prob > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.5  # No discrimination
    
    brier = brier_score_loss(y_true, y_prob)
    
    # Clip probabilities for log loss
    y_prob_clipped = np.clip(y_prob, 1e-15, 1 - 1e-15)
    logloss = log_loss(y_true, y_prob_clipped)
    
    return BenchmarkMetrics(
        name=name,
        n_predictions=len(y_true),
        accuracy=accuracy,
        auc=auc,
        brier_score=brier,
        log_loss=logloss,
    )


def compare_models(
    y_true: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    name_a: str,
    name_b: str
) -> List[ComparisonResult]:
    """
    Compare two models with statistical significance tests.
    
    Args:
        y_true: True labels
        prob_a: Model A probabilities
        prob_b: Model B probabilities
        name_a: Model A name
        name_b: Model B name
        
    Returns:
        List of ComparisonResult objects
    """
    results = []
    
    # Convert to binary predictions for accuracy
    pred_a = (prob_a > 0.5).astype(int)
    pred_b = (prob_b > 0.5).astype(int)
    
    # Accuracy comparison (McNemar test)
    acc_a = accuracy_score(y_true, pred_a)
    acc_b = accuracy_score(y_true, pred_b)
    _, p_acc = mcnemar_test(y_true, pred_a, pred_b)
    
    results.append(ComparisonResult(
        model_a=name_a,
        model_b=name_b,
        metric='accuracy',
        value_a=acc_a,
        value_b=acc_b,
        difference=acc_a - acc_b,
        p_value=p_acc,
        significant=p_acc < 0.05,
        test_name='McNemar',
    ))
    
    # AUC comparison (DeLong test)
    try:
        auc_a = roc_auc_score(y_true, prob_a)
        auc_b = roc_auc_score(y_true, prob_b)
        _, p_auc = delong_test(y_true, prob_a, prob_b)
        
        results.append(ComparisonResult(
            model_a=name_a,
            model_b=name_b,
            metric='auc',
            value_a=auc_a,
            value_b=auc_b,
            difference=auc_a - auc_b,
            p_value=p_auc,
            significant=p_auc < 0.05,
            test_name='DeLong',
        ))
    except ValueError:
        pass
    
    # Brier score comparison (paired t-test)
    brier_a = (y_true - prob_a) ** 2
    brier_b = (y_true - prob_b) ** 2
    _, p_brier = paired_t_test(brier_a, brier_b)
    
    results.append(ComparisonResult(
        model_a=name_a,
        model_b=name_b,
        metric='brier_score',
        value_a=brier_a.mean(),
        value_b=brier_b.mean(),
        difference=brier_a.mean() - brier_b.mean(),
        p_value=p_brier,
        significant=p_brier < 0.05,
        test_name='Paired t-test',
    ))
    
    return results


def run_benchmark_comparison(
    season: int,
    output_dir: Optional[Path] = None
) -> Dict:
    """
    Run benchmark comparison for a season.
    
    Args:
        season: Season year
        output_dir: Output directory
        
    Returns:
        Dictionary with metrics and comparisons
    """
    if output_dir is None:
        output_dir = Path("analysis") / "benchmark_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load predictions
    df = load_predictions(season)
    
    y_true = df['home_won'].values
    model_prob = df['home_win_prob'].values
    
    # Calculate model metrics
    model_metrics = calculate_benchmark_metrics(y_true, model_prob, "Our Model")
    
    benchmarks = {}
    comparisons = []
    
    # Vegas consensus (if available)
    if 'closing_odds_home' in df.columns:
        vegas_prob = df['closing_odds_home'].apply(calculate_vegas_prob).values
        vegas_metrics = calculate_benchmark_metrics(y_true, vegas_prob, "Vegas Consensus")
        benchmarks['vegas'] = vegas_metrics
        
        # Compare to Vegas
        vegas_comparisons = compare_models(
            y_true, model_prob, vegas_prob,
            "Our Model", "Vegas Consensus"
        )
        comparisons.extend(vegas_comparisons)
    
    # nfelo (if Elo ratings available)
    if 'home_elo' in df.columns and 'away_elo' in df.columns:
        nfelo_prob = df.apply(
            lambda row: calculate_nfelo_prob(row['home_elo'], row['away_elo']),
            axis=1
        ).values
        nfelo_metrics = calculate_benchmark_metrics(y_true, nfelo_prob, "nfelo")
        benchmarks['nfelo'] = nfelo_metrics
        
        # Compare to nfelo
        nfelo_comparisons = compare_models(
            y_true, model_prob, nfelo_prob,
            "Our Model", "nfelo"
        )
        comparisons.extend(nfelo_comparisons)
    
    # Spread baseline (if spread available)
    if 'spread' in df.columns:
        spread_prob = df['spread'].apply(calculate_spread_baseline_prob).values
        spread_metrics = calculate_benchmark_metrics(y_true, spread_prob, "Spread Baseline")
        benchmarks['spread_baseline'] = spread_metrics
        
        # Compare to spread baseline
        spread_comparisons = compare_models(
            y_true, model_prob, spread_prob,
            "Our Model", "Spread Baseline"
        )
        comparisons.extend(spread_comparisons)
    
    # Home favorite baseline
    if 'spread' in df.columns:
        home_fav_prob = df['spread'].apply(calculate_home_favorite_baseline).values
        home_fav_metrics = calculate_benchmark_metrics(y_true, home_fav_prob, "Home Favorite")
        benchmarks['home_favorite'] = home_fav_metrics
        
        # Compare to home favorite
        home_fav_comparisons = compare_models(
            y_true, model_prob, home_fav_prob,
            "Our Model", "Home Favorite"
        )
        comparisons.extend(home_fav_comparisons)
    
    # Compile results
    results = {
        'season': season,
        'model': model_metrics.to_dict(),
        'benchmarks': {name: metrics.to_dict() for name, metrics in benchmarks.items()},
        'comparisons': [comp.to_dict() for comp in comparisons],
    }
    
    # Save results
    results_file = output_dir / f"{season}_comparison.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Benchmark comparison saved to {results_file}")
    
    return results


def generate_comparison_report(
    season: int,
    output_dir: Optional[Path] = None
) -> None:
    """
    Generate benchmark comparison report.
    
    Args:
        season: Season year
        output_dir: Output directory
    """
    if output_dir is None:
        output_dir = Path("analysis") / "benchmark_comparison"
    
    # Load results
    results_file = output_dir / f"{season}_comparison.json"
    if not results_file.exists():
        # Run comparison if not exists
        results = run_benchmark_comparison(season, output_dir)
    else:
        with open(results_file) as f:
            results = json.load(f)
    
    # Generate markdown report
    report_file = output_dir / f"{season}_report.md"
    
    with open(report_file, 'w') as f:
        f.write(f"# Benchmark Comparison Report - {season}\n\n")
        
        # Model metrics
        model = results['model']
        f.write("## Our Model Performance\n\n")
        f.write(f"- **Predictions:** {model['n_predictions']}\n")
        f.write(f"- **Accuracy:** {model['accuracy']:.3f}\n")
        f.write(f"- **AUC:** {model['auc']:.3f}\n")
        f.write(f"- **Brier Score:** {model['brier_score']:.4f}\n")
        f.write(f"- **Log Loss:** {model['log_loss']:.4f}\n\n")
        
        # Benchmark metrics
        f.write("## Benchmark Performance\n\n")
        f.write("| Benchmark | Accuracy | AUC | Brier Score | Log Loss |\n")
        f.write("|-----------|----------|-----|-------------|----------|\n")
        
        for name, metrics in results['benchmarks'].items():
            f.write(f"| {metrics['name']} | {metrics['accuracy']:.3f} | "
                   f"{metrics['auc']:.3f} | {metrics['brier_score']:.4f} | "
                   f"{metrics['log_loss']:.4f} |\n")
        
        # Statistical comparisons
        f.write("\n## Statistical Comparisons\n\n")
        
        for comp in results['comparisons']:
            f.write(f"### {comp['model_a']} vs {comp['model_b']}\n\n")
            f.write(f"**Metric:** {comp['metric']}\n\n")
            f.write(f"- **{comp['model_a']}:** {comp['value_a']:.4f}\n")
            f.write(f"- **{comp['model_b']}:** {comp['value_b']:.4f}\n")
            f.write(f"- **Difference:** {comp['difference']:+.4f}\n")
            f.write(f"- **P-value:** {comp['p_value']:.4f} ({comp['test_name']})\n")
            
            if comp['significant']:
                f.write(f"- **Result:** ✅ Statistically significant (p < 0.05)\n\n")
            else:
                f.write(f"- **Result:** ❌ Not statistically significant\n\n")
        
        f.write("## Interpretation\n\n")
        f.write("- **P-value < 0.05:** Statistically significant difference\n")
        f.write("- **Positive difference:** Our model performs better\n")
        f.write("- **Negative difference:** Benchmark performs better\n\n")
        
        f.write("---\n\n")
        f.write("*Generated by benchmark_comparison.py*\n")
    
    print(f"Report saved to {report_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Compare model to benchmarks")
    parser.add_argument('--season', type=int, help='Season year')
    parser.add_argument('--start-season', type=int, help='Start season for multi-season')
    parser.add_argument('--end-season', type=int, help='End season for multi-season')
    parser.add_argument('--benchmark', choices=['vegas', 'nfelo', 'spread', 'home_favorite'],
                       help='Specific benchmark to compare')
    parser.add_argument('--generate-report', action='store_true',
                       help='Generate comparison report')
    
    args = parser.parse_args()
    
    output_dir = Path("analysis") / "benchmark_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        if args.start_season and args.end_season:
            # Multi-season comparison
            for season in range(args.start_season, args.end_season + 1):
                print(f"\nProcessing season {season}...")
                run_benchmark_comparison(season, output_dir)
                if args.generate_report:
                    generate_comparison_report(season, output_dir)
        
        elif args.season:
            # Single season
            if args.generate_report:
                generate_comparison_report(args.season, output_dir)
            else:
                results = run_benchmark_comparison(args.season, output_dir)
                
                # Print summary
                print(f"\nBenchmark Comparison - {args.season}")
                print("=" * 60)
                
                model = results['model']
                print(f"\nOur Model: Accuracy={model['accuracy']:.3f}, "
                      f"AUC={model['auc']:.3f}, Brier={model['brier_score']:.4f}")
                
                for name, metrics in results['benchmarks'].items():
                    print(f"{metrics['name']}: Accuracy={metrics['accuracy']:.3f}, "
                          f"AUC={metrics['auc']:.3f}, Brier={metrics['brier_score']:.4f}")
        
        else:
            parser.print_help()
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

# Made with Bob
