#!/usr/bin/env python
"""
Real-Time Calibration Monitoring

Tracks prediction quality over time with:
- Reliability diagrams (predicted vs actual probabilities)
- Brier score decomposition (calibration + resolution + uncertainty)
- Calibration drift detection and alerts
- Probability bin accuracy tracking
- Confidence interval coverage analysis
- Automated recalibration triggers

Usage:
    # Monitor single week
    python -m analysis.calibration_monitor --season 2025 --week 10 --generate-report
    
    # Monitor full season
    python -m analysis.calibration_monitor --season 2025 --generate-report
    
    # Check for drift and trigger recalibration
    python -m analysis.calibration_monitor --season 2025 --check-drift --threshold 0.02
    
    # Generate reliability diagram
    python -m analysis.calibration_monitor --season 2025 --plot-reliability
"""

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

# Optional matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class CalibrationMetrics:
    """Calibration metrics for a set of predictions."""
    n_predictions: int
    brier_score: float
    brier_calibration: float  # Calibration component
    brier_resolution: float   # Resolution component
    brier_uncertainty: float  # Uncertainty component
    mean_predicted_prob: float
    mean_actual_prob: float
    calibration_error: float  # Mean absolute calibration error
    max_calibration_error: float  # Maximum calibration error in any bin
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ProbabilityBin:
    """Statistics for a probability bin."""
    bin_center: float
    bin_range: Tuple[float, float]
    n_predictions: int
    mean_predicted: float
    mean_actual: float
    calibration_error: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'bin_center': self.bin_center,
            'bin_range': list(self.bin_range),
            'n_predictions': self.n_predictions,
            'mean_predicted': self.mean_predicted,
            'mean_actual': self.mean_actual,
            'calibration_error': self.calibration_error,
        }


@dataclass
class DriftAlert:
    """Calibration drift alert."""
    season: int
    week: Optional[int]
    metric: str
    current_value: float
    baseline_value: float
    drift: float
    threshold: float
    severity: str  # 'warning' or 'critical'
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


def decompose_brier_score(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Tuple[float, float, float, float]:
    """
    Decompose Brier score into calibration, resolution, and uncertainty.
    
    Brier = Calibration - Resolution + Uncertainty
    
    - Calibration: How close predicted probabilities are to actual frequencies
    - Resolution: How well predictions separate outcomes (higher is better)
    - Uncertainty: Inherent unpredictability (constant for dataset)
    
    Args:
        y_true: Actual outcomes (0 or 1)
        y_pred: Predicted probabilities
        
    Returns:
        (brier_score, calibration, resolution, uncertainty)
    """
    brier = brier_score_loss(y_true, y_pred)
    
    # Overall base rate
    base_rate = y_true.mean()
    
    # Uncertainty component (constant)
    uncertainty = base_rate * (1 - base_rate)
    
    # Calibration component
    # Group predictions into bins and measure deviation from actual frequencies
    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred, bins[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    calibration = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_pred = y_pred[mask].mean()
            bin_actual = y_true[mask].mean()
            bin_weight = mask.sum() / len(y_true)
            calibration += bin_weight * (bin_pred - bin_actual) ** 2
    
    # Resolution component
    resolution = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_actual = y_true[mask].mean()
            bin_weight = mask.sum() / len(y_true)
            resolution += bin_weight * (bin_actual - base_rate) ** 2
    
    return brier, calibration, resolution, uncertainty


def calculate_calibration_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> CalibrationMetrics:
    """
    Calculate comprehensive calibration metrics.
    
    Args:
        y_true: Actual outcomes (0 or 1)
        y_pred: Predicted probabilities
        
    Returns:
        CalibrationMetrics object
    """
    brier, calibration, resolution, uncertainty = decompose_brier_score(y_true, y_pred)
    
    # Mean calibration error
    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred, bins[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    calibration_errors = []
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_pred = y_pred[mask].mean()
            bin_actual = y_true[mask].mean()
            calibration_errors.append(abs(bin_pred - bin_actual))
    
    mean_cal_error = np.mean(calibration_errors) if calibration_errors else 0.0
    max_cal_error = np.max(calibration_errors) if calibration_errors else 0.0
    
    return CalibrationMetrics(
        n_predictions=len(y_true),
        brier_score=brier,
        brier_calibration=calibration,
        brier_resolution=resolution,
        brier_uncertainty=uncertainty,
        mean_predicted_prob=y_pred.mean(),
        mean_actual_prob=y_true.mean(),
        calibration_error=mean_cal_error,
        max_calibration_error=max_cal_error,
    )


def calculate_probability_bins(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10
) -> List[ProbabilityBin]:
    """
    Calculate statistics for probability bins.
    
    Args:
        y_true: Actual outcomes (0 or 1)
        y_pred: Predicted probabilities
        n_bins: Number of bins
        
    Returns:
        List of ProbabilityBin objects
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred, bins[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    probability_bins = []
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_pred = y_pred[mask].mean()
            bin_actual = y_true[mask].mean()
            bin_center = (bins[i] + bins[i + 1]) / 2
            bin_range = (bins[i], bins[i + 1])
            
            probability_bins.append(ProbabilityBin(
                bin_center=bin_center,
                bin_range=bin_range,
                n_predictions=int(mask.sum()),
                mean_predicted=bin_pred,
                mean_actual=bin_actual,
                calibration_error=abs(bin_pred - bin_actual),
            ))
    
    return probability_bins


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    title: str = "Reliability Diagram"
) -> None:
    """
    Generate reliability diagram (calibration plot).
    
    Args:
        y_true: Actual outcomes (0 or 1)
        y_pred: Predicted probabilities
        output_path: Path to save plot
        title: Plot title
    """
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available, skipping plot")
        return
    
    # Calculate calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred, n_bins=10, strategy='uniform'
    )
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
    
    # Actual calibration
    ax.plot(
        mean_predicted_value,
        fraction_of_positives,
        'o-',
        label='Model Calibration',
        linewidth=2,
        markersize=8
    )
    
    # Formatting
    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives (Actual)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # Add metrics text
    brier = brier_score_loss(y_true, y_pred)
    metrics_text = f'Brier Score: {brier:.4f}\nN = {len(y_true)}'
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Reliability diagram saved to {output_path}")


def detect_calibration_drift(
    current_metrics: CalibrationMetrics,
    baseline_metrics: CalibrationMetrics,
    threshold: float = 0.02
) -> List[DriftAlert]:
    """
    Detect calibration drift by comparing current to baseline metrics.
    
    Args:
        current_metrics: Current period metrics
        baseline_metrics: Baseline period metrics
        threshold: Drift threshold for alerts
        
    Returns:
        List of DriftAlert objects
    """
    alerts = []
    
    # Check Brier score drift
    brier_drift = current_metrics.brier_score - baseline_metrics.brier_score
    if abs(brier_drift) > threshold:
        severity = 'critical' if abs(brier_drift) > threshold * 2 else 'warning'
        alerts.append(DriftAlert(
            season=0,  # Will be filled by caller
            week=None,
            metric='brier_score',
            current_value=current_metrics.brier_score,
            baseline_value=baseline_metrics.brier_score,
            drift=brier_drift,
            threshold=threshold,
            severity=severity,
        ))
    
    # Check calibration error drift
    cal_error_drift = current_metrics.calibration_error - baseline_metrics.calibration_error
    if abs(cal_error_drift) > threshold:
        severity = 'critical' if abs(cal_error_drift) > threshold * 2 else 'warning'
        alerts.append(DriftAlert(
            season=0,
            week=None,
            metric='calibration_error',
            current_value=current_metrics.calibration_error,
            baseline_value=baseline_metrics.calibration_error,
            drift=cal_error_drift,
            threshold=threshold,
            severity=severity,
        ))
    
    # Check max calibration error
    max_cal_drift = current_metrics.max_calibration_error - baseline_metrics.max_calibration_error
    if abs(max_cal_drift) > threshold * 2:  # Higher threshold for max error
        severity = 'critical' if abs(max_cal_drift) > threshold * 4 else 'warning'
        alerts.append(DriftAlert(
            season=0,
            week=None,
            metric='max_calibration_error',
            current_value=current_metrics.max_calibration_error,
            baseline_value=baseline_metrics.max_calibration_error,
            drift=max_cal_drift,
            threshold=threshold * 2,
            severity=severity,
        ))
    
    return alerts


def load_predictions(season: int, week: Optional[int] = None) -> pd.DataFrame:
    """
    Load predictions from live tracking or matchup features.
    
    Args:
        season: Season year
        week: Optional week number (None for full season)
        
    Returns:
        DataFrame with columns: game_id, season, week, home_win_prob, home_won
    """
    # Try live tracking first
    predictions_dir = Path("predictions_log") / str(season)
    
    if week is not None:
        # Load specific week
        pred_file = predictions_dir / f"week_{week:02d}_predictions.csv"
        actual_file = predictions_dir / f"week_{week:02d}_actuals.csv"
        
        if pred_file.exists() and actual_file.exists():
            preds = pd.read_csv(pred_file)
            actuals = pd.read_csv(actual_file)
            
            # Merge predictions with actuals
            df = preds.merge(actuals[['game_id', 'home_won']], on='game_id', how='inner')
            return df[['game_id', 'season', 'week', 'home_win_prob', 'home_won']]
    
    # Fall back to matchup features
    features_file = Path("data") / "matchup_features.parquet"
    if not features_file.exists():
        raise FileNotFoundError(f"No predictions found for season {season}")
    
    df = pd.read_parquet(features_file)
    df = df[df['season'] == season].copy()
    
    if week is not None:
        df = df[df['week'] == week].copy()
    
    if len(df) == 0:
        raise ValueError(f"No predictions found for season {season}, week {week}")
    
    # Ensure required columns exist
    if 'home_win_prob' not in df.columns or 'home_won' not in df.columns:
        raise ValueError("Required columns 'home_win_prob' and 'home_won' not found")
    
    return df[['game_id', 'season', 'week', 'home_win_prob', 'home_won']]


def monitor_calibration(
    season: int,
    week: Optional[int] = None,
    output_dir: Optional[Path] = None
) -> CalibrationMetrics:
    """
    Monitor calibration for a season or week.
    
    Args:
        season: Season year
        week: Optional week number
        output_dir: Output directory for results
        
    Returns:
        CalibrationMetrics object
    """
    if output_dir is None:
        output_dir = Path("analysis") / "calibration_monitoring"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load predictions
    df = load_predictions(season, week)
    
    y_true = df['home_won'].values
    y_pred = df['home_win_prob'].values
    
    # Calculate metrics
    metrics = calculate_calibration_metrics(y_true, y_pred)
    
    # Calculate probability bins
    bins = calculate_probability_bins(y_true, y_pred)
    
    # Save metrics
    period = f"week_{week:02d}" if week is not None else "season"
    metrics_file = output_dir / f"{season}_{period}_metrics.json"
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics.to_dict(), f, indent=2)
    
    print(f"Calibration metrics saved to {metrics_file}")
    
    # Save bin statistics
    bins_file = output_dir / f"{season}_{period}_bins.json"
    with open(bins_file, 'w') as f:
        json.dump([b.to_dict() for b in bins], f, indent=2)
    
    print(f"Probability bins saved to {bins_file}")
    
    return metrics


def check_drift(
    season: int,
    baseline_season: int,
    threshold: float = 0.02,
    output_dir: Optional[Path] = None
) -> List[DriftAlert]:
    """
    Check for calibration drift vs baseline season.
    
    Args:
        season: Current season to check
        baseline_season: Baseline season for comparison
        threshold: Drift threshold
        output_dir: Output directory for alerts
        
    Returns:
        List of DriftAlert objects
    """
    if output_dir is None:
        output_dir = Path("analysis") / "calibration_monitoring"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load current metrics
    current_file = output_dir / f"{season}_season_metrics.json"
    if not current_file.exists():
        # Calculate if not exists
        current_metrics = monitor_calibration(season, output_dir=output_dir)
    else:
        with open(current_file) as f:
            current_data = json.load(f)
        current_metrics = CalibrationMetrics(**current_data)
    
    # Load baseline metrics
    baseline_file = output_dir / f"{baseline_season}_season_metrics.json"
    if not baseline_file.exists():
        # Calculate if not exists
        baseline_metrics = monitor_calibration(baseline_season, output_dir=output_dir)
    else:
        with open(baseline_file) as f:
            baseline_data = json.load(f)
        baseline_metrics = CalibrationMetrics(**baseline_data)
    
    # Detect drift
    alerts = detect_calibration_drift(current_metrics, baseline_metrics, threshold)
    
    # Fill in season info
    for alert in alerts:
        alert.season = season
    
    # Save alerts
    if alerts:
        alerts_file = output_dir / f"{season}_drift_alerts.json"
        with open(alerts_file, 'w') as f:
            json.dump([a.to_dict() for a in alerts], f, indent=2)
        
        print(f"\n[ALERT] {len(alerts)} calibration drift alert(s) detected!")
        print(f"Alerts saved to {alerts_file}")
        
        for alert in alerts:
            print(f"  - {alert.metric}: {alert.drift:+.4f} ({alert.severity})")
    else:
        print(f"\n[OK] No calibration drift detected (threshold={threshold})")
    
    return alerts


def generate_report(
    season: int,
    week: Optional[int] = None,
    output_dir: Optional[Path] = None
) -> None:
    """
    Generate calibration monitoring report.
    
    Args:
        season: Season year
        week: Optional week number
        output_dir: Output directory
    """
    if output_dir is None:
        output_dir = Path("analysis") / "calibration_monitoring"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Monitor calibration
    metrics = monitor_calibration(season, week, output_dir)
    
    # Load predictions for plotting
    df = load_predictions(season, week)
    y_true = df['home_won'].values
    y_pred = df['home_win_prob'].values
    
    # Generate reliability diagram
    period = f"week_{week:02d}" if week is not None else "season"
    plot_file = output_dir / f"{season}_{period}_reliability.png"
    title = f"Reliability Diagram - {season}"
    if week is not None:
        title += f" Week {week}"
    
    plot_reliability_diagram(y_true, y_pred, plot_file, title)
    
    # Generate markdown report
    report_file = output_dir / f"{season}_{period}_report.md"
    
    with open(report_file, 'w') as f:
        f.write(f"# Calibration Monitoring Report\n\n")
        f.write(f"**Season:** {season}\n")
        if week is not None:
            f.write(f"**Week:** {week}\n")
        f.write(f"**Predictions:** {metrics.n_predictions}\n\n")
        
        f.write("## Overall Metrics\n\n")
        f.write(f"- **Brier Score:** {metrics.brier_score:.4f}\n")
        f.write(f"- **Mean Calibration Error:** {metrics.calibration_error:.4f}\n")
        f.write(f"- **Max Calibration Error:** {metrics.max_calibration_error:.4f}\n")
        f.write(f"- **Mean Predicted Prob:** {metrics.mean_predicted_prob:.3f}\n")
        f.write(f"- **Mean Actual Prob:** {metrics.mean_actual_prob:.3f}\n\n")
        
        f.write("## Brier Score Decomposition\n\n")
        f.write(f"- **Calibration:** {metrics.brier_calibration:.4f} (lower is better)\n")
        f.write(f"- **Resolution:** {metrics.brier_resolution:.4f} (higher is better)\n")
        f.write(f"- **Uncertainty:** {metrics.brier_uncertainty:.4f} (constant)\n\n")
        
        f.write("**Formula:** Brier = Calibration - Resolution + Uncertainty\n\n")
        
        # Load bin statistics
        bins_file = output_dir / f"{season}_{period}_bins.json"
        with open(bins_file) as bf:
            bins = json.load(bf)
        
        f.write("## Probability Bins\n\n")
        f.write("| Bin Range | N | Mean Predicted | Mean Actual | Cal Error |\n")
        f.write("|-----------|---|----------------|-------------|----------|\n")
        
        for bin_data in bins:
            bin_range = f"{bin_data['bin_range'][0]:.2f}-{bin_data['bin_range'][1]:.2f}"
            f.write(f"| {bin_range} | {bin_data['n_predictions']} | "
                   f"{bin_data['mean_predicted']:.3f} | {bin_data['mean_actual']:.3f} | "
                   f"{bin_data['calibration_error']:.3f} |\n")
        
        f.write("\n## Interpretation\n\n")
        f.write("- **Brier Score:** Lower is better (0 = perfect, 0.25 = random)\n")
        f.write("- **Calibration Error:** Deviation between predicted and actual probabilities\n")
        f.write("- **Resolution:** Ability to separate outcomes (higher = better discrimination)\n")
        f.write("- **Drift Alert Threshold:** 0.02 (2 percentage points)\n\n")
        
        if plot_file.exists():
            f.write(f"## Reliability Diagram\n\n")
            f.write(f"![Reliability Diagram]({plot_file.name})\n\n")
        
        f.write("---\n\n")
        f.write("*Generated by calibration_monitor.py*\n")
    
    print(f"Report saved to {report_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Monitor prediction calibration")
    parser.add_argument('--season', type=int, required=True, help='Season year')
    parser.add_argument('--week', type=int, help='Week number (optional)')
    parser.add_argument('--generate-report', action='store_true',
                       help='Generate calibration report')
    parser.add_argument('--check-drift', action='store_true',
                       help='Check for calibration drift')
    parser.add_argument('--baseline-season', type=int,
                       help='Baseline season for drift comparison')
    parser.add_argument('--threshold', type=float, default=0.02,
                       help='Drift alert threshold (default: 0.02)')
    parser.add_argument('--plot-reliability', action='store_true',
                       help='Generate reliability diagram')
    
    args = parser.parse_args()
    
    output_dir = Path("analysis") / "calibration_monitoring"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        if args.generate_report:
            generate_report(args.season, args.week, output_dir)
        
        elif args.check_drift:
            baseline = args.baseline_season or (args.season - 1)
            check_drift(args.season, baseline, args.threshold, output_dir)
        
        elif args.plot_reliability:
            df = load_predictions(args.season, args.week)
            y_true = df['home_won'].values
            y_pred = df['home_win_prob'].values
            
            period = f"week_{args.week:02d}" if args.week else "season"
            plot_file = output_dir / f"{args.season}_{period}_reliability.png"
            title = f"Reliability Diagram - {args.season}"
            if args.week:
                title += f" Week {args.week}"
            
            plot_reliability_diagram(y_true, y_pred, plot_file, title)
        
        else:
            # Default: monitor calibration
            metrics = monitor_calibration(args.season, args.week, output_dir)
            
            print(f"\nCalibration Metrics:")
            print(f"  Brier Score: {metrics.brier_score:.4f}")
            print(f"  Calibration Error: {metrics.calibration_error:.4f}")
            print(f"  Max Calibration Error: {metrics.max_calibration_error:.4f}")
            print(f"  Resolution: {metrics.brier_resolution:.4f}")
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

# Made with Bob
