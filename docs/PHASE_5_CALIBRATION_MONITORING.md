# Phase 5: Real-Time Calibration Monitoring - Implementation Guide

## Overview

The calibration monitoring system (`analysis/calibration_monitor.py`) tracks prediction quality over time, ensuring the model's predicted probabilities match actual outcomes. This is critical for maintaining trust in the model and detecting when recalibration is needed.

## Key Features

### 1. Brier Score Decomposition
Breaks down overall prediction error into three components:

**Formula:** `Brier = Calibration - Resolution + Uncertainty`

- **Calibration Component** (lower is better)
  - Measures deviation between predicted probabilities and actual frequencies
  - Perfect calibration = 0
  - High values indicate overconfidence or underconfidence

- **Resolution Component** (higher is better)
  - Measures ability to separate outcomes
  - High resolution = predictions distinguish between wins and losses
  - Low resolution = predictions cluster around base rate

- **Uncertainty Component** (constant)
  - Inherent unpredictability in the data
  - Equals `base_rate × (1 - base_rate)`
  - Cannot be reduced by better modeling

### 2. Reliability Diagrams
Visual calibration assessment:
- X-axis: Mean predicted probability (binned)
- Y-axis: Actual frequency of positive outcomes
- Perfect calibration: Points lie on diagonal line
- Overconfident: Points below diagonal
- Underconfident: Points above diagonal

### 3. Calibration Drift Detection
Compares current metrics to baseline:
- **Warning drift:** 0.02-0.04 change (increase monitoring)
- **Critical drift:** >0.04 change (recalibrate immediately)
- Tracks Brier score, calibration error, and max error

### 4. Probability Bin Analysis
Divides predictions into 10 bins (0.0-0.1, 0.1-0.2, etc.):
- Counts predictions per bin
- Calculates mean predicted vs actual
- Measures calibration error per bin
- Identifies problematic probability ranges

## Installation

No additional dependencies required beyond main project requirements. Optional matplotlib for plotting:

```bash
pip install matplotlib
```

## Usage

### Monitor Single Week

```bash
python -m analysis.calibration_monitor --season 2025 --week 10 --generate-report
```

Generates:
- Calibration metrics JSON
- Probability bin statistics
- Reliability diagram
- Markdown report

### Monitor Full Season

```bash
python -m analysis.calibration_monitor --season 2025 --generate-report
```

Aggregates all weeks in the season for comprehensive analysis.

### Check for Drift

```bash
# Compare 2025 to 2024 baseline
python -m analysis.calibration_monitor --season 2025 --check-drift --baseline-season 2024 --threshold 0.02
```

Detects:
- Brier score drift
- Calibration error drift
- Max calibration error drift

Generates alerts JSON if drift exceeds threshold.

### Generate Reliability Diagram Only

```bash
python -m analysis.calibration_monitor --season 2025 --plot-reliability
```

Creates visual calibration plot without full report.

### Calculate Metrics Only

```bash
python -m analysis.calibration_monitor --season 2025
```

Calculates and saves metrics without generating report or plots.

## Output Files

### Metrics JSON
`analysis/calibration_monitoring/YYYY_season_metrics.json`

```json
{
  "n_predictions": 267,
  "brier_score": 0.2015,
  "brier_calibration": 0.0023,
  "brier_resolution": 0.0458,
  "brier_uncertainty": 0.2450,
  "mean_predicted_prob": 0.5234,
  "mean_actual_prob": 0.5131,
  "calibration_error": 0.0156,
  "max_calibration_error": 0.0423
}
```

### Probability Bins JSON
`analysis/calibration_monitoring/YYYY_season_bins.json`

Contains array of bin objects:
```json
[
  {
    "bin_center": 0.05,
    "bin_range": [0.0, 0.1],
    "n_predictions": 12,
    "mean_predicted": 0.0523,
    "mean_actual": 0.0833,
    "calibration_error": 0.0310
  },
  ...
]
```

### Reliability Diagram
`analysis/calibration_monitoring/YYYY_season_reliability.png`

Visual plot with:
- Perfect calibration line (diagonal)
- Actual calibration curve
- Brier score annotation
- Sample size

### Drift Alerts JSON
`analysis/calibration_monitoring/YYYY_drift_alerts.json`

Generated when drift detected:
```json
[
  {
    "season": 2025,
    "week": null,
    "metric": "brier_score",
    "current_value": 0.2234,
    "baseline_value": 0.2015,
    "drift": 0.0219,
    "threshold": 0.02,
    "severity": "warning"
  }
]
```

### Calibration Report
`analysis/calibration_monitoring/YYYY_season_report.md`

Comprehensive markdown report with:
- Overall metrics summary
- Brier decomposition
- Probability bin table
- Interpretation guide
- Reliability diagram (if generated)

## Interpretation Guide

### Brier Score Ranges

| Range | Quality | Action |
|-------|---------|--------|
| 0.19-0.21 | Excellent | Continue monitoring |
| 0.21-0.23 | Good | Minor improvements possible |
| 0.23-0.25 | Acceptable | Consider recalibration |
| >0.25 | Poor | Recalibrate immediately |

### Calibration Error Ranges

| Range | Quality | Action |
|-------|---------|--------|
| <0.02 | Excellent | Well-calibrated |
| 0.02-0.05 | Good | Monitor for drift |
| 0.05-0.10 | Acceptable | Plan recalibration |
| >0.10 | Poor | Recalibrate now |

### Resolution Component

| Range | Quality | Interpretation |
|-------|---------|----------------|
| >0.04 | High | Strong discrimination |
| 0.02-0.04 | Moderate | Acceptable separation |
| <0.02 | Low | Predictions too similar |

### Drift Severity

| Drift | Severity | Action |
|-------|----------|--------|
| <0.02 | None | Continue normal monitoring |
| 0.02-0.04 | Warning | Increase monitoring frequency |
| >0.04 | Critical | Recalibrate immediately |

## Calibration Maintenance Workflow

### Weekly Monitoring (During Season)

**Thursday (Pre-Games):**
```bash
# Lock predictions
python -m analysis.live_tracking --lock-predictions --season 2025 --week 10
```

**Tuesday (Post-Games):**
```bash
# Fetch actuals
python -m analysis.live_tracking --fetch-actuals --season 2025 --week 10

# Monitor calibration
python -m analysis.calibration_monitor --season 2025 --week 10 --generate-report

# Check for drift (vs rolling 4-week baseline)
python -m analysis.calibration_monitor --season 2025 --check-drift --threshold 0.02
```

### Monthly Review

```bash
# Generate full season report
python -m analysis.calibration_monitor --season 2025 --generate-report

# Compare to previous season
python -m analysis.calibration_monitor --season 2025 --check-drift --baseline-season 2024
```

### Off-Season Recalibration

```bash
# Analyze full season
python -m analysis.calibration_monitor --season 2025 --generate-report

# If drift detected, recalibrate using isotonic regression
# (Manual step - run src/evaluation/calibrate_winprob.py)

# Verify recalibration improved metrics
python -m analysis.calibration_monitor --season 2025 --generate-report
```

## Recalibration Triggers

### Automatic Triggers
- Brier score increases >0.02 vs baseline
- Calibration error >0.05
- Max calibration error >0.10
- 3+ consecutive weeks of drift warnings

### Manual Triggers
- Major data source changes
- Feature engineering updates
- Model architecture changes
- New season starts

## Recalibration Methods

### 1. Isotonic Regression (Recommended)
**Location:** `src/evaluation/calibrate_winprob.py`

**Advantages:**
- Non-parametric, monotonic
- Preserves rank order
- Works well for NFL predictions
- Already implemented

**Usage:**
```python
from src.evaluation.calibrate_winprob import calibrate_predictions

# Load predictions
df = pd.read_parquet('data/matchup_features.parquet')

# Calibrate
calibrated_df = calibrate_predictions(df, method='isotonic')
```

### 2. Platt Scaling
**Method:** Logistic regression on predictions

**Advantages:**
- Simple, parametric
- Good for small datasets
- Fast to compute

**Disadvantages:**
- May not preserve rank order
- Assumes sigmoid relationship

### 3. Beta Calibration
**Method:** Three-parameter generalization of Platt

**Advantages:**
- More flexible than Platt
- Better for skewed distributions

**Disadvantages:**
- Requires more data
- More complex to implement

## Integration with Validation Pipeline

### Data Flow

```
1. live_tracking.py → Lock predictions with timestamps
2. live_tracking.py → Fetch actual results
3. calibration_monitor.py → Calculate metrics
4. calibration_monitor.py → Check for drift
5. (If drift) → Recalibrate using isotonic regression
6. (Verify) → Re-run calibration monitoring
```

### Automated Weekly Pipeline

```bash
#!/bin/bash
# weekly_validation.sh

SEASON=2025
WEEK=10

# Lock predictions (Thursday)
python -m analysis.live_tracking --lock-predictions --season $SEASON --week $WEEK

# Wait for games to complete...

# Fetch actuals (Tuesday)
python -m analysis.live_tracking --fetch-actuals --season $SEASON --week $WEEK

# Monitor calibration
python -m analysis.calibration_monitor --season $SEASON --week $WEEK --generate-report

# Check drift
python -m analysis.calibration_monitor --season $SEASON --check-drift --threshold 0.02

# If drift alerts exist, send notification
if [ -f "analysis/calibration_monitoring/${SEASON}_drift_alerts.json" ]; then
    echo "ALERT: Calibration drift detected!"
    # Send email/Slack notification
fi
```

## Testing

Run unit tests to verify functionality:

```bash
python test_calibration_monitor.py
```

Tests cover:
- Brier score decomposition
- Calibration metrics calculation
- Probability bin creation
- Drift detection
- Module function availability

## Troubleshooting

### No predictions found
**Error:** `FileNotFoundError: No predictions found`

**Solution:** Ensure predictions are available:
1. Check `predictions_log/YYYY/week_NN_predictions.csv` exists
2. Or verify `data/matchup_features.parquet` contains season data

### Matplotlib not available
**Warning:** `matplotlib not available, skipping plot`

**Solution:** Install matplotlib:
```bash
pip install matplotlib
```

### High calibration error
**Issue:** Calibration error >0.05

**Diagnosis:**
1. Check Brier decomposition - is calibration component high?
2. Review reliability diagram - overconfident or underconfident?
3. Examine probability bins - which ranges have high error?

**Solutions:**
1. Apply isotonic recalibration
2. Review feature importance for drift
3. Check data quality
4. Retrain model if necessary

### Constant drift alerts
**Issue:** Drift warnings every week

**Diagnosis:**
1. Is threshold too strict? (0.02 may be low for weekly data)
2. High week-to-week variance? (small sample sizes)
3. Genuine model degradation?

**Solutions:**
1. Increase threshold to 0.03-0.04
2. Use rolling 4-week average instead of single week
3. Investigate root cause of degradation

## Best Practices

### 1. Establish Baseline
- Use previous season as baseline
- Calculate on 200+ predictions minimum
- Update baseline annually

### 2. Monitor Continuously
- Check calibration weekly during season
- Generate monthly comprehensive reports
- Track trends over time

### 3. Set Appropriate Thresholds
- **Brier drift:** 0.02 (2 percentage points)
- **Calibration error:** 0.05 (5 percentage points)
- **Max error:** 0.10 (10 percentage points)

### 4. Recalibrate Proactively
- Don't wait for critical drift
- Recalibrate at season start
- Recalibrate after major changes

### 5. Document Changes
- Log all recalibrations with dates
- Track before/after metrics
- Maintain calibration history

## Advanced Topics

### Rolling Window Calibration

Monitor calibration over rolling 4-week windows:

```python
from analysis.calibration_monitor import monitor_calibration

# Calculate for each 4-week window
for week in range(4, 18):
    # Get predictions for weeks (week-3) to week
    # Calculate metrics
    # Track trend
```

### Stratified Calibration

Analyze calibration by subgroups:

```python
# By probability range
high_prob = df[df['home_win_prob'] > 0.6]
low_prob = df[df['home_win_prob'] < 0.4]

# By team strength
favorites = df[df['spread'] < -3]
underdogs = df[df['spread'] > 3]

# Calculate metrics for each group
```

### Confidence Interval Coverage

Check if confidence intervals are well-calibrated:

```python
# For 80% confidence interval
lower = df['home_win_prob'] - 1.28 * df['home_win_prob_std']
upper = df['home_win_prob'] + 1.28 * df['home_win_prob_std']

coverage = ((df['home_won'] >= lower) & (df['home_won'] <= upper)).mean()
# Should be ~0.80 for well-calibrated intervals
```

## References

- **Brier Score:** G. W. Brier, "Verification of forecasts expressed in terms of probability" (1950)
- **Decomposition:** A. H. Murphy, "A New Vector Partition of the Probability Score" (1973)
- **Reliability Diagrams:** D. S. Wilks, "Statistical Methods in the Atmospheric Sciences" (2011)
- **Isotonic Regression:** Zadrozny & Elkan, "Transforming classifier scores into accurate multiclass probability estimates" (2002)
- **Calibration Methods:** Niculescu-Mizil & Caruana, "Predicting good probabilities with supervised learning" (2005)

## Next Steps

After implementing calibration monitoring:

1. **Phase 6:** Compare against public benchmarks (nfelo, Vegas consensus)
2. **Phase 7:** Integrate calibration metrics into Streamlit dashboard
3. **Phase 8:** Automate weekly validation pipeline with cron jobs
4. **Phase 9:** Document complete validation methodology

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review test file (`test_calibration_monitor.py`) for examples
3. Examine `analysis/calibration_monitoring/README.md` for interpretation guide
4. Verify predictions are available in expected locations

---

**Remember:** Calibration monitoring is about maintaining trust in your model's probabilities. A well-calibrated model saying "70% chance of winning" should win approximately 70% of the time. Regular monitoring and proactive recalibration ensure this relationship holds over time.