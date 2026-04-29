# Calibration Monitoring

Real-time tracking of prediction quality with reliability diagrams, Brier score decomposition, and drift detection.

## Overview

Calibration monitoring ensures your model's predicted probabilities match actual outcomes. A well-calibrated model predicting 70% win probability should win approximately 70% of the time.

## Key Concepts

### Brier Score
**Formula:** `BS = (1/N) Σ(predicted_prob - actual_outcome)²`

- **Range:** 0 (perfect) to 1 (worst)
- **Typical NFL models:** 0.19-0.22
- **Interpretation:** Mean squared error of probability predictions

### Brier Score Decomposition
**Formula:** `Brier = Calibration - Resolution + Uncertainty`

1. **Calibration Component** (lower is better)
   - Measures how close predicted probabilities are to actual frequencies
   - Perfect calibration = 0
   - High calibration error = model is overconfident or underconfident

2. **Resolution Component** (higher is better)
   - Measures how well predictions separate outcomes
   - High resolution = model distinguishes between likely wins and losses
   - Low resolution = predictions cluster around base rate

3. **Uncertainty Component** (constant)
   - Inherent unpredictability in the data
   - Equals `base_rate × (1 - base_rate)`
   - Cannot be reduced by better modeling

### Calibration Error
**Mean Absolute Calibration Error (MACE):**
- Average deviation between predicted and actual probabilities across bins
- **<0.02:** Excellent calibration
- **0.02-0.05:** Good calibration
- **0.05-0.10:** Acceptable calibration
- **>0.10:** Poor calibration (recalibration needed)

### Reliability Diagram
Visual representation of calibration:
- **X-axis:** Mean predicted probability (binned)
- **Y-axis:** Actual frequency of positive outcomes
- **Perfect calibration:** Points lie on diagonal line
- **Overconfident:** Points below diagonal
- **Underconfident:** Points above diagonal

## Usage

### Monitor Single Week
```bash
python -m analysis.calibration_monitor --season 2025 --week 10 --generate-report
```

### Monitor Full Season
```bash
python -m analysis.calibration_monitor --season 2025 --generate-report
```

### Check for Drift
```bash
# Compare 2025 to 2024 baseline
python -m analysis.calibration_monitor --season 2025 --check-drift --baseline-season 2024 --threshold 0.02
```

### Generate Reliability Diagram Only
```bash
python -m analysis.calibration_monitor --season 2025 --plot-reliability
```

### Calculate Metrics Only
```bash
python -m analysis.calibration_monitor --season 2025
```

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

Contains statistics for each probability bin (0.0-0.1, 0.1-0.2, etc.):
- Number of predictions in bin
- Mean predicted probability
- Mean actual outcome
- Calibration error

### Reliability Diagram
`analysis/calibration_monitoring/YYYY_season_reliability.png`

Visual plot showing predicted vs actual probabilities.

### Drift Alerts JSON
`analysis/calibration_monitoring/YYYY_drift_alerts.json`

Generated when drift exceeds threshold:
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
- Overall metrics
- Brier decomposition
- Probability bin table
- Interpretation guide
- Reliability diagram

## Interpretation Guide

### Brier Score Analysis

**Good Brier Score (0.19-0.21):**
- Model is well-calibrated
- Predictions are sharp (not clustering around 50%)
- Continue monitoring for drift

**Moderate Brier Score (0.21-0.23):**
- Acceptable but room for improvement
- Check calibration component (should be <0.01)
- Consider isotonic recalibration

**Poor Brier Score (>0.23):**
- Model needs recalibration
- Check for data quality issues
- Review feature engineering

### Calibration Component Analysis

**Low Calibration (<0.01):**
- Excellent! Predicted probabilities match actual frequencies
- Model is trustworthy for decision-making

**Moderate Calibration (0.01-0.03):**
- Good calibration
- Minor adjustments may help
- Monitor for drift

**High Calibration (>0.03):**
- Poor calibration
- Model is overconfident or underconfident
- **Action required:** Apply isotonic calibration or Platt scaling

### Resolution Component Analysis

**High Resolution (>0.04):**
- Model effectively separates outcomes
- Predictions are informative
- Good discrimination between wins and losses

**Moderate Resolution (0.02-0.04):**
- Acceptable separation
- Model has some predictive power
- Consider adding features to improve

**Low Resolution (<0.02):**
- Poor separation
- Predictions cluster around base rate
- Model may not be learning meaningful patterns

### Drift Detection

**No Drift (change <0.02):**
- Model calibration is stable
- Continue normal monitoring

**Warning Drift (0.02-0.04):**
- Calibration is degrading
- Increase monitoring frequency
- Prepare for recalibration

**Critical Drift (>0.04):**
- Significant calibration degradation
- **Action required:** Recalibrate immediately
- Investigate root cause (data shift, feature drift, etc.)

## Calibration Maintenance

### Weekly Monitoring Schedule

**During Season:**
1. **Thursday (pre-games):** Lock predictions
2. **Tuesday (post-games):** Calculate calibration metrics
3. **Weekly:** Check for drift vs rolling 4-week baseline
4. **Monthly:** Generate comprehensive calibration report

**Off-Season:**
1. **Full season review:** Analyze calibration trends
2. **Recalibration:** Apply isotonic scaling if needed
3. **Baseline update:** Set new baseline for next season

### Recalibration Triggers

**Automatic recalibration when:**
- Brier score increases >0.02 vs baseline
- Calibration error >0.05
- Max calibration error >0.10
- 3+ consecutive weeks of drift warnings

**Manual recalibration when:**
- Major data source changes
- Feature engineering updates
- Model architecture changes
- New season starts

### Recalibration Methods

**1. Isotonic Regression (Recommended)**
- Non-parametric, monotonic calibration
- Preserves rank order of predictions
- Works well for NFL predictions
- Already implemented in `src/evaluation/calibrate_winprob.py`

**2. Platt Scaling**
- Logistic regression on predictions
- Parametric, assumes sigmoid relationship
- Good for small datasets
- May not preserve rank order

**3. Beta Calibration**
- Generalization of Platt scaling
- Three parameters (a, b, c)
- More flexible than Platt
- Requires more data

## Integration with Pipeline

### Automated Workflow

```bash
# 1. Lock predictions (Thursday)
python -m analysis.live_tracking --lock-predictions --season 2025 --week 10

# 2. Fetch actuals (Tuesday)
python -m analysis.live_tracking --fetch-actuals --season 2025 --week 10

# 3. Monitor calibration
python -m analysis.calibration_monitor --season 2025 --week 10 --generate-report

# 4. Check for drift
python -m analysis.calibration_monitor --season 2025 --check-drift --threshold 0.02

# 5. If drift detected, recalibrate
# (Manual step - run isotonic calibration in src/evaluation/calibrate_winprob.py)
```

### Dashboard Integration

Calibration metrics should be displayed in Streamlit dashboard:
- Current week Brier score
- Rolling 4-week calibration trend
- Reliability diagram
- Drift alerts
- Recalibration status

## Troubleshooting

### No predictions found
**Error:** `FileNotFoundError: No predictions found`

**Solution:** Ensure predictions are logged via `live_tracking.py` or available in `matchup_features.parquet`

### Matplotlib not available
**Warning:** `matplotlib not available, skipping plot`

**Solution:** Install matplotlib: `pip install matplotlib`

### High calibration error
**Issue:** Calibration error >0.05

**Causes:**
1. Model overconfident (predicting too many extremes)
2. Model underconfident (predicting too many 50/50s)
3. Data distribution shift
4. Feature drift

**Solutions:**
1. Apply isotonic recalibration
2. Review feature importance for drift
3. Check data quality
4. Retrain model if necessary

### Drift alerts every week
**Issue:** Constant drift warnings

**Causes:**
1. Threshold too strict (0.02 may be too low)
2. High week-to-week variance (small sample sizes)
3. Genuine model degradation

**Solutions:**
1. Increase threshold to 0.03-0.04
2. Use rolling 4-week average instead of single week
3. Investigate root cause of degradation

## Best Practices

### 1. Establish Baseline
- Use previous season as baseline
- Calculate metrics on 200+ predictions
- Update baseline annually

### 2. Monitor Continuously
- Check calibration weekly during season
- Generate monthly reports
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
- Log all recalibrations
- Track before/after metrics
- Maintain calibration history

## References

- **Brier Score:** G. W. Brier, "Verification of forecasts expressed in terms of probability" (1950)
- **Calibration:** A. H. Murphy, "A New Vector Partition of the Probability Score" (1973)
- **Reliability Diagrams:** D. S. Wilks, "Statistical Methods in the Atmospheric Sciences" (2011)
- **Isotonic Regression:** Zadrozny & Elkan, "Transforming classifier scores into accurate multiclass probability estimates" (2002)

## Next Steps

After implementing calibration monitoring:
1. **Phase 6:** Compare against public benchmarks (nfelo, Vegas)
2. **Phase 7:** Integrate into Streamlit dashboard
3. **Phase 8:** Automate weekly validation pipeline
4. **Phase 9:** Document complete validation methodology