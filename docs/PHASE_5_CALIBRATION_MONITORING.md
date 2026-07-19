# Phase 5: Calibration Monitoring System - Implementation Complete

## Overview

Phase 5 of the Live Validation Infrastructure provides real-time monitoring of prediction calibration quality. This enables early detection of calibration drift and automated triggers for recalibration, ensuring probabilities remain trustworthy over time.

## What Was Implemented

### Core Module: `analysis/calibration_monitor.py`

A comprehensive calibration monitoring system that:

1. **Tracks Calibration Quality**
   - Reliability diagrams (predicted vs actual probabilities)
   - Brier score decomposition (calibration + resolution + uncertainty)
   - Probability bin accuracy tracking
   - Confidence interval coverage analysis

2. **Detects Calibration Drift**
   - Compares current vs baseline calibration
   - Statistical tests for significant drift
   - Automated alerts when thresholds exceeded
   - Severity classification (warning/critical)

3. **Triggers Recalibration**
   - Automated recalibration when drift detected
   - Configurable thresholds and windows
   - Integration with model training pipeline
   - Audit trail of recalibration events

4. **Generates Reports**
   - Markdown calibration reports
   - Reliability plots and visualizations
   - Historical calibration trends
   - Bin-level diagnostics

## Key Features

### Brier Score Decomposition

The Brier score can be decomposed into three components:

**Brier = Calibration + Resolution - Uncertainty**

#### 1. Calibration Component
- Measures how close predicted probabilities are to actual frequencies
- Lower is better (0 = perfect calibration)
- **Target:** <0.01 for well-calibrated model

#### 2. Resolution Component
- Measures ability to separate outcomes (discrimination)
- Higher is better (more separation)
- **Target:** >0.15 for good discrimination

#### 3. Uncertainty Component
- Baseline uncertainty in the data (irreducible)
- Depends on outcome distribution
- **Typical:** ~0.25 for NFL (50/50 outcomes)

**Example:**
```
Brier Score: 0.197
├─ Calibration: 0.008 (excellent)
├─ Resolution:  0.058 (good)
└─ Uncertainty: 0.247 (typical)
```

### Reliability Diagrams

Visual representation of calibration quality:

```
Perfect Calibration (diagonal line):
  Predicted 70% → Actual 70%
  Predicted 50% → Actual 50%
  Predicted 30% → Actual 30%

Overconfident (above diagonal):
  Predicted 70% → Actual 60%
  (Model too confident)

Underconfident (below diagonal):
  Predicted 60% → Actual 70%
  (Model not confident enough)
```

### Probability Bins

Predictions grouped into bins for analysis:

| Bin | Range | Count | Predicted | Actual | Error |
|-----|-------|-------|-----------|--------|-------|
| 1 | 0.0-0.1 | 15 | 0.08 | 0.07 | 0.01 |
| 2 | 0.1-0.2 | 23 | 0.15 | 0.17 | -0.02 |
| 3 | 0.2-0.3 | 31 | 0.25 | 0.26 | -0.01 |
| ... | ... | ... | ... | ... | ... |
| 10 | 0.9-1.0 | 18 | 0.94 | 0.94 | 0.00 |

**Calibration Error:** Mean absolute difference between predicted and actual

### Drift Detection

Monitors calibration over time and alerts when drift exceeds thresholds:

**Warning Threshold:** Brier drift >0.01  
**Critical Threshold:** Brier drift >0.02

**Example Alert:**
```
⚠️ CALIBRATION DRIFT WARNING
Season: 2025, Week: 10
Metric: Brier Score
Current: 0.215
Baseline: 0.197
Drift: +0.018 (9.1%)
Threshold: 0.010
Action: Monitor closely, consider recalibration
```

## Architecture

```
Calibration Monitoring Process
==============================

Input:
  - Predictions (predictions/history/*.csv)
  - Actuals (matchup_features.parquet)
  - Baseline metrics (from training)

For each monitoring period:
  1. Load predictions and actuals
  2. Calculate calibration metrics
  3. Decompose Brier score
  4. Generate reliability diagram
  5. Check for drift vs baseline
  6. Generate alerts if needed
  7. Trigger recalibration if critical

Output:
  - analysis/calibration_monitoring/
    ├── season_YYYY_week_WW_calibration.json
    ├── season_YYYY_week_WW_reliability.png
    ├── season_YYYY_drift_alerts.json
    └── recalibration_log.json
```

## Usage

### Monitor Single Week

Check calibration for a specific week:

```bash
python -m analysis.calibration_monitor \
    --season 2025 \
    --week 10 \
    --generate-report
```

**Output:**
```
analysis/calibration_monitoring/
├── season_2025_week_10_calibration.json
└── season_2025_week_10_reliability.png
```

### Monitor Full Season

Check calibration across entire season:

```bash
python -m analysis.calibration_monitor \
    --season 2025 \
    --generate-report
```

**Output:**
```
analysis/calibration_monitoring/
├── season_2025_calibration.json
├── season_2025_reliability.png
└── season_2025_trend.png
```

### Check for Drift

Detect calibration drift and trigger alerts:

```bash
python -m analysis.calibration_monitor \
    --season 2025 \
    --check-drift \
    --threshold 0.02
```

**Output:**
```
Checking calibration drift...
✓ No significant drift detected
  Current Brier: 0.199
  Baseline Brier: 0.197
  Drift: +0.002 (within threshold)
```

### Generate Reliability Diagram

Create visual calibration plot:

```bash
python -m analysis.calibration_monitor \
    --season 2025 \
    --plot-reliability
```

**Output:**
```
analysis/calibration_monitoring/
└── season_2025_reliability.png
```

### Automated Recalibration

Trigger recalibration when drift exceeds threshold:

```bash
python -m analysis.calibration_monitor \
    --season 2025 \
    --check-drift \
    --threshold 0.02 \
    --auto-recalibrate
```

**Output:**
```
⚠️ CRITICAL DRIFT DETECTED
  Current Brier: 0.220
  Baseline Brier: 0.197
  Drift: +0.023 (exceeds threshold)

🔄 Triggering automatic recalibration...
  ✓ Isotonic calibrator retrained
  ✓ New baseline: 0.198
  ✓ Recalibration logged
```

## Output Format

### JSON Metrics

```json
{
  "season": 2025,
  "week": 10,
  "n_predictions": 14,
  "brier_score": 0.197,
  "brier_calibration": 0.008,
  "brier_resolution": 0.058,
  "brier_uncertainty": 0.247,
  "mean_predicted_prob": 0.523,
  "mean_actual_prob": 0.500,
  "calibration_error": 0.012,
  "max_calibration_error": 0.035,
  "bins": [
    {
      "bin_center": 0.05,
      "bin_range": [0.0, 0.1],
      "n_predictions": 2,
      "mean_predicted": 0.08,
      "mean_actual": 0.00,
      "calibration_error": 0.08
    }
  ]
}
```

### Drift Alert JSON

```json
{
  "season": 2025,
  "week": 10,
  "metric": "brier_score",
  "current_value": 0.215,
  "baseline_value": 0.197,
  "drift": 0.018,
  "threshold": 0.010,
  "severity": "warning",
  "timestamp": "2025-11-12T10:30:00Z",
  "action_taken": "alert_sent"
}
```

### Markdown Report

```markdown
# Calibration Monitoring Report - 2025 Week 10

## Overall Metrics

| Metric | Value | Status |
|--------|-------|--------|
| Brier Score | 0.197 | ✓ Good |
| Calibration Error | 0.012 | ✓ Good |
| Max Bin Error | 0.035 | ✓ Good |
| Mean Predicted | 0.523 | ✓ Balanced |
| Mean Actual | 0.500 | ✓ Balanced |

## Brier Decomposition

| Component | Value | Interpretation |
|-----------|-------|----------------|
| Calibration | 0.008 | Excellent (low) |
| Resolution | 0.058 | Good (high) |
| Uncertainty | 0.247 | Typical |

## Probability Bins

| Bin | Range | Count | Predicted | Actual | Error |
|-----|-------|-------|-----------|--------|-------|
| 1 | 0.0-0.1 | 2 | 0.08 | 0.00 | 0.08 |
| 2 | 0.1-0.2 | 1 | 0.15 | 0.00 | 0.15 |
| 3 | 0.2-0.3 | 2 | 0.25 | 0.50 | -0.25 |
| ... | ... | ... | ... | ... | ... |

## Drift Analysis

✓ No significant drift detected
- Current Brier: 0.197
- Baseline Brier: 0.197
- Drift: 0.000 (0.0%)
```

## Integration with Other Phases

### Phase 1: Walk-Forward Validation
- Monitor calibration across different training windows
- Detect if calibration degrades over time
- Identify optimal recalibration frequency

### Phase 2: Live Tracking
- Monitor calibration weekly during season
- Alert when live predictions drift from training
- Track calibration trends over season

### Phase 3: Benchmark Comparison
- Compare calibration to benchmark models
- Ensure model calibration is competitive
- Learn from benchmark calibration patterns

### Phase 4: Paper Trading
- Use calibration quality to adjust bet sizing
- Reduce stakes when calibration is poor
- Increase stakes when calibration is excellent

## Interpretation Guide

### Excellent Calibration
- Brier calibration component <0.01
- Mean calibration error <0.015
- Max bin error <0.05
- **Action:** Maintain current approach

### Good Calibration
- Brier calibration component 0.01-0.02
- Mean calibration error 0.015-0.025
- Max bin error 0.05-0.10
- **Action:** Monitor closely

### Poor Calibration
- Brier calibration component >0.02
- Mean calibration error >0.025
- Max bin error >0.10
- **Action:** Recalibrate immediately

### Overconfident Model
- Predicted probabilities too extreme
- High-confidence predictions often wrong
- Reliability curve above diagonal
- **Fix:** Apply Platt scaling or isotonic regression

### Underconfident Model
- Predicted probabilities too moderate
- Low-confidence predictions often right
- Reliability curve below diagonal
- **Fix:** Adjust probability scaling factor

## Recalibration Strategies

### 1. Isotonic Regression
**When to use:** Non-monotonic calibration errors  
**How it works:** Fits step function to reliability curve  
**Pros:** Flexible, no assumptions  
**Cons:** Can overfit with small data

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_model = CalibratedClassifierCV(
    base_model,
    method='isotonic',
    cv='prefit'
)
calibrated_model.fit(X_cal, y_cal)
```

### 2. Platt Scaling
**When to use:** Monotonic calibration errors  
**How it works:** Fits logistic regression to probabilities  
**Pros:** Simple, stable  
**Cons:** Assumes sigmoid shape

```python
calibrated_model = CalibratedClassifierCV(
    base_model,
    method='sigmoid',
    cv='prefit'
)
calibrated_model.fit(X_cal, y_cal)
```

### 3. Temperature Scaling
**When to use:** Model consistently over/underconfident  
**How it works:** Divides logits by temperature parameter  
**Pros:** Very simple, one parameter  
**Cons:** Limited flexibility

```python
def temperature_scale(logits, temperature):
    return logits / temperature

# Find optimal temperature
from scipy.optimize import minimize
result = minimize(
    lambda T: brier_score_loss(y_true, softmax(logits / T)),
    x0=1.0
)
optimal_temp = result.x[0]
```

### 4. Beta Calibration
**When to use:** Skewed probability distributions  
**How it works:** Fits beta distribution to probabilities  
**Pros:** Handles skew well  
**Cons:** More complex

## Performance Targets

### Calibration Metrics
- **Brier Calibration:** <0.01 (excellent), <0.02 (good)
- **Mean Calibration Error:** <0.015 (excellent), <0.025 (good)
- **Max Bin Error:** <0.05 (excellent), <0.10 (good)

### Drift Thresholds
- **Warning:** Brier drift >0.01 (0.5%)
- **Critical:** Brier drift >0.02 (1.0%)
- **Recalibration:** Brier drift >0.03 (1.5%)

### Monitoring Frequency
- **Training:** After each model retrain
- **Weekly:** During active season
- **Monthly:** During off-season
- **On-demand:** After major data changes

## Troubleshooting

### Issue: Poor calibration in specific bins

**Symptom:**
```
Bin 8 (0.7-0.8): Error = 0.15 (high)
Other bins: Error < 0.05 (good)
```

**Diagnosis:** Overconfident in 70-80% range

**Solution:**
1. Check if bin has enough samples (n>10)
2. Apply bin-specific recalibration
3. Review features driving high confidence
4. Consider ensemble with more conservative model

### Issue: Calibration drift over time

**Symptom:**
```
Week 1-5: Brier = 0.197
Week 6-10: Brier = 0.215 (+0.018)
```

**Diagnosis:** Model calibration degrading

**Solution:**
1. Check if data distribution changed
2. Retrain isotonic calibrator on recent data
3. Use rolling window for calibration
4. Investigate feature drift

### Issue: Reliability plot shows S-curve

**Symptom:**
```
Low probs: Underconfident (below diagonal)
Mid probs: Well-calibrated (on diagonal)
High probs: Overconfident (above diagonal)
```

**Diagnosis:** Non-monotonic calibration

**Solution:**
1. Use isotonic regression (not Platt scaling)
2. Increase calibration data size
3. Check for class imbalance
4. Review probability transformation

## Advanced Usage

### Custom Drift Detection

Implement custom drift detection logic:

```python
from analysis.calibration_monitor import CalibrationMonitor

monitor = CalibrationMonitor()

# Load predictions
preds = pd.read_csv('predictions.csv')
actuals = pd.read_csv('actuals.csv')

# Calculate metrics
metrics = monitor.calculate_metrics(preds, actuals)

# Custom drift check
baseline_brier = 0.197
current_brier = metrics.brier_score
drift = current_brier - baseline_brier

if drift > 0.02:
    print("⚠️ CRITICAL DRIFT")
    # Trigger recalibration
    monitor.recalibrate(preds, actuals)
elif drift > 0.01:
    print("⚠️ WARNING: Monitor closely")
else:
    print("✓ Calibration stable")
```

### Automated Monitoring Pipeline

Set up automated calibration monitoring:

```bash
#!/bin/bash
# monitor_calibration.sh

SEASON=2025
WEEK=$(date +%V)

# Run calibration check
python -m analysis.calibration_monitor \
    --season $SEASON \
    --week $WEEK \
    --check-drift \
    --threshold 0.02 \
    --auto-recalibrate

# Send alert if drift detected
if [ $? -ne 0 ]; then
    echo "Calibration drift detected!" | mail -s "Alert" admin@example.com
fi
```

### Integration with CI/CD

Add calibration checks to automated testing:

```yaml
# .github/workflows/calibration.yml
name: Calibration Check
on:
  schedule:
    - cron: '0 0 * * 2'  # Every Tuesday
jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Check calibration
        run: |
          python -m analysis.calibration_monitor \
            --season 2025 \
            --check-drift \
            --threshold 0.02
      - name: Fail if drift detected
        run: exit $?
```

## Future Enhancements

### Planned Features
- [ ] Multi-model calibration comparison
- [ ] Adaptive recalibration frequency
- [ ] Confidence interval calibration
- [ ] Quantile calibration for spreads
- [ ] Real-time calibration dashboard

### Research Directions
- [ ] Online calibration (update after each game)
- [ ] Context-specific calibration (weather, rest, etc.)
- [ ] Ensemble calibration methods
- [ ] Uncertainty-aware calibration

## References

### Calibration Methods
- Platt, J. (1999). "Probabilistic outputs for support vector machines"
- Zadrozny, B., & Elkan, C. (2002). "Transforming classifier scores into accurate multiclass probability estimates"
- Guo, C., et al. (2017). "On calibration of modern neural networks"

### Evaluation Metrics
- Brier, G. W. (1950). "Verification of forecasts expressed in terms of probability"
- Murphy, A. H. (1973). "A new vector partition of the probability score"
- DeGroot, M. H., & Fienberg, S. E. (1983). "The comparison and evaluation of forecasters"

## Summary

Phase 5 provides continuous monitoring of prediction calibration quality. This enables:

✅ **Trust** - Know when probabilities are reliable  
✅ **Early Detection** - Catch calibration drift before it impacts decisions  
✅ **Automation** - Trigger recalibration automatically  
✅ **Transparency** - Visual and statistical calibration diagnostics  

**Status:** ✅ Implementation Complete  
**Next Steps:** Set up Phase 2A infrastructure for enhancement implementation