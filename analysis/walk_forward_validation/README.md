# Walk-Forward Validation Results

This directory contains the results of walk-forward validation, which tests the model's performance on truly unseen data using rolling training windows.

## Methodology

Instead of a single train/test split, walk-forward validation:
1. Trains on N years of data (e.g., 2016-2020)
2. Tests on the next season (e.g., 2021)
3. Rolls forward and repeats (train 2017-2021, test 2022, etc.)

This provides multiple independent test periods to assess:
- **Consistency**: Does performance remain stable across different test seasons?
- **Overfitting**: Do metrics degrade significantly on new data?
- **Variance**: How much do results fluctuate between test periods?

## Files

### Summary Files
- `walk_forward_summary.json` - Complete results with aggregated statistics
- `winprob_by_window.csv` - Win probability metrics for each validation window
- `spread_by_window.csv` - Spread prediction metrics for each validation window

### Individual Window Predictions
- `winprob_train_YYYY_YYYY_test_YYYY.csv` - Win probability predictions for each test season
- `spread_train_YYYY_YYYY_test_YYYY.csv` - Spread predictions for each test season

## Key Metrics

### Win Probability
- **Accuracy**: Percentage of correct winner predictions
- **AUC**: Area under ROC curve (discrimination ability)
- **Brier Score**: Calibration quality (lower is better)
- **LogLoss**: Probability prediction quality (lower is better)

### Spread
- **MAE**: Mean absolute error in points
- **RMSE**: Root mean squared error in points

## Interpreting Results

**Good Performance Indicators:**
- Consistent metrics across test windows (low std deviation)
- AUC > 0.70 across all test seasons
- Brier score < 0.22 across all test seasons
- MAE < 11 points across all test seasons

**Warning Signs:**
- Large variance in metrics between windows
- Degrading performance in recent test seasons
- Metrics significantly worse than training performance

## Running Validation

```bash
# Default: 5-year training windows, test 2021-2025
python -m analysis.walk_forward_validation

# Custom configuration
python -m analysis.walk_forward_validation \
    --start-season 2016 \
    --end-season 2025 \
    --train-window-years 5 \
    --min-test-season 2021

# Quiet mode (no progress output)
python -m analysis.walk_forward_validation --quiet
```

## Example Output

```
Walk-Forward Validation Framework
================================================================================

Loaded 4850 games from 2002 to 2025

Generated 5 validation windows:
  - Train 2016-2020 (5 years) → Test 2021
  - Train 2017-2021 (5 years) → Test 2022
  - Train 2018-2022 (5 years) → Test 2023
  - Train 2019-2023 (5 years) → Test 2024
  - Train 2020-2024 (5 years) → Test 2025

================================================================================
Summary Statistics
================================================================================

Win Probability (across all test windows):
  ACCURACY  : 0.714 ± 0.023 (range: 0.685 - 0.742)
  AUC       : 0.774 ± 0.018 (range: 0.751 - 0.795)
  BRIER     : 0.215 ± 0.012 (range: 0.198 - 0.231)
  LOGLOSS   : 0.621 ± 0.035 (range: 0.578 - 0.665)

Spread (across all test windows):
  MAE       : 10.04 ± 0.45 (range: 9.42 - 10.68)
  RMSE      : 13.21 ± 0.62 (range: 12.35 - 14.01)
```

## Comparison to Single-Split Validation

The existing `forward_validation.py` uses a single split (train ≤2023, test ≥2024). Walk-forward validation provides:
- **Multiple test periods** instead of one
- **Variance estimates** to assess stability
- **Temporal robustness** by testing across different NFL eras
- **Overfitting detection** by comparing early vs late test windows

## Next Steps

After reviewing walk-forward results:
1. If metrics are consistent → Model is robust, proceed to live tracking
2. If variance is high → Investigate feature stability and model complexity
3. If recent seasons underperform → Update features for contemporary NFL trends
4. If all looks good → Implement Phase 2 (Live Prediction Tracking)