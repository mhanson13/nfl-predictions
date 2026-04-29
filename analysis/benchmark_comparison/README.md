# Benchmark Comparison

Compare model performance against public benchmarks and baselines with statistical significance testing.

## Overview

This module compares your NFL prediction model against:
1. **nfelo** - FiveThirtyEight Elo ratings
2. **Vegas Consensus** - Closing odds implied probabilities
3. **Spread Baseline** - Simple spread-to-probability conversion
4. **Home Favorite** - Always pick home team if favored

Includes statistical tests to determine if differences are significant.

## Benchmarks Explained

### 1. nfelo (FiveThirtyEight Elo)
**Method:** Elo rating system with ~65 point home field advantage

**Formula:** `P(home wins) = 1 / (1 + 10^(-(elo_diff + 65)/400))`

**Strengths:**
- Simple, interpretable
- Proven track record
- Publicly available

**Weaknesses:**
- Doesn't account for injuries, weather, etc.
- Slow to adapt to team changes
- No game-specific context

### 2. Vegas Consensus
**Method:** Closing moneyline odds converted to implied probability

**Formula:** 
- Favorite (-150): `150 / (150 + 100) = 0.60`
- Underdog (+200): `100 / (200 + 100) = 0.333`

**Strengths:**
- Incorporates all public information
- Efficient market hypothesis
- Real money backing

**Weaknesses:**
- Includes vig (bookmaker margin)
- May not reflect true probabilities
- Can be influenced by public betting

### 3. Spread Baseline
**Method:** Convert point spread to win probability

**Formula:** `P(home wins) = 0.50 + (spread × -0.03)`

**Example:** Home favored by 7 → 0.50 + (7 × 0.03) = 0.71

**Strengths:**
- Simple, interpretable
- Uses market information
- No complex modeling

**Weaknesses:**
- Linear assumption (not accurate at extremes)
- Ignores game context
- Rough approximation

### 4. Home Favorite
**Method:** Always predict home team wins if favored (spread < 0)

**Formula:** `P(home wins) = 1.0 if spread < 0 else 0.0`

**Strengths:**
- Simplest possible baseline
- Easy to beat (hopefully!)

**Weaknesses:**
- No probability calibration
- Binary predictions only
- Ignores magnitude of advantage

## Statistical Tests

### McNemar Test (Accuracy)
**Purpose:** Compare two classifiers on same dataset

**Null Hypothesis:** Both models have same error rate

**Interpretation:**
- P < 0.05: Significant difference in accuracy
- Focuses on disagreements between models
- Accounts for paired nature of predictions

### DeLong Test (AUC)
**Purpose:** Compare ROC curves (discrimination ability)

**Null Hypothesis:** Both models have same AUC

**Interpretation:**
- P < 0.05: Significant difference in discrimination
- Tests if one model better separates outcomes
- More powerful than comparing AUC point estimates

### Paired T-Test (Brier Score)
**Purpose:** Compare calibration quality

**Null Hypothesis:** Both models have same mean squared error

**Interpretation:**
- P < 0.05: Significant difference in calibration
- Tests per-prediction errors
- Sensitive to both calibration and resolution

## Usage

### Compare Single Season

```bash
python -m analysis.benchmark_comparison --season 2025 --generate-report
```

### Compare Multiple Seasons

```bash
python -m analysis.benchmark_comparison --start-season 2023 --end-season 2025 --generate-report
```

### Compare Specific Benchmark

```bash
python -m analysis.benchmark_comparison --season 2025 --benchmark vegas
```

## Output Files

### Comparison JSON
`analysis/benchmark_comparison/YYYY_comparison.json`

```json
{
  "season": 2025,
  "model": {
    "name": "Our Model",
    "n_predictions": 267,
    "accuracy": 0.714,
    "auc": 0.774,
    "brier_score": 0.2015,
    "log_loss": 0.621
  },
  "benchmarks": {
    "vegas": {
      "name": "Vegas Consensus",
      "accuracy": 0.682,
      "auc": 0.745,
      "brier_score": 0.2134,
      "log_loss": 0.658
    },
    "nfelo": {
      "name": "nfelo",
      "accuracy": 0.671,
      "auc": 0.728,
      "brier_score": 0.2201,
      "log_loss": 0.682
    }
  },
  "comparisons": [
    {
      "model_a": "Our Model",
      "model_b": "Vegas Consensus",
      "metric": "accuracy",
      "value_a": 0.714,
      "value_b": 0.682,
      "difference": 0.032,
      "p_value": 0.023,
      "significant": true,
      "test_name": "McNemar"
    }
  ]
}
```

### Comparison Report
`analysis/benchmark_comparison/YYYY_report.md`

Markdown report with:
- Model performance summary
- Benchmark performance table
- Statistical comparison results
- Significance indicators
- Interpretation guide

## Interpretation Guide

### Accuracy Comparison

| Difference | Interpretation |
|------------|----------------|
| +0.05 (5%) | Excellent improvement |
| +0.03 (3%) | Good improvement |
| +0.01 (1%) | Marginal improvement |
| 0.00 | No difference |
| -0.01 (-1%) | Marginal decline |

**Note:** Even 1-2% accuracy improvement is valuable in NFL prediction.

### AUC Comparison

| Difference | Interpretation |
|------------|----------------|
| +0.05 | Substantial improvement |
| +0.03 | Good improvement |
| +0.01 | Marginal improvement |
| 0.00 | No difference |

**Note:** AUC measures discrimination ability, not calibration.

### Brier Score Comparison

| Difference | Interpretation |
|------------|----------------|
| -0.02 | Excellent improvement (lower is better) |
| -0.01 | Good improvement |
| -0.005 | Marginal improvement |
| 0.00 | No difference |
| +0.01 | Decline |

**Note:** Brier combines calibration and resolution.

### Statistical Significance

**P-value < 0.05:**
- Difference is statistically significant
- Less than 5% chance of observing this difference by random chance
- Can confidently say models perform differently

**P-value >= 0.05:**
- Difference is not statistically significant
- Could be due to random variance
- Need more data or larger effect size

**Important:** Statistical significance ≠ practical significance. A 0.5% accuracy improvement might be statistically significant but not practically useful.

## Expected Results

### Realistic Expectations

**vs Vegas Consensus:**
- **Accuracy:** Tie or slightly better (+0-2%)
- **AUC:** Slightly better (+0.01-0.03)
- **Brier:** Slightly better (-0.01-0.02)
- **Why:** Vegas is efficient but includes vig and public bias

**vs nfelo:**
- **Accuracy:** Better (+2-4%)
- **AUC:** Better (+0.03-0.05)
- **Brier:** Better (-0.02-0.03)
- **Why:** nfelo is simple, doesn't use game-specific features

**vs Spread Baseline:**
- **Accuracy:** Better (+3-5%)
- **AUC:** Much better (+0.05-0.08)
- **Brier:** Much better (-0.03-0.05)
- **Why:** Spread baseline is crude approximation

**vs Home Favorite:**
- **Accuracy:** Much better (+10-15%)
- **AUC:** Much better (+0.15-0.20)
- **Brier:** Much better (-0.05-0.08)
- **Why:** Home favorite is simplest possible baseline

### Red Flags

**If your model is worse than Vegas:**
- Check data quality and feature engineering
- Verify no data leakage
- Consider if model is overfit to training data

**If your model is worse than nfelo:**
- Major issue - nfelo is simple baseline
- Review model architecture
- Check for bugs in prediction pipeline

**If your model is worse than spread baseline:**
- Critical issue - spread baseline is crude
- Fundamental problem with model
- Start debugging immediately

## Troubleshooting

### Missing benchmark data

**Issue:** `KeyError: 'closing_odds_home'` or `KeyError: 'home_elo'`

**Solution:** Ensure required columns exist in `matchup_features.parquet`:
- Vegas: `closing_odds_home`, `closing_odds_away`
- nfelo: `home_elo`, `away_elo`
- Spread: `spread`

### No significant differences

**Issue:** All p-values > 0.05

**Causes:**
1. Sample size too small (need 200+ predictions)
2. Models truly perform similarly
3. High variance in predictions

**Solutions:**
1. Combine multiple seasons
2. Accept that models are similar
3. Focus on practical differences, not just statistical

### Unexpected results

**Issue:** Model performs worse than expected

**Diagnosis:**
1. Check data quality (missing values, outliers)
2. Verify no data leakage (future information in features)
3. Review feature importance (are key features being used?)
4. Compare to walk-forward validation results

**Solutions:**
1. Clean data and re-run
2. Remove leaky features
3. Add missing important features
4. Retrain model if necessary

## Best Practices

### 1. Use Multiple Seasons
- Single season can be noisy
- Combine 2-3 seasons for robust comparison
- Check consistency across seasons

### 2. Focus on Practical Significance
- Statistical significance ≠ practical value
- 0.5% accuracy improvement may not be worth complexity
- Consider cost/benefit of model improvements

### 3. Compare to Multiple Benchmarks
- Don't just compare to one benchmark
- Vegas is hardest to beat
- nfelo and spread baseline should be easier

### 4. Document Assumptions
- Note which benchmarks are available
- Document any data limitations
- Explain missing comparisons

### 5. Update Regularly
- Re-run comparisons each season
- Track trends over time
- Detect if model edge is degrading

## Integration with Validation Pipeline

### Workflow

```bash
# 1. Run walk-forward validation
python -m analysis.walk_forward_validation --start-season 2021 --end-season 2025

# 2. Compare to benchmarks
python -m analysis.benchmark_comparison --start-season 2021 --end-season 2025 --generate-report

# 3. Review results
cat analysis/benchmark_comparison/2025_report.md
```

### Dashboard Integration

Display in Streamlit:
- Current season comparison table
- Multi-season trend charts
- Statistical significance indicators
- Benchmark performance over time

## Advanced Topics

### Custom Benchmarks

Add your own benchmark:

```python
def calculate_custom_prob(row):
    # Your custom logic
    return probability

custom_prob = df.apply(calculate_custom_prob, axis=1).values
custom_metrics = calculate_benchmark_metrics(y_true, custom_prob, "Custom")
```

### Ensemble Comparison

Compare ensemble of models:

```python
ensemble_prob = (model_prob + vegas_prob + nfelo_prob) / 3
ensemble_metrics = calculate_benchmark_metrics(y_true, ensemble_prob, "Ensemble")
```

### Stratified Comparison

Compare by subgroups:

```python
# By team strength
favorites = df[df['spread'] < -3]
underdogs = df[df['spread'] > 3]

# Run comparison on each group
```

## References

- **McNemar Test:** McNemar, Q. "Note on the sampling error of the difference between correlated proportions or percentages" (1947)
- **DeLong Test:** DeLong, E. R., et al. "Comparing the areas under two or more correlated receiver operating characteristic curves" (1988)
- **FiveThirtyEight Elo:** https://fivethirtyeight.com/methodology/how-our-nfl-predictions-work/
- **Vegas Odds:** Levitt, S. D. "Why are gambling markets organised so differently from financial markets?" (2004)

## Next Steps

After implementing benchmark comparison:
1. **Phase 7:** Integrate into Streamlit dashboard
2. **Phase 8:** Automate weekly validation pipeline
3. **Phase 9:** Document complete validation methodology

---

**Remember:** The goal is not to beat Vegas by 10% (impossible), but to demonstrate your model has edge and is improving over time. Even 1-2% improvement over Vegas is excellent and potentially profitable.