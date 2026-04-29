# Streamlit Dashboard User Guide

## Overview

The NFL Analytics Command Center is a comprehensive Streamlit dashboard that provides operational control, transparency, predictions, and performance retrospectives in one place. This guide covers all features and how to use them effectively.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Dashboard Overview](#dashboard-overview)
3. [Tab 1: Pipeline Ops](#tab-1-pipeline-ops)
4. [Tab 2: Transparency](#tab-2-transparency)
5. [Tab 3: Upcoming Predictions](#tab-3-upcoming-predictions)
6. [Tab 4: Performance Retro](#tab-4-performance-retro)
7. [Tab 5: Live Validation](#tab-5-live-validation)
8. [Troubleshooting](#troubleshooting)
9. [Best Practices](#best-practices)

---

## Getting Started

### Prerequisites

- Python 3.8+ installed
- All project dependencies installed (`pip install -r requirements.txt`)
- Data pipeline has been run at least once

### Starting the Dashboard

```bash
# From project root directory
streamlit run streamlit_app.py
```

The dashboard will automatically open in your default web browser at `http://localhost:8501`

### First-Time Setup

If this is your first time running the dashboard:

1. **Run the pipeline** to generate initial data:
   ```bash
   python -m tools.run_pipeline --debug
   ```

2. **Generate predictions** for upcoming games:
   ```bash
   python -m src.predict.predict_upcoming --season 2025 --week 11 --overwrite --debug
   ```

3. **Refresh the dashboard** by clicking the refresh button or pressing `R` in the browser

---

## Dashboard Overview

The dashboard consists of 5 main tabs:

| Tab | Purpose | Key Features |
|-----|---------|--------------|
| **Pipeline Ops** | Run commands and check system status | Command runner, artifact status |
| **Transparency** | View model diagnostics and metrics | Latest metrics, analysis files |
| **Upcoming Predictions** | See predictions for next week | Team and player projections |
| **Performance Retro** | Historical performance analysis | Weekly trends, player evaluation |
| **Live Validation** | Real-time validation metrics | Calibration, CLV, benchmarks |

### Navigation

- Click tab names at the top to switch between sections
- Use browser back/forward buttons to navigate
- Press `R` to refresh the entire dashboard
- Use `Ctrl+R` (Windows/Linux) or `Cmd+R` (Mac) for hard refresh

---

## Tab 1: Pipeline Ops

### Purpose
Operational control center for running data pipelines and checking system health.

### Features

#### 1. Artifact Status
**Location:** Left column

**What it shows:**
- List of key prediction and model files
- Last modified timestamp
- File age in hours
- File size in KB
- Missing file indicators

**How to use:**
- Check "age_hours" to see data freshness
- Look for "missing" status to identify gaps
- Use this to determine if pipeline needs to run

#### 2. Latest Metrics Summary
**Location:** Left column, below artifacts

**Displays:**
- Accuracy (% correct predictions)
- AUC (discrimination ability)
- Brier Score (calibration quality)
- LogLoss (confidence penalty)

**Interpretation:**
- **Accuracy:** >0.65 is good, >0.70 is excellent
- **AUC:** >0.70 is good, >0.75 is excellent
- **Brier:** <0.22 is good, <0.20 is excellent
- **LogLoss:** Lower is better, <0.65 is good

#### 3. Command Runner
**Location:** Right column

**How to use:**
1. Select a template command from dropdown
2. Edit command in text area if needed
3. Click "Execute" button
4. Wait for completion (spinner shows progress)
5. Review stdout/stderr output
6. Check exit code (0 = success)

**Available Templates:**
- **Run full pipeline:** Complete data refresh and model training
- **Refresh data only:** Update data without retraining
- **Rebuild matchup features:** Regenerate feature engineering
- **Retrain win probability model:** Train model with latest data
- **Generate upcoming predictions:** Create predictions for next week
- **Generate historical predictions:** Backfill historical predictions

**Tips:**
- Commands run from project root directory
- Adjust `--season` and `--week` parameters as needed
- Use `--debug` flag for verbose output
- Check logs if command fails

---

## Tab 2: Transparency

### Purpose
Model diagnostics, feature importance, and analysis documentation.

### Features

#### 1. Latest Calibrated Run Metrics
**Location:** Top section, 6 metric cards

**Metrics Displayed:**
- **N (Sample Size):** Number of predictions evaluated
- **Accuracy:** Percentage of correct winner predictions
- **AUC:** Area Under ROC Curve (discrimination)
- **Brier:** Mean squared error of probabilities
- **LogLoss:** Logarithmic loss (confidence penalty)
- **MAE Margin:** Mean absolute error in point spread

**How to interpret:**
- Hover over metrics for detailed tooltips
- Compare to benchmarks in GOALS.md
- Track changes over time
- Look for consistency across runs

#### 2. Analysis Folder Reference
**Location:** Bottom section, expandable files

**Available Files:**
- **Feature Importance:** Which features matter most
- **Market ROI:** Backtest results vs closing odds
- **Volatility Classifier:** High-variance game detection
- **Feature Lift:** Improvement from new features

**How to use:**
1. Click file name to expand
2. Read description to understand purpose
3. View data preview (first 200 rows)
4. Use for debugging and model understanding

**Key Files:**
- `feature_importance_winprob_full.csv` - Current model drivers
- `market_roi_moneyline.csv` - Betting performance vs odds
- `volatility_classifier_metrics.json` - Uncertainty detection

---

## Tab 3: Upcoming Predictions

### Purpose
View predictions for the upcoming week's games.

### Features

#### 1. Player Projection Leaders
**Location:** Top section with filters

**Filters:**
- **Minimum games sampled:** Filter low-confidence projections (default: 3)
- **Maximum player rank:** Show only top N players (default: 2)

**How to use:**
1. Adjust filters to refine results
2. View top 10 players in each category
3. Check "games_sampled" for confidence level
4. Review "player_rank" for position depth

**Projection Categories:**

**Offensive Leaders:**
- Projected total yards (rushing + receiving)
- Projected total TDs
- Games sampled (confidence indicator)

**QB Leaders:**
- Projected passing yards
- Projected passing TDs
- Projected interceptions

**Defensive Leaders:**
- Projected sacks
- Projected QB hits
- Games sampled

**Tips:**
- Higher "games_sampled" = more reliable projection
- Lower "player_rank" = starter (more playing time)
- Compare projections to betting lines
- Use for DFS lineup construction

#### 2. Team Win Probabilities
**Location:** Bottom section, full table

**Columns:**
- **favorite:** Team favored to win
- **favorite_prob:** Win probability for favorite
- **favorite_margin:** Expected point margin
- **home_team / away_team:** Matchup teams
- **home_win_prob:** Home team win probability
- **pred_home_margin:** Predicted home margin
- **volatility_prob:** Uncertainty indicator
- **kickoff_mt:** Game time (Mountain Time)
- **pick_expl:** Explanation of prediction

**How to use:**
1. Sort by "favorite_prob" to see most confident picks
2. Check "volatility_prob" for high-variance games
3. Compare "pred_home_margin" to betting spreads
4. Use "pick_expl" to understand model reasoning

**Interpretation:**
- **Win Prob >70%:** Strong favorite
- **Win Prob 55-70%:** Moderate favorite
- **Win Prob 50-55%:** Toss-up game
- **Volatility >0.15:** High uncertainty (weather, injuries, etc.)

---

## Tab 4: Performance Retro

### Purpose
Historical performance analysis and prediction evaluation.

### Features

#### Sub-Tab 1: Team Performance
**What it shows:**
- Weekly accuracy trends
- Brier score over time
- MAE margin errors
- Season-by-season breakdown

**How to use:**
1. Review weekly performance table
2. Check accuracy trends (should be stable)
3. Monitor Brier score (should be <0.22)
4. Look for MAE margin consistency

**Charts:**
- **Accuracy Trend:** Should hover around 65-70%
- **MAE Margin Trend:** Should be around 10 points

#### Sub-Tab 2: Offense Evaluation
**What it shows:**
- Offensive player projection accuracy
- Actual vs predicted yards
- Error analysis by player

**How to use:**
1. Select season and week from dropdowns
2. Review prediction errors
3. Identify systematic biases
4. Check "abs_error" for accuracy

**Columns:**
- **projected_total_yards:** Model prediction
- **actual_total_yards:** Actual performance
- **total_yards_error:** Difference (actual - predicted)
- **abs_error:** Absolute error magnitude

#### Sub-Tab 3: QB Evaluation
**What it shows:**
- QB passing yard predictions
- Actual vs predicted performance
- Error analysis by quarterback

**How to use:**
1. Select season and week
2. Review QB projection accuracy
3. Identify over/under predictions
4. Check for systematic errors

#### Sub-Tab 4: Defense Evaluation
**What it shows:**
- Defensive player projections (sacks, QB hits)
- Actual vs predicted performance
- Error analysis by player

**How to use:**
1. Select season and week
2. Review defensive projection accuracy
3. Check sack predictions
4. Analyze QB hit projections

---

## Tab 5: Live Validation

### Purpose
Real-time validation metrics, calibration monitoring, and market edge analysis.

### Features

#### Season Selector
**Location:** Top of tab

**How to use:**
1. Select season from dropdown (2021-2025)
2. Dashboard automatically loads data for selected season
3. All sections update to show season-specific metrics

---

### Section 1: Current Season Performance

**What it shows:**
- Total games tracked this season
- Average accuracy across all weeks
- Average Brier score
- Average AUC

**Metrics:**
- **Total Games:** Number of predictions made
- **Avg Accuracy:** Mean weekly accuracy
- **Avg Brier Score:** Mean calibration quality
- **Avg AUC:** Mean discrimination ability

**Weekly Trend Chart:**
- X-axis: Week number
- Y-axis: Accuracy and Brier score
- Shows performance consistency over season

**How to interpret:**
- Accuracy should be stable (65-70%)
- Brier should be consistent (<0.22)
- Look for sudden drops (investigate causes)
- Upward accuracy trend = model improving

**If no data:**
Run live tracking:
```bash
python -m analysis.live_tracking --season 2025 --weekly-cycle --week 10
```

---

### Section 2: Calibration Health

**What it shows:**
- Brier score decomposition
- Calibration error metrics
- Drift alerts
- Reliability diagram

**Metrics:**

**Brier Score:**
- Current value
- Delta from baseline (0.20)
- Green delta = better than baseline
- Red delta = worse than baseline

**Calibration Error:**
- Mean absolute error across probability bins
- <0.02 = Excellent (green)
- 0.02-0.05 = Good
- >0.05 = Needs attention (red)

**Resolution:**
- Ability to separate outcomes
- >0.04 = High (good discrimination)
- 0.02-0.04 = Moderate
- <0.02 = Low (predictions too similar)

**Max Cal Error:**
- Worst bin calibration error
- <0.10 = OK
- >0.10 = High (investigate)

**Drift Alerts:**
- Appears if calibration degrading
- Shows metric, drift amount, severity
- Warning = monitor closely
- Critical = recalibrate immediately

**Reliability Diagram:**
- X-axis: Predicted probability
- Y-axis: Actual frequency
- Perfect calibration = diagonal line
- Points below line = overconfident
- Points above line = underconfident

**How to use:**
1. Check Brier score delta (should be negative)
2. Verify calibration error <0.02
3. Review drift alerts (none is best)
4. Examine reliability diagram for bias

**If no data:**
```bash
python -m analysis.calibration_monitor --season 2025 --generate-report
```

---

### Section 3: Closing Line Value (CLV)

**What it shows:**
- Model edge vs betting market
- CLV distribution
- Win rate vs break-even

**Metrics:**

**Avg CLV:**
- Average closing line value
- Positive = model has edge
- >2% = significant edge
- <0% = no edge (model worse than market)

**Positive CLV %:**
- Percentage of games with positive CLV
- >50% = model finding value
- <50% = market beating model

**Win Rate:**
- Actual win percentage
- Compare to break-even (52.4% for -110 odds)
- Above break-even = profitable
- Below break-even = losing

**High CLV Games:**
- Count of games with CLV >2%
- These are best betting opportunities
- More high CLV games = more edge

**CLV Distribution Chart:**
- Shows spread of CLV values
- Should be centered above 0
- Wide distribution = inconsistent edge
- Narrow distribution around positive = consistent edge

**How to interpret:**
- **Avg CLV +3%:** Excellent edge
- **Avg CLV +1-2%:** Good edge
- **Avg CLV 0-1%:** Marginal edge
- **Avg CLV <0%:** No edge (don't bet)

**If no data:**
```bash
python -m analysis.clv_tracker --season 2025 --generate-report
```

---

### Section 4: Paper Trading Results

**What it shows:**
- Simulated betting performance
- Strategy comparison
- Risk-adjusted returns

**Strategy Table Columns:**
- **Strategy:** Conservative, Moderate, Kelly, Aggressive
- **ROI:** Return on investment (%)
- **Win Rate:** Percentage of winning bets
- **Sharpe Ratio:** Risk-adjusted return
- **Max Drawdown:** Largest peak-to-trough decline
- **Total Bets:** Number of bets placed
- **Final Bankroll:** Ending capital

**How to interpret:**

**ROI:**
- >5%: Excellent
- 2-5%: Good
- 0-2%: Marginal
- <0%: Losing

**Sharpe Ratio:**
- >2.0: Exceptional
- 1.0-2.0: Good
- 0.5-1.0: Acceptable
- <0.5: Poor

**Max Drawdown:**
- <10%: Low risk
- 10-20%: Moderate risk
- 20-30%: High risk
- >30%: Very high risk

**Strategy Selection:**
- **Conservative:** Safest, slowest growth
- **Moderate:** Recommended for most users
- **Kelly:** Higher variance, optimal growth
- **Aggressive:** Highest risk/reward

**If no data:**
```bash
python -m analysis.paper_trading --season 2025 --compare-strategies
```

---

### Section 5: Benchmark Comparison

**What it shows:**
- Model vs public benchmarks
- Statistical significance tests
- Competitive positioning

**Benchmark Table:**
- **Our Model:** Your model's performance
- **Vegas Consensus:** Closing odds implied probabilities
- **nfelo:** FiveThirtyEight Elo ratings
- **Spread Baseline:** Simple spread-to-probability conversion
- **Home Favorite:** Always pick home if favored

**Metrics:**
- **Accuracy:** % correct predictions
- **AUC:** Discrimination ability
- **Brier Score:** Calibration quality

**Statistical Significance Table:**
- **Comparison:** Which models being compared
- **Metric:** What's being tested
- **Difference:** Performance gap
- **P-value:** Statistical significance
- **Significant:** ✅ = significant, ❌ = not significant
- **Test:** Statistical test used

**How to interpret:**

**vs Vegas:**
- Tie or slightly better is excellent
- Vegas is efficient market
- Even 1-2% edge is valuable

**vs nfelo:**
- Should beat by 2-4% accuracy
- nfelo is simple baseline
- Larger gap = better model

**vs Spread Baseline:**
- Should beat by 3-5% accuracy
- Crude approximation
- Easy to beat

**P-value:**
- <0.05 = Statistically significant
- ≥0.05 = Not significant (could be luck)

**If no data:**
```bash
python -m analysis.benchmark_comparison --season 2025 --generate-report
```

---

### Section 6: Walk-Forward Validation

**What it shows:**
- Multi-season consistency
- Rolling window validation
- Overfitting detection

**Aggregated Metrics:**
- **Mean Accuracy ± Std:** Average across test windows
- **Mean AUC ± Std:** Average discrimination
- **Mean Brier ± Std:** Average calibration
- **Test Windows:** Number of independent tests

**Per-Window Results Table:**
- **Test Season:** Year being predicted
- **Train Period:** Years used for training
- **Accuracy:** Test season accuracy
- **AUC:** Test season AUC
- **Brier:** Test season Brier

**How to interpret:**

**Standard Deviation:**
- Low std (<0.03) = consistent performance
- High std (>0.05) = unstable (overfitting?)

**Per-Window Consistency:**
- All windows similar = robust model
- One window much worse = investigate that season
- Trend over time = model improving/degrading

**If no data:**
```bash
python -m analysis.walk_forward_validation --start-season 2021 --end-season 2025
```

---

### Section 7: Quick Actions

**Buttons:**

**🔄 Refresh All Metrics:**
- Clears Streamlit cache
- Reloads all data from disk
- Use after running validation scripts
- Forces complete dashboard refresh

**📊 Generate Reports:**
- Info message with instructions
- Run validation scripts manually
- Future: automated report generation

**📥 Export Data:**
- Coming soon
- Will export metrics to CSV/JSON
- For external analysis

---

## Troubleshooting

### Dashboard Won't Start

**Error:** `ModuleNotFoundError: No module named 'streamlit'`

**Solution:**
```bash
pip install streamlit
# or
pip install -r requirements.txt
```

### No Data Showing

**Issue:** All sections show "No data available"

**Solution:**
1. Run the pipeline first:
   ```bash
   python -m tools.run_pipeline --debug
   ```

2. Generate predictions:
   ```bash
   python -m src.predict.predict_upcoming --season 2025 --week 11
   ```

3. Run validation modules:
   ```bash
   python -m analysis.live_tracking --season 2025 --weekly-cycle --week 10
   python -m analysis.calibration_monitor --season 2025 --generate-report
   python -m analysis.clv_tracker --season 2025 --generate-report
   python -m analysis.paper_trading --season 2025 --compare-strategies
   python -m analysis.benchmark_comparison --season 2025 --generate-report
   ```

### Dashboard is Slow

**Issue:** Dashboard takes long to load

**Solutions:**
1. Clear cache: Click "Refresh All Metrics" button
2. Reduce data size: Filter to recent seasons only
3. Close other browser tabs
4. Restart Streamlit:
   ```bash
   # Press Ctrl+C to stop
   # Then restart
   streamlit run streamlit_app.py
   ```

### Charts Not Displaying

**Issue:** Trend charts are blank

**Causes:**
1. Insufficient data (need multiple weeks)
2. Missing columns in data files
3. All values are NaN

**Solution:**
1. Check data files exist and have content
2. Run pipeline to regenerate data
3. Verify column names match expected format

### Command Runner Fails

**Issue:** Commands return non-zero exit code

**Solutions:**
1. Check stdout/stderr for error messages
2. Verify command syntax is correct
3. Ensure required data files exist
4. Check Python environment is activated
5. Run command manually in terminal to debug

---

## Best Practices

### Daily Workflow

**Morning (Before Games):**
1. Open dashboard
2. Check "Upcoming Predictions" tab
3. Review team win probabilities
4. Note high-confidence picks
5. Check volatility indicators

**Evening (After Games):**
1. Run live tracking to log results
2. Check "Live Validation" tab
3. Review accuracy for the week
4. Monitor calibration drift
5. Update CLV analysis

### Weekly Workflow

**Tuesday (Post-Weekend):**
1. Run complete validation cycle:
   ```bash
   python -m analysis.live_tracking --season 2025 --weekly-cycle --week 10
   python -m analysis.calibration_monitor --season 2025 --generate-report
   python -m analysis.clv_tracker --season 2025 --generate-report
   ```

2. Review "Live Validation" tab
3. Check for drift alerts
4. Analyze CLV trends
5. Update paper trading simulation

**Thursday (Pre-Games):**
1. Run pipeline to get latest data
2. Generate upcoming predictions
3. Review "Upcoming Predictions" tab
4. Check player projections
5. Note team win probabilities

### Monthly Workflow

**End of Month:**
1. Run walk-forward validation
2. Compare to benchmarks
3. Review paper trading results
4. Check calibration consistency
5. Update documentation

### Season Workflow

**Start of Season:**
1. Run full pipeline with all historical data
2. Retrain models
3. Generate baseline metrics
4. Set up live tracking
5. Initialize paper trading

**Mid-Season:**
1. Monitor weekly performance
2. Check for calibration drift
3. Compare to benchmarks
4. Adjust strategies if needed

**End of Season:**
1. Run complete validation suite
2. Generate annual reports
3. Compare to previous seasons
4. Document lessons learned
5. Plan improvements

### Data Management

**Keep Data Fresh:**
- Run pipeline weekly minimum
- Update predictions before each game week
- Log results after games complete
- Archive old predictions

**Monitor Data Quality:**
- Check artifact ages in Pipeline Ops
- Verify file sizes are reasonable
- Look for missing data indicators
- Validate prediction counts

**Backup Important Data:**
- Export predictions regularly
- Save validation reports
- Archive model files
- Document configuration changes

### Performance Monitoring

**Key Metrics to Track:**
1. **Accuracy:** Should be 65-70%
2. **Brier Score:** Should be <0.22
3. **Calibration Error:** Should be <0.02
4. **CLV:** Should be positive
5. **Win Rate:** Should be >52.4%

**Red Flags:**
- Accuracy drops below 60%
- Brier score increases above 0.25
- Calibration error >0.05
- Negative CLV for multiple weeks
- Win rate <50%

**When to Investigate:**
- Sudden performance drop
- Drift alerts appear
- Negative CLV trend
- Benchmark comparisons worsen
- High volatility in metrics

---

## Keyboard Shortcuts

- **R:** Refresh dashboard
- **Ctrl+R / Cmd+R:** Hard refresh (clear cache)
- **Ctrl+C:** Stop Streamlit server (in terminal)
- **Ctrl+Shift+R:** Rerun script
- **Esc:** Close sidebar (if open)

---

## Additional Resources

- **GOALS.md:** Performance targets and roadmap
- **README.md:** Project overview and setup
- **docs/VALIDATION_METHODOLOGY.md:** Complete validation guide
- **analysis/*/README.md:** Module-specific documentation

---

## Support

For issues or questions:
1. Check this user guide
2. Review module-specific READMEs
3. Check GOALS.md for context
4. Review error messages in dashboard
5. Run validation scripts manually to debug

---

**Remember:** The dashboard is a visualization tool. Data quality depends on running the underlying validation modules regularly. Keep data fresh for accurate insights!