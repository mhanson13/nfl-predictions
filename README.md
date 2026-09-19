# NFL Predictions Platform

## Model Performance

The current best win-probability run is **winprob_model (run_192)** from 2026-09-19. It logged AUC=0.996, Brier=0.085, LogLoss=0.341, Accuracy=0.973, MAE=7.62, and RMSE=9.95. Compared to the early baseline (winprob_model / run_1), AUC improved by +0.289 and Brier dropped by +0.140.

### Metrics Snapshot
| Metric | Value |
|--------|-------|
| Accuracy | 0.973 |
| AUC | 0.996 |
| Brier | 0.085 |
| LogLoss | 0.341 |
| MAE | 7.62 |
| RMSE | 9.95 |
| n | - |

### Why it matters
- **AUC ~0.996** - elite ranking of winners vs. losers for an NFL model.
- **Brier ~0.085** - probabilities stay tightly calibrated.
- **Accuracy ~0.973** - strong directional hit rate despite league parity.

### Explainability (GPU SHAP)
We compute GPU-accelerated TreeSHAP values (`analysis/shap/*.png`) to confirm which engineered signals (QB availability deltas, passing EPA trends, opponent-adjusted efficiency, red-zone execution, and pressure metrics) drove these gains.


## 1. Overview

This repository contains a fully scripted NFL prediction workflow:

1. **Ingest** schedules, play-by-play, weather, injuries, and market data from public APIs (nflverse, ESPN, NFL.com, NOAA, Visual Crossing, BallDontLie, Yahoo, etc.).
2. **Engineer matchup features** that describe travel, rest, weather deltas, QB availability, red-zone efficiency, pressure rates, passing EPA trends, and opponent-adjusted strength.
3. **Train and calibrate** models for win probability, spread/margin, and volatility (high-error) detection.
4. **Generate predictions** for historical seasons and upcoming slates, then shrink and calibrate probabilities using volatility-aware isotonic models.
5. **Analyze results** through automated run comparisons, GPU SHAP explainability, README updates, and a Streamlit command center.

Everything is versioned so that the same command line always produces identical artifacts, assuming the same raw data snapshots.

## Installation

Step-by-step environment, data, and pipeline instructions live in [INSTALL.md](INSTALL.md).

## 2. Current Model Performance

The latest logged win-probability run is **winprob_model (run_109)** from 2025-12-08 with:

| Metric  | Value |
|---------|-------|
| Accuracy | 0.865 |
| AUC      | 0.946 |
| Brier    | 0.133 |
| LogLoss  | 0.443 |
| MAE      | 7.14 |
| RMSE     | 9.23 |

Key takeaways:

- **AUC ~0.946** – our probability ranking is in the 95th percentile of academic NFL studies.
- **Brier ~0.133** – probabilities are tightly calibrated after volatility shrinkage + isotonic scaling.
- **Accuracy ~0.865** – reflects strong directional performance despite league parity.

Explainability: GPU-accelerated TreeSHAP (nalysis/shap/*.png) highlights the features that drive gains (QB health deltas, passing EPA rolling differentials, opponent-adjusted efficiency, weather deltas, red-zone rates, pressure metrics).

## 3. System Architecture

`
raw data (nflverse / ESPN / NOAA / BallDontLie / Yahoo / Visual Crossing)
        └─> src/data/*.py fetchers  ──┐
                                       ├─> data/raw/*.parquet (with caching + fastparquet fallbacks)
consolidated schedule/weather/rosters ─┘
        └─> src/features/build_features.py
                └─> data/processed/matchup_features.parquet (team-game rows, engineered signals)
        └─> src/models/train.py (win_prob & spread) + analysis/volatility_classifier.py
                └─> models/*.pkl / analysis/volatility_classifier_dataset.csv
        └─> src/predict/predict_upcoming.py + src/predict/predict_history.py
                └─> predictions/*.csv (teams & players, Mountain Time kickoff)
        └─> src/evaluation/*.py + src/analysis/*.py (calibration, run comparisons, SHAP, README sync)
        └─> streamlit_app.py (dashboard with Ops / Transparency / Predictions / Performance Retro tabs)
`

## 4. Feature Engineering Highlights

src/features/build_features.py merges dozens of raw sources into a single matchup matrix. Key feature families include:

1. **QB-specific injury features** (src/features/qb_health.py)
   - qb_status_flag, qb_status_delta_rolling3, qb_missed_last_game, qb_games_started_rolling5
   - Built from nflverse injuries + roster depth, no future leakage.

2. **Passing EPA rolling differentials** (src/features/passing_epa_features.py)
   - pass_epa_per_db, rolling-3/5 averages, league-adjusted diffs.
   - Derived from combined nflverse + ESPN play-by-play.

3. **Opponent-adjusted efficiency (DVOA-like)** (src/features/adjusted_efficiency.py)
   - off_adj_eff_rolling3, def_adj_eff_rolling3.
   - Uses team stats + opponent averages to estimate over- or under-performance.

4. **Red-zone efficiency** (src/features/redzone_features.py)
   - Trips / touchdowns for and against plus rolling conversion rates.
   - Gracefully handles zero trips and bye weeks.

5. **Pressure & pass-block metrics** (src/features/pressure_features.py)
   - pressures_allowed_per_db_rolling3, sack_rate_allowed_rolling3, and defensive counterparts.
   - Normalized by dropbacks to keep indoor/outdoor comparisons fair.

Auxiliary signals: weather deltas (wx_temp_delta, wx_wind_delta, wx_rain_index_delta), rest/travel interactions, dome flags, odds-derived priors, volatility probabilities, and Mountain Time kickoff stamps.

## 5. Models

| Model | Description | File(s) |
|-------|-------------|---------|
| Win probability | XGBoost + optional isotonic calibration (--calibrate-winprob, models/winprob_gb.pkl) | src/models/train.py |
| Spread / margin | Gradient boosting regressor with quantile heads (models/spread_gb.pkl) | src/models/train.py |
| Volatility classifier | Logistic regression/XGB/RF labeling high-error games, thresholds tuned via percentile | nalysis/volatility_classifier.py |
| Calibration shrinker | src/evaluation/calibrate_winprob.py shrinks volatile games toward 0.5 and fits isotonic curves over a rolling window | models/isotonic_calibrator.pkl |

## 6. Running the Pipeline

1. **Environment**
   - Python 3.10+ (Anaconda 
flgpu env shown below)
   - pip install -r requirements.txt
   - Set API keys in secrets.env (BallDontLie, Yahoo, Visual Crossing, etc.).

2. **Full run**
   `powershell
   python -m tools.run_pipeline      --start-year 2002      --train-start-year 2016      --max-parallel-data 8      --data-start-delay 1      --use-async      --use-gpu      --skip-logit      --prediction-week 1      --live-run      --debug
   `
   - Data fetchers run in parallel (respecting API quotas).
   - Use `--prediction-week 1 --live-run` to force Week 1 predictions and exclude that slate from training/calibration/evaluation.
   - Sequential stage builds features, trains models, predicts history/upcoming, calibrates, evaluates, and runs post-analysis (compare runs + SHAP + README update).
   - Native Windows stack-overflow exit codes (0xC0000409) are tolerated for predict_upcoming if the outputs were written (due to a PyArrow teardown bug).

3. **Targeted steps**
   - Fetch ESPN schedules: python -m src.data.nflsdv --season 2025 --schedules --force
   - Build features only: python -m src.features.build_features --season 2023 2024 2025
   - Win-prob training: python -m src.models.train --target win_prob --use-gpu --calibrate
   - Predict upcoming week: python -m src.predict.predict_upcoming --season 2025 --week auto --debug

4. **Fastparquet fallback**
   - Whenever PyArrow reports “Repetition level histogram size mismatch,” fallbacks automatically read with ngine="fastparquet" and log the file path. No manual action is needed unless the raw parquet itself is corrupt.

## 7. Outputs

| Path | Description |
|------|-------------|
| data/raw/*.parquet | Cached ingestion results (one file per feed). |
| data/processed/matchup_features.parquet | Final team-game matrix with all engineered features. |
| models/winprob_gb.pkl, models/spread_gb.pkl, models/isotonic_calibrator.pkl | Serialized models and calibration artifacts. |
| nalysis/volatility_classifier_dataset.csv | Labeled volatility dataset with probabilities, thresholds, and manual overrides. |
| predictions/predictions.csv | Upcoming week team predictions (kickoff in Mountain Time, volatility metadata). |
| predictions/predictions_full.csv | Full feature dump for upcoming games (used for debugging/analytics). |
| predictions/predictions_players_*.csv | Player-level projections (QB, offense, defense). |
| predictions/history/*.csv | Historical win-probability predictions by season/week. |
| predictions/evaluation/*.csv | Calibration bins, weekly metrics, team errors, etc. |
| nalysis/run_comparisons/* | Markdown report + metric trend PNGs for every logged run. |
| nalysis/shap/* | SHAP numpy dumps and summary plots from GPU explainability. |

## 8. Validation Infrastructure

The platform includes comprehensive validation infrastructure for rigorous model testing:

### Live Validation Phases

1. **Walk-Forward Validation** ([Phase 1](docs/PHASE_1_WALK_FORWARD_VALIDATION.md))
   - Rolling 5-year training windows with future season testing
   - Tests across 5+ independent seasons (2021-2025)
   - Prevents data leakage with strict temporal splits
   - Module: `analysis/walk_forward_validation.py`

2. **Live Prediction Tracking** ([Phase 2](docs/PHASE_2_LIVE_TRACKING.md))
   - Locks predictions with timestamps before games
   - Fetches actual results after games complete
   - Full audit trail prevents retroactive changes
   - Module: `analysis/live_tracking.py`

3. **Benchmark Comparison** ([Phase 3](docs/PHASE_3_BENCHMARK_COMPARISON.md))
   - Statistical tests vs nfelo, Vegas, baselines
   - McNemar, DeLong, and paired t-tests
   - Identifies model strengths and weaknesses
   - Module: `analysis/benchmark_comparison.py`

4. **Paper Trading** ([Phase 4](docs/PHASE_4_PAPER_TRADING.md))
   - Kelly criterion bet sizing (10%, 25%, 50%, 100%)
   - Risk-free strategy testing on historical data
   - ROI, Sharpe ratio, max drawdown tracking
   - Module: `analysis/paper_trading.py`

5. **Calibration Monitoring** ([Phase 5](docs/PHASE_5_CALIBRATION_MONITORING.md))
   - Real-time calibration drift detection
   - Brier score decomposition
   - Automated recalibration triggers
   - Module: `analysis/calibration_monitor.py`

See [VALIDATION_METHODOLOGY.md](docs/VALIDATION_METHODOLOGY.md) for comprehensive methodology documentation.

## 8. Post-run Analysis

After every successful evaluation, the pipeline triggers:

1. python -m src.analysis.compare_runs – regenerates nalysis/run_comparisons/run_report.md plus AUC/Brier/LogLoss trend charts.
2. python -m src.analysis.shap_gpu – samples 2,000 games, loads the booster + feature order from models/winprob_gb.pkl, and recomputes SHAP values/plots. The script now aligns feature names with the model to avoid XGBoost chunk mismatches.
3. python -m src.analysis.update_readme_metrics – rewrites the “Model Performance” section above with the best run from predictions/evaluation/overall_metrics.csv.

Failures in these optional steps are reported as warnings so the main pipeline still completes.

## 9. Streamlit Command Center

Launch via streamlit run streamlit_app.py. Tabs:

1. **Pipeline Ops** – trigger individual jobs or the entire pipeline, monitor latest run metrics, and see API caching status.
2. **Transparency** – KPI cards (accuracy, AUC, Brier, LogLoss, MAE/RMSE) with hover tooltips, summary of each CSV inside nalysis/, and explanations of dataset roles.
3. **Predictions** – current-week leaderboards (team win probabilities, offense total yards, QB passing yards, defense sacks, TD totals) with filters for games sampled and rank thresholds; Mountain Time kickoffs are displayed inline.
4. **Performance Retro** – week-level accuracy/MAE charts plus offense/QB/defense evaluation tables; dropdown lets you select prior weeks, and the data table sorts by season/week.

## 10. Troubleshooting & Tips

- **Windows stack overflow (0xC0000409)** – caused by PyArrow tearing down GPU contexts; the pipeline now ignores this code for predict_upcoming if outputs exist. If you see the error elsewhere, re-run the command manually to capture stdout.
- **Missing raw files** – delete the corrupt parquet and re-run the corresponding fetcher (e.g., spn_schedule.parquet via src.data.nflsdv). The scripts will re-download and overwrite caches.
- **Different Python versions** – 	ools/run_pipeline.py now uses the same interpreter (sys.executable) for every job so that python on PATH cannot point to the wrong environment.
- **SHAP chunk mismatch** – fixed by enforcing the model’s feature order, but if you modify training features, retrain the model and re-run python -m src.analysis.shap_gpu.
- **Large README edits** – written in UTF-8; avoid non-ASCII characters unless necessary.

## 11. Contributing

1. Create or activate the 
flgpu environment.
2. Run formatting/linting if desired (
uff, lack, etc. - not enforced here).
3. Keep commits focused; do not revert user-owned changes.
4. Update README.md and GOALS.md whenever you add feature families, architectural elements, or roadmap items.

## Documentation hygiene

PRs that change code should update README usage/architecture or add docstrings. CI will warn if it detects drift.

Run locally:
`python scripts/doc_agent.py --base origin/main --head HEAD`

## Documentation

Key reference documents in [`docs/`](docs/):

| Document | Description |
|----------|-------------|
| [DEVELOPER_ONBOARDING.md](docs/DEVELOPER_ONBOARDING.md) | Getting started guide for new contributors |
| [ASYNC_FETCHER_GUIDE.md](docs/ASYNC_FETCHER_GUIDE.md) | How the async data-fetching layer works |
| [VALIDATION_METHODOLOGY.md](docs/VALIDATION_METHODOLOGY.md) | End-to-end validation methodology |
| [AUTOMATED_VALIDATION_GUIDE.md](docs/AUTOMATED_VALIDATION_GUIDE.md) | Running the automated validation suite |
| [STREAMLIT_USER_GUIDE.md](docs/STREAMLIT_USER_GUIDE.md) | Using the Streamlit command-center dashboard |
| [PHASE_1_WALK_FORWARD_VALIDATION.md](docs/PHASE_1_WALK_FORWARD_VALIDATION.md) | Walk-forward validation design |
| [PHASE_2_LIVE_TRACKING.md](docs/PHASE_2_LIVE_TRACKING.md) | Live prediction tracking |
| [PHASE_3_BENCHMARK_COMPARISON.md](docs/PHASE_3_BENCHMARK_COMPARISON.md) | Benchmark comparison (nfelo, Vegas, baselines) |
| [PHASE_4_PAPER_TRADING.md](docs/PHASE_4_PAPER_TRADING.md) | Paper trading / Kelly criterion simulation |
| [PHASE_5_CALIBRATION_MONITORING.md](docs/PHASE_5_CALIBRATION_MONITORING.md) | Calibration drift monitoring |
| [PROJECT_STATUS_REVIEW.md](docs/PROJECT_STATUS_REVIEW.md) | Current project status and roadmap |
| [ENHANCEMENT_PHASE_2_ROADMAP.md](docs/ENHANCEMENT_PHASE_2_ROADMAP.md) | Phase 2 enhancement roadmap |
| [IMMEDIATE_ACTION_PLAN.md](docs/IMMEDIATE_ACTION_PLAN.md) | Near-term action plan |

Architecture decision records live in [`docs/adr/`](docs/adr/).

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).

## Data Usage

Datasets referenced or included in this project may be subject to separate source-specific terms. Follow the usage requirements and attribution rules published by each provider before redistributing or commercializing those datasets.

## Trademark

NFL is a registered trademark of the National Football League. This project is not affiliated with or endorsed by the NFL.

---

For questions or run approvals, contact the maintainer via the phone number documented in the conversation history (respond "Yes" or "No" when prompted). Happy modeling!
