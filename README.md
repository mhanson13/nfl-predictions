# NFL Predictions Platform

## Model Performance

The latest validated win-probability evaluation run is **winprob_model (run_331)** from 2026-10-08. It logged AUC=0.703, Brier=0.218, LogLoss=0.625, Accuracy=0.652, F1=0.691, MAE=10.14, and RMSE=13.08. Compared to the earliest current-schema baseline (winprob_model / run_212), AUC changed by -0.005 and Brier changed by -0.001 (lower Brier is better).

### Metrics Snapshot
| Metric | Value |
|--------|-------|
| Accuracy | 0.652 |
| AUC | 0.703 |
| Brier | 0.218 |
| LogLoss | 0.625 |
| MAE | 10.14 |
| RMSE | 13.08 |
| n | 2283 |

### Why it matters
- **AUC ~0.703** - ranking quality for winners vs. losers.
- **Brier ~0.218** - probability calibration/error for game winners.
- **F1 ~0.691** - balance between precision and recall on home-win calls.

### Explainability (GPU SHAP)
We compute GPU-accelerated TreeSHAP values (`analysis/shap/*.png`) to confirm which engineered signals (QB availability deltas, passing EPA trends, opponent-adjusted efficiency, red-zone execution, and pressure metrics) drove these gains.


## 1. Overview

This repository contains a fully scripted NFL prediction workflow:

1. **Ingest** schedules, play-by-play, weather, injuries, player props, and market data from nflverse, ESPN, NFL.com, NOAA, Visual Crossing, BallDontLie, PropLine, The Odds API, and Yahoo when approved.
2. **Engineer matchup features** that describe travel, rest, weather deltas, QB availability, red-zone efficiency, pressure rates, passing EPA trends, and opponent-adjusted strength.
3. **Train and calibrate** models for win probability, spread/margin, volatility detection, and player-prop markets.
4. **Generate predictions** for historical seasons and upcoming slates, with live-run guards that exclude already-started games.
5. **Analyze results** through feature ablation, market benchmarking, CLV diagnostics, GPU SHAP explainability, README updates, and MattyTheBookie publishing.

Raw data, processed data, model files, predictions, and generated analysis outputs are intentionally ignored by Git. Source code, tests, docs, and pipeline definitions are versioned; generated artifacts are reproducible from the pipeline and stay local unless published to MattyTheBookie.

## Installation

Step-by-step environment, data, and pipeline instructions live in [INSTALL.md](INSTALL.md).

## 2. Current Operating State

The current validated model is usable but below the aspirational targets in [GOALS.md](GOALS.md). Treat older claims of near-perfect AUC/Brier values as stale unless they are reproduced by the current evaluation pipeline.

Current highlights:

- BallDontLie is the preferred commercial feed; SportsDataIO is a fallback and is disabled by default unless explicitly enabled.
- BallDontLie `bdl_*` features are generated and ablated, but kept out of production training unless `--include-bdl-features` is passed because recent Week 4/Week 5 ablations did not show repeatable lift.
- Team prediction outputs include market-policy guardrails so downstream consumers can distinguish strong moneyline policy picks, review-only moneyline edges, blocked spread signals, and no-bet rows.
- Player-prop models publish QB, offense, defense, and novelty CSVs with line consensus, injury/roster validation, line variance, over-probability, EV, and conservative edge-policy fields.
- Market benchmarking publishes ROI, model-vs-Vegas calibration, spread key-number diagnostics, CLV detail/summary artifacts, and optional per-game CLV fields in `analysis/market_benchmark.csv` when evaluated rows match pick-time snapshots.

## 3. System Architecture

```text
raw data (nflverse / ESPN / NOAA / Visual Crossing / BallDontLie / PropLine / Odds API / Yahoo)
        -> src/data/*.py fetchers
        -> data/raw/*.parquet (cached locally; ignored by Git)
        -> src/data/player_identity.py + src/player_availability.py
        -> data/processed/player_identity.parquet + current_player_availability.parquet
        -> src/features/build_features.py + src/features/balldontlie_team_form.py
        -> data/processed/matchup_features.parquet
        -> src/models/train.py + analysis/volatility_classifier.py
        -> models/*.pkl
        -> src/player_props/*.py
        -> predictions/player_props_*.csv + predictions/evaluation/player_props/*.csv
        -> src/predict/predict_upcoming.py + src/predict/predict_history.py
        -> predictions/*.csv
        -> analysis.market_clv / analysis.market_roi / analysis.market_blend / analysis.model_metrics_export
        -> analysis/*.csv (generated locally; ignored by Git)
        -> tools.publish_mtb_csvs + tools.import_mtb_predictions
        -> C:\Code\mtb\data\prediction-csvs\...
```

## 4. Feature Engineering Highlights

`src/features/build_features.py` merges dozens of raw sources into a single matchup matrix. Key feature families include:

1. **QB-specific injury features** (`src/features/qb_health.py`)
   - qb_status_flag, qb_status_delta_rolling3, qb_missed_last_game, qb_games_started_rolling5
   - Built from nflverse injuries + roster depth, no future leakage.

2. **Passing EPA rolling differentials** (`src/features/passing_epa_features.py`)
   - pass_epa_per_db, rolling-3/5 averages, league-adjusted diffs.
   - Derived from combined nflverse + ESPN play-by-play.

3. **Opponent-adjusted efficiency (DVOA-like)** (`src/features/adjusted_efficiency.py`)
   - off_adj_eff_rolling3, def_adj_eff_rolling3.
   - Uses team stats + opponent averages to estimate over- or under-performance.

4. **Red-zone efficiency** (`src/features/redzone_features.py`)
   - Trips / touchdowns for and against plus rolling conversion rates.
   - Gracefully handles zero trips and bye weeks.

5. **Pressure & pass-block metrics** (`src/features/pressure_features.py`)
   - pressures_allowed_per_db_rolling3, sack_rate_allowed_rolling3, and defensive counterparts.
   - Normalized by dropbacks to keep indoor/outdoor comparisons fair.

Auxiliary signals: weather deltas (wx_temp_delta, wx_wind_delta, wx_rain_index_delta), rest/travel interactions, dome flags, odds-derived priors, volatility probabilities, and Mountain Time kickoff stamps.

## 5. Models

| Model | Description | File(s) |
|-------|-------------|---------|
| Win probability | XGBoost + optional isotonic calibration (`--calibrate-winprob`, `models/winprob_gb.pkl`) | `src/models/train.py` |
| Spread / margin | Gradient boosting regressor with quantile heads (`models/spread_gb.pkl`) | `src/models/train.py` |
| Volatility classifier | Random forest labeling high-log-loss games by default; margin-error labels are analysis-only | `analysis/volatility_classifier.py` |
| Calibration shrinker | Shrinks volatile games toward 0.5 and fits isotonic curves over a rolling window | `src/evaluation/calibrate_winprob.py`, `models/isotonic_calibrator.pkl` |
| Player props | Split offense/defense/novelty models with historical line joins, over-probability calibration, edge backtests, and quality reports | `src/player_props/*.py` |
| Market policy | Model-vs-Vegas benchmarking, CLV diagnostics, market-aware blending, spread calibration, and policy gates | `analysis/market_roi.py`, `analysis/market_clv.py`, `analysis/market_blend.py` |

## 6. Running the Pipeline

1. **Environment**
   - Python 3.10+; the active local environment is typically the Anaconda `nflgpu` env.
   - Install dependencies with `pip install -r requirements.txt`.
   - Set API keys in `secrets.env` for BallDontLie, PropLine, Visual Crossing, The Odds API, Yahoo when approved, and any optional fallback feeds.

2. **Full run**
   ```powershell
   python -m tools.run_pipeline `
     --start-year 2017 `
     --train-start-year 2017 `
     --max-parallel-data 8 `
     --data-start-delay 1 `
     --use-async `
     --use-gpu `
     --skip-logit `
     --prediction-week 5 `
     --live-run `
     --publish-mtb `
     --mtb-repo-dir C:\Code\mtb `
     --debug
   ```
   - Replace `5` with the target NFL week.
   - Data fetchers run in parallel, with `--max-parallel-data 8` and `--data-start-delay 1` as the current operating defaults.
   - Use `--prediction-week 5 --live-run` with the target week substituted to force a live slate, exclude that week from training/calibration/evaluation, and avoid regenerating predictions for games that already kicked off.
   - Use `--publish-mtb` to copy current prediction, evaluation, ROI, and volatility CSVs into the local MattyTheBookie repo and refresh its normalized JSON.
   - BallDontLie is preferred when `BALLDONTLIE_API_KEY` exists. SportsDataIO is skipped unless `--enable-sportsdataio` is passed.
   - PropLine player-prop odds run for explicit prediction weeks when `PROPLINE_API_KEY` is configured.
   - Sequential stages build features, train team and prop models, predict history/upcoming, evaluate, run market CLV/ROI/blend diagnostics, export website metrics, compare runs, generate SHAP artifacts, update this README, and publish to MTB when requested.
   - Native Windows stack-overflow exit codes (0xC0000409) are tolerated for predict_upcoming if the outputs were written (due to a PyArrow teardown bug).

3. **Targeted steps**
   - Refresh BallDontLie live context: `python -m src.data.balldontlie --seasons 2026 --weeks 5 --feeds active_players games standings injuries stats season_stats team_stats team_season_stats --overwrite --debug`
   - Build player identity: `python -m src.data.player_identity --debug`
   - Build features only: `python -m src.features.build_features --season 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026 --debug`
   - Win-prob training: `python -m src.models.train --target win_prob --train-start-year 2017 --use-gpu --calibrate`
   - Predict upcoming week: `python -m src.predict.predict_upcoming --season 2026 --week 5 --exclude-started --debug`
   - Rebuild market diagnostics: `python -m analysis.market_clv --season 2026 --week 5` then `python -m analysis.market_roi --debug`
   - Rebuild player-prop quality reports: `python -m src.player_props.quality_report --season 2026 --week 5 --debug`

4. **Fastparquet fallback**
   - Whenever PyArrow reports "Repetition level histogram size mismatch," fallbacks automatically read with `engine="fastparquet"` and log the file path. No manual action is needed unless the raw parquet itself is corrupt.

## 7. Outputs

| Path | Description |
|------|-------------|
| `data/raw/*.parquet` | Cached ingestion results, one file per feed; generated locally and ignored by Git. |
| `data/processed/matchup_features.parquet` | Final team-game matrix with engineered features; generated locally and ignored by Git. |
| `data/processed/player_identity.parquet` | Cross-source player identity table linking GSIS, ESPN, BallDontLie, Sportradar, and sportsbook aliases. |
| `data/processed/current_player_availability.parquet` | Weekly availability table built from ESPN, nflverse, and BallDontLie context. |
| `models/winprob_gb.pkl`, `models/spread_gb.pkl`, `models/isotonic_calibrator.pkl` | Serialized team models and calibration artifacts; ignored by Git. |
| `models/player_props/*/*_model.pkl` | Serialized player-prop market models; ignored by Git. |
| `predictions/predictions.csv` | Upcoming week team predictions with Mountain Time kickoff, market policy, and volatility metadata. |
| `predictions/predictions_full.csv` | Full feature dump for upcoming games, used for debugging and analytics. |
| `predictions/predictions_players_*.csv` | Legacy rate-based player projections for QB, offense, and defense. |
| `predictions/player_props_qb.csv`, `predictions/player_props_offense.csv`, `predictions/player_props_defense.csv`, `predictions/player_props_novelty.csv` | Trained player-prop prediction exports with sportsbook lines, quality flags, EV fields, and conservative recommendation policy. |
| `predictions/history/*.csv` | Historical win-probability predictions by season/week. |
| `predictions/evaluation/*.csv` | Team model evaluation artifacts, calibration bins, weekly metrics, team errors, feature-quality reports, and feature-family ablations. |
| `predictions/evaluation/player_props/*.csv` | Player-prop model metrics, historical line joins, over-probability diagnostics, edge backtests, weekly quality reports, line coverage, and roster-line audits. |
| `analysis/market_*.csv`, `analysis/model_metrics_week_NN.csv` | Market ROI, model-vs-Vegas benchmark, CLV, blend, spread calibration, policy, and website-ready metrics exports; generated locally and ignored by Git. |
| `analysis/run_comparisons/*` | Markdown report and metric trend PNGs for logged runs; generated locally and ignored by Git. |
| `analysis/shap/*` | SHAP numpy dumps and summary plots from GPU explainability; generated locally and ignored by Git. |
| C:\Code\mtb\data\prediction-csvs\current\*.csv | Published CSV copies for the local MattyTheBookie site when `--publish-mtb` is used. |
| C:\Code\mtb\data\predictions\current.json | Normalized MattyTheBookie prediction JSON refreshed after CSV publish unless `--skip-mtb-import` is used. |

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

## 9. Post-run Analysis

After every successful evaluation, the pipeline triggers:

1. `python -m analysis.market_clv` writes weekly pick-time-vs-close/latest movement diagnostics.
2. `python -m analysis.market_roi` writes moneyline/spread ROI, model-vs-Vegas benchmarks, CLV-enriched benchmark rows, spread calibration, and market policy artifacts.
3. `python -m analysis.market_blend` evaluates market-aware model/market probability blends.
4. `python -m analysis.model_metrics_export` writes `analysis/model_metrics_week_NN.csv` for MattyTheBookie About-page consumption.
5. `python -m src.analysis.compare_runs` regenerates `analysis/run_comparisons/run_report.md` plus AUC/Brier/LogLoss trend charts.
6. `python -m src.analysis.shap_gpu` samples evaluated games, loads the booster and feature order from `models/winprob_gb.pkl`, and recomputes SHAP values/plots.
7. `python -m src.analysis.update_readme_metrics` rewrites the "Model Performance" section above with the latest valid run from `predictions/evaluation/overall_metrics.csv`.

Failures in these optional steps are reported as warnings so the main pipeline still completes.

## 10. Streamlit Command Center

Launch via streamlit run streamlit_app.py. Tabs:

1. **Pipeline Ops** - trigger individual jobs or the entire pipeline, monitor latest run metrics, and see API caching status.
2. **Transparency** - KPI cards (accuracy, AUC, Brier, LogLoss, MAE/RMSE) with hover tooltips, summary of generated `analysis/` outputs, and explanations of dataset roles.
3. **Predictions** - current-week leaderboards for team win probabilities, player projections, player props, defensive sacks, and touchdown markets with filters for games sampled and rank thresholds; Mountain Time kickoffs are displayed inline.
4. **Performance Retro** - week-level accuracy/MAE charts plus offense/QB/defense evaluation tables; dropdown lets you select prior weeks, and the data table sorts by season/week.

## 11. Troubleshooting & Tips

- **Windows stack overflow (0xC0000409)** - caused by PyArrow tearing down GPU contexts; the pipeline now ignores this code for predict_upcoming if outputs exist. If you see the error elsewhere, re-run the command manually to capture stdout.
- **Missing raw files** - delete the corrupt parquet and re-run the corresponding fetcher. The scripts will re-download and overwrite caches.
- **Different Python versions** - `tools/run_pipeline.py` uses the same interpreter (`sys.executable`) for every job so that `python` on PATH cannot point to the wrong environment.
- **SHAP chunk mismatch** - fixed by enforcing the model's feature order, but if you modify training features, retrain the model and re-run `python -m src.analysis.shap_gpu`.
- **Dirty Git status after pipeline runs** - `data/`, `models/`, `predictions/`, and generated `analysis` artifacts are ignored. If Git is dirty after a run, it should usually be source/docs/tests rather than regenerated outputs.
- **Large README edits** - written in UTF-8; avoid non-ASCII characters unless necessary.

## 12. Contributing

1. Create or activate the `nflgpu` environment.
2. Run formatting/linting if desired (`ruff`, `black`, etc. - not enforced here).
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
| [PLAYER_PROP_PREDICTION_PLAN.md](docs/PLAYER_PROP_PREDICTION_PLAN.md) | Player prop prediction goals, target markets, novelty bets, and implementation plan |
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
