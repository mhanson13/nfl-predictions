# NFL Predictions Goals

Last reviewed: 2026-09-23

This file is the live roadmap and status board for the NFL prediction pipeline. Keep it tied to current artifacts, not historical plans. Do not mark work complete unless the code, generated output, or documentation exists in the repo.

## Maintenance Rules

- Update this file after any production pipeline run that materially changes metrics, data sources, model behavior, publishing flow, or roadmap priority.
- Use current artifact paths when marking a goal complete.
- Keep historical implementation notes in `docs/`; keep this file focused on current goals and status.
- Do not keep inactive phase labels. If a task is still worth doing, keep it on the roadmap. If not, remove it.

## Latest Validated Run

Source artifacts:

- `predictions/evaluation/overall_metrics.csv`
- `analysis/market_benchmark.csv`
- `analysis/market_benchmark_summary.csv`
- `analysis/market_disagreement_roi.csv`
- `analysis/market_roi_spread.csv`
- `analysis/market_roi_spread_edge_bins.csv`
- `analysis/model_metrics_week_03.csv`
- `data/processed/player_prop_labels.parquet`
- `data/processed/player_prop_features_offense.parquet`
- `data/processed/player_prop_features_defense.parquet`
- `predictions/evaluation/player_props/baseline_metrics.csv`
- `predictions/evaluation/player_props/baseline_predictions.csv`
- `predictions/evaluation/player_props/model_metrics.csv`
- `predictions/evaluation/player_props/model_predictions.csv`
- `predictions/evaluation/player_props/model_vs_baseline.csv`
- `predictions/evaluation/player_props/prop_quality_week_03.csv`
- `predictions/evaluation/player_props/prop_quality_summary_week_03.csv`
- `predictions/evaluation/player_props/prop_line_coverage_week_03.csv`
- `predictions/player_props_qb.csv`
- `predictions/player_props_offense.csv`
- `predictions/player_props_defense.csv`
- `predictions/player_props_novelty.csv`
- `data/raw/propline_player_props_2026_wk03.parquet`
- `data/processed/player_prop_historical_line_eval.parquet`
- `data/processed/player_prop_over_probability_eval.parquet`
- `predictions/evaluation/player_props/historical_line_join_summary.csv`
- `predictions/evaluation/player_props/over_probability_metrics.csv`
- `predictions/evaluation/player_props/over_probability_bins.csv`
- `data/raw/espn_rosters_<season>_wkNN.parquet`
- `data/raw/espn_depthcharts_<season>_wkNN.parquet`
- `data/raw/espn_injuries_<season>_wkNN.parquet`
- `data/processed/current_player_availability.parquet`
- `analysis/run_comparisons/run_report.md`
- `README.md`

Latest full evaluation snapshot from 2026-09-21:

| Metric | Current | Near-Term Goal |
|---|---:|---:|
| Games / samples | 2250 | Track per run |
| Accuracy | 0.654 | >= 0.670 |
| F1 | 0.696 | >= 0.710 |
| AUC | 0.707 | >= 0.720 |
| Brier | 0.218 | <= 0.210 |
| LogLoss | 0.627 | <= 0.610 |
| Margin MAE | 10.24 | <= 9.75 |
| Margin RMSE | 13.27 | <= 12.75 |

Current read: the model is usable but below the earlier aspirational AUC/Brier targets. Treat prior AUC values near 1.0 as invalid or stale unless backed by the current evaluation pipeline.

## What Is In Place

- Core pipeline command with async data ingestion, GPU training, live-week exclusion, evaluation, SHAP, run comparison, README refresh, and optional MTB publishing.
- Current operational defaults: `--start-year 2017`, `--max-parallel-data 8`, `--data-start-delay 1`, `--live-run`, and `--publish-mtb` when exporting to the website repo.
- NFLVerse, ESPN, NFL.com, NOAA/weather, Visual Crossing, BallDontLie, PropLine, Yahoo, SportsDataIO fallback, and local reference-data ingestion paths.
- BallDontLie is the preferred commercial feed. SportsDataIO remains a fallback and should not run by default when BallDontLie is configured.
- Current team prediction outputs:
  - `predictions/predictions.csv`
  - `predictions/predictions_full.csv`
  - `predictions/history/*.csv`
  - `predictions/evaluation/*.csv`
- Current rate-based player projection outputs:
  - `predictions/predictions_players_qb.csv`
  - `predictions/predictions_players_offense.csv`
  - `predictions/predictions_players_defense.csv`
- First player-prop label table exists for QB passing yards, RB rushing yards, WR/TE receiving yards, and defensive player sacks:
  - `data/processed/player_prop_labels.parquet`
- Split player-prop feature tables exist so offensive and defensive player models do not share one mixed feature matrix:
  - `data/processed/player_prop_features_offense.parquet`
  - `data/processed/player_prop_features_defense.parquet`
- First player-prop baseline evaluation exists for current rate-based projection CSVs:
  - `predictions/evaluation/player_props/baseline_metrics.csv`
  - `predictions/evaluation/player_props/baseline_predictions.csv`
- First trained player-prop model evaluation exists for the first four markets:
  - `predictions/evaluation/player_props/model_metrics.csv`
  - `predictions/evaluation/player_props/model_predictions.csv`
  - `predictions/evaluation/player_props/model_vs_baseline.csv`
  - `models/player_props/offense/*_model.pkl`
  - `models/player_props/defense/*_model.pkl`
- Weekly player-prop quality reports exist for the live board:
  - `predictions/evaluation/player_props/prop_quality_week_NN.csv`
  - `predictions/evaluation/player_props/prop_quality_summary_week_NN.csv`
  - `predictions/evaluation/player_props/prop_line_coverage_week_NN.csv`
- Weekly trained-model player-prop prediction exports exist for the first four markets:
  - `predictions/player_props_qb.csv`
  - `predictions/player_props_offense.csv`
  - `predictions/player_props_defense.csv`
  - `predictions/player_props_novelty.csv` currently writes the target schema only; novelty markets are not modeled yet.
- Player-prop line snapshots are available through PropLine:
  - `data/raw/propline_player_props_<season>_wkNN.parquet`
  - Week 3 currently matched 22 trained-model output rows with lines after sportsbook team aliases and safer player-name matching were added.
- Manual historical player-prop odds backfill scaffolding exists through The Odds API:
  - `src.data.odds_api_historical`
  - event discovery outputs: `data/raw/oddsapi_historical_events_<snapshot>.parquet`
  - event market discovery outputs: `data/raw/oddsapi_historical_event_markets_<season>_wkNN_<event>_<snapshot>.parquet`
  - event player-prop outputs: `data/raw/oddsapi_historical_player_props_<season>_wkNN_<event>_<snapshot>.parquet`
- Resumable Odds API historical player-prop backfill controller exists:
  - `src.data.odds_api_backfill`
  - manifest output: `data/processed/oddsapi_player_prop_backfill_manifest.parquet`
  - aggregate events output: `data/raw/oddsapi_historical_events_backfill.parquet`
  - aggregate event-market output: `data/raw/oddsapi_historical_event_markets_backfill.parquet`
  - aggregate player-prop odds output: `data/raw/oddsapi_historical_player_props_backfill.parquet`
  - normalized line output: `data/processed/oddsapi_historical_player_prop_lines.parquet`
  - 2025-current backfill completed for 317 games and normalized 24,202 sportsbook line rows across DraftKings, FanDuel, and Hard Rock.
- Historical prop-line join artifact exists:
  - `data/processed/player_prop_historical_line_eval.parquet`
  - `predictions/evaluation/player_props/historical_line_join_summary.csv`
  - Current join matched 11,361 sportsbook line rows to walk-forward model predictions and actuals.
- Per-market historical over-probability evaluation exists:
  - `src.player_props.over_probability`
  - `data/processed/player_prop_over_probability_eval.parquet`
  - `predictions/evaluation/player_props/over_probability_metrics.csv`
  - `predictions/evaluation/player_props/over_probability_bins.csv`
  - Current first-pass residual-CDF probabilities are published for diagnostics, but market implied probabilities still beat them overall on Brier and LogLoss.
- Weekly player-prop CSVs now include line-specific probability fields when historical calibration data and a matched line are available:
  - `prob_over`
  - `prob_over_raw`
  - `prob_edge`
  - `prob_over_method`
  - `prob_over_sample_size`
  - `over_break_even_probability`, `under_break_even_probability`, `over_ev`, `under_ev`, `edge_side`, `edge_odds`, `edge_break_even_probability`, and `edge_ev`
  - Price-aware EV fields account for sportsbook vig, but remain diagnostic until over-probability calibration beats sportsbook implied probabilities.
- Current-week ESPN roster, depth chart, and injury snapshots are now wired into the pipeline for explicit prediction weeks:
  - `src.data.espn_rosters`
  - `data/raw/espn_rosters_<season>_wkNN.parquet`
  - `data/raw/espn_depthcharts_<season>_wkNN.parquet`
  - `data/raw/espn_injuries_<season>_wkNN.parquet`
- Unified current-player availability is now built from ESPN, nflverse, and BallDontLie sources:
  - `src.player_availability`
  - `data/processed/current_player_availability.parquet`
- Weekly player-prop CSVs can now include roster validation fields:
  - `current_team`
  - `active_current_roster`
  - `roster_validation_flag`
  - `roster_validation_source`
  - `availability_status`
  - `depth_chart_position`
  - `depth_chart_rank`
- MattyTheBookie publishing flow:
  - `tools.publish_mtb_csvs`
  - `tools.import_mtb_predictions`
  - `C:\Code\mtb\data\prediction-csvs\current\*.csv`
  - `C:\Code\mtb\data\prediction-csvs\seasons\<season>\week_NN\*.csv`
  - `C:\Code\mtb\data\predictions\current.json`
  - `C:\Code\mtb\data\predictions\seasons\<season>\week_NN.json`
- Schema models and validation framework exist in `src/utils/pydantic_schemas.py`, with validation hooks in several ingestion jobs.
- Run comparison, market ROI, volatility classifier, calibration, evaluation, and SHAP tooling exist and are integrated into the pipeline where appropriate.
- Market benchmark artifacts exist for model-vs-no-vig-Vegas comparison:
  - `analysis/market_benchmark.csv`
  - `analysis/market_benchmark_summary.csv`
  - `analysis/market_disagreement_roi.csv`
  - `analysis/market_roi_spread_edge_bins.csv`
- Website-ready model metrics export exists for MattyTheBookie About-page consumption:
  - `analysis/model_metrics_week_NN.csv`

## Current Gaps

- `GOALS.md` and some older docs previously contained stale metrics and completed-status claims. This file should now be the source of current roadmap truth.
- Live tracking, CLV, paper trading, calibration monitoring, and benchmark modules exist, but their generated artifacts are not currently maintained as live weekly outputs.
- Current measured coverage is low. Treat any old >70% coverage claim as stale until a fresh full-suite coverage run proves otherwise.
- Player prop labels, split feature tables, baseline scoring, trained-model evaluation, weekly trained-model CSV exports, optional PropLine sportsbook ingestion, historical line joins, and first-pass over-probability diagnostics now exist for the first four markets. Remaining prop gaps are confidence-tier calibration, prop-line coverage expansion, edge backtesting, and true novelty-market models.
- Current-week roster validation is newly integrated and needs Week 3/Week 4 production-output review for false positives, missing ESPN matches, and player-name/team alias mismatches.
- Yahoo Fantasy remains blocked pending API access approval.
- SportsDataIO is retained only as a fallback while BallDontLie coverage is verified.
- Market benchmarking now publishes model-vs-Vegas probability comparisons and disagreement ROI. CLV is still not maintained as a live weekly artifact.
- Corrected spread ROI no longer shows broad profitability. Treat the 3-to-5 point edge bucket as a research lead, not a production betting rule.

## Operating Roadmap

### 1. Metrics And Documentation Hygiene

- [x] Auto-update README headline metrics from latest valid evaluation rows.
- [x] Generate run-comparison report after pipeline runs.
- [ ] Add season-by-season metric table to README or a dedicated evaluation doc.
- [ ] Add a lightweight changelog for major pipeline/data/model changes.
- [ ] Update `GOALS.md` after each meaningful pipeline run or roadmap change.
- [ ] Reconcile old docs that still claim stale AUC/Brier/coverage values.

### 2. Data Source Direction

- [x] Prefer BallDontLie over SportsDataIO when `BALLDONTLIE_API_KEY` is present.
- [x] Keep SportsDataIO disabled by default unless `--enable-sportsdataio` is passed.
- [x] Document current datasource roles in `datasources.md`.
- [ ] Verify BallDontLie coverage is sufficient for current-season player/team context.
- [ ] Remove or downgrade SportsDataIO dependencies once BallDontLie replacements are proven.
- [ ] Enable Yahoo Fantasy feeds only after API approval is confirmed.

### 3. Team Model Improvement

- [x] Use walk-forward historical predictions for calibration/evaluation.
- [x] Exclude the live prediction week from training/calibration/evaluation during live runs.
- [x] Apply volatility-aware calibration/shrinkage.
- [ ] Improve current AUC from ~0.707 to >=0.720 without leakage.
- [ ] Improve Brier from ~0.218 to <=0.210.
- [ ] Improve margin MAE from ~10.24 to <=9.75.
- [ ] Re-check feature lift after every material feature-family change.
- [x] Rework volatility classifier to RF, log-loss-only labels, and 0.5 percentile default.
- [ ] Add a separate spread/margin-volatility model instead of mixing margin-error labels into probability volatility.
- [ ] Improve probability calibration, especially high-confidence picks where LogLoss/Brier penalties are largest.
- [ ] Build calibration diagnostics by model confidence tier and identify overconfident ranges that need shrinkage or recalibration.

### 4. Market Benchmarking And Betting Edge

Current read from the Week 3 evaluation set:

- SU model accuracy is 65.0% over 2250 games.
- No-vig Vegas favorite accuracy is about 66.2% over the same games.
- Model and Vegas favorite agree on about 87.0% of games.
- In 293 favorite-disagreement games, the model is 45.4% SU and Vegas is 54.6%.
- Overall model-side moneyline ROI is -3.1%; model-vs-Vegas disagreement ROI is -0.6% at threshold 0.00.
- Corrected spread ROI is not broadly positive. The best current overall spread signals are home edge >3 points at +4.7% ROI and the 3-to-5 point absolute edge bucket at +7.5% ROI; larger edge buckets are mixed and have smaller samples.

Roadmap:

- [x] Add a `market_benchmark.csv` artifact comparing model probability to no-vig Vegas implied probability for every evaluated game.
- [x] Publish model-vs-Vegas favorite agreement/disagreement metrics by season and overall.
- [x] Add moneyline disagreement ROI by probability-edge threshold.
- [x] Fix and test `analysis.market_roi` spread-line sign convention.
- [x] Recompute spread ROI using corrected edge: `pred_home_margin - market_home_margin`.
- [x] Add edge-bin ROI tables for corrected spread edges, including by season and minimum sample thresholds.
- [ ] Compare model probability directly to no-vig Vegas probability by probability bin and publish the calibration/edge table.
- [ ] Improve model performance in favorite-disagreement games versus Vegas before treating disagreements as actionable.
- [ ] Build and backtest a market-aware blend or meta-model using model probability, no-vig moneyline probability, market spread, and current model features.
- [ ] Track whether any market-aware blend adds genuine edge versus simply mirroring Vegas favorite picks.
- [ ] Keep spread edge analysis as a separate betting-signal track from straight-up winner model accuracy.
- [ ] Add closing-line movement / CLV fields to `market_benchmark.csv` when reliable opening and closing snapshots are available.
- [ ] Promote CLV reports into the weekly validation flow before using ROI results as decision criteria.

### 5. Player Prop Prediction Goals

Current state: legacy player CSVs remain available as rate-based projections, and the first trained-model player prop CSVs now publish weekly for QB passing yards, RB rushing yards, WR/TE receiving yards, and defensive player sacks. PropLine line ingestion is wired through DraftKings, Hard Rock, and FanDuel when `PROPLINE_API_KEY` is configured, with line-consensus/variance diagnostics, injury quality controls, and current-roster validation included in the weekly prop CSVs. Goal: expand line availability reporting, calibrate confidence tiers, backtest prop edges, broaden markets, and score the outputs weekly.

#### First Implementation Slice

- [x] Build normalized player-game label table for the first four markets from historical weekly player stats: `data/processed/player_prop_labels.parquet`.
- [x] Build split offensive and defensive player-prop feature tables:
  - `data/processed/player_prop_features_offense.parquet`
  - `data/processed/player_prop_features_defense.parquet`
- [x] Score existing rate-based projections as baseline models: `predictions/evaluation/player_props/baseline_metrics.csv`.
- [x] Train/evaluate first high-volume markets:
  - [x] QB passing yards
  - [x] RB rushing yards
  - [x] WR/TE receiving yards
  - [x] Defensive player sacks
- [x] Publish baseline evaluation artifacts under `predictions/evaluation/player_props/`.
- [x] Publish trained-model evaluation artifacts under `predictions/evaluation/player_props/`.
- [x] Build weekly trained-model player prop prediction exports.
- [x] Add optional PropLine player-prop line ingestion for `line`, `implied_probability`, and projection-vs-line `edge`.
- [x] Add sportsbook line consensus/variance fields to prop exports.
- [x] Add injury status, injury-exclusion, and bettable flags to prop exports.
- [x] Add current-week ESPN roster/depthchart/injury snapshots for explicit prediction weeks.
- [x] Build unified current-player availability from ESPN, nflverse, and BallDontLie.
- [x] Add roster validation fields to prop exports and block confirmed roster mismatch, practice-squad, and inactive-roster rows from `bettable_flag`.
- [x] Add weekly player-prop quality report artifacts for board usability review.
- [x] Add roster validation counts to weekly player-prop quality reports.
- [x] Report PropLine availability and match quality by week, sportsbook, market, and player naming.
- [x] Add manual Odds API historical event/event-market/player-prop fetcher for backfill experiments.
- [x] Build historical prop-line backfill manifest by season/week/game/snapshot time before running quota-heavy pulls.
- [x] Add resumable Odds API historical backfill controller with paid-key confirmation, event discovery, market discovery, odds fetch, and normalized historical line output.
- [ ] Calibrate player-prop confidence tiers before treating them as recommendations.
- [x] Execute the paid one-time Odds API historical player-prop backfill in resumable batches.
- [x] Join historical prop lines to walk-forward model predictions and actual player results.
- [x] Add first-pass per-market residual-CDF over probabilities and publish model-vs-market calibration metrics.
- [ ] Backtest player-prop edge thresholds before treating projection-vs-line deltas as betting signals.
- [ ] Improve per-market over-probability calibration versus sportsbook implied probabilities before using `prob_edge` as a recommendation signal.

#### Short-Term Roster Availability Plan

- [ ] Review Week 3 and Week 4 player-prop outputs for `roster_validation_flag` false positives before surfacing the flag prominently on MattyTheBookie.
- [ ] Audit unmatched `unknown` roster rows by player, team, market, and sportsbook line availability.
- [ ] Add a small weekly exception list only if legitimate active players are repeatedly misclassified by source data.
- [ ] Confirm ESPN roster/depthchart snapshots are refreshed before `src.player_availability` in every explicit-week pipeline run.
- [ ] Publish roster-validation summary counts in the weekly debug review alongside PropLine coverage and injury exclusions.

#### Long-Term Roster And Player Identity Plan

- [ ] Build a durable cross-source player identity table linking GSIS, ESPN, BallDontLie, PropLine name keys, and sportsbook display names.
- [ ] Add confidence scoring for player identity matches so initials/name collisions are visible before they affect bettable rows.
- [ ] Track roster transitions by week so trades, signings, practice-squad elevations, and IR activations can be audited historically.
- [ ] Use depth chart rank and active roster status as model features after enough weekly snapshots are collected.
- [ ] Backtest whether depth chart rank, roster section, and late-week injury status improve prop-line edge calibration.

#### Serious Prop Markets

- [ ] QB: passing yards, passing TDs, interceptions, completions, attempts, rushing yards, rushing TD probability.
- [ ] RB: rushing yards, carries, rushing TD probability, receiving yards, receptions, targets, total scrimmage yards, anytime TD probability.
- [ ] WR/TE: receiving yards, receptions, targets, receiving TD probability, total yards, anytime TD probability.
- [ ] Defense: sacks, QB hits, tackles for loss, total tackles, solo tackles, passes defended, interception probability, forced fumble probability.

#### Fun Novelty Bets

These should stay labeled as novelty longshot watchlists until backtests prove stable calibration:

- [ ] WR/TE passing yards
- [ ] Pick-six / interception touchdown probability
- [ ] Defensive touchdown probability
- [ ] Blocked punt probability
- [ ] Blocked field goal / blocked kick probability
- [ ] Return touchdown probability
- [ ] Fumble recovery touchdown probability

#### Planned Player Prop Outputs

- [x] `predictions/player_props_qb.csv`
- [x] `predictions/player_props_offense.csv`
- [x] `predictions/player_props_defense.csv`
- [x] `predictions/player_props_novelty.csv` schema placeholder.
- [x] MattyTheBookie CSV publishing support for player prop files.
- [ ] MattyTheBookie website/import consumption for player prop files, if needed by the MTB repo.

Implementation details live in `docs/PLAYER_PROP_PREDICTION_PLAN.md`.

### 6. Validation Operations

- [x] Evaluation pipeline writes weekly and overall metrics.
- [x] Run comparison pipeline writes leaderboard and metric trend charts.
- [x] Live tracking module exists.
- [x] CLV, paper trading, calibration monitoring, benchmark comparison modules exist.
- [ ] Refresh live tracking artifacts for the active 2026 season.
- [ ] Re-run CLV reports with current predictions and odds data.
- [ ] Re-run paper trading reports after CLV artifacts are current.
- [ ] Re-run benchmark comparison with current-schema predictions.
- [ ] Decide whether validation outputs should be generated by the main weekly pipeline or a separate scheduled validation command.

### 7. Testing And Quality

- [x] Unit tests exist for data fetchers, schema validation, prediction helpers, pipeline helpers, publish/import helpers, and analysis utilities.
- [ ] Run a fresh full test suite after the current Week 3 pipeline finishes.
- [ ] Establish a realistic current coverage baseline from the full suite.
- [x] Add focused tests for the first player-prop label builder.
- [x] Add focused tests for the split player-prop feature builder.
- [x] Add focused tests for the first player-prop baseline evaluator.
- [x] Add focused tests for the first player-prop trained-model evaluator.
- [x] Add focused tests for player-prop model outputs after those outputs exist.
- [x] Add regression tests for MTB archive publishing.
