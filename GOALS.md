# NFL Predictions Goals

Last reviewed: 2026-10-07

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
- `analysis/market_clv_detail.csv`
- `analysis/market_clv_summary.csv`
- `analysis/market_disagreement_roi.csv`
- `analysis/market_disagreement_season_stability.csv`
- `analysis/market_disagreement_edge_bins.csv`
- `analysis/market_probability_edge_calibration.csv`
- `analysis/market_blend_backtest.csv`
- `analysis/market_blend_summary.csv`
- `analysis/market_blend_policy.csv`
- `analysis/market_roi_spread.csv`
- `analysis/market_roi_spread_edge_bins.csv`
- `analysis/market_spread_key_numbers.csv`
- `analysis/market_spread_calibration.csv`
- `analysis/market_spread_cover_recalibration.csv`
- `analysis/market_utility_segments.csv`
- `analysis/market_edge_policy.csv`
- `predictions/evaluation/feature_family_ablation.csv`
- `analysis/model_metrics_week_05.csv`
- `data/processed/player_prop_labels.parquet`
- `data/processed/player_prop_features_offense.parquet`
- `data/processed/player_prop_features_defense.parquet`
- `data/processed/player_prop_features_novelty.parquet`
- `data/processed/player_prop_novelty_td_labels.parquet`
- `predictions/evaluation/player_props/baseline_metrics.csv`
- `predictions/evaluation/player_props/baseline_predictions.csv`
- `predictions/evaluation/player_props/model_metrics.csv`
- `predictions/evaluation/player_props/model_predictions.csv`
- `predictions/evaluation/player_props/model_vs_baseline.csv`
- `predictions/evaluation/player_props/prop_quality_week_05.csv`
- `predictions/evaluation/player_props/prop_quality_summary_week_05.csv`
- `predictions/evaluation/player_props/prop_line_coverage_week_05.csv`
- `predictions/evaluation/player_props/prop_roster_line_audit_week_05.csv`
- `predictions/player_props_qb.csv`
- `predictions/player_props_offense.csv`
- `predictions/player_props_defense.csv`
- `predictions/player_props_novelty.csv`
- `data/raw/propline_player_props_2026_wk05.parquet`
- `data/processed/player_prop_historical_line_eval.parquet`
- `data/processed/player_prop_over_probability_eval.parquet`
- `data/processed/oddsapi_historical_player_prop_lines_novelty.parquet`
- `predictions/evaluation/player_props/historical_line_join_summary.csv`
- `predictions/evaluation/player_props/over_probability_metrics.csv`
- `predictions/evaluation/player_props/over_probability_bins.csv`
- `predictions/evaluation/player_props/edge_backtest_detail.csv`
- `predictions/evaluation/player_props/edge_backtest_summary.csv`
- `predictions/evaluation/player_props/edge_backtest_market_thresholds.csv`
- `predictions/evaluation/player_props/edge_recommendation_policy.csv`
- `data/raw/espn_rosters_<season>_wkNN.parquet`
- `data/raw/espn_depthcharts_<season>_wkNN.parquet`
- `data/raw/espn_injuries_<season>_wkNN.parquet`
- `data/processed/current_player_availability.parquet`
- `analysis/run_comparisons/run_report.md`
- `README.md`

Latest full evaluation snapshot from the Week 5 pipeline run on 2026-10-06:

| Metric | Current | Near-Term Goal |
|---|---:|---:|
| Games / samples | 2283 | Track per run |
| Accuracy | 0.649 | >= 0.670 |
| F1 | 0.687 | >= 0.710 |
| AUC | 0.703 | >= 0.720 |
| Brier | 0.220 | <= 0.210 |
| LogLoss | 0.630 | <= 0.610 |
| Margin MAE | 10.23 | <= 9.75 |
| Margin RMSE | 13.28 | <= 12.75 |

Current read: the model is usable but below the earlier aspirational AUC/Brier targets. Treat prior AUC values near 1.0 as invalid or stale unless backed by the current evaluation pipeline.

## What Is In Place

- Core pipeline command with async data ingestion, GPU training, live-week exclusion, evaluation, SHAP, run comparison, README refresh, and optional MTB publishing.
- Current operational defaults: `--start-year 2017`, `--max-parallel-data 8`, `--data-start-delay 1`, `--live-run`, and `--publish-mtb` when exporting to the website repo.
- Live prediction reruns now pass `--exclude-started` into `src.predict.predict_upcoming` when `--live-run` is active, so Friday/Saturday reruns do not regenerate post-kickoff Thursday game picks.
- NFLVerse, ESPN, NFL.com, NOAA/weather, Visual Crossing, BallDontLie, PropLine, Yahoo, SportsDataIO fallback, and local reference-data ingestion paths.
- BallDontLie is the preferred commercial feed. SportsDataIO remains a fallback and should not run by default when BallDontLie is configured.
- BallDontLie current-season team game stats are now converted into modular weekly form features through `src.features.balldontlie_team_form`. The features ignore non-final live rows, expose prior/rolling values only for modeling, and are merged week-to-week from `data/raw/balldontlie_team_stats_<season>.parquet`.
- Production team-model feature selection now gates `bdl_*` features off by default while preserving BDL ingestion and feature generation. Training and tuning can opt in with `--include-bdl-features`, and feature-family ablation still opts in automatically so BDL lift can keep being measured.
- Team-model feature-family ablation is now part of evaluation output at `predictions/evaluation/feature_family_ablation.csv`. It compares `baseline_no_bdl`, `baseline_plus_bdl`, and `baseline_plus_bdl_no_sparse_sack_rate` on the same CV folds, so BallDontLie feature lift can be measured without retraining the production model variants by hand.
- Current team prediction outputs:
  - `predictions/predictions.csv`
  - `predictions/predictions_full.csv`
  - `predictions/history/*.csv`
- Current team prediction CSVs now include market-policy guardrail fields so the website can distinguish automatic moneyline recommendations, review-only moneyline edges, lean spread research, blocked spread edges, and no-bet rows without reimplementing policy logic.
  - `predictions/evaluation/*.csv`
- Current rate-based player projection outputs:
  - `predictions/predictions_players_qb.csv`
  - `predictions/predictions_players_offense.csv`
  - `predictions/predictions_players_defense.csv`
- Player-prop label table exists for QB passing yards, QB passing TDs, RB rushing yards, RB rushing TDs, WR/TE receiving yards, WR/TE receiving TDs, and defensive player sacks:
  - `data/processed/player_prop_labels.parquet`
- Split player-prop feature tables exist so offensive, defensive, and novelty player models do not share one mixed feature matrix:
  - `data/processed/player_prop_features_offense.parquet`
  - `data/processed/player_prop_features_defense.parquet`
  - `data/processed/player_prop_features_novelty.parquet`
- First player-prop baseline evaluation exists for current rate-based projection CSVs:
  - `predictions/evaluation/player_props/baseline_metrics.csv`
  - `predictions/evaluation/player_props/baseline_predictions.csv`
- Trained player-prop model evaluation exists for the currently modeled core and novelty markets:
  - `predictions/evaluation/player_props/model_metrics.csv`
  - `predictions/evaluation/player_props/model_predictions.csv`
  - `predictions/evaluation/player_props/model_vs_baseline.csv`
  - `models/player_props/offense/*_model.pkl`
  - `models/player_props/defense/*_model.pkl`
  - `models/player_props/novelty/*_model.pkl`
- Weekly player-prop quality reports exist for the live board:
  - `predictions/evaluation/player_props/prop_quality_week_NN.csv`
  - `predictions/evaluation/player_props/prop_quality_summary_week_NN.csv`
  - `predictions/evaluation/player_props/prop_line_coverage_week_NN.csv`
  - `predictions/evaluation/player_props/prop_roster_line_audit_week_NN.csv`
- Weekly trained-model player-prop prediction exports exist for the currently modeled markets:
  - `predictions/player_props_qb.csv`
  - `predictions/player_props_offense.csv`
  - `predictions/player_props_defense.csv`
  - `predictions/player_props_novelty.csv` publishes first TD, last TD, 2+ TD, and 3+ TD scorer research-only rows when current projections, novelty models, and matching lines are available.
- Player-prop line snapshots are available through PropLine:
  - `data/raw/propline_player_props_<season>_wkNN.parquet`
  - Week 5 current prop exports have 2,817 rows, 1,497 rows with lines, 419 bettable rows, 0 recommended rows, and 5 line-variance rows across DraftKings, FanDuel, and Hard Rock.
  - Default current PropLine requests now include `player_pass_tds`, `player_rush_tds`, `player_reception_tds`, and `player_anytime_td`. Week 5 has current line coverage for QB passing TD, anytime TD, first/last TD, 2+/3+ TD, yardage markets, and sparse FanDuel defensive sacks. Defensive sack outputs now expand weekly candidates from active sportsbook-offered players with prior sack history, improving Week 5 `def_sacks` published line coverage from 14/290 to 21/297. PropLine returned zero exact `player_rush_tds` and `player_reception_tds` rows, so RB rushing TD and WR/TE receiving TD outputs now attach identity-aware `anytime_td_fallback` lines for display/coverage while remaining non-bettable.
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
  - novelty normalized line output: `data/processed/oddsapi_historical_player_prop_lines_novelty.parquet`
  - Current combined historical line universe has 124,013 sportsbook line rows across DraftKings, FanDuel, and Hard Rock after the main and novelty backfills are normalized.
- Historical prop-line join artifact exists:
  - `data/processed/player_prop_historical_line_eval.parquet`
  - `predictions/evaluation/player_props/historical_line_join_summary.csv`
  - Current join matched 53,274 sportsbook line rows to walk-forward model predictions and actuals from 124,013 available historical line rows.
- Per-market historical over-probability evaluation exists:
  - `src.player_props.over_probability`
  - `data/processed/player_prop_over_probability_eval.parquet`
  - `predictions/evaluation/player_props/over_probability_metrics.csv`
  - `predictions/evaluation/player_props/over_probability_bins.csv`
  - Current probabilities use walk-forward residual-CDF estimates blended toward no-vig sportsbook implied probabilities when prior same-market calibration shows the market is better calibrated.
  - Latest artifact scores 53,268 joined rows. Current blended player-prop probabilities have 0.0978 overall Brier and 0.3097 LogLoss; sportsbook implied probabilities remain slightly better overall at 0.0972 Brier and about 0.307 LogLoss.
- Weekly player-prop CSVs now include line-specific probability fields when historical calibration data and a matched line are available:
  - `prob_over`
  - `prob_over_raw`
  - `prob_edge`
  - `prob_over_method`
  - `prob_over_sample_size`
  - `over_break_even_probability`, `under_break_even_probability`, `over_ev`, `under_ev`, `edge_side`, `edge_odds`, `edge_break_even_probability`, and `edge_ev`
  - `edge_probability_edge`, `edge_recommendation_flag`, `edge_recommendation_tier`, and `edge_policy_*`
  - Price-aware EV fields account for sportsbook vig. Recommendation flags now require a bettable current row, a historically positive market threshold from `edge_recommendation_policy.csv`, and positive current expected value after juice.
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
  - `analysis/market_clv_detail.csv`
  - `analysis/market_clv_summary.csv`
  - `analysis/market_disagreement_roi.csv`
  - `analysis/market_disagreement_season_stability.csv`
  - `analysis/market_disagreement_edge_bins.csv`
  - `analysis/market_probability_edge_calibration.csv`
  - `analysis/market_blend_backtest.csv`
  - `analysis/market_blend_summary.csv`
  - `analysis/market_blend_policy.csv`
  - `analysis/market_roi_spread_edge_bins.csv`
  - `analysis/market_spread_key_numbers.csv`
  - `analysis/market_spread_calibration.csv`
  - `analysis/market_spread_cover_recalibration.csv`
  - `analysis/market_utility_segments.csv`
  - `analysis/market_edge_policy.csv`
- Website-ready model metrics export exists for MattyTheBookie About-page consumption:
  - `analysis/model_metrics_week_NN.csv`

## Current Gaps

- `GOALS.md` and some older docs previously contained stale metrics and completed-status claims. This file should now be the source of current roadmap truth.
- Live tracking, paper trading, calibration monitoring, and benchmark modules exist, but their generated artifacts are not all currently maintained as live weekly outputs. CLV detail/summary CSVs are now generated in the weekly pipeline from local pick-time and close/latest odds snapshots.
- Current measured coverage is low. Treat any old >70% coverage claim as stale until a fresh full-suite coverage run proves otherwise.
- Player prop labels, split feature tables, baseline scoring, trained-model evaluation, weekly trained-model CSV exports, optional PropLine sportsbook ingestion, historical line joins, market-aware over-probability diagnostics, and first edge-threshold backtests now exist for the core yardage/sacks markets. Direct QB/RB/WRTE touchdown markets and anytime TD scorer are now published as projections; first/last/2+/3+ TD scorer novelty markets now have PBP-derived labels, a separate novelty feature/model family, and research-only weekly output. QB passing TD and anytime TD have exact current line coverage; RB rushing TD and WR/TE receiving TD now have non-bettable anytime-TD fallback lines, but still need true exact sportsbook line coverage before recommendation gates can use them. Remaining prop gaps are confidence-tier calibration, broader defensive-sack sportsbook/vendor coverage, historical touchdown line calibration, profitable edge validation, and historical novelty-market calibration.
- Current-week roster validation is integrated and needs ongoing production-output review for false positives, missing ESPN matches, and player-name/team alias mismatches. Week 5 currently has 0 unknown roster rows, 15 team mismatches, and 168 inactive/practice-squad roster exclusions.
- Yahoo Fantasy remains blocked pending API access approval.
- SportsDataIO is retained only as a fallback while BallDontLie coverage is verified.
- Current-season nflverse PBP can still lag or be unavailable. BallDontLie team-form features now cover current-season team context, but PBP-specific feature families such as EPA, pressure, and red-zone detail still depend on nflverse/ESPN PBP until equivalent replacement logic is proven.
- BallDontLie team-form features are useful as a monitored fallback source, but latest Week 4/Week 5 ablation runs showed win-probability regression and only marginal spread benefit. Keep them out of production model features until repeatable same-split lift appears.
- Market benchmarking now publishes model-vs-Vegas probability comparisons, model-vs-no-vig-Vegas probability calibration by bin, walk-forward market-aware blend backtests, moneyline disagreement threshold ROI, moneyline disagreement season-stability gates, bounded moneyline disagreement edge-bin ROI, corrected spread ROI, spread edge-bin ROI, spread key-number diagnostics, spread cover-probability calibration, and weekly CLV detail/summary artifacts. Historical per-game CLV fields are still not merged into `market_benchmark.csv` until reliable pick-time snapshots exist for the evaluated rows.
- Corrected spread ROI no longer shows broad profitability. Current Week 5 policy artifacts mark broad spread threshold strategies as `no_bet`; only the 2-to-3 point spread-edge bucket remains a low-confidence lean/research signal.

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
- [x] Exclude already-started games from live prediction output reruns to avoid publishing post-kickoff regenerated picks.
- [x] Apply volatility-aware calibration/shrinkage.
- [x] Stabilize live volatility shrinkage with automatic per-slate coverage floors/ceilings when the learned threshold under- or over-selects the current week.
- [x] Classify expected sparse injury/news/weather/schedule feature nulls separately from actionable high-null data-quality warnings.
- [x] Add BallDontLie current-season team-form fallback features with final-only raw stats and prior/rolling week windows.
- [x] Backfill enough historical BallDontLie team-stat seasons for model training support. Regular-season team stats now exist for 2017-2025, with 2026 current-season rows refreshed by the weekly live feed.
- [x] Add a feature-family ablation artifact for baseline vs `bdl_*` feature lift: `predictions/evaluation/feature_family_ablation.csv`.
- [x] Gate `bdl_*` out of default production model training/tuning while keeping an explicit `--include-bdl-features` experiment switch and ablation opt-in.
- [x] Review Week 5 `feature_family_ablation.csv`; `baseline_plus_bdl` and `baseline_plus_bdl_no_sparse_sack_rate` both underperformed `baseline_no_bdl` on win-probability AUC and spread MAE.
- [ ] Continue reviewing `feature_family_ablation.csv` after each full pipeline run.
- [ ] Treat `bdl_*` features as production-proven only after at least 3 full live-week runs show same-split lift or no material regression: target `baseline_plus_bdl` or `baseline_plus_bdl_no_sparse_sack_rate` should improve or hold AUC/Brier/LogLoss while also improving spread MAE by a repeatable margin.
- [ ] If later ablation proves only a narrow BDL subset is useful, add a named subset gate instead of enabling the entire `bdl_*` family.
- [ ] Improve current AUC from ~0.703 to >=0.720 without leakage.
- [ ] Improve Brier from ~0.220 to <=0.210.
- [ ] Improve margin MAE from ~10.23 to <=9.75.
- [ ] Re-check feature lift after every material feature-family change.
- [x] Rework volatility classifier to RF, log-loss-only labels, and 0.5 percentile default.
- [ ] Add a separate spread/margin-volatility model instead of mixing margin-error labels into probability volatility.
- [ ] Improve probability calibration, especially high-confidence picks where LogLoss/Brier penalties are largest.
- [ ] Build calibration diagnostics by model confidence tier and identify overconfident ranges that need shrinkage or recalibration.

### 4. Market Benchmarking And Betting Edge

Priority: high. Protect published picks from known bad market signals before adding new model complexity.

Current read from the Week 5 evaluation set:

- SU model accuracy is 65.4% over 2,283 market-matched games.
- No-vig Vegas favorite accuracy is about 66.1% over the same games.
- Model and Vegas favorite agree on about 86.9% of games.
- In 299 favorite-disagreement games, the model is 47.2% SU and Vegas is 52.8%, but model-side moneyline ROI is +4.3% overall because the model side is usually plus-money.
- The historical moneyline disagreement candidate `prob_edge >= 0.02` has 293 bets, +5.1% ROI, and a 47.4% hit rate at average model-side odds near +102, but the current live policy is `review` because the CLV sample is still sparse.
- Moneyline disagreement policy now requires season-stability and current-week CLV gates before publication. The current `prob_edge >= 0.02` policy passes stability with 7 qualified seasons, a 71.4% positive qualified-season rate, a 100.0% recent positive qualified-season rate over the two qualified recent seasons, and a 14.3% severe-negative season rate, but fails the minimum CLV sample gate with only 6 tracked Week 5 games.
- Higher moneyline disagreement thresholds are not automatically safer: `prob_edge >= 0.08` currently fails due to insufficient recent qualified seasons, `prob_edge >= 0.10` fails due to weak positive-season rate, and `prob_edge >= 0.15` fails due to insufficient qualified seasons.
- Model-vs-no-vig-Vegas probability calibration is now published. Current selected-side Brier is worse than Vegas overall (`0.218` vs `0.211`), and the `15%+` model edge bin is materially overconfident: 62.2% average model probability, 43.8% average no-vig Vegas same-side probability, 42.6% actual win rate, and -5.3% model-side ROI.
- A first walk-forward market-aware blend backtest is now published. Current blend Brier improves raw model Brier (`0.214` vs `0.219`) but still trails no-vig Vegas (`0.212`), and blend selected-side ROI is -3.6%, so both blend probability and blend moneyline policies remain `no_bet`.
- Bounded moneyline disagreement bins are now published. The early Week 5 read shows the strongest bounded samples in the 2-5% and 10-15% bins, while the 15%+ bin is negative; do not scale confidence purely by larger probability edge until season-stability and CLV checks support it.
- Weekly CLV detail and summary artifacts are now published from local pick-time and close/latest odds snapshots, and `analysis/market_edge_policy.csv` consumes the threshold-specific CLV summary before allowing a moneyline policy to become `strong`. Current Week 5 favorite-disagreement rows have 6 tracked games, +0.8 percentage-point average no-vig implied-probability CLV, and a 66.7% positive CLV rate, so the policy remains `review` until it reaches at least 10 tracked games.
- Broad spread threshold strategies are now policy-blocked as `no_bet`. Away edge >2 points is -1.5% ROI despite 61.0% average modeled cover probability, which confirms spread cover-probability overconfidence.
- Corrected spread edge-bin ROI is only positive in the 2-to-3 point bucket: 368 bets, +1.2% ROI, and 52.5% hit rate. That bucket is now review-only until the empirical spread cover calibration gate passes; spread edges of 5+ points are negative and should not be promoted.
- Spread key-number diagnostics are now published for 3, 6, 7, and 10. In the current 2-to-3 point spread-edge bucket, key-window ROI is -5.4% around 3, +9.0% around 6, +10.9% around 7, and roughly flat around 10. Treat this as a monitoring slice, not a promotion rule, until season stability and CLV confirm it.
- Spread cover-probability calibration and empirical recalibration gates are now published. The 70-75% bucket has a 33.3% actual cover rate, and the 80%+ bucket has a 0.0% actual cover rate in the current sample, so raw spread `cover_prob` remains diagnostic-only and blocks spread edge-bin promotion while high-confidence calibration errors are unacceptable.

Roadmap:

- [x] Add a `market_benchmark.csv` artifact comparing model probability to no-vig Vegas implied probability for every evaluated game.
- [x] Publish model-vs-Vegas favorite agreement/disagreement metrics by season and overall.
- [x] Add moneyline disagreement ROI by probability-edge threshold.
- [x] Fix and test `analysis.market_roi` spread-line sign convention.
- [x] Recompute spread ROI using corrected edge: `pred_home_margin - market_home_margin`.
- [x] Add edge-bin ROI tables for corrected spread edges, including by season and minimum sample thresholds.
- [x] Add market utility segment table by agreement, model confidence, probability edge, and model-side moneyline bucket.
- [x] Add conservative market-edge policy artifact that can explicitly say `no_bet` when historical ROI/sample gates fail.
- [x] Add bounded moneyline disagreement edge-bin ROI artifact: `analysis/market_disagreement_edge_bins.csv`.
- [x] Add spread cover-probability calibration artifact: `analysis/market_spread_calibration.csv`.
- [x] Add empirical spread cover recalibration artifact: `analysis/market_spread_cover_recalibration.csv`.
- [x] Hard-gate broad spread threshold policy to `no_bet` until calibrated spread probabilities and CLV justify promotion.
- [x] Restrict spread edge-bin policy recommendations to the 2-to-3 point bucket as lean/research; block 5+ point spread edges from promotion.
- [x] Add season-stability gates for the moneyline disagreement policy before raising confidence beyond the current strong research signal.
- [x] Add current-week publishing guardrails so MattyTheBookie can distinguish strong moneyline policy picks, review-only high moneyline edges, lean spread research picks, blocked spread signals, and no-bet rows.
- [x] Compare model probability directly to no-vig Vegas probability by probability bin and publish the calibration/edge table.
- [ ] Improve model performance in favorite-disagreement games versus Vegas before treating disagreements as actionable.
- [x] Build and backtest a market-aware blend or meta-model using model probability, no-vig moneyline probability, market spread, and current model features.
- [x] Track whether any market-aware blend adds genuine edge versus simply mirroring Vegas favorite picks.
- [x] Keep spread edge analysis as a separate betting-signal track from straight-up winner model accuracy.
- [x] Add key-number-aware spread diagnostics for 3, 6, 7, and 10 before promoting any spread signal.
- [x] Recalibrate spread cover probabilities empirically; suppress raw high `cover_prob` as a confidence signal until calibration errors are acceptable.
- [x] Add weekly CLV detail/summary artifacts from local pick-time and close/latest odds snapshots: `analysis/market_clv_detail.csv` and `analysis/market_clv_summary.csv`.
- [ ] Add closing-line movement / CLV fields to `market_benchmark.csv` when reliable opening and closing snapshots are available.
- [x] Promote CLV reports into the weekly validation flow before using ROI results as decision criteria.

### 5. Player Prop Prediction Goals

Current state: legacy player CSVs remain available as rate-based projections, and trained-model player prop CSVs now publish weekly for QB passing yards, QB passing TDs, RB rushing yards, RB rushing TDs, WR/TE receiving yards, WR/TE receiving TDs, anytime TD scorer, defensive player sacks, and research-only first/last/2+/3+ TD scorer novelty rows. PropLine line ingestion is wired through DraftKings, Hard Rock, and FanDuel when `PROPLINE_API_KEY` is configured, with line-consensus/variance diagnostics, injury quality controls, current-roster validation, and conservative edge-recommendation policy fields included in the weekly prop CSVs. Default current PropLine requests include yardage, stat-specific TD, anytime TD, first/last/2+/3+ TD scorer novelty keys, and defensive sack market keys; the weekly coverage artifact now separates exact line rows from `anytime_td_fallback` display rows so RB rushing TD and WR/TE receiving TD coverage is visible without making those rows bettable. A durable cross-source player identity table now publishes to `data/processed/player_identity.parquet` and links GSIS player IDs to nflverse, ESPN, BallDontLie, Sportradar, and sportsbook-style player-name aliases for safer historical and current prop-line matching. Goal: expand line availability reporting, calibrate confidence tiers, backtest prop edges, broaden markets, and score the outputs weekly.

#### First Implementation Slice

- [x] Build normalized player-game label table for the current modeled prop markets from historical weekly player stats: `data/processed/player_prop_labels.parquet`.
- [x] Build split offensive and defensive player-prop feature tables:
  - `data/processed/player_prop_features_offense.parquet`
  - `data/processed/player_prop_features_defense.parquet`
- [x] Score existing rate-based projections as baseline models: `predictions/evaluation/player_props/baseline_metrics.csv`.
- [x] Train/evaluate first high-volume markets:
  - [x] QB passing yards
  - [x] QB passing TDs
  - [x] RB rushing yards
  - [x] RB rushing TDs
  - [x] WR/TE receiving yards
  - [x] WR/TE receiving TDs
  - [x] Defensive player sacks
- [x] Publish baseline evaluation artifacts under `predictions/evaluation/player_props/`.
- [x] Publish trained-model evaluation artifacts under `predictions/evaluation/player_props/`.
- [x] Build weekly trained-model player prop prediction exports.
- [x] Publish direct touchdown projections where current stats/projection data supports them.
- [x] Request stat-specific TD market keys from current PropLine weekly line ingestion.
- [x] Add first anytime TD scorer market with current PropLine line matching.
- [x] Request first/last/2+/3+ TD scorer novelty market keys from current PropLine weekly line ingestion.
- [x] Add explicit `coverage_status` diagnostics to `prop_line_coverage_week_NN.csv` for requested markets with no vendor rows.
- [x] Split exact sportsbook line counts from `anytime_td_fallback` display counts in weekly prop quality and line-coverage artifacts.
- [x] Add optional PBP-derived first/last/2+/3+ TD novelty label output: `data/processed/player_prop_novelty_td_labels.parquet`.
- [x] Wire PBP-derived novelty TD labels into separate novelty features, model training, and research-only weekly prediction output.
- [x] Add optional PropLine player-prop line ingestion for `line`, `implied_probability`, and projection-vs-line `edge`.
- [x] Add identity-aware anytime-TD fallback mapping for RB rushing TD and WR/TE receiving TD current outputs; fallback rows use `line_type=anytime_td_fallback`, `prop_quality_flag=fallback_line`, and are excluded from bettable/recommendation gates.
- [x] Add line-offered defensive sack candidate expansion using PropLine offers plus player identity, active-roster availability, and prior non-leaking sack history.
- [x] Add sportsbook line consensus/variance fields to prop exports.
- [x] Add injury status, injury-exclusion, and bettable flags to prop exports.
- [x] Add current-week ESPN roster/depthchart/injury snapshots for explicit prediction weeks.
- [x] Build unified current-player availability from ESPN, nflverse, and BallDontLie.
- [x] Add roster validation fields to prop exports and block confirmed roster mismatch, practice-squad, and inactive-roster rows from `bettable_flag`.
- [x] Add weekly player-prop quality report artifacts for board usability review.
- [x] Add roster validation counts to weekly player-prop quality reports.
- [x] Report PropLine availability and match quality by week, sportsbook, market, and player naming.
- [x] Add player-prop edge recommendation counts and EV/probability-edge diagnostics to weekly quality summaries.
- [x] Add manual Odds API historical event/event-market/player-prop fetcher for backfill experiments.
- [x] Build historical prop-line backfill manifest by season/week/game/snapshot time before running quota-heavy pulls.
- [x] Add resumable Odds API historical backfill controller with paid-key confirmation, event discovery, market discovery, odds fetch, and normalized historical line output.
- [x] Add Odds API historical market plumbing for first TD, last TD, and `player_tds_over` milestone rows so novelty TD lines can normalize into the first/last/2+/3+ TD model markets.
- [x] Add player-prop confidence-tier diagnostics at model-prediction and historical-line evaluation scopes: `predictions/evaluation/player_props/confidence_tier_metrics.csv`.
- [ ] Calibrate player-prop confidence tiers before treating them as recommendations.
- [x] Execute the paid one-time Odds API historical player-prop backfill in resumable batches.
- [x] Join historical prop lines to walk-forward model predictions and actual player results.
- [x] Add first-pass per-market residual-CDF over probabilities and publish model-vs-market calibration metrics.
- [x] Add market-aware residual/implied-probability blending for current player-prop `prob_over` outputs.
- [x] Backtest player-prop edge thresholds before treating projection-vs-line deltas as betting signals.
- [x] Add conservative edge-recommendation policy fields to current prop CSVs.
- [ ] Prove player-prop `prob_edge` and `edge_ev` add durable lift versus sportsbook implied probabilities before expanding recommendation signals beyond sparse market-specific policies.
- [ ] Add true current and historical sportsbook line coverage for RB rushing TDs and WR/TE receiving TDs before treating those stat-specific TD props as bettable recommendations; current anytime-TD fallback lines are display/coverage aids only.
- [ ] Add historical sportsbook line coverage and edge backtests for anytime TD before treating it as a recommendation signal.
- [ ] Backfill historical first/last/2+/3+ TD scorer lines before promoting novelty TD markets beyond research-only output.

Current edge-backtest read: the policy selects only market-level thresholds with at least 100 historical bets and positive ROI: strong anytime TD at `prob_edge >= 0.10` and lean RB rushing yards at `prob_edge >= 0.02`. Current rows expose these thresholds, but no Week 5 row is flagged as recommended because current EV/threshold gates are not met after price/juice.

Current confidence-tier read: higher historical sample size should remain diagnostic-only. The latest historical-line evaluation shows high-tier rows with more volume but weaker line MAE/model-pick accuracy than low/medium tiers, so confidence tier must not boost published recommendations until calibrated by market and line context.

#### Short-Term Roster Availability Plan

- [x] Review Week 3 through Week 5 player-prop outputs for `roster_validation_flag` false positives before surfacing the flag prominently on MattyTheBookie.
- [x] Audit unmatched `unknown` roster rows by player, team, market, and sportsbook line availability. Weekly `prop_roster_line_audit_week_NN.csv` now expands unknown roster rows across requested sportsbooks with raw/prepared/matchable/published line status.
- [ ] Add a small weekly exception list only if legitimate active players are repeatedly misclassified by source data.
- [x] Confirm ESPN roster/depthchart snapshots are refreshed before `src.player_availability` in every explicit-week pipeline run.
- [x] Publish roster-validation summary counts in the weekly debug review alongside PropLine coverage and injury exclusions.

#### Long-Term Roster And Player Identity Plan

- [x] Build a durable cross-source player identity table linking GSIS, ESPN, BallDontLie, PropLine name keys, and sportsbook display names.
- [x] Add first-pass confidence scoring and ambiguity filtering for player identity matches before historical/current line attachment.
- [ ] Publish identity-match diagnostics in the weekly quality report so initials/name collisions are visible before they affect bettable rows.
- [ ] Track roster transitions by week so trades, signings, practice-squad elevations, and IR activations can be audited historically.
- [ ] Use depth chart rank and active roster status as model features after enough weekly snapshots are collected.
- [ ] Backtest whether depth chart rank, roster section, and late-week injury status improve prop-line edge calibration.

#### Serious Prop Markets

- [x] QB: passing yards and passing TDs.
- [ ] QB: interceptions, completions, attempts, rushing yards, rushing TD probability.
- [x] RB: rushing yards and rushing TD probability.
- [x] RB: anytime TD probability.
- [ ] RB: carries, receiving yards, receptions, targets, total scrimmage yards.
- [x] WR/TE: receiving yards and receiving TD probability.
- [x] WR/TE: anytime TD probability.
- [ ] WR/TE: receptions, targets, total yards.
- [x] Defense: sacks.
- [ ] Defense: QB hits, tackles for loss, total tackles, solo tackles, passes defended, interception probability, forced fumble probability.

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
- [x] `predictions/player_props_novelty.csv` research-only first/last/2+/3+ TD scorer output.
- [x] MattyTheBookie CSV publishing support for player prop files.
- [x] MattyTheBookie website/import consumption for player prop files.

Implementation details live in `docs/PLAYER_PROP_PREDICTION_PLAN.md`.

### 6. Validation Operations

- [x] Evaluation pipeline writes weekly and overall metrics.
- [x] Run comparison pipeline writes leaderboard and metric trend charts.
- [x] Live tracking module exists.
- [x] CLV, paper trading, calibration monitoring, benchmark comparison modules exist.
- [x] Weekly CLV detail/summary artifacts are generated, exported to model metrics, included in the main pipeline, and published to MattyTheBookie CSVs.
- [ ] Refresh live tracking artifacts for the active 2026 season.
- [x] Re-run CLV reports with current predictions and odds data.
- [ ] Re-run paper trading reports after CLV artifacts are current.
- [ ] Re-run benchmark comparison with current-schema predictions.
- [ ] Decide whether validation outputs should be generated by the main weekly pipeline or a separate scheduled validation command.

### 7. Testing And Quality

- [x] Unit tests exist for data fetchers, schema validation, prediction helpers, pipeline helpers, publish/import helpers, and analysis utilities.
- [ ] Run a fresh full test suite after the next full pipeline stabilization point.
- [ ] Establish a realistic current coverage baseline from the full suite.
- [x] Add focused tests for the first player-prop label builder.
- [x] Add focused tests for the split player-prop feature builder.
- [x] Add focused tests for the first player-prop baseline evaluator.
- [x] Add focused tests for the first player-prop trained-model evaluator.
- [x] Add focused tests for player-prop model outputs after those outputs exist.
- [x] Add regression tests for MTB archive publishing.
- [x] Add regression tests for production BDL feature gating and feature-family ablation opt-in.
- [x] Add regression tests for roster-line audit artifact generation and MTB publish coverage.
