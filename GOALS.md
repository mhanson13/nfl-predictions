# NFL Predictions Goals

Last reviewed: 2026-11-21

Primary goal: team win-probability AUC and Brier at or better than the no-vig market, measured on
walk-forward history (`predictions/evaluation/walk_forward_vs_market.csv`, final published probability).
The pipeline's published outputs (files, columns, MTB publishing) stay stable while the model improves.
Open items and watch items are listed in [Open Items And Watch List](#open-items-and-watch-list).
Superseded plans and status docs are in [`docs/archive/`](docs/archive/README.md).

This file is the live roadmap and status board for the NFL prediction pipeline. Keep it tied to current artifacts, not historical plans. Do not mark work complete unless the code, generated output, or documentation exists in the repo.

## Maintenance Rules

- Update this file after any production pipeline run that materially changes metrics, data sources, model behavior, publishing flow, or roadmap priority.
- Record every open item, watch item, decision, data issue, or finding that comes up in a working session here in the same session: in [Open Items And Watch List](#open-items-and-watch-list), [Known Data Issues](#known-data-issues), or the relevant roadmap section. Ship the update in the same PR as the related change, or in a small docs PR. Roadmap state must not live only in chat, plan files, or assistant memory.
- When an item is done, remove it from the open list, record the result in its section, and update `Last reviewed`.
- Use current artifact paths when marking a goal complete.
- Keep historical implementation notes in `docs/`; keep this file focused on current goals and status.
- Do not keep inactive phase labels. If a task is still worth doing, keep it on the roadmap. If not, remove it.
- **Model-change ship rule (from 2026-11-18):** compare against live on walk-forward with at least 5 model seeds (paired: same seed in both arms). Brier, log loss and calibration error (ECE, 10 bins): ship if the mean change is within one standard deviation of the live arm's per-seed values (or better), and the change is no worse in a majority of seeds. Accuracy and AUC: not meaningfully worse (same test). Report the per-seed values and standard deviations with every comparison. This replaces the strict "no worse" rule.
- **Pick-of-record transparency rule (from 2026-11-18, MTB PR #14):** a game counts toward the record only if its pick was published on the MTB site before kickoff. If the only pre-kickoff publish was faulty (placeholder probabilities, no odds), or nothing was published before kickoff, the game has no pick of record. Never substitute a later or unpublished run, even one that picks the same side. Grading writes a row for every final game: a no-pick game gets its final score, `prediction_source_type = no_pick_of_record`, and blank `model_pick_team` and grade columns (SU, moneyline, ATS). It is left out of every record, summary and metric. Known no-pick games are listed with a reason in `docs/checks/no_pick_of_record_<season>.csv` (committed). A final game with no pick that is not on that list still gets the score-only row, but grading raises an ALERT in its report and the RESULTS email (subject "RESULTS ALERT"): check the archive history before adding it to the list. Restores of dropped games come only from MTB's published git history, never from local run backups.
  - **2026 Week 11 IND-HOU (2026_11_IND_HOU), added 2026-11-20:** no pick of record. Thursday's Week 11 slot (2026-11-19) ran, but its MTB publish failed at `npm run test`: two blog tests assumed Week 10 was the current week (fixed in MTB PR #16). Nothing was published before kickoff (2026-11-20 01:15 UTC). The run's ledger row (HOU 84.7%, recorded 2026-11-19 22:13 UTC) was removed from `predictions/picks_of_record/picks_ledger_2026.csv` and the records were re-resolved. A backup of both files was kept outside the repo. The dispatcher's MTB checkout was cleaned of Thursday's unpublished files (week 11 archive, blog preview) and a manual Week 11 slot was published on Fri 11-20 (MTB 4194824; GKE deploy run 44754280494 timed out waiting for an old pod to terminate, then succeeded on re-run); its exclude-started filter left IND-HOU out (12 games). Week 11 grades 12 games plus one score-only row.
  - **2026 Week 12 GB-LA (2026_12_GB_LA), added 2026-11-26:** no pick of record. Wednesday's slot (2026-11-25) ran twice; both MTB publishes failed at `npm run test` (`tests/publish-lock.test.ts` inherited the dispatcher's lock token; fixed in MTB #30, 3e395c9). Nothing was published before kickoff (2026-11-26 01:00 UTC). MTB 8ad47ca shows it as a no-pick game (`pick: null` with a `pickNote`). This time the ledger rows were **kept** (LA 53.9%, recorded 2026-11-25 21:48:54 and 22:18:49 UTC): from this change a game on `docs/checks/no_pick_of_record_<season>.csv` resolves as `status = not_published` with the list's reason in `note` and blank `home_won`/`pick_correct`/`brier`/`market_brier` in `picks_of_record_<season>.csv`, and grading drops a listed game from every source (ledger, published override, MTB archive CSV), so it only gets the score-only row. The ledger is never edited. Not yet done: a general check that a pick counts only if its slot's publish succeeded before kickoff (proposed separately); IND-HOU's removed ledger row could be restored from the backup now that a listed game is safe in the ledger.

## Latest Validated Run

Source artifacts:

- `predictions/evaluation/walk_forward_vs_market.csv`
- `predictions/evaluation/overall_metrics.csv`
- `predictions/evaluation/live_prediction_vs_actual.csv` / `live_prediction_vs_actual_summary.csv`
- `analysis/market_benchmark.csv`
- `analysis/market_benchmark_summary.csv`
- `analysis/market_clv_detail.csv`
- `analysis/market_clv_summary.csv`
- `analysis/market_disagreement_roi.csv`
- `analysis/market_disagreement_season_stability.csv`
- `analysis/market_disagreement_edge_bins.csv`
- `analysis/market_disagreement_segments.csv`
- `analysis/market_probability_edge_calibration.csv`
- `analysis/market_model_confidence_calibration.csv`
- `analysis/market_model_confidence_recalibration.csv`
- `analysis/market_confidence_recalibration_detail.csv`
- `analysis/market_confidence_recalibration_backtest.csv`
- `analysis/market_confidence_recalibration_decision.csv`
- `analysis/market_blend_backtest.csv`
- `analysis/market_blend_summary.csv`
- `analysis/market_blend_policy.csv`
- `analysis/market_disagreement_games.csv`
- `analysis/market_disagreement_diagnostics.csv`
- `analysis/market_disagreement_recommendations.csv`
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
- `predictions/evaluation/player_props/prop_line_gap_audit_week_NN.csv`
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

Walk-forward history vs the no-vig market, manual Week 11 run of 2026-11-20 (run_512, 2,369 games,
2018-2026 through Week 10; `model_final` = the published probability):

| Scope | Model AUC | Market AUC | Model Brier | Market Brier | Model LogLoss | Market LogLoss |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.7241 | 0.7250 | 0.2106 | 0.2108 | 0.6088 | 0.6091 |
| 2026 season (150 games) | 0.7334 | 0.7422 | 0.2122 | 0.2110 | 0.6153 | 0.6123 |

By season, model AUC beats the market in 2022-2024 (+0.001 to +0.003) and trails in 2018 (-0.005),
2019 (-0.004), 2021 (-0.004), 2025 (-0.002) and 2026 (-0.009). By week bucket it leads in weeks 9-13
(+0.001) and the postseason (+0.006) and trails most in weeks 14-18 (-0.004).

Full evaluation (`overall_metrics.csv`, same run):

| Metric | Current (2026-11-20) | Previous (2026-11-14) | Goal |
|---|---:|---:|---:|
| Games / samples | 2368 | 2354 | Track per run |
| Accuracy | 0.668 | 0.664 | >= 0.670 |
| F1 | 0.711 | 0.709 | >= 0.710 |
| AUC | 0.7239 | 0.7226 | >= market (0.725) |
| Brier | 0.2107 | 0.2110 | <= market (0.211) |
| LogLoss | 0.6090 | 0.6096 | <= market (0.609) |
| Margin MAE | 9.83 | 9.83 | <= 9.75 |
| Margin RMSE | 12.73 | 12.73 | <= 12.75 |

Current read: Brier and LogLoss are slightly better than the market overall; AUC trails it by about
0.001 overall and by about 0.009 in 2026 (150 games, within noise at that size). The AUC gap, mostly
in 2026 and late-season weeks, is the main thing to close. Note that live heavy-favorite picks are
held at the 0.85 cap (Open Items: live win-probability cap), which walk-forward does not apply the same way. Calibration currently
selects `none` (2026-11-14 run: walk-forward Brier 0.2108 none / 0.2117 sigmoid / 0.2135 isotonic).

Live published-pick record (graded on what MTB showed, `live_prediction_vs_actual.csv`): 2026 weeks 2-10,
85-45-1 straight-up (65.4%) over 131 graded games, 3 no-pick games not counted (pick-of-record rule);
moneyline +4.49 units; against the spread (graded from week 5) 39-36-3 (52%), about break-even after vig. See
[Archive-To-Results Feedback Loop](#archive-to-results-feedback-loop) for which games are not graded.

## What Is In Place

- Core pipeline command with async data ingestion, GPU training, live-week exclusion, evaluation, SHAP, run comparison, README refresh, and optional MTB publishing.
- Current operational defaults: `--start-year 2017`, `--max-parallel-data 8`, `--data-start-delay 1`, `--live-run`, and `--publish-mtb` when exporting to the website repo.
- Market-anchored win and spread models with weekly walk-forward evaluation against the no-vig market: `predictions/evaluation/walk_forward_vs_market.csv` (PR #1).
- Walk-forward calibration selection (`none` / `sigmoid` / `isotonic`, scored out of sample; replaced the always-on isotonic step), market-anchored volatility shrink, and injury grid fill (PR #2).
- Week-aware runner with a week-named debug log (`tools.weekly_run`, [docs/weekly_run.md](docs/weekly_run.md)), LA odds mapping, and the home-vs-market bias monitor ([docs/market_bias_watch.md](docs/market_bias_watch.md)) (PR #3).
- Weather at the actual venue and kickoff time for every game, Visual Crossing refresh, Xweather trial snapshots ([docs/weather_sources_review.md](docs/weather_sources_review.md)) (PR #4); international travel features `travel_km_diff` / `travel_tz_shift_diff` and corrected nflverse 2025 international venues (PR #5); NOAA skips non-US venues and is optional (PR #6).
- Per-game picks-of-record ledger for live runs: `predictions/picks_of_record/` (`src.evaluation.picks_of_record`). Pick of record = latest row recorded at least 2h before that game's kickoff, else the first late row; same rule as MTB's week archive (PR #7).
- Manual tool, not in the pipeline: pick-time vs closing-line report for graded picks of record (`python -m src.evaluation.pick_time_vs_close --season 2026` -> `predictions/evaluation/pick_time_vs_close_{detail,summary}.csv`): market move from pick time to the nflverse close, CLV of the model's lean, and Brier/log loss/accuracy of model vs pick-time and closing market. Run it by hand after grading; wiring it into the dispatcher's weekly grading is a later option (user's call, 2026-11-24).
- Unattended game-day runs: Windows task + `tools.dispatcher` run the pipeline each game day, publish MTB, write a Claude review, and alert via Bird SMTP; on Tuesday it grades the completed week and republishes results ([docs/automation.md](docs/automation.md)) (PRs #8-#10). First automated weekend reviewed 2026-11-17: Sun 11-15 early slot failed twice (run-log encoding crash, fixed in PR #12), main slot published (MTB 8a38c37); Mon 11-16 slot succeeded first try (MTB 36d23e9); Tue 11-17 grading ran (Week 10 SU 10-4) but the blog recap failure did not reach the alert (fixed in PR #18). Reviews: `logs/reviews/2026/`.
- Live prediction reruns now pass `--exclude-started` into `src.predict.predict_upcoming` when `--live-run` is active, so Friday/Saturday reruns do not regenerate post-kickoff Thursday game picks.
- NFLVerse, ESPN, NFL.com, NOAA/weather, Visual Crossing, BallDontLie, PropLine, Yahoo, SportsDataIO fallback, and local reference-data ingestion paths.
- Weekly publish runs can now skip Tomorrow.io live weather with `--skip-tomorrow-weather` while retaining the paid Visual Crossing source plus NOAA/static fallback, reducing rate-limit noise without disabling weather features.
- BallDontLie is the preferred commercial feed. SportsDataIO remains a fallback and should not run by default when BallDontLie is configured.
- BallDontLie current-season team game stats are now converted into modular weekly form features through `src.features.balldontlie_team_form`. The features ignore non-final live rows, expose prior/rolling values only for modeling, and are merged week-to-week from `data/raw/balldontlie_team_stats_<season>.parquet`.
- Production team-model feature selection now gates `bdl_*` features off by default while preserving BDL ingestion and feature generation. Training and tuning can opt in with `--include-bdl-features`, and feature-family ablation still opts in automatically so BDL lift can keep being measured.
- Team-model feature-family ablation is now part of evaluation output at `predictions/evaluation/feature_family_ablation.csv`. It compares `baseline_no_bdl`, `baseline_plus_bdl`, and `baseline_plus_bdl_no_sparse_sack_rate` on the same CV folds, so BallDontLie feature lift can be measured without retraining the production model variants by hand.
- Current team prediction outputs:
  - `predictions/predictions.csv`
  - `predictions/predictions_full.csv`
  - `predictions/history/*.csv`
- Current team prediction CSVs now include market-policy guardrail fields and public recommendation labels so the website can distinguish `recommended`, `lean`, `research_only`, `review`, and `no_bet` rows without reimplementing policy logic.
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
  - `predictions/evaluation/player_props/prop_line_gap_audit_week_NN.csv`
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
  - `edge_publish_status`, `edge_reason`, `edge_required_threshold`, `edge_current_probability_edge`, `edge_policy_sample_size`, and `edge_policy_historical_roi`
  - Price-aware EV fields account for sportsbook vig. Recommendation flags now require a bettable current row, a historically positive market threshold from `edge_recommendation_policy.csv`, and positive current expected value after juice.
- Current-week ESPN roster, depth chart, and injury snapshots are now wired into the pipeline for explicit prediction weeks:
  - `src.data.espn_rosters`
  - `data/raw/espn_rosters_<season>_wkNN.parquet`
  - `data/raw/espn_depthcharts_<season>_wkNN.parquet`
  - `data/raw/espn_injuries_<season>_wkNN.parquet`
- Unified current-player availability is now built from ESPN, nflverse, and BallDontLie sources:
  - `src.player_availability`
  - `data/processed/current_player_availability.parquet`
  - Output rows are stamped to the requested prediction season/week even when the nflverse fallback uses the latest known roster row from an earlier source week.
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
- MattyTheBookie archive pages preserve immutable prediction snapshots by season/week and show graded team results (final scores, SU, ATS, moneyline) as an overlay. Player-prop result overlays are not built yet.
- Archive-to-results grading (`src.evaluation.live_results`):
  - `src.evaluation.live_results`
  - `predictions/evaluation/live_prediction_vs_actual_week_NN.csv`
  - `predictions/evaluation/live_prediction_vs_actual.csv`
  - `predictions/evaluation/live_prediction_vs_actual_summary_week_NN.csv`
  - `predictions/evaluation/live_prediction_vs_actual_summary.csv`
  - `predictions/evaluation/live_prediction_vs_actual_segments_week_NN.csv`
  - Each game is graded on its pick of record when the ledger has one; other games fall back to `predictions_log/<season>/week_NN_published_override.csv`, then the MTB season/week CSV, then `predictions_log` locked predictions. Final scores come from the nflverse schedule (`--refresh-scores`). Re-grading a week replaces that week's rows in the cumulative files; archived snapshots are never mutated.
  - `--completed-weeks` backfills every finished week; an explicit recovery pick set is graded with `grade_week(..., prediction_file=...)` (the CLI's `--prediction-file` applies only without `--completed-weeks` / `--require-complete` / `--refresh-scores`, and then takes scores from `matchup_features`).
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
  - `analysis/market_disagreement_segments.csv`
  - `analysis/market_probability_edge_calibration.csv`
  - `analysis/market_model_confidence_calibration.csv`
  - `analysis/market_model_confidence_recalibration.csv`
  - `analysis/market_confidence_recalibration_detail.csv`
  - `analysis/market_confidence_recalibration_backtest.csv`
  - `analysis/market_confidence_recalibration_decision.csv`
  - `analysis/market_blend_backtest.csv`
  - `analysis/market_blend_summary.csv`
  - `analysis/market_blend_policy.csv`
  - `analysis/market_disagreement_games.csv`
  - `analysis/market_disagreement_diagnostics.csv`
  - `analysis/market_disagreement_recommendations.csv`
  - `analysis/market_roi_spread_edge_bins.csv`
  - `analysis/market_spread_key_numbers.csv`
  - `analysis/market_spread_calibration.csv`
  - `analysis/market_spread_cover_recalibration.csv`
  - `analysis/market_utility_segments.csv`
  - `analysis/market_edge_policy.csv`
- Website-ready model metrics export exists for MattyTheBookie About-page consumption:
  - `analysis/model_metrics_week_NN.csv`

## Current Gaps

- `GOALS.md` is the source of current roadmap truth. Older plans and status reviews with stale metrics were moved to `docs/archive/` on 2026-11-14.
- The model still trails the market on 2026 AUC (0.733 vs 0.742, 150 games; 0.7241 vs 0.7250 overall). See Team Model Improvement below.
- The Phase 1-5 validation modules (live tracking, paper trading, calibration monitoring, benchmark comparison) and the Streamlit dashboard were removed on 2026-11-21 (cleanup PR C); `src.evaluation` (walk-forward vs market, picks of record, live results) replaces them. CLV detail/summary CSVs are now generated in the weekly pipeline from local pick-time and close/latest odds snapshots.
- Current measured coverage is low. Treat any old >70% coverage claim as stale until a fresh full-suite coverage run proves otherwise.
- Player prop labels, split feature tables, baseline scoring, trained-model evaluation, weekly trained-model CSV exports, optional PropLine sportsbook ingestion, historical line joins, market-aware over-probability diagnostics, and first edge-threshold backtests now exist for the core yardage/sacks markets. Direct QB/RB/WRTE touchdown markets and anytime TD scorer are now published as projections; first/last/2+/3+ TD scorer novelty markets now have PBP-derived labels, a separate novelty feature/model family, and research-only weekly output. QB passing TD and anytime TD have exact current line coverage; RB rushing TD and WR/TE receiving TD now have non-bettable anytime-TD fallback lines and explicit `research_only` publish reasons, but still need true exact sportsbook line coverage before recommendation gates can use them. Remaining prop gaps are confidence-tier calibration, broader defensive-sack sportsbook/vendor coverage, historical touchdown line calibration, profitable edge validation, historical novelty-market calibration, and closing coverage gaps where offered PropLine player/game/market pairs sit outside the prediction universe.
- Current-week roster validation is integrated and needs ongoing production-output review for false positives, missing ESPN matches, and player-name/team alias mismatches. Week 5 currently has 0 unknown roster rows, 15 team mismatches, and 168 inactive/practice-squad roster exclusions.
- Yahoo Fantasy remains blocked pending API access approval.
- SportsDataIO is retained only as a fallback while BallDontLie coverage is verified.
- NFLVerse ingestion now prefers maintained GitHub release assets for PBP, schedules, rosters, injuries, snap counts, depth charts, and player stats, with `nfl_data_py` retained as fallback. Current-season nflverse release files can still lag or be unavailable, but the recurring wrapper-only PBP failure path is no longer the primary path. BallDontLie team-form features still cover some current-season team context, but PBP-specific feature families such as EPA, pressure, and red-zone detail depend on nflverse/ESPN PBP until equivalent replacement logic is proven.
- BallDontLie team-form features are useful as a monitored fallback source, but latest Week 4/Week 5 ablation runs showed win-probability regression and only marginal spread benefit. Keep them out of production model features until repeatable same-split lift appears.
- Archived MattyTheBookie prediction snapshots can now be joined to finalized game results in Python, MattyTheBookie displays the result overlay without mutating archive snapshots, and `analysis/model_metrics_week_NN.csv` ingests the live-published summary when `live_prediction_vs_actual_summary_week_NN.csv` exists.
- Market benchmarking now publishes model-vs-Vegas probability comparisons, model-vs-no-vig-Vegas probability calibration by bin, model-confidence calibration and empirical recalibration by agreement/disagreement segment, walk-forward confidence-recalibration strategy decisions, walk-forward market-aware blend backtests, favorite-disagreement game diagnostics/recommendations, moneyline disagreement threshold ROI, moneyline disagreement season-stability gates, bounded moneyline disagreement edge-bin ROI, disagreement-only segment diagnostics by side/odds/confidence/edge/CLV, segment-aware and confidence-aware moneyline disagreement policy gates, corrected spread ROI, spread edge-bin ROI, spread key-number diagnostics, spread cover-probability calibration, weekly CLV detail/summary artifacts, and optional per-game CLV fields in `market_benchmark.csv` when matching pick-time and close/latest snapshots exist.
- Corrected spread ROI no longer shows broad profitability. Current Week 5 policy artifacts mark broad spread threshold strategies as `no_bet`; only the 2-to-3 point spread-edge bucket remains a low-confidence lean/research signal.

## Open Items And Watch List

As of 2026-11-24. Remove items when done; move lasting results into the sections below.

| Item | When | Notes |
|---|---|---|
| 2026 play-by-play and player-stats duplicate rows | **Merged 2026-11-18** (PR #17, merge 862f494), before the Thu 11-19 Week 11 slot. Moved up from Tue 11-24 because the live site showed inflated player projections and novelty TD props. Pre-merge copy of `data/processed/matchup_features.parquet`: `C:\Code\pr17-baseline\matchup_features_pre_pr17.parquet` (sha256 b0d5b3a5...c87d0f, size and mtime match the original). **Post-merge check ran 2026-11-20 (result: ALERT, reviewed 2026-11-20 and accepted: the 3 ALERT columns are not model inputs, and the TB_MIA news change is unrelated to PR #17; report: `docs/checks/pbp_dedupe_week11_report.md`):** the manual Week 11 slot (MTB 4194824) was compared. 14 of the 17 expected model inputs changed on 2026+ rows, none outside the expected set; drive_completion_rate_rolling3_diff, rz_trips_for/against_rolling3_diff did not change, as expected: both count unique drives (drive features group plays by drive and take the max of the drive's scored flag; red-zone trips keep one row per game, drive and team), so a duplicated play row adds no drive. The rates built on play counts (rz_td_rate, sack rates) did change. 3 ALERT lines for non-model columns (pstats_def_fumbles_pct_change_away, pstats_pt_downed_pct_change_home, pstats_pt_out_of_bounds_pct_change_away; 1-2 rows each). Pre-2026 vs Monday's copy: qb_recovery_score_rolling3_* is float noise (no difference above 1e-6); 2025_17_TB_MIA news features changed (2 more news items, latest now 2.0 h before kickoff instead of 26.2 h). **Checked 2026-11-20, no leak:** both items are Rotowire notes on Dee Eskridge (ESPN 4043016): 62524528 "doubtful for Sunday's game against the Buccaneers" (2025-12-26 20:38 UTC) and 62543328 "placed on IR Saturday" (2025-12-27 21:58 UTC). ESPN's own published/lastModified match our stamps, and both are before kickoff (2025-12-28 18:00 UTC). They arrived when the 2026 news fetch pulled his feed. Side note: news hours are measured from game-day midnight UTC (the schedule's `gameday` has no time), not from kickoff, so "2.0 h" means about 20 h before kickoff, and items from midnight UTC to kickoff are dropped. That is conservative, not a leak. Board: no side, recommendation or status flips; 2 tier flips (1 defense, 1 public). Original plan: after the first Week 11 run, `bash C:/Code/pr17-baseline/thursday_compare.sh 11` runs `tools/check_pbp_dedupe_changes.py` against that copy and against a pre-merge-code build of the same data, and compares the Week 11 public board and projections. Its step 0 refuses to start until the run, the MTB publish and the deploy have finished. | Found 2026-11-17 (MTB review item 2). Cause: `_merge_and_write` in `src/data/nflverse.py` concatenated cache + download and dropped only rows identical in every column; nflverse revises in-season rows, so every revision was kept. Fix: a refetched season replaces its cached slice for PBP and player stats. Row counts, 2026, before -> after: PBP 82,296 -> 26,246 (all unique); player stats 12,014 -> 10,446. 2025 clean. 17 of 78 pbp-derived model features change on 2026 games (drive EPA rolling3 ~4.3x inflated, red-zone TD rate ~3.3x, adj-efficiency/pass EPA per dropback ~1.15-1.2x, sack rates ~0.9x); pre-2026 unchanged. Player-stats dedupe has no effect on the win model. Decision 2026-11-18 (user, option a): ship PBP + player-stats dedupe + sorted column order; injuries held. Walk-forward 2026 wk2-10 (n=132), 5 model seeds, live -> fixed: accuracy 0.6818 -> 0.6833; Brier 0.2123 -> 0.2117 (better 4/5); log loss 0.6160 -> 0.6144 (4/5); AUC 0.7359 -> 0.7388 (4/5); ECE 0.0758 (sd 0.0054) -> 0.0773 (sd 0.0063), change +0.0015 (sd 0.0067), no worse in 3/5 -> passes the noise rule. Per-seed ECE live/fixed: 0.0743/0.0843, 0.0758/0.0739, 0.0752/0.0819, 0.0694/0.0685, 0.0845/0.0780. Market: Brier 0.2104, log loss 0.6110, AUC 0.7474, ECE 0.0750. Calibrator check 2026-11-18: refit on each arm's walk-forward history (5 seeds x 2 arms, with and without 2026 wk2+) selects `none` every time, as live does, so ECE with refit = without (0.0773 vs live 0.0758). Forced sigmoid (runner-up), fit on 2024-2026 wk1: Brier 0.2117 -> 0.2104 but ECE 0.0773 -> 0.0896 (live 0.2123 -> 0.2111, ECE 0.0758 -> 0.0885); selection still prefers `none` over 2019-2026. Volatility shrink touched <= 1 of 132 games. **Before merging:** save a copy of `data/processed/matchup_features.parquet`. **After the first run post-merge:** run `python -m tools.check_pbp_dedupe_changes --before <copy>` and send the user the result. ALERT (exit 1) on: any change in a pre-2026 row (any column); a changed column not in `docs/checks/pbp_dedupe_expected_feature_changes.csv` (1,556 columns: 150 pbp, 126 `sr_*` built from pbp, 1,280 `pstats_*`); a non-negative count/sum that rises (exceptions, reported as notes: signed yardage sums, and teams whose `sr_games_sampled` rose because the refetch has a game the old cache lacked); any model input other than the 17. Validated on the walk-forward feature sets (OK; negative control with the injuries fix alerts). The picks-of-record warning itself will name a subset (its fingerprint skips names with pick-time markers such as `kickoff`, `qb_`). **Props impact (Week 10 rerun, main vs PR #17, full props chain in isolated copies; main copy reproduces the published board exactly):** QB/offense/defense props identical; novelty two-plus/three-plus TD props all fall (duplicated TD plays counted as several TDs; e.g. Diggs 3+ TDs 0.592 -> 0.004), first/last TD unchanged; public board 42 of 147 rows change probability (max 0.013), no side/recommendation/status/tier flips, top 10 unchanged, 93 rows re-sort; player projection table: 18 of 63 QBs (Murray pass yds 359.8 -> 170.3), 98 of 273 offense, 222 of 246 defense change, top-10 membership changes for 7 offense / 34 defense players. |
| Walk-forward and live models are not reproducible | Merged with PR #17 (2026-11-18, 862f494) | Found 2026-11-17. `make_feature_diffs` (`src/models/feature_columns.py:104`) builds `_diff` columns by iterating a `set`, so column order follows the per-process string hash seed; with `colsample_bytree` ~0.34 the seeded sampler then picks different features each run. Same data, two runs: mean 0.004 per-game probability difference (CPU and GPU), as large as the PBP fix. Every live run is affected (part of run-to-run jitter). Fix: iterate in sorted order (one fixed ordering; changes outputs once). Fixed in PR #17 (sorted order) with `tests/test_model_reproducibility.py` (3 interpreters, PYTHONHASHSEED 1/2/3, identical probabilities). Verified: same data under hash seeds 11 and 999 gives identical walk-forward output. Pre-2026 ECE 0.0209 -> 0.0230 (all 5 seeds) from the fixed ordering alone; Brier/log loss there unchanged. |
| Injuries dedupe scores worse (held) | **Recheck Tue 12-15** (after Week 14 is graded) | `nfl_injuries` 2026: 5,085 rows -> 3,168 (1,917 repeated (season, week, gsis_id, team)); 2025 clean. Fix is on branch `fix/injuries-replace` (pushed, 0a13785, on top of PR #17; no PR), not shipped. Walk-forward 2026 wk2-10, 5 seeds, PBP-fixed -> PBP + injuries fixed: Brier +0.0001, log loss +0.0003, ECE +0.0108 (better 1/5) -> fails the noise rule. Investigation 2026-11-18: only 2026 rows change (pre-2026 identical), so the model never trained on duplicated injury history. Duplicates inflate practice-limited counts ~2.4x (2026 mean per team 6.0 dup vs 2.5 clean; 2025 2.44) and practice-full ~1.1x; out/questionable/listed barely move. 25 of 42 model injury features change, mostly the practice-limited family (~100 of 132 eval games). Report-frequency hypothesis rejected: duplication excess vs home win corr -0.04 to +0.07 (SE ~0.087). Likely cause: small-sample noise. History 2018-2025: more home limited-practice players -> slightly worse home result (corr -0.066 for `inj_practice_limited_diff`); 2026 wk2-10 clean values run the other way (+0.118, ~1.4 SE), while inflation blurs them (dup +0.037). Clean data matches the training distribution; duplicates keep growing with each refetch. **Which version the fix keeps (checked 2026-11-18):** the duplicates are revised reports for the same player-week (e.g. Wednesday Limited -> Friday Full, then a game status), and `date_modified` is empty for 2026, so the file itself can't date versions. The branch as written keeps the version from the latest fetch, i.e. the week's final report, not the latest report available at pick time. Evidence: no week 1-9 row has a substantive revision after the next week's reports appear (only 4 `full_name` spelling fixes); but comparing each game's last pre-kickoff `predictions_full` dump with today's cache, 7 of 518 game-status counts (out/questionable) rose after the pick, in 5 of 132 games (NO-DET wk1, PIT-CLE wk4 1.7 h lead, PHI-JAX wk5 and HOU-JAX wk6 picked ~66 h early, CIN-ATL wk9). So for ~4% of 2026 games the final report includes post-pick information (the duplicated cache sums all versions, so it includes it too). **Required before shipping:** record a fetch timestamp per injury version and have the dedupe keep, for each game, the latest version fetched before that game's pick time (live runs are unaffected: they only see versions fetched so far). Versions already cached have no timestamps, so this needs logging to start in production. **Logging merged 2026-11-18 (PR #20):** each distinct injury download is saved per season to `data/raw/history/nfl_injuries/injuries_<season>_<UTC>.parquet` with `manifest.csv` (identical downloads skipped by content hash); first entries come from the first Week 11 run, and every pick run downloads before predicting, so the log holds the version each pick saw. PR #20 merge commit: eba0b2b. **Done on `fix/injuries-replace` (0a13785, 2026-11-18):** `src/data/injury_versions.py` keeps, for each game, the latest logged version downloaded before that game's pick (the pick of record's `recorded_at`, otherwise kickoff minus 2 h, never later than now). Team-weeks with no logged version before the pick (everything before logging starts 11-19) keep their cached rows. `build_features` prints how many team-weeks came from each source. Tests: `tests/test_injury_versions.py`. **For the Week 14 comparison:** live picks only ever saw versions downloaded before the pick, so the after-pick information affects backtests only, including the earlier "clean injury data scores worse" result (~4% of games). Rebuild the injury backtest with dated versions before drawing conclusions (weeks 11+ from the log; weeks 1-10 have no dated versions, so use the pick-time `predictions_full` dumps to identify and handle the affected games). Next, Tue 12-15: rerun the 5-seed walk-forward on 2026 wk2-14 (~190 games) under the noise rule (branch `fix/injuries-replace` vs PR #17 as merged), and ship if it passes. **Also measure duplication growth for that decision.** Method (raw, repeatable, no feature rebuild): in live `data/raw/nfl_injuries.parquet`, 2026 rows; practice-limited = `practice_status` contains "limited"; mean count per (week, team) using all rows vs one row per (season, week, gsis_id, team), keeping the last. Baseline 2026-11-18 (through wk10): 2.24x overall (5.39 vs 2.40 per team-week; feature-level ~2.4x on game rows); rows 5,085 vs 3,168 unique keys. By week: wk1 1.00, wk2 2.05, wk3 3.19, wk4 2.30, wk5 1.90, wk6 1.97, wk7 2.48, wk8 2.75, wk9 2.20, wk10 2.65. Unlike PBP, older weeks are not more inflated, so copies seem to come from in-week report revisions rather than piling up on each refetch. If the overall figure has risen by 12-15, holding the fix is moving live features further from the training data, and that counts toward shipping. |
| Picks-of-record warning names changed features | Done (this branch) | The ledger fingerprint is unchanged; the fingerprinted values are now also kept locally in `predictions/picks_of_record/feature_snapshots_<season>.parquet` (not published), and the warning lists each flagged game's changed columns with old->new values. LAC-BAL itself cannot be itemized (no Sunday snapshot); the published CSVs show the `def_*_allowed_*` group changed. |
| Early-slot retries | Done (this branch) | Cause of the 11-15 Sun-early failures was the run-log encoding crash (PR #12; reproduced 2026-11-17 with the old wrapper under cp1252: exits 1 with UnicodeEncodeError; the current wrapper exits 0). PR #14 logs and retries dead data jobs. Dispatcher now allows 3 attempts and retries an attempt that failed within 10 min after 60 s in the same tick. The missing 04:15 tick on 11-15 is unexplained (no Task Scheduler history); most likely the crashed wrapper's orphaned child held the task's output pipe so the task still counted as running (IgnoreNew). |
| Venue columns for MTB | Done; first published in the Fri 11-20 manual Week 11 slot (MTB 4194824; all 12 games have venue fields), rendered by MTB since its PR #15 (361748d, deployed 2026-11-21) | `predictions.csv` and `predictions_full.csv` gain `venue_name`, `venue_city`, `venue_country`, `neutral_site`, `venue_lat`, `venue_lon` (from `src.utils.venues.game_venue_table`, output only). `predictions/game_venues_<season>.csv` (every game of the season) is published to MTB `data/prediction-csvs/reference/game_venues_2026.csv` for past weeks. Coordinates in features and published files have been correct since PR #4 (2026-11-13); the blank/0,0 values MTB saw were in October publishes, and Ford Field for NE-DET only in the Thursday 11-12 file. No prediction changes. |
| Blog failures don't reach the alert (Week 10 recap was missed) | Done 2026-11-17 (PR #18) | Recap republished by hand 2026-11-17 06:33 MST via `scripts/publish_site.py` (MTB b263f27, deploy passed, 2 attempts / 106 s). 2026-11-17 grading (~04:15 ET): graded 14 games, SU 10-4 (picks_of_record 12, mtb_archive 2; the Thursday and London games); season 85-43-1 (66.4%, 129 games). MTB published 127a929 and the deploy passed, but `blog:auto` logged `[blog] recap week 10 failed: terminated` (the Claude API stream dropped mid-response; `lib/blog/claude/provider.ts` does not retry a dropped stream) and `scripts/publish_site.py` still exited 0, so the SUCCESS alert didn't mention it. Fixes to consider: (1) retry a dropped stream once in the MTB provider; (2) have `tools.review_run` / the alert flag `[blog] ... failed` lines from the publish log. |
| Tomorrow.io 429s on next-week games | Merged 2026-11-24 (PR #38, 8772cca); first live use Wed 11-25 slot | Cause: Tomorrow.io's per-second limit; calls went back to back and a 429 was retried 0.6-1.2 s later (the Mon 11-23 run lost `2026_12_NE_LAC`, next week, no pick affected). Monday windows hold many games beyond the 120 h hourly horizon, each costing a useless hourly call plus a daily one (33 requests on 11-23). Fix: >= 0.4 s between calls; on 429 honour Retry-After (cap 10 s) else 1/2/4 s, 4 attempts; skip the hourly call beyond 133 h. Live check: 17 requests, 0 429s, 16/16 games. Watch the Wed/Thu debug logs' `Tomorrow.io request summary`. |
| nflverse PBP merge ran out of memory | Merged 2026-11-24 (PR #35, 6c686ab) | Mon 11-23 slot: `src/data/nflverse.py` `_drop_refetched_seasons` failed to allocate 1.54 GiB copying the cached PBP (1.14M rows); the data-job retry succeeded and the cache is intact (1,171,328 rows, 2002-2026; `saved PBP rows=` logs only the re-fetched season). Fix: drop refetched seasons in place before normalising, shallow copy in `_normalize_pbp_columns`, dedupe only the refetched rows for replace-on-refetch datasets, and in `src/utils/io.py` `_sanitize_for_parquet` a shallow copy plus column-wise NaN->None instead of a whole-frame `where`. Measured on the real cache (nflgpu, 2026 refetch): peak commit 24.5 -> 14.6 GiB (merge phase 23.7 -> 11.8), 280 s -> 237 s. Output identical: merged PBP and player stats files exactly equal old vs new, and `_sanitize_for_parquet` old vs new identical on every `data/raw` parquet plus `matchup_features.parquet`. |
| Fingerprint warnings from nflverse EPA revisions | Merged 2026-11-24 (PR #36, 45de35c), before the Wed 11-25 slot | The Mon 11-23 review flagged 8 games. Of the 7 with snapshots, 6 had no model-input change at all (only `roster_*` sums); `2026_11_CIN_WAS` had 58 changed columns but only 2 model inputs (`off_adj_eff_rolling3_diff`, `def_adj_eff_rolling3_diff`), from nflverse rewriting past-play EPA (24,062 of 26,071 week <= 10 plays differ from the 11-16 snapshot; not leakage: no week 11 rows feed either team, pick moved 28.0% -> 27.7%). Fix (user's choice): the fingerprint covers only the saved models' input columns minus pick-time inputs (74 of 2,348 columns in Week 12; `*_diff` inputs rebuilt with `make_feature_diffs`), falling back to every column if the models can't be read. No week 12 ledger rows existed, so no spurious flags at the switch. A real EPA revision can still flag a game through the rolling adj-eff diffs; that is now a model-input change worth the review's attention. |
| `roster_*` sum features | Checked 2026-11-24: not model inputs | `_aggregate_team_week` (`src/features/build_features.py`) sums every numeric column per team-week for rosters, snap counts, depth charts and player stats, so `roster_*` holds sums of player IDs, jersey numbers, heights, weights. No model uses them: the training allowlist (`src/models/feature_columns.py` `PREGAME_DIFF_PREFIXES` / `ROLLING_PERFORMANCE_PREFIXES`) has no `roster_`, `snaps_`, `depth_` or `pstats_` prefix, and the saved win/spread models (182 inputs, tree + logistic) contain none; `roster_injured_pct_diff` is a logistic candidate but not selected. No model change or ship test needed. They only fed the picks-of-record fingerprint: see the next row. Removing them from the feature build is optional cleanup. |
| Pick-time vs closing-line report (Step 4) | Merged 2026-11-24 as a manual tool (PR #37); re-run by hand after each grading | `python -m src.evaluation.pick_time_vs_close --season 2026` (not in the pipeline) compares each graded pick of record's pick-time market (Odds API, `market_*`) with the nflverse schedule close, the line the walk-forward benchmark uses. Weeks 10-11, 23 games (CIN_WAS waits for the schedule cache to get its score): picked 4.5h before kickoff on average (3h for a slot's first games, 6-10h for later ones). Market moved 1.0 no-vig prob pts on average (max 3.4), spread 0.24 pts, moved in 30% of games; bigger for 6-10h picks (1.3 pts, 0.44 spread) than ~3h picks (0.8, 0.13). Brier: model 0.1733, pick-time market 0.1769, closing market 0.1720. So the model beats the market it was picked against (-0.0037) but trails the close (+0.0013), and the close is ~0.005 Brier harder than the pick-time market: live 'beats the market' figures are not comparable to walk-forward vs close. Moves toward the model's lean: 13 of 22 (p=0.52), avg CLV +0.13 prob pts: no signal yet. Caveats: two sources (Odds API vs nflverse consensus), so part of the 'move' may be source difference; tiny sample. Kept as a manual tool (user, 2026-11-24): run it by hand after each grading; consider adding it to the dispatcher's weekly grading later. |
| Xweather keep/drop decision | Around Week 13 | After 3-4 weeks of trial snapshots; see [docs/weather_sources_review.md](docs/weather_sources_review.md). |
| Home-vs-market bias watch | Every run | `[market_bias]` lines in the debug log; see [docs/market_bias_watch.md](docs/market_bias_watch.md). 2026-11-15 Week 10 Sunday slate: WATCH, home below market in 12/12 games, mean -2.2 pts (t=-7.4); season (-0.6 pts, t=-3.8) and last 8 weeks also WATCH; all history -0.45 pts (INFO). Decide on a home-gap correction if it persists for another slate. |
| Close the 2026 AUC gap to the market | Ongoing | Step 5 feature pruning closed 2026-11-24, not shipped (see the Step 5 row in [Team Model Improvement](#3-team-model-improvement)): leak-free per-fold pruning of the win model gains AUC +0.0004 / Brier -0.00014 overall (about 40% of the 0.001 overall AUC gap) but worsens calibration (2026 ECE +0.0045, 5/5 seeds), and the pipeline's calibrator rule keeps choosing `none`. Next model step not yet chosen. |
| News features cut off at midnight UTC on game day | Held 2026-11-24 (user); recheck after the 2026 season | Player and team news features count stories before the schedule's `gameday` (midnight UTC), so same-day news (inactives) never reaches a pick. Branch `feat/news-pick-time-cutoff` (df5cd0c, pushed, no PR): `_news_pick_time_utc` uses the dispatcher slot start (first kickoff of the slot minus 3h30; morning international games their own slot: Thu 16:45, London 06:00, Sun 09:30, Mon 16:45 ET), not kickoff minus 1-2 h, which would show walk-forward news a live pick never sees. Experiment (same raw-data snapshot, baseline features identical to live): news columns change for 397 games (2024: 29, 2025: 189, 2026: 179), but walk-forward probabilities move only for 81 games in 2026, because folds before 2026 train on seasons with no news (history starts 2025). Walk-forward 2018-2026, 5 paired seeds (42, 1-4), news - main: Brier -0.000004 (sd 0.00009, worse 0/5), log loss -0.00001 (sd 0.00022, 1/5), AUC +0.00001, accuracy 0, ECE +0.0001 (sd 0.0020, 'worse' 3/5 by ~1e-5 to 5e-4): no measurable effect; ECE fails the majority clause on near-zero margins. 2025-26 subset and changed games: same picture. Kept off per this row's own rule (ship only if walk-forward improves). Recheck when more news history exists. Side finding: the seed-42 baseline matched the live walk-forward on 2,245 of 2,389 games; the other 144 (1-3 per week, every season, max 0.013) look like run-to-run GPU nondeterminism. |
| Unused-code cleanup (PRs A-D) | All merged. A 2026-11-21 (#26, f79f11a); B, C, D 2026-11-22 (#27 a12b25e, #28, #29 bee9429), merged on the user's call during the Week 11 Sunday slot instead of after grading. Re-reviewed before merging: 1032 tests pass with only the 3 known failures, all 51 pipeline job modules import, and the merged `main` tree is identical to the reviewed stack (dbc15af). Deleting #27's branch closed #28 again (restored, reopened, retargeted), so retarget the next stacked PR before deleting. The dispatcher checkout was pulled to bee9429 after the Sunday slot finished (08:10 MST, slot succeeded on the old code); the 51 job modules import in `nflgpu`; the `C:/Code/nfl-cleanup` worktree is removed. Monday's slot and Tuesday's Week 11 grading are the first runs on the cleaned code | Started 2026-11-21. An import graph from the pipeline, dispatcher and publish entry points found modules nothing runs. Plan, in four PRs: A tracked junk (`tmp/`, old test backups, root `tuning.db`, `docker-compose.yml`); B the unused April 2026 framework layer (async pipeline, base fetchers, data loader, cache, retry, circuit breaker and similar) plus Sportradar; C the Phase 1-5 validation suite and Streamlit app (replaced by `src.evaluation`); D the shelved long-memory strength experiment (Step 6). Files are deleted, not moved; each is listed with its restore commit in [`docs/archive/removed_code_2026-11.md`](docs/archive/removed_code_2026-11.md). Pipeline outputs must not change; B and C merge after Week 11 grading and never during a dispatcher slot. |
| ESPN player news speed-up | Fix in PR (2026-11-25), merge on the user's OK | Measured 2026-11-25: the fetch took 17-19 min in every run since 11-14 (~2,150 players, one request at a time, 0.15 s sleep) and finished last among the data jobs, so the whole run waited on it (next slowest: balldontlie_live, ~3 min). The 11-13 run that skipped it (4 h cache) took 19.3 min vs ~30. The news features are model inputs but weak (best SHAP rank 85 of 171; `news_count3/7_diff` and `news_injury_kw7_diff` zero), and ~2,000 of the players have news, so the fix is concurrency, not fewer players. Benchmark (300 ids per setting): 1/4/8/16 workers -> est. 12.2/3.1/1.5/0.8 min, all HTTP 200, no 429s, latency flat. Fix: the pipeline passes `--max-workers 8`; one HTTP session per worker thread; one retry on 429/5xx/timeouts; failed requests are counted and logged as a `Warning: ... requests failed (...)` line, which the run summary lists as an issue (before, any error was silently treated as "no news"). Full fetch in nflgpu with 8 workers, written to a worktree copy: 88 s, 2,155/2,155 requests OK, 7,428 rows (Mon 11-23 run: 7,414). Expected saving ~11 min per run. |
| Weather rows missing coordinates | Merged 2026-11-24 (PR #41, 5d92dfb) | The `[weather][diag] feature-overlap missing venue(lat,lon)=(2,2)` line in every run: NOAA rows for Rams home games. `_load_weather_sources` fills NOAA rows' venue coordinates from `team_locations.csv` by home team, but the schedule calls the Rams `LA` and the file has `LAR` (`_norm_abbr` keeps `LA`). Visual Crossing rows win once their forecast exists, so only the far-future `2026_15_DAL_LA` and `2026_18_SEA_LA` showed. Fix: apply the `LA` -> `LAR` alias (as `src/utils/venues.py` does) in that join. Live data, old vs new: only those 2 rows' `venue_lat`/`venue_lon` change; no model input changes (travel uses `_load_venue_travel_context`, else the home base, which is SoFi either way). |
| Dependabot alerts (pyarrow, lxml, requests) | Tested in a clone 2026-11-24 (user's choice); live switch waits for the user's OK, after the Wed 11-25 slot | Clone `nflgpu-dep` (conda clone of nflgpu + pyarrow 23.0.1, lxml 6.1.0, requests 2.33.0; pandas 2.3.3, numpy 2.1.2, xgboost 2.1.4 unchanged). Results: full suite on `main` 1080 passed / 1 skipped in both nflgpu and nflgpu-dep (run via `conda run`; calling an env's python.exe without activation crashes scipy on missing DLLs, and `tests/test_blog_failure_alerts.py` can't be collected in either conda env because a site-packages `tests` package from myql/yahoo_oauth shadows the repo's `tests`; CI and .venv are unaffected). Every parquet under `data/` (379 files, 359 non-empty) reads with identical content, rows and dtypes under pyarrow 22 and 23; copies re-written by pyarrow 23 read back identically under both 22 and 23, so switching back is safe. `pd.read_html(flavor="lxml")` and requests work. To ship: upgrade nflgpu in place (or swap envs), update `requirements-ci-constraints.txt` / `requirements.txt` pins in one PR, then watch one slot. 6 open alerts (4 high, 2 moderate) for 3 packages, on `requirements.txt` and the CI pins (which name the live versions). Live nflgpu: pyarrow 22.0.0 (fix 23.0.1: use-after-free reading IPC files; we read parquet), lxml 6.0.2 (fix 6.1.0: XXE via `iterparse` / `ETCompatXMLParser`; HTML scraping doesn't use those), requests 2.32.5 (fix 2.33.0: temp-file reuse in `extract_zipped_paths`, local). Low exposure. Upgrading means changing the live env and the CI pins together, then a full test run and one watched slot. |
| Step 8 cleanup | Minor | Split `src/models/train.py` and `src/predict/predict_upcoming.py`. |
| Find the stage that sometimes un-stepped final probabilities in September/early-October runs | Record only | No current effect; see Known Data Issues. |
| Season-to-date published-pick stats on MTB | NFL side merged 2026-11-24 (PR #43, 53352f1); MTB display change open | 2026-11-24: `analysis/model_metrics_export.py` now adds a `live_published_season` section (sort orders 101-109) built from the cumulative `predictions/evaluation/live_prediction_vs_actual.csv` with the grader's own `summarize_detail`, picks of record only, weeks up to the export week: `live_season_su_record`, `_accuracy`, `_auc`, `_brier`, `_log_loss`, `_ats_record`, `_ats_hit_rate`, `_moneyline_units`, `_moneyline_roi`. The record row's description names the weeks and the no-pick count. Week 12 dry run on live data: 93-49-1 (65.5%), AUC 0.712, Brier 0.215, ATS 48-39-3 (55.2%), ML +2.93 units on 90 priced games (+3.3%), weeks 2-11, 4 no-pick games not counted; matches the week 11 grading report. Weekly rows unchanged (their labels already say "Latest graded"). MTB looks metrics up by key and no MTB page reads `live_published_*` today, so the new rows show nowhere until MTB adds them; that display change is MTB's to make. Before: `analysis/model_metrics_week_NN.csv` (the site's stats page) only has latest-graded-week published-pick rows. On 2026-11-15 that was Week 9, 15 games: 86.7% SU, AUC 0.98, +19% ML ROI, too small a sample to headline. The season record (75-39-1, 65.8%, 115 games, +5.92 units) appears only in blog posts. Idea: add season-to-date rows (record, accuracy, Brier, ATS, ML units) and label the weekly rows as one week's sample. Additive rows only; MTB needs a small display change. |
| Player props and projections show abbreviated names | Done (PR fix/prop-full-names), from Week 11 on | 2026-11-18: user saw it again on mattythebookie.com/picks ("J.HERBERT", "T.HARRIS" next to "LAJOHNTAY WESTER"). Fix: new `src/player_props/display_names.py` maps player_id to nflverse `player_display_name` (latest, multi-word only), falling back to roster `full_name`. Applied (a) in `src/player_props/predict.py` to the qb/offense/defense/novelty frames after odds matching, before the 5 props CSVs are written, and (b) in `src/predict/predict_upcoming.py` when writing `predictions_players_{qb,offense,defense}.csv` (MTB's player projection table). Checked on the Week 10 outputs: props 112 of 147 public rows abbreviated, 0 after; projections 46 of 46, 0 after. A/B of the props step with short vs full projection names: every prop value identical; only ties on the public board reorder by name (13 of 147 rows, top 10 unchanged). Nothing joins on player_name (baseline evaluation joins on player_id). Weeks 1-10 files not regenerated. New `--player-stats` arg; tests/test_player_prop_display_names.py. 2026-11-15: 311 of 481 players (1,273 of 1,894 rows) in `player_props_public.csv` show nflverse short names ("E.Johnson", "C.McCaffrey"). Cause: `src/player_props/labels.py:239-242` prefers nflverse `player_name` (abbreviated) over `player_display_name` (full); full names only appear for roster-expansion rows. Fix: use `player_display_name`, falling back to the identity table's full name, for the output name only, keeping internal matching keys as they are. MTB uses `playerName` only for display. Also: `player_identity.parquet` has some first-name-only canonical rows ("Emmett", "Christian"). |
| Props ignore a starter's absence (vacated role) | Gate merged 2026-11-24 (PR #44, b4f7f9a); feature still open | 2026-11-24 gate: `src/player_props/predict.py` `_attach_vacated_role_flags` flags a row when a different depth-chart rank-1 player at the same position (QB/RB/WR/TE) on the same team is Questionable or worse (injury report by gsis id, then name+team, then the availability frame's status/IR flag) and the line is at least 2x the projection (`VACATED_ROLE_LINE_RATIO`); not applied to novelty or sacks markets. Flagged rows get `edge_publish_status = review`, so they cannot be official picks and the signal caps at 64; `edge_reason` names the starter and status (the player's own injury review and exclusions still take precedence). No new output columns. Week 11 dry run (Monday game, main vs branch, same inputs): 1 of 153 public rows changed, WAS Robert Henry Jr. anytime TD (line 0.5, projection 0.13; RB1 Jacory Croskey-Merritt Questionable), no_bet to review. The week 10 Emmett Johnson case cannot be replayed (availability holds only the current week), so it is reproduced in tests/test_props_vacated_role.py. Still open: (a) the vacated-usage feature (needs retraining and the 5-seed ship rule); (b) a starter already moved off the depth chart (long-term IR) is not caught, because the backup is then rank 1; (c) "unlisted" starters (no injury report row) are not caught; (d) TD markets fire whenever a starter is Questionable because a 0.5 line is always at least 2x a backup's projection; watch how many rows that moves to review. 2026-11-15: the top "official pick" was Emmett Johnson (KC) Under 73.5 rush yds, signal 85, projection 24.4. KC's RB1 Kenneth Walker is Questionable with no line posted, so books priced Johnson as the starter. The model projects from Johnson's backup history (0-8 carries/week) and has no feature for a teammate's absence, and the pick gate does not flag "line far above projection after a role change". Ideas: a vacated-usage feature from teammate injury status; a gate that downgrades picks where the line is more than about 2x the projection and the starter at the position is out/questionable/unlisted. |
| Model-vs-market AUC on MTB | NFL side done (branch `feat/metrics-auc-vs-market`); MTB display change open | 2026-11-24: `analysis/model_metrics_export.py` adds a `walk_forward_vs_market` section (sort orders 161-169) from `predict_history`'s `predictions/evaluation/walk_forward_vs_market.csv`, written every run before the export: `wf_vs_market_games`, `wf_model_auc`, `wf_market_auc`, `wf_auc_minus_market`, `wf_model_brier`, `wf_market_brier`, and for the export season `wf_season_model_auc`, `wf_season_market_auc`, `wf_season_auc_minus_market`. Model = `model_final` (published walk-forward probability), market = no-vig close, same games. Week 12 dry run on live data: overall AUC 0.7244 vs 0.7252 (-0.0009), Brier 0.2104 vs 0.2106, 2,380 games; 2026 AUC 0.7383 vs 0.7425 (-0.0042), 161 games. Like the season-to-date rows, nothing shows until MTB reads these keys. The plain 'no betting edge yet' wording is an MTB copy decision; the inputs are already exported (agreement rate, disagreement games, model-side ML ROI, policy statuses). Before: the stats page compares Brier to Vegas (0.2109 vs 0.2110 when checked 2026-11-14) but not AUC (0.7241 vs 0.7250 overall, 0.733 vs 0.742 in 2026, from `walk_forward_vs_market.csv`, 2026-11-20 run). Idea: add an AUC-vs-market row. Also consider saying plainly that there is no betting edge yet: model/Vegas agreement is 98.2% (43 disagreement games), model-side ML ROI -3.1%, and the policy rows read `unavailable`/`no_bet`. |
| Week 11 Sunday check | Done 2026-11-22 | Reported OK (IND-HOU no pick shown, MTB cb9d124 deployed, run 45036232561). Task Scheduler task `MTB-Week11-SundayCheck` removed 2026-11-24. |
| Week 11 grading | Done 2026-11-24 | Graded 12 games, SU 8-4 (picks of record 12); IND-HOU score-only `no_pick_of_record`. Season 93-49-1 over 143 (includes the 11-18 weeks 5-7 transparency regrade: +2 losses, +3 no-pick rows). Sandbox worktree `C:/Code/nfl-props-main11` removed. |
| Live win-probability cap ignores the market | Merged 2026-11-24 (PR #40, bae1fa1) | Cause confirmed: `apply_probability_caps` (`src/predict/utils.py`) caps at 0.85/0.15 unless the market is more extreme (then market +/- 0.05, bounds 0.02/0.98). Walk-forward (`predict_history`) reads the market from the feature rows (`sched_implied_prob_home`, else `home_moneyline`); live `predict_upcoming` read it from `preds`, which holds only teams, weather and context, so the cap never relaxed and heavy favourites were held at 85% live. Fix: `_market_cap_input` reads the same feature-row columns (`this_week_nodup`, same index as `preds`). Walk-forward is unchanged (`predict_history.py` untouched). Replay on the Week 11 Sunday predictions (backup_153): 2 of 12 games change, MIA-BUF 0.85 -> 0.903 and ARI-KC 0.85 -> 0.906 (no-vig market ~0.90); the other 10 are identical. Affects future live outputs for heavy favourites only; past picks of record stay as published. |
| CI runs the tests | Done 2026-11-24 (PR #39, a7f110f) | New `test` job in `.github/workflows/ci.yml`: Python 3.10, `requirements.txt` installed with `requirements-ci-constraints.txt` (pins = the live nflgpu env; unpinned, pip resolved pandas 3.1 / numpy 2.4 / scikit-learn 1.9), `pip check`, then `pytest --no-cov`. Coverage stays off in CI because the 70% gate in pyproject is above the current ~65%; decide later whether to lower the gate or add tests. The `lint` job still runs `ruff check . || true` (never fails; many existing findings). Fixed for CI: `_load_sr_pbp_subset` returns an empty frame when `nfl_pbp.parquet` is missing (what its test expects; live runs always have it), `test_noaa_venues` points `RAW_DIR` at a temp dir, and the flaky `test_rolling_average_properties` allows float rounding from the running sum (passed 6 Hypothesis seeds). The first CI run also showed `requirements.txt` lacked `pydantic` (imported by `src/utils/pydantic_schemas.py`; 9 test modules failed to import), now added. First green run: 1072 passed, 1 skipped on Linux. Update the pins when nflgpu is upgraded. |
| Test environment issues | Fixed 2026-11-24 (branch `fix/test-import-shadowing`) | Found 2026-11-21. The nflgpu env has a stray top-level `tests` package in site-packages (it came with `yahoo_oauth`/myql). The repo's `tests/` has no `__init__.py`, so it is a namespace package and a regular site-packages `tests` always wins over it; the only casualty was `tests/test_blog_failure_alerts.py`'s `from tests.test_dispatcher import ...`, which stopped collection. It now imports `test_dispatcher` directly (pytest puts `tests/` on the path), so the full suite runs in nflgpu too: 1096 passed, same as `.venv`. Run conda envs through `conda run` (or an activated shell): calling an env's `python.exe` directly skips activation and crashes scipy on missing DLLs. The data-dependent test failures were fixed 2026-11-24 (see CI row). |
| MTB shared publish lock (dispatcher) | Merged 2026-11-23 (PR #31, 3e1228d), live checkout pulled before the Mon 11-23 slot; first locked run is Mon 11-23 (watch its publish log) | NFL and MLB both publish from `C:/Code/mtb`; MTB PR #22 (merged 2026-11-24, MTB 8b753dd; `C:\Code\mtb` is on it) adds `.publish.lock` to `publish_site.py`; first NFL publish checked against it: the Wed 11-25 slot. The dispatcher now holds that lock from the first CSV copy into MTB until `publish_site.py` finishes (`src/utils/publish_lock.py`, vendored from MTB e7df1ea; `sport=nfl`, stale after 90 min) and passes the token in `MTB_PUBLISH_LOCK_TOKEN`. To make that possible, `run_pipeline --mtb-publish-plan` defers its last two jobs (CSV copy, MTB JSON import) to the dispatcher. Lock busy or exit 75: slot `publish_pending`, no attempt used, no alert, the next tick publishes the same run (re-runs the pipeline if a game kicked off since); still pending when the window closes: failed + alert. Exit 77: failed + alert. 2026-11-23 (user OK): the lock wait went from 0 to 300 s (`--publish-lock-wait-sec`, `PUBLISH_LOCK_WAIT_SEC`), so an MLB publish (~8-10 min) already in progress usually clears within the same tick; uncapped, because the wait comes after launch (the 2h45m cutoff is unaffected), NFL has no MLB-style `lock_utc - 20 min` freeze, and picks of record are recorded >= 2h before kickoff during the pipeline. Dry run against `2026_wk11_Sun_2026-11-22`: the plan's commands are byte-identical to the copy/import commands main runs, and every other pipeline job is unchanged. Follow-ups: (1) after #22 merges, check the first locked publish log shows "Running under the caller's publish lock"; (2) re-sync `src/utils/publish_lock.py` if MTB changes the lock; (3) manual `weekly_run`: done 2026-11-24 (user chose to lock it; branch `feat/weekly-run-publish-lock`). A publishing manual run passes its own `--mtb-publish-plan` (next to the debug log), then runs the deferred copy/import under the lock via `dispatcher.run_locked_publish` (300 s wait); busy lock: nothing copied, exit 75, retry with `--publish-only <plan>`. Runs that already pass `--mtb-publish-plan` (the dispatcher) and `--no-publish` runs are unchanged. Manual runs still don't deploy the site. A bare `run_pipeline --publish-mtb` remains unlocked. |
| MTB deploy rollout timeout | MTB side (user fixing the health check) | 2026-11-20: deploy run 44754280494 timed out waiting for an old pod to terminate while the new pods were serving; the re-run passed. Our dispatcher state still records the manual Fri slot as failed (`publish_rc` 1), and a FAILED alert went out. 2026-11-21: deploy 44804510390 (MTB 361748d) passed on the first attempt. Nothing to change on this side. |

## Known Data Issues

- **Step-valued published probabilities, 2026 weeks 2-10 (fixed 2026-11-13, PR #2).** Before PR #2, `src.evaluation.calibrate_winprob` always fit an isotonic calibrator and prediction applied it, so `home_win_prob_calibrated` took only about 5-10 distinct values (e.g. 0.2245, 0.7377, 0.15). Visible repeats in the published probability: weeks 5-9 in every version, week 10's Thursday file (MTB 158762b), 2 of 8 week 3 versions, and the early week 4 versions (25eea5a, 0adf06d). Other September/early-October versions had distinct final values because a later stage spread them out. Since PR #2, calibration is chosen walk-forward and currently `none`; the 2026-11-14 Week 10 run is clean. Published picks are not regraded: they are what readers saw.
- **Run-log encoding crash (fixed 2026-11-15, PR #12).** Under the scheduled task, `tools.weekly_run`'s stdout is a cp1252 file; a non-cp1252 character in pipeline output raised `UnicodeEncodeError` in `src/utils/run_log.py` and killed both Week 10 Sunday-early attempts 30 seconds in. The NE-DET London game therefore got no fresh pick. Manual runs never showed it because the VS Code terminal is UTF-8.
- **Week 2 version MTB 7e1d595** repeats values in the raw model output itself (13 distinct of 16). Superseded the next day; low priority.
- **MTB archive overwrites.** Later publishes replaced some weeks' archive CSVs with only the games not yet kicked off (e.g. week 4 at MTB 9d230e2 kept only ATL-NO). Weeks 4 and 9 were rebuilt per game from MTB git history (MTB PR #7). From the Week 10 Sunday main slot on, grading uses the picks-of-record ledger, so a later overwrite of the archive CSV no longer changes which pick is graded. If grading ever shows fewer games than the schedule, compare against the archive file's git history. **Weeks 5-7 (found 2026-11-18):** five more games were dropped the same way: TB-DAL and PHI-JAX (week 5), SEA-DEN and HOU-JAX (week 6), NE-CHI (week 7). Weeks 5-7 graded 13/15, 12/14 and 13/14 games, and nobody compared those counts with the schedule. Resolved under the pick-of-record transparency rule (Maintenance Rules; MTB PR #14). SEA-DEN and HOU-JAX count, from MTB fdab881 (recorded 2026-10-15 19:42 UTC); both lost, so week 6 goes 7-5 -> 7-7. TB-DAL and PHI-JAX have no pick of record (the only pre-kickoff publish, MTB 0547f57, was faulty: no odds, both at 0.5761). NE-CHI has none either (Week 7 was first published after kickoff). Those three get score-only rows (`no_pick_of_record`). Season record 85-43-1 over 129 -> 85-45-1 over 131. A later local run (`predictions/w5_*.csv.backup_11`) was not used. Week 1 has no picks of record: every local run publishes a constant 0.15/0.2725, and MTB starts 09-19. Weeks 8-10 are complete on the site. Since MTB 083b133 (11-14) the site archive merges per game (`mergeWeekArchive`: locked games stay), so a later run can no longer drop a Thursday or London game. Week 10's WAS-NYG Thursday pick survived the Sunday and Monday runs. **Regraded and published 2026-11-18:** nfl-predictions PR #21 (19dda47) writes score-only `no_pick_of_record` rows and alerts on any no-pick game missing from `docs/checks/no_pick_of_record_2026.csv`. Weeks 5-7 were regraded from MTB 870c50c (PR #14): week 5 8-4-1 over 13 (+2 no-pick rows), week 6 7-5 over 12 -> 7-7 over 14 (SEA-DEN and HOU-JAX both lost), week 7 9-4 over 13 (+1 no-pick row). Season 85-43-1 over 129 -> **85-45-1 over 131**, 3 no-pick games not counted. Weeks 2-4 and 8-10 unchanged; no ALERT. Published via `tools.publish_mtb_csvs --results-only` and `scripts/publish_site.py` (MTB **9e7d36a**, GKE deploy run 44388115579 succeeded; MTB verified it on the live site: the 3 no-pick pages show the score, the no-pick card and noindex; week 6 shows 7-7; SEA-DEN and HOU-JAX are graded). Logs: `logs/checks/regrade_2026_wk0{5,6,7}.json`, `logs/checks/publish_*_wk05-07.txt`.

## Operating Roadmap

### 1. Metrics And Documentation Hygiene

- [x] Auto-update README headline metrics from latest valid evaluation rows.
- [x] Generate run-comparison report after pipeline runs.
- [ ] Add season-by-season metric table to README or a dedicated evaluation doc.
- [ ] Add a lightweight changelog for major pipeline/data/model changes.
- [ ] Update `GOALS.md` after each meaningful pipeline run or roadmap change (ongoing; last 2026-11-14).
- [x] Reconcile old docs that still claim stale AUC/Brier/coverage values: superseded plans/status docs moved to `docs/archive/` (2026-11-14).

### 2. Data Source Direction

- [x] Prefer maintained nflverse-data release assets over `nfl_data_py` for core nflverse datasets.
- [x] Normalize NFLVerse schedule external ID columns before parquet writes so release schedules do not fall back to stale CSV output.
- [x] Correct NFLVerse PBP validation so signed `wpa` values are accepted while true win probabilities remain bounded to `[0, 1]`.
- [x] Suppress/soften recurring low-level noise for known optional-source paths: pandas concat FutureWarning during nflverse merges, optional PFR helper/feed misses, and Yahoo Fantasy authorization skips while API approval is pending.
- [ ] Monitor two full weekly runs after the release-first switch for PBP/schedule/player-stat row-count drift and feature-distribution drift.
- [x] Prefer BallDontLie over SportsDataIO when `BALLDONTLIE_API_KEY` is present.
- [x] Keep SportsDataIO disabled by default unless `--enable-sportsdataio` is passed.
- [x] Document current datasource roles in `datasources.md`.
- [ ] Verify BallDontLie coverage is sufficient for current-season player/team context.
- [ ] Remove or downgrade SportsDataIO dependencies once BallDontLie replacements are proven.
- [ ] Enable Yahoo Fantasy feeds only after API approval is confirmed.

### 3. Team Model Improvement

Market-gap plan (started 2026-11-12). Metric: walk-forward AUC/Brier/LogLoss vs no-vig market.

| Step | Status |
|---|---|
| 1. Evaluation harness: weekly walk-forward vs market | Done (PR #1) |
| 2. Retire stale params; walk-forward tuning | Done |
| 3. Market-anchored win + spread models | Done (PR #1) |
| 4. Pick-time lines | Partial: Wednesday line history backfilled; pick-time vs close reported from 2026-11-24 (`src/evaluation/pick_time_vs_close.py`; weeks 10-11: close is ~0.005 Brier harder than the pick-time market, model between them). Next: more weeks, then decide whether later games in a slot need a later pick |
| 5. Prune features | Closed 2026-11-24, not shipped (fails the ship rule on calibration). Walk-forward 2018-2026, 5 paired seeds (42, 1-4), same feature snapshot, vs `main` (2,380 games). Saved models: 49 of 176 win inputs and 68 of 175 spread inputs have zero importance (`inj_qb_*`, `qb_*` status/games, `news_*` counts, many `sched_*` rest/travel flags; current-game market diffs are zero because the anchor masks them). (1) Fixed lists from the saved models (selection uses data through 2026 wk11, so optimistic): dropping 41 inputs dead in both models changes nothing (all deltas within noise; fails the majority clause on ~1e-5 margins); dropping 76 dead in either: Brier -0.00010, AUC +0.00033, but ECE worse 3/5 overall and 2026 ECE +0.0057 (4/5). (2) Leak-free per fold (each weekly refit fits both models on its own training rows, drops zero-importance inputs, refits; a model with no splits is ignored, which matters in 2018 when one model barely splits): "dead in either" (~104 dropped per refit) gives Brier -0.00014 (better 4/5), log loss -0.00028, AUC +0.00042 (4/5); 2026 AUC +0.0008; but ECE +0.0023 overall (worse 3/5) and +0.0045 in 2026 (worse 5/5, ~6 sd), margin MAE +0.0003 (worse 4/5). "Dead in both" (~66 dropped) is slightly worse on nearly everything. So the gain in (1) was not leakage. (3) Win-only per-fold prune (spread inputs unchanged): win probabilities identical to (2), margin MAE identical to `main`, ECE still fails. Calibration re-check with `select_calibration_method` (window 3) on raw walk-forward probabilities: picks `none` for every seed in both `main` and the pruned arm; with sigmoid on both, the pruned arm passes in 2026 (ECE -0.006, 5/5) but sigmoid itself is worse than `none` (Brier 0.2111 vs 0.2101). Possible later retest: win-only per-fold pruning if calibration is reworked (e.g. selected on log loss or ECE, not Brier). Experiment code was an uncommitted hook in `src/predict/predict_history.py`; not kept. |
| 6. Long-memory team strength | Shelved (no lift); code removed 2026-11-21 (cleanup PR D), restore steps in `docs/archive/removed_code_2026-11.md` |
| 7. Calibration simplification | Done: walk-forward selection (currently `none`); volatility shrinks toward market (PR #2) |
| 8. Hygiene | Partial: weekly runner + debug log done; `train.py` / `predict_upcoming.py` size untouched |
| Data quality | Done: injury grid, LA odds, Visual Crossing refresh, actual venue + kickoff, 2025 international venues, travel. Open: Xweather decision, home-bias watch |

Older checklist:

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
- [x] Improve AUC from ~0.706 to >=0.720 without leakage (0.7226 walk-forward, 2026-11-14).
- [ ] Match or beat the no-vig market AUC (0.7250 overall; 0.7422 in 2026; 2026-11-20 run).
- [x] Reach market Brier parity overall (0.2110 vs 0.2110).
- [ ] Reach market Brier parity in the current season (2026: 0.2169 vs 0.2157).
- [ ] Improve margin MAE from 9.83 to <=9.75.
- [ ] Re-check feature lift after every material feature-family change.
- [x] Rework volatility classifier to RF, log-loss-only labels, and 0.5 percentile default.
- [ ] Add a separate spread/margin-volatility model instead of mixing margin-error labels into probability volatility.
- [ ] Improve probability calibration, especially high-confidence picks where LogLoss/Brier penalties are largest.
- [x] Build calibration diagnostics by model confidence tier and identify overconfident ranges that need shrinkage or recalibration: `analysis/market_model_confidence_calibration.csv`.
- [x] Use confidence-tier calibration diagnostics to define a tested shrinkage/recalibration rule for overconfident ranges: `analysis/market_model_confidence_recalibration.csv`, `model_confidence_gate_*` fields in `analysis/market_edge_policy.csv`, and user-facing moneyline policy statuses.
- [x] Use the confidence recalibration gate to decide whether the next model work should adjust the base win-probability calibrator, the market policy only, or both: `analysis/market_confidence_recalibration_decision.csv`.
- [x] Export raw-vs-empirically-calibrated moneyline confidence diagnostics in weekly prediction CSVs (`market_moneyline_model_pick_prob_raw`, `market_moneyline_calibrated_model_pick_prob`, `market_moneyline_calibrated_prob_edge`, and calibration source/sample/error fields) without changing the base pick path.
- [x] Refresh upcoming predictions after market ROI/confidence artifacts are regenerated so published CSVs use the current run's confidence-policy diagnostics rather than the previous run's artifacts.
- [ ] If the confidence-recalibration decision artifact shows durable overall Brier/LogLoss lift, wire the chosen strategy into the base win-probability calibration path behind an explicit experiment flag before defaulting it on.
- [ ] Decide whether MattyTheBookie should display calibrated confidence as primary public confidence or keep raw model confidence primary with calibrated confidence as tooltip/context.
- [x] Build a live-published prediction-vs-actual artifact from archived/locked predictions and finalized game results before using archive outcomes for model decisions (`live_prediction_vs_actual*.csv`; see Archive-To-Results). This artifact should grade SU winner, margin error, probability loss metrics, ATS/spread outcomes when market lines exist, and market-policy result fields without rewriting the original prediction snapshot.
- [ ] Add live holdout segment diagnostics from the prediction-vs-actual artifact by confidence bin, favorite/underdog, home/away, rest, weather, injury/roster status, volatility label, market agreement/disagreement, team, division, venue, and key spread buckets.
- [ ] Convert repeated live holdout errors into explicit action classes: recalibrate probabilities, improve or remove a feature family, tighten publication policy gates, or leave as monitored variance. Do not train future models on prior predictions as features unless a separately validated stacking/meta-model is designed.

### 4. Market Benchmarking And Betting Edge

Priority: high. Protect published picks from known bad market signals before adding new model complexity.

Current read from the Week 5 evaluation set (2026-10-11; not refreshed since the market-anchored models, so treat these numbers as historical until the market artifacts are re-reviewed):

- SU model accuracy is 65.4% over 2,283 market-matched games.
- No-vig Vegas favorite accuracy is about 66.1% over the same games.
- Model and Vegas favorite agree on about 86.9% of games.
- In 305 favorite-disagreement games, the model is 47.2% SU and Vegas is 52.8%, but model-side moneyline ROI is +4.2% overall because the model side is usually plus-money.
- The current historical moneyline disagreement policy candidate is `prob_edge >= 0.05`: 266 bets, +5.3% ROI, 47.0% hit rate, and average model-side odds near +112. It remains `no_bet` because season stability, CLV sample, coarse segment, diagnostic segment, and confidence-overestimate gates do not all pass.
- Moneyline disagreement policy now requires season-stability and current-week CLV gates before publication. The current `prob_edge >= 0.05` policy has 6 qualified seasons, a 50.0% positive qualified-season rate, a 0.0% recent positive qualified-season rate, and a 33.3% severe-negative season rate, and it has only 2 tracked Week 5 CLV games.
- Higher moneyline disagreement thresholds are not automatically safer: `prob_edge >= 0.08` is roughly flat (+0.1% ROI), `prob_edge >= 0.10` is positive but lower-sample (+4.6% ROI), and `prob_edge >= 0.15` is negative (-8.1% ROI).
- Model-vs-no-vig-Vegas probability calibration is now published. Current selected-side Brier is worse than Vegas overall (`0.218` vs `0.211`), and the `15%+` model edge bin is materially overconfident: 62.2% average model probability, 43.8% average no-vig Vegas same-side probability, 42.6% actual win rate, and -5.3% model-side ROI.
- A first walk-forward market-aware blend backtest is now published. Current blend Brier improves raw model Brier (`0.214` vs `0.218`) but still trails no-vig Vegas (`0.212`), and blend selected-side ROI improves raw model and Vegas ROI but remains negative (-2.3%), so `market_blend_policy.csv` still marks blend probability and blend moneyline as `no_bet`. The confidence-recalibration step now annotates `market_blend_policy.csv` with `confidence_recalibration_*` fields when the 50/50 blend is an experiment candidate, keeping the live recommendation off while making the experiment path explicit.
- Bounded moneyline disagreement bins are now published. The current Week 5 read shows the strongest bounded samples in the 5-8% and 10-15% bins, while the 8-10% and 15%+ bins are negative; do not scale confidence purely by larger probability edge until season-stability and CLV checks support it.
- Weekly CLV detail and summary artifacts are now published from local pick-time and close/latest odds snapshots, and `analysis/market_edge_policy.csv` consumes the threshold-specific CLV summary before allowing a moneyline policy to become `strong`. Current Week 5 favorite-disagreement rows have 2 tracked games, +0.7 percentage-point average no-vig implied-probability CLV, and a 100.0% positive CLV rate, so the policy remains blocked until the tracked-game sample is large enough.
- Model-confidence empirical recalibration is now published for moneyline disagreement picks. High-confidence disagreement bins with enough sample and model overestimate above the policy threshold hard-block moneyline promotion and surface `blocked_model_confidence_overconfidence` in weekly prediction policy fields.
- Walk-forward confidence recalibration strategy decisions are now published. Current read: no-vig Vegas remains the probability benchmark to beat, while empirical confidence shrinkage is now surfaced as a candidate diagnostic path. Weekly predictions export calibrated moneyline confidence fields for downstream review, but the base public pick probability is not changed until repeated live runs prove the strategy is durable.
- Favorite-disagreement diagnostics are now published and reconciled back into `analysis/market_edge_policy.csv`. The strongest current model-side research slices are `prob_edge_bin=5-8%` (47 games, +29.2% ROI, 61.7% SU), `model_side_odds_bucket=+100 to +120` (121 games, +14.5% ROI), and `prob_edge_bin=10-15%` (112 games, +13.3% ROI). The policy reconciliation now tracks probability-edge candidate/avoid bins separately from model-side moneyline bucket candidate/avoid bins, so a near-pick plus-money slice can be evaluated without promoting all broad disagreement rows. For the current `prob_edge >= 0.05` policy window, `5-8%` is a diagnostic candidate segment and `8-10%` is a diagnostic avoid segment, so the reconciled moneyline policy remains `no_bet` with `blocked_diagnostic_avoid_segment`. The segment gates now also handle a selected `prob_edge >= 0.10` threshold by evaluating the 10%+ bins instead of treating the policy window as empty.
- Broad spread threshold strategies are now policy-blocked as `no_bet`. Away edge >2 points is -1.5% ROI despite 61.0% average modeled cover probability, which confirms spread cover-probability overconfidence.
- Corrected spread edge-bin ROI is only positive in the 2-to-3 point bucket: 368 bets, +1.2% ROI, and 52.5% hit rate. That bucket is now `research_only` until the empirical spread cover calibration gate passes; spread edges of 5+ points are negative and should not be promoted.
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
- [x] Add disagreement-only moneyline segment diagnostics by side, odds bucket, confidence, probability edge, and CLV: `analysis/market_disagreement_segments.csv`.
- [x] Add a segment-aware moneyline disagreement policy gate so weak bounded probability-edge slices can downgrade broad threshold candidates before publication.
- [x] Add spread cover-probability calibration artifact: `analysis/market_spread_calibration.csv`.
- [x] Add empirical spread cover recalibration artifact: `analysis/market_spread_cover_recalibration.csv`.
- [x] Hard-gate broad spread threshold policy to `no_bet` until calibrated spread probabilities and CLV justify promotion.
- [x] Restrict spread edge-bin policy recommendations to the 2-to-3 point bucket as lean/research; block 5+ point spread edges from promotion.
- [x] Add season-stability gates for the moneyline disagreement policy before raising confidence beyond the current strong research signal.
- [x] Add current-week publishing guardrails so MattyTheBookie can distinguish `recommended`, `lean`, `research_only`, `review`, and `no_bet` rows.
- [x] Compare model probability directly to no-vig Vegas probability by probability bin and publish the calibration/edge table.
- [x] Add a confidence-aware moneyline disagreement policy gate that uses empirical recalibration bins before promoting disagreement picks.
- [x] Add a walk-forward confidence recalibration strategy backtest and decision artifact before making any live probability change.
- [x] Add favorite-disagreement diagnostic and research-recommendation artifacts: `analysis/market_disagreement_games.csv`, `analysis/market_disagreement_diagnostics.csv`, and `analysis/market_disagreement_recommendations.csv`.
- [x] Reconcile favorite-disagreement diagnostic candidate/avoid probability-edge segments with `market_edge_policy.csv` before any segment can move from research to live recommendation.
- [ ] Improve model performance in favorite-disagreement games versus Vegas before treating disagreements as actionable.
- [x] Reconcile `market_confidence_recalibration_decision.csv` and `market_blend_policy.csv` so the 50/50 blend candidate has one explicit experiment/policy path.
- [x] Build and backtest a market-aware blend or meta-model using model probability, no-vig moneyline probability, market spread, and current model features.
- [x] Track whether any market-aware blend adds genuine edge versus simply mirroring Vegas favorite picks.
- [x] Keep spread edge analysis as a separate betting-signal track from straight-up winner model accuracy.
- [x] Add key-number-aware spread diagnostics for 3, 6, 7, and 10 before promoting any spread signal.
- [x] Recalibrate spread cover probabilities empirically; suppress raw high `cover_prob` as a confidence signal until calibration errors are acceptable.
- [x] Add weekly CLV detail/summary artifacts from local pick-time and close/latest odds snapshots: `analysis/market_clv_detail.csv` and `analysis/market_clv_summary.csv`.
- [x] Add closing-line movement / CLV fields to `market_benchmark.csv` when reliable opening and closing snapshots are available; rows without matching CLV detail or with model-side mismatches are explicitly marked unavailable.
- [x] Promote CLV reports into the weekly validation flow before using ROI results as decision criteria.

### 5. Player Prop Prediction Goals

Current state: legacy player CSVs remain available as rate-based projections, and trained-model player prop CSVs now publish weekly for QB passing yards, QB passing TDs, RB rushing yards, RB rushing TDs, WR/TE receiving yards, WR/TE receiving TDs, anytime TD scorer, defensive player sacks, and research-only first/last/2+/3+ TD scorer novelty rows. PropLine line ingestion is wired through DraftKings, Hard Rock, and FanDuel when `PROPLINE_API_KEY` is configured, with line-consensus/variance diagnostics, injury quality controls, current-roster validation, and conservative edge-recommendation policy fields included in the weekly prop CSVs. Default current PropLine requests include yardage, stat-specific TD, anytime TD, first/last/2+/3+ TD scorer novelty keys, and defensive sack market keys; the weekly coverage artifact now separates exact line rows from `anytime_td_fallback` display rows so RB rushing TD and WR/TE receiving TD coverage is visible without making those rows bettable. Published prop CSVs now include `edge_publish_status`, `edge_reason`, and public-board policy fields for user-facing policy clarity, while `predictions/player_props_public.csv` provides a cleaner website board limited to exact sportsbook-line rows that pass active-roster, injury-exclusion, roster-mismatch, and fallback-line gates. The weekly quality suite includes `prop_line_gap_audit_week_NN.csv` to separate model predictions without offered lines from offered player/game/market pairs outside the prediction universe, and offered-line candidate expansion now uses PropLine offers plus identity, active-roster, and prior pre-week label history beyond defensive sacks. A durable cross-source player identity table now publishes to `data/processed/player_identity.parquet` and links GSIS player IDs to nflverse, ESPN, BallDontLie, Sportradar, and sportsbook-style player-name aliases for safer historical and current prop-line matching. Goal: expand line availability reporting, calibrate confidence tiers, backtest prop edges, broaden markets, and score the outputs weekly.

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
- [x] Add weekly PropLine timing and selection diagnostics to `prop_line_coverage_week_NN.csv`, including raw game counts, kickoff/update timestamps, hours until kickoff, and `available_not_selected` rows for books that had matchable lines but were not chosen by the best-line selector.
- [x] Split exact sportsbook line counts from `anytime_td_fallback` display counts in weekly prop quality and line-coverage artifacts.
- [x] Add optional PBP-derived first/last/2+/3+ TD novelty label output: `data/processed/player_prop_novelty_td_labels.parquet`.
- [x] Wire PBP-derived novelty TD labels into separate novelty features, model training, and research-only weekly prediction output.
- [x] Add optional PropLine player-prop line ingestion for `line`, `implied_probability`, and projection-vs-line `edge`.
- [x] Add identity-aware anytime-TD fallback mapping for RB rushing TD and WR/TE receiving TD current outputs; fallback rows use `line_type=anytime_td_fallback`, `prop_quality_flag=fallback_line`, and are excluded from bettable/recommendation gates.
- [x] Publish fallback-only TD split rows as `edge_publish_status=research_only` with an explicit true-line-required reason so consumer apps do not treat inferred lines as bettable edges.
- [x] Add line-offered defensive sack candidate expansion using PropLine offers plus player identity, active-roster availability, and prior non-leaking sack history.
- [x] Generalize line-offered candidate expansion for modeled main prop markets when PropLine offers the player/game/market and identity, active-roster, and prior pre-week label history are available.
- [x] Extend the same line-offered candidate expansion to research-only novelty TD markets so first/last/2+/3+ TD rows are scored when a sportsbook offers the player and the player has prior pre-week novelty labels.
- [x] Add sportsbook line consensus/variance fields to prop exports.
- [x] Add injury status, injury-exclusion, and bettable flags to prop exports.
- [x] Add current-week ESPN roster/depthchart/injury snapshots for explicit prediction weeks.
- [x] Build unified current-player availability from ESPN, nflverse, and BallDontLie.
- [x] Add roster validation fields to prop exports and block confirmed roster mismatch, practice-squad, and inactive-roster rows from `bettable_flag`.
- [x] Add weekly player-prop quality report artifacts for board usability review.
- [x] Add roster validation counts to weekly player-prop quality reports.
- [x] Report PropLine availability and match quality by week, sportsbook, market, and player naming.
- [x] Add player-prop edge recommendation counts and EV/probability-edge diagnostics to weekly quality summaries.
- [x] Add weekly player-prop line-gap audit output: `predictions/evaluation/player_props/prop_line_gap_audit_week_NN.csv`.
- [x] Add line-gap taxonomy with `offer_gap_category` and offered identity/position fields so sportsbook offers outside predictions are separated into non-player outcomes, unsupported position/market pairs, inactive/IR/practice-squad rows, missing prior market history, identity misses, and true `candidate_expansion_gap` rows.
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
- [x] Add website-ready player-prop policy explanation fields to current prop CSVs: `edge_publish_status`, `edge_reason`, `edge_required_threshold`, `edge_current_probability_edge`, `edge_policy_sample_size`, and `edge_policy_historical_roi`.
- [x] Add website-facing `predictions/player_props_public.csv` with `public_board_status`, `public_board_reason`, `public_bettable_flag`, `public_pick_flag`, `public_pick_tier`, `public_pick_reason`, and `public_sort_rank`, while keeping full diagnostic prop CSVs intact.
- [x] Keep injury-review player props visible in the public export for transparency, but demote them below clean no-bet/research rows so default website sorting is not dominated by availability warnings.
- [x] Split current recommendation blockers into `below_threshold` versus `no_positive_ev` so rows that pass the historical edge threshold but fail the current price/juice gate are no longer mislabeled.
- [x] Add `predictions/evaluation/player_props/edge_policy_diagnostics.csv` so unsupported player-prop markets have explicit rejection reasons such as insufficient sample, non-positive historical ROI, or below-minimum probability-edge threshold instead of silently staying `research_only`.
- [x] Add live-week leakage guardrails to downstream player-prop policy stages: historical line joins, confidence-tier metrics, over-probability calibration, and edge-policy backtests now receive the pipeline live cutoff.
- [ ] Prove player-prop `prob_edge` and `edge_ev` add durable lift versus sportsbook implied probabilities before expanding recommendation signals beyond sparse market-specific policies.
- [ ] Add true current and historical sportsbook line coverage for RB rushing TDs and WR/TE receiving TDs before treating those stat-specific TD props as bettable recommendations; current anytime-TD fallback lines are display/coverage aids only.
- [ ] Add historical sportsbook line coverage and edge backtests for anytime TD before treating it as a recommendation signal.
- [ ] Backfill historical first/last/2+/3+ TD scorer lines before promoting novelty TD markets beyond research-only output.

Current edge-backtest read: the policy selects only market-level thresholds with at least 100 historical bets and positive ROI: strong anytime TD at `prob_edge >= 0.10` and lean RB rushing yards at `prob_edge >= 0.02`. Current rows expose these thresholds and explain blocked picks through `edge_publish_status` / `edge_reason`. The latest Week 8 review has zero current recommended rows because recommendation flags now require both the historical probability-edge threshold and positive current EV after sportsbook juice; rows that clear the threshold but fail the price gate publish as `no_positive_ev`, not `below_threshold`.

Current line-gap read: the latest Week 8 quality rerun has 728 model prediction rows without an offered sportsbook line and 237 offered player/game/market pairs outside predictions. The LA/LAR team-alias mismatch in line-offered candidate expansion is fixed, and the true `candidate_expansion_gap` bucket is now 0. Remaining offered rows are non-player D/ST/no-touchdown outcomes, unsupported market/position pairs such as QB rushing yards under the RB rushing model, inactive/IR/practice-squad rows, missing-history rows, and identity misses; handle those separately without loosening public pick gates.

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
- [x] `predictions/player_props_public.csv` cleaner website-facing board derived from full diagnostic prop exports.
- [x] MattyTheBookie CSV publishing support for player prop files.
- [x] MattyTheBookie website/import consumption for player prop files.

Implementation details live in `docs/PLAYER_PROP_PREDICTION_PLAN.md`.

### 6. Validation Operations

- [x] Evaluation pipeline writes weekly and overall metrics.
- [x] Run comparison pipeline writes leaderboard and metric trend charts.
- [x] Live tracking and the Phase 1-5 validation modules: replaced by `src.evaluation` and removed 2026-11-21 (cleanup PR C).
- [x] Weekly CLV detail/summary artifacts are generated, exported to model metrics, included in the main pipeline, and published to MattyTheBookie CSVs.
- [x] Re-run CLV reports with current predictions and odds data.
- [x] Decide whether validation outputs should be generated by the main weekly pipeline or a separate scheduled validation command: the main pipeline (`src.evaluation`); the separate validation suite was removed 2026-11-21 (cleanup PR C).
- [x] Make a dying data job or run-log wrapper leave a clear error in the debug log: decoded exit status, one retry for required data jobs, every failed job named, wrapper failures logged with the traceback and the pipeline stopped ([docs/automation.md](docs/automation.md#when-something-dies), 2026-11-15).
- [ ] Decide whether data jobs need a timeout. A hung job currently blocks the run until the dispatcher's `--pipeline-timeout-min` (120 min).

#### Archive-To-Results Feedback Loop

Principle: keep archived prediction snapshots immutable as the public record of what was published before kickoff. Add result grading as a separate overlay that can be regenerated after games finalize.

- [x] Define the canonical live-published prediction source for grading: per-game pick of record (ledger) first, then `week_NN_published_override.csv`, then the MTB archive CSV, then `predictions_log` locked predictions, with an explicit file override for recovery cases.
- [x] Grade automatically: the dispatcher grades each week from 8h after its last kickoff, once every game is final, then publishes results to MTB and alerts (PR #10).
- [x] Backfill 2026 weeks 2-9. Week 8 grades from a published override built from the site JSON. Weeks 4 and 9 were regraded on 2026-11-14 from per-game picks of record rebuilt from MTB git history (MTB restored its archive files to match, MTB PR #7).
- Not graded on purpose: week 1 (no published picks); week 5 TB-DAL and PHI-JAX and week 6 SEA-DEN and HOU-JAX (left out by decision on 2026-11-14; their only pre-kickoff versions, MTB 0547f57 and fdab881, came from early-week runs whose probabilities were badly step-valued, see Known Data Issues); week 7 NE-CHI (first publish was after kickoff).
- Week 10 WAS-NYG (Thursday) and NE-DET (London, 09:30 ET Sunday) have no ledger rows: the ledger started with the Sunday main slot. They will grade on the MTB archive CSV (Thursday's file, MTB 158762b, which still has step-valued probabilities), which is what the site showed.
- [x] Implement a weekly join from published predictions to finalized game results using `season`, `week`, and `game_id`.
- [x] Publish weekly and cumulative artifacts: `predictions/evaluation/live_prediction_vs_actual_week_NN.csv`, `predictions/evaluation/live_prediction_vs_actual.csv`, `predictions/evaluation/live_prediction_vs_actual_summary_week_NN.csv`, `predictions/evaluation/live_prediction_vs_actual_summary.csv`, and `predictions/evaluation/live_prediction_vs_actual_segments_week_NN.csv`.
- [x] Include core per-game grading fields: final scores, actual winner, model pick result, SU correctness, predicted margin error, Brier/log-loss contribution, ATS result where a valid line exists, moneyline result, moneyline unit profit when odds are available, and market-policy context fields.
- [x] Publish a weekly summary artifact for model-improvement review: live accuracy, F1, AUC, Brier, LogLoss, MAE, ATS hit rate, moneyline ROI, and segment-level misses.
- [x] Add MattyTheBookie CSV publishing support for live prediction-vs-actual artifacts when those CSVs exist.
- [x] Export the weekly live-published summary into `analysis/model_metrics_week_NN.csv` so the About page can distinguish historical evaluation from true published-pick holdout results.
- [x] Add CLV result/status fields to the live prediction-vs-actual artifact when reliable pick-time and close/latest odds snapshots match.
- [x] Add MTB archive support for displaying prediction-vs-actual as a result overlay, not by mutating the stored prediction fields.
- [x] Protect MTB season/week archive CSVs during live reruns: preserve existing snapshots and skip creating partial archive snapshots when already-started games have been excluded.
- [x] Keep latest finalized live-published holdout metrics available in `analysis/model_metrics_week_NN.csv` when the active prediction week is not finalized yet.
- [x] Add `live_published_source_week` and latest-graded wording to `analysis/model_metrics_week_NN.csv` so the website does not present an earlier finalized holdout as current-week grading.
- [x] Defer player-prop archive grading until per-market player actuals and sportsbook line mappings are reliable enough to avoid false grading.
- [ ] Review the first finalized week with MTB result overlays and live-published holdout metrics together, then decide which repeated live misses should become model/policy work items.

### 7. Testing And Quality

- [x] Unit tests exist for data fetchers, schema validation, prediction helpers, pipeline helpers, publish/import helpers, and analysis utilities.
- [x] Run a fresh full test suite after the next full pipeline stabilization point (2026-11-14: 1235 passed, 1 known failure, run in `.venv`; the `nflgpu` env crashes in `test_player_prop_train`).
- [x] Establish a realistic current coverage baseline from the full suite (about 66% of `src`; `pyproject.toml` sets `--cov-fail-under=70`, so local full runs report a coverage failure until coverage rises or the gate is lowered).
- [ ] Fix `test_rolling_average_properties` (fails on main too).
- [x] Add focused tests for the first player-prop label builder.
- [x] Add focused tests for the split player-prop feature builder.
- [x] Add focused tests for the first player-prop baseline evaluator.
- [x] Add focused tests for the first player-prop trained-model evaluator.
- [x] Add focused tests for player-prop model outputs after those outputs exist.
- [x] Add regression tests for MTB archive publishing.
- [x] Add regression tests for production BDL feature gating and feature-family ablation opt-in.
- [x] Add regression tests for roster-line audit artifact generation and MTB publish coverage.
- [x] Add regression tests for the archive-to-results join, including immutable prediction preservation, finalized-result overlays, duplicate `game_id` protection, and no postgame leakage into same-week predictions.
