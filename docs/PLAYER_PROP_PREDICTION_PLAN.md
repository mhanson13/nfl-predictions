# Player Prop Prediction Plan

This plan turns the existing player projection exports into validated player prop prediction markets. The first priority is to build reliable labels and evaluation loops before adding more model complexity.

## Current State

The weekly pipeline already writes:

- `predictions/predictions_players_qb.csv`
- `predictions/predictions_players_offense.csv`
- `predictions/predictions_players_defense.csv`
- `data/processed/player_prop_labels.parquet`
- `data/processed/player_prop_features_offense.parquet`
- `data/processed/player_prop_features_defense.parquet`
- `predictions/evaluation/player_props/baseline_metrics.csv`
- `predictions/evaluation/player_props/baseline_predictions.csv`
- `predictions/evaluation/player_props/model_metrics.csv`
- `predictions/evaluation/player_props/model_predictions.csv`
- `predictions/evaluation/player_props/model_vs_baseline.csv`
- `predictions/evaluation/player_props/prop_quality_week_NN.csv`
- `predictions/evaluation/player_props/prop_quality_summary_week_NN.csv`
- `predictions/evaluation/player_props/prop_line_coverage_week_NN.csv`
- `predictions/evaluation/player_props/prop_roster_line_audit_week_NN.csv`
- `data/processed/player_prop_historical_line_eval.parquet`
- `data/processed/player_prop_over_probability_eval.parquet`
- `predictions/evaluation/player_props/historical_line_join_summary.csv`
- `predictions/evaluation/player_props/over_probability_metrics.csv`
- `predictions/evaluation/player_props/over_probability_bins.csv`
- `predictions/player_props_qb.csv`
- `predictions/player_props_offense.csv`
- `predictions/player_props_defense.csv`
- `predictions/player_props_novelty.csv`
- Optional `data/raw/propline_player_props_<season>_wkNN.parquet` snapshots when `PROPLINE_API_KEY` is configured.
- Optional `data/raw/oddsapi_historical_event_markets_<season>_wkNN_<event>_<snapshot>.parquet` and `data/raw/oddsapi_historical_player_props_<season>_wkNN_<event>_<snapshot>.parquet` snapshots from The Odds API historical backfill when `odds_api_paid_key` is configured and manual historical pulls are run.

The legacy `predictions_players_*.csv` files remain useful projection tables based on historical player rates, roster filtering, and schedule context. The `player_props_*.csv` files are the first trained-model weekly prop outputs. The novelty output publishes first TD, last TD, 2+ TD, and 3+ TD scorer rows as research-only outputs until historical line calibration and edge backtests exist.

Repository findings from the Week 3 implementation review:

- `src.predict.predict_upcoming` creates the current QB, offense, and defense projection CSVs.
- QB and offensive projections use historical rows from `data/raw/nfl_player_stats.parquet`.
- Defensive projections are PBP-derived from `data/raw/nfl_pbp.parquet`.
- `data/processed/player_actuals.parquet` already exists as a narrow actuals layer with `passing`, `rushing`, `receiving`, and `defense` stat categories.
- Since 2017, `data/raw/nfl_player_stats.parquet` has enough volume for the currently modeled serious markets:
  - QB passing rows: about 6k
  - RB rushing rows: about 14k
  - WR/TE receiving rows: about 31k
  - Defensive rows: about 22k, with sacks as a sparse-event target
- The current player CSVs should remain published as baseline projections until trained prop models prove lift.
- `src.player_props.labels` is wired into `tools.run_pipeline` after `player_actuals` and before feature generation.
- `src.player_props.features` is wired into `tools.run_pipeline` after `player_prop_labels` and before team feature generation.
- `src.player_props.evaluate_baselines` is wired into `tools.run_pipeline` after `predict_upcoming` and before team prediction evaluation.
- `src.player_props.train` is wired into `tools.run_pipeline` after baseline evaluation, so trained model comparison uses baseline artifacts from the same run.
- The current generated label table contains 372,354 rows across QB passing yards, QB passing TDs, RB rushing yards, RB rushing TDs, WR/TE receiving yards, WR/TE receiving TDs, anytime TD scorer, and defensive player sacks.
- Split player-prop feature tables contain 348,120 offensive rows and 11,640 defensive rows with the Week 3 live cutoff applied.
- Invalid source rows with missing opponents or `team == opponent` are filtered before label generation so the current output has no missing `game_id` values.
- Current baseline scoring, using the Week 3 live cutoff, scores QB passing yards/TDs, RB rushing yards/TDs, WR/TE receiving yards/TDs, anytime TD, and defensive sack projections when baseline projections exist.
- First trained models use per-market walk-forward evaluation and write serialized models under `models/player_props/offense/`, `models/player_props/defense/`, and `models/player_props/novelty/`.
- Weekly trained-model export is implemented in `src.player_props.predict` and is wired into `tools.run_pipeline` after trained prop models are refreshed.
- Weekly trained-model export writes QB, offense, defense, and novelty CSVs. Novelty rows are kept separate in `predictions/player_props_novelty.csv` and remain research-only.
- Player-prop line ingestion is wired through `src.data.propline` for explicit weekly pipeline runs. The default PropLine pull uses DraftKings, Hard Rock, and FanDuel and requests yardage, stat-specific TD, anytime TD, first/last/2+/3+ TD scorer novelty keys, and defensive sack market keys. The prediction exporter maps PropLine event/player names back to local game/player rows and fills `line`, `implied_probability`, `sportsbook`, odds columns, and projection-vs-line `edge` when a matching prop exists.
- The latest Week 3 PropLine pull using DraftKings, Hard Rock, and FanDuel produced 11,099 raw rows and 9,110 prepared rows. Current Week 3 prop exports have 542 rows with lines: 295 anytime TD, 29 QB passing TD, 29 QB passing-yard, 53 RB rushing-yard, 118 WR/TE receiving-yard, and 18 defensive sack rows. Stat-specific RB rushing TD and WR/TE receiving TD markets are published as model projections but have no matched PropLine rows yet.
- Weekly prop exports now carry line-consensus diagnostics (`line_consensus`, `line_min`, `line_max`, `line_range`, `line_std`, `line_book_count`, `line_options_count`, `line_variance_flag`) computed from comparable two-way lines when possible.
- Weekly prop exports now carry injury quality controls from nflverse injuries with BallDontLie fallback (`injury_status`, practice/body-part fields, `injury_reported`, `injury_exclusion_flag`, `prop_quality_flag`, `bettable_flag`). Rows are still published, but injury-excluded players are not marked bettable.
- `src.player_props.quality_report` is wired into the weekly pipeline after prop prediction and writes detail/summary quality artifacts for rows with lines, missing-line rows, bettable rows, edge recommendation counts, edge EV/probability diagnostics, injury review/exclusion rows, line-variance rows, PropLine market/book/player match coverage, and row-level unknown-roster line audits. `prop_line_coverage_week_NN.csv` now includes `coverage_status` so requested markets with no sportsbook rows are explicitly marked `vendor_no_rows`; `prop_roster_line_audit_week_NN.csv` expands unknown roster rows by player, team, market, and sportsbook line availability.
- `src.player_props.labels --include-novelty-td-labels` builds a separate PBP-derived novelty TD label table for first TD, last TD, 2+ TDs, and 3+ TDs from `data/raw/nfl_pbp.parquet`; `src.player_props.features`, `src.player_props.train`, and `src.player_props.predict` wire those labels into a separate novelty model family and weekly research-only output.
- `src.data.odds_api_historical` can manually fetch individual historical event IDs, historical event market availability, and historical event player-prop odds from The Odds API using `odds_api_paid_key`.
- `src.data.odds_api_backfill` is the resumable one-time backfill controller. It builds `data/processed/oddsapi_player_prop_backfill_manifest.parquet`, writes aggregate historical events/markets/odds files under `data/raw/`, and normalizes completed odds into `data/processed/oddsapi_historical_player_prop_lines.parquet`. Paid historical API modes require `--confirm-paid-backfill`; always dry-run first. Historical novelty support maps `player_1st_td`, `player_last_td`, and `player_tds_over` into first TD, last TD, 2+ TD, and 3+ TD model markets.
- `src.player_props.historical_lines` joins normalized historical sportsbook lines to walk-forward model predictions and actual player results. It writes `data/processed/player_prop_historical_line_eval.parquet` and `predictions/evaluation/player_props/historical_line_join_summary.csv`.
- `src.player_props.over_probability` converts projection-vs-line deltas into walk-forward per-market over probabilities using prior residual distributions, then blends toward no-vig sportsbook implied probabilities when prior same-market calibration shows the market is better calibrated. It writes `data/processed/player_prop_over_probability_eval.parquet`, `predictions/evaluation/player_props/over_probability_metrics.csv`, and `predictions/evaluation/player_props/over_probability_bins.csv`.
- `src.player_props.edge_backtest` simulates flat-unit historical Over/Under bets from calibrated probabilities and sportsbook prices. It writes `predictions/evaluation/player_props/edge_backtest_detail.csv`, `predictions/evaluation/player_props/edge_backtest_summary.csv`, `predictions/evaluation/player_props/edge_backtest_market_thresholds.csv`, and a conservative `predictions/evaluation/player_props/edge_recommendation_policy.csv`.
- Weekly trained-model prop CSVs now include `prob_over_raw`, `prob_edge`, `prob_over_method`, and `prob_over_sample_size` when a sportsbook line is matched. `prob_edge` is the calibrated over probability minus the sportsbook no-vig implied probability; the legacy `edge` column remains projection minus line.
- Weekly trained-model prop CSVs also include price-aware break-even and expected-value diagnostics (`over_break_even_probability`, `under_break_even_probability`, `over_ev`, `under_ev`, `edge_side`, `edge_odds`, `edge_break_even_probability`, `edge_ev`). These account for prices such as `-114`, where the raw break-even probability is about 53.3%, not 50.0%.
- Weekly trained-model prop CSVs now separate diagnostics from recommendations with `edge_probability_edge`, `edge_recommendation_flag`, `edge_recommendation_tier`, and `edge_policy_*` fields. A current row is recommended only when it is bettable, clears a historically profitable market threshold from `edge_recommendation_policy.csv`, and has positive current expected value after price/juice.
- Anytime TD now has historical line calibration from the Odds API backfill, but current recommendations are still gated by `edge_recommendation_policy.csv` and positive current EV after price/juice.

## Target Markets

### QB

- Passing yards
- Passing touchdowns
- Interceptions
- Completions
- Attempts
- Rushing yards
- Rushing touchdown probability

### RB

- Rushing yards
- Carries
- Rushing touchdown probability
- Receiving yards
- Receptions
- Targets
- Total scrimmage yards
- Anytime touchdown probability

### WR/TE

- Receiving yards
- Receptions
- Targets
- Receiving touchdown probability
- Total yards
- Anytime touchdown probability

### Defensive Players

- Sacks
- QB hits
- Tackles for loss
- Total tackles
- Solo tackles
- Passes defended
- Interception probability
- Forced fumble probability

### Fun Novelty Bets

These should be treated as longshot watchlists, not core betting recommendations:

- WR/TE passing yards
- Pick-six / interception touchdown probability
- Defensive touchdown probability
- Blocked punt probability
- Blocked field goal / blocked kick probability
- Return touchdown probability
- Fumble recovery touchdown probability

## Data Plan

Use only data available before kickoff for features. Labels come from completed games only.

Primary historical label sources:

- `data/raw/nfl_player_stats.parquet` for weekly player passing, rushing, receiving, defensive, kicking, and return stats.
- `data/raw/nfl_pbp.parquet` for play-level labels, rare-event extraction, defensive events, and novelty markets.
- `data/processed/player_actuals.parquet` as the existing aggregated player-game actuals layer.
- BallDontLie current-season stat feeds as supplemental current-season refresh data.

Primary feature sources:

- NFLVerse rosters, injuries, depth charts, snap counts, schedules, and PBP-derived team context.
- ESPN player and team news for injury/news pressure.
- BallDontLie players, active players, injuries, and current-season stats.
- Team model outputs such as projected margin, total, pace, win probability, weather, and volatility labels.

## Modeling Plan

### Phase 1: Labels and Baselines

1. Build a normalized player-game target table keyed by `season`, `week`, `game_id`, `team`, `player_id`, and `position`.
2. Add labels for each serious market.
3. Add labels for novelty markets, while preserving rare-event counts.
4. Score the current rate-based projection files as baseline models.
5. Add weekly evaluation output under `predictions/evaluation/player_props/`.

### Phase 2: Feature Builder

1. Create pre-game rolling player features: last 3, last 5, season-to-date, prior season, career baseline.
2. Add playing-time features: snap share, route/carry/target share where available, depth chart rank, active/injury status.
3. Add matchup context: opponent defensive rates, pressure, run defense, pass defense, red-zone profile, projected pace/total.
4. Add recency controls so target week never leaks into features.

### Phase 3: Models

Use simple, auditable models first:

- Yardage/count markets: regularized linear models, gradient boosting, Poisson/negative-binomial style count models, and quantile models for distribution ranges.
- Binary event markets: calibrated logistic regression and gradient boosting classifiers.
- Rare novelty markets: shrinkage-heavy probability estimates with hard confidence caps.

### Phase 4: Publishing

Create stable outputs after backtests are available:

- `predictions/player_props_qb.csv`
- `predictions/player_props_offense.csv`
- `predictions/player_props_defense.csv`
- `predictions/player_props_novelty.csv`

Each row should include:

- Player and game identifiers
- Market name
- Projection or probability
- Confidence tier
- Sample size
- Baseline projection
- Model projection
- Optional sportsbook line, implied probability, edge, and CLV fields

## Validation Rules

- Serious markets need per-market metrics before being called recommendations.
- Novelty markets must remain labeled as novelty until they show stable calibration and enough sample size.
- Publish sample size and confidence tier with every player prop.
- Never train on the target week or future weeks.
- Keep rate-based projections as benchmarks so model lift is measurable.

## Suggested First Implementation Slice

Start with four high-volume markets:

1. QB passing yards
2. RB rushing yards
3. WR/TE receiving yards
4. Defensive player sacks

These covered the major data paths and gave enough historical volume to validate the architecture. The current implementation has since expanded to direct touchdown markets that are available in the existing stats and projection data.

## Concrete Implementation Plan

### Step 1: Label Builder

Status: implemented for the current seven modeled markets in `src/player_props/labels.py`.

CLI entry point:

```bash
python -m src.player_props.labels --seasons 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026
```

For live weekly runs, pass the active-week cutoff so labels do not include the prediction week:

```bash
python -m src.player_props.labels --seasons 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026 --exclude-from-season 2026 --exclude-from-week 3
```

Output:

- `data/processed/player_prop_labels.parquet`

Minimum columns:

- `season`
- `week`
- `game_id`
- `team`
- `opponent`
- `player_id`
- `player_name`
- `position`
- `position_group`
- `market`
- `actual_value`
- `actual_over_zero`
- `played_flag`
- `active_flag`
- `value_type`
- `source`

Current markets:

- `qb_passing_yards`
- `qb_passing_tds`
- `rb_rushing_yards`
- `rb_rushing_tds`
- `wrte_receiving_yards`
- `wrte_receiving_tds`
- `anytime_td`
- `def_sacks`

Implementation notes:

- Use `data/raw/nfl_player_stats.parquet` as the primary label source because it contains the richest completed-game player stat fields.
- Use `data/processed/player_actuals.parquet` as a cross-check and fallback for PBP-derived labels.
- For sacks, include both `actual_value` as sack count and a derived binary label `actual_over_zero`.
- Keep labels long-format by market so adding new markets does not require widening every downstream table.
- Fill missing `game_id` values from the schedule table when `season`, `week`, `team`, and `opponent` can be matched.

### Step 2: Feature Builder

Status: implemented in `src/player_props/features.py` with separate offensive and defensive outputs.

CLI entry point:

```bash
python -m src.player_props.features --seasons 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026
```

For live weekly runs, pass the active-week cutoff so target-week labels cannot enter features:

```bash
python -m src.player_props.features --seasons 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026 --exclude-from-season 2026 --exclude-from-week 3
```

Output:

- `data/processed/player_prop_features_offense.parquet`
- `data/processed/player_prop_features_defense.parquet`

Feature groups:

- Rolling player production: last 1, last 3, last 5, career-to-date, and season-to-date shifted history.
- Team market context: prior weekly total, mean, and player-count rates for the player's team and market.
- Opponent allowed context: prior weekly total, mean, and player-count rates allowed by the opponent for the same market.
- Game context: home/away flags derived from `game_id`.

Planned feature additions after the first trained-model pass:

- Opportunity features: attempts, carries, targets, receptions, snap share where available.
- Role features: roster status, position group, depth chart rank, recent team, games sampled.
- Matchup features: opponent defensive rates, team projected margin, total, pace, weather, volatility flag.
- Availability features: injury status, active roster flag, recent news count where available.

Leakage rule:

- A row for `season/week` may only use games strictly before that week, plus prior seasons.
- Offensive and defensive features are written to separate files to avoid mixing incompatible player populations in one model family.

### Step 3: Baseline Evaluation

Status: implemented in `src/player_props/evaluate_baselines.py`.

CLI entry point:

```bash
python -m src.player_props.evaluate_baselines --prediction-dir predictions --output-dir predictions/evaluation/player_props --start-season 2017 --exclude-from-season 2026 --exclude-from-week 3
```

Inputs:

- Existing projection CSVs under `predictions/predictions_players_*.csv` and archived `wN_predictions_players_*.csv`.
- `data/processed/player_prop_labels.parquet`.

Outputs:

- `predictions/evaluation/player_props/baseline_metrics.csv`
- `predictions/evaluation/player_props/baseline_predictions.csv`

Metrics:

- Yardage markets: MAE, RMSE, median absolute error, directional accuracy against common synthetic lines when available.
- Sack market: MAE for expected sacks, Brier/LogLoss/AUC for `sack > 0`.
- Coverage: number of labeled rows, number predicted, join rate, active-player rate.

Current baseline snapshot with the Week 3 live cutoff:

| Market | Scored | MAE | RMSE | Label Coverage |
|---|---:|---:|---:|---:|
| QB passing yards | 277 | 68.52 | 89.50 | 54.3% |
| RB rushing yards | 818 | 22.81 | 33.36 | 64.8% |
| WR/TE receiving yards | 1,857 | 22.37 | 30.64 | 63.9% |
| Defensive sacks | 2,946 | 0.43 | 0.53 | 39.7% |

For sacks, the evaluator also converts expected sacks to `P(sack > 0)` using a Poisson-style transform and reports Brier, LogLoss, and AUC.

### Step 4: First Trained Models

Status: implemented in `src/player_props/train.py`.

CLI entry point:

```bash
python -m src.player_props.train --start-season 2017 --exclude-from-season 2026 --exclude-from-week 3
```

Initial model choices:

- Yardage markets: gradient boosting regressor by default, with ridge mode available for fast deterministic tests.
- Sack market: gradient boosting expected-count model plus binary `sack > 0` classifier.

Outputs:

- `models/player_props/offense/<market>_model.pkl`
- `models/player_props/offense/<market>_metadata.json`
- `models/player_props/defense/<market>_model.pkl`
- `models/player_props/defense/<market>_metadata.json`
- `predictions/evaluation/player_props/model_metrics.csv`
- `predictions/evaluation/player_props/model_predictions.csv`
- `predictions/evaluation/player_props/model_vs_baseline.csv`

Current walk-forward trained-model snapshot:

| Market | Walk-Forward Rows | Model MAE | Model RMSE | Notes |
|---|---:|---:|---:|---|
| QB passing yards | 5,200 | 69.50 | 87.77 | Full walk-forward MAE is slightly worse than the limited baseline sample, but same-baseline-row comparison improved MAE by 8.07 yards. |
| RB rushing yards | 11,833 | 20.98 | 29.15 | Beats the rate-based baseline on native and same-baseline-row comparisons. |
| WR/TE receiving yards | 26,339 | 21.61 | 28.79 | Beats the rate-based baseline on native and same-baseline-row comparisons. |
| Defensive sacks | 1,201 | 0.20 | 0.34 | `sack > 0` AUC is 0.739 with Brier 0.099. |

Same-baseline-row comparison:

| Market | Rows | Baseline MAE | Model MAE | MAE Delta |
|---|---:|---:|---:|---:|
| QB passing yards | 277 | 68.52 | 60.45 | -8.07 |
| RB rushing yards | 818 | 22.81 | 20.07 | -2.74 |
| WR/TE receiving yards | 1,857 | 22.37 | 20.71 | -1.66 |
| Defensive sacks | 516 | 0.41 | 0.27 | -0.14 |

Acceptance gate before publishing trained model outputs:

- Walk-forward evaluation exists.
- Same-baseline-row comparison beats the current rate-based baseline across all four first markets.
- Sample sizes are published in `model_metrics.csv` and `model_vs_baseline.csv`.
- For binary/event markets, calibration must still be checked before confidence tiers are exposed.

### Step 5: Weekly Prediction Export

Status: implemented in `src/player_props/predict.py` and wired into `tools.run_pipeline` after trained prop model refresh.

Outputs:

- `predictions/player_props_qb.csv`
- `predictions/player_props_offense.csv`
- `predictions/player_props_defense.csv`
- `predictions/player_props_novelty.csv`

Recommended output schema:

- `season`
- `week`
- `game_id`
- `team`
- `opponent`
- `player_id`
- `player_name`
- `position`
- `market`
- `projection`
- `prob_over`
- `confidence_tier`
- `model_name`
- `baseline_projection`
- `model_projection`
- `sample_size`
- `feature_count`
- `line`
- `implied_probability`
- `edge`
- `sportsbook`
- `over_odds`
- `under_odds`
- `line_updated_at`
- `games_sampled`
- `player_rank`
- `generated_at`

Status: implemented for sportsbook lines available from PropLine, defaulting to DraftKings, Hard Rock, and FanDuel. `edge` is currently `model_projection - line`, so it is a yard/count delta rather than a probability edge. PropLine can return standard O/U rows and milestone ladders such as `70+ Rushing Yards`; the exporter parses both, using Over-only implied probability when no Under price is available.

Status: implemented for market-aware calibrated over probabilities. When `data/processed/player_prop_historical_line_eval.parquet` exists, `src.player_props.predict` uses prior same-market residuals and market-implied blend diagnostics to fill line-specific `prob_over`, `prob_over_raw`, `prob_edge`, `prob_over_method`, and `prob_over_sample_size` for rows with matched lines. Current Week 3 metrics improved overall Brier to 0.2497 and LogLoss to 0.6925, but sportsbook implied probabilities still remain slightly better overall at 0.2481 Brier and 0.6890 LogLoss, so `prob_edge` and `edge_ev` are diagnostic until edge-threshold backtests prove lift.

Status: first edge-threshold backtest and conservative recommendation policy implemented. The Week 3 policy currently selects only market-level thresholds with at least 100 historical bets and positive ROI: strong anytime TD at `prob_edge >= 0.10` and lean RB rushing yards at `prob_edge >= 0.02`. Current Week 3 rows expose these thresholds, but no row is flagged as recommended unless it also has positive current EV after price/juice.

Remaining work:

- Use `prop_line_coverage_week_NN.csv` to drive targeted coverage work by sportsbook, market, player naming, and unmodeled market shape.
- Use the new `coverage_status` field to separate true vendor no-row gaps from prepared-but-unmatched player-name/game matching problems.
- Backfill historical first/last/2+/3+ TD scorer lines and add edge backtests before publishing bettable first/last/2+/3+ TD outputs.
- Use the Odds API backfill manifest by season/week/game/snapshot time, discover event-market availability first, then pull historical event player-prop snapshots only for markets that existed at that snapshot.
- Re-run local historical prop-line snapshots and probability diagnostics after each material player-prop model change:

```bash
python -m src.player_props.historical_lines --debug
python -m src.player_props.over_probability --debug
```

- Expand edge backtesting beyond the overall rows: inspect market, sportsbook, season, and best-price slices before exposing confidence tiers or bet recommendations.

### Step 6: MattyTheBookie Publishing

Status: CSV publishing support is implemented in `tools.publish_mtb_csvs`.

Published current and weekly archive files now include:

- `predictions/player_props_qb.csv`
- `predictions/player_props_offense.csv`
- `predictions/player_props_defense.csv`
- `predictions/player_props_novelty.csv`

Keep the current `predictions/predictions_players_*.csv` files available as legacy projection tables until the website has migrated. Website-side consumption and display in the MattyTheBookie repo can be handled separately from CSV publishing.
