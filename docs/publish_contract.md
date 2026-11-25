# NFL → MTB publish contract

**Version 1.1.0** (1.0.0 is the baseline; 1.1.0 adds previews). See the [changelog](#changelog).

This contract covers what nfl-predictions publishes into MattyTheBookie (MTB):
- the files `tools/publish_mtb_csvs.py` copies into `C:\Code\mtb\data\prediction-csvs\`;
- the columns MTB depends on;
- the formats of those columns.

It describes the format in use today; writing it down changed no format.

**Confirmation.** MTB confirms each version through the user. Its index of cross-repo agreements is `mtb/docs/contracts.md`. Once the version check ships (a separate PR), `publish_mtb_csvs.py` refuses to copy into MTB when `CONTRACT_VERSION` differs from `contract_confirmed_version`.

**Readers checked.** The column lists below come from MTB's readers, read on 2026-11-25 at MTB `b07a21c`:
- `scripts/import-predictions.ts`
- `lib/result-overlays.ts`
- `lib/picks-of-record.ts`
- `lib/venues.ts`
- `lib/model-metrics.ts`
- `lib/blog/generator.ts`, which also reads several of these files.

## Rules for MTB

- **Required columns** are listed per file below. nfl-predictions keeps them present and keeps their meaning.
- **Every other column is informational.** MTB must ignore any column it doesn't use. Columns may be added at any time.
- **Read by header name**, never by position.
- **Empty files and missing optional files** are normal. For example, there are no props before lines are posted, and no graded weeks before Week 1 is final.

## Files

All paths are relative to `data/prediction-csvs/` in MTB.

There are two kinds of run.
- **Slot run.** Every game-day pipeline run calls `publish_mtb_csvs --season S --week W` under the MTB publish lock.
- **Results run.** Tuesday grading (after the week's last game) calls `publish_mtb_csvs --season S --results-only`, also under the lock.

Each file is either:
- **overwritten**, replaced on every run that writes it; or
- **write-once**, written by the first run that has a full slate, then never replaced unless someone passes `--overwrite-week-archive`.

### Game predictions and player outputs (slot runs)

| MTB path | Written | Mode | MTB reader |
|---|---|---|---|
| `current/predictions.csv` | every slot run (**required**; the copy fails without it) | overwritten | import-predictions, blog |
| `current/predictions_full.csv` | every slot run | overwritten | none (informational) |
| `current/predictions_players_qb.csv`, `_offense.csv`, `_defense.csv` | every slot run | overwritten | import-predictions |
| `current/player_props_public.csv` | every slot run | overwritten | import-predictions, blog |
| `current/player_props_qb.csv`, `_offense.csv`, `_defense.csv`, `_novelty.csv` | every slot run | overwritten | import-predictions legacy fallback only (used when `player_props_public.csv` is absent) |
| `seasons/<season>/week_<nn>/<each of the 10 files above>` | the first slot run of the week whose predictions cover the full scheduled slate | **write-once** | blog (`player_props_public.csv`) |

**Gap in the week archive.** `seasons/<season>/week_<nn>/` is skipped whenever a run's slate is partial (started games are dropped). If the week's full-slate runs fail to publish, the week gets no CSV archive. This happened in 2026 Week 11: the Thursday and Friday manual publishes failed. MTB's own JSON archive (`data/predictions/seasons/...`) is built separately, at import.

### Model, evaluation and analysis outputs (slot runs)

| MTB path | Written | Mode | MTB reader |
|---|---|---|---|
| `analysis/model_metrics_week_<nn>.csv` | every slot run of week `nn` | overwritten (one file per week) | model-metrics, blog |
| `analysis/market_benchmark_summary.csv`, `analysis/market_clv_summary.csv`, `analysis/market_edge_policy.csv` | every slot run | overwritten | blog |
| `evaluation/player_props/prop_quality_week_<nn>.csv`, `prop_quality_summary_week_<nn>.csv`, `prop_line_coverage_week_<nn>.csv` | every slot run of week `nn` | overwritten | import-predictions, blog |
| `evaluation/player_props/prop_roster_line_audit_week_<nn>.csv`, `prop_line_gap_audit_week_<nn>.csv` | every slot run of week `nn` | overwritten | none |
| `evaluation/player_props/edge_backtest_detail.csv`, `edge_backtest_summary.csv`, `edge_backtest_market_thresholds.csv`, `edge_recommendation_policy.csv`, `edge_policy_diagnostics.csv`, `confidence_tier_metrics.csv` | every slot run | overwritten | none |
| `evaluation/overall_metrics.csv`, `weekly_metrics.csv`, `calibration_by_prob_bin.csv`, `team_error_summary.csv`, `high_confidence_misclassifications.csv`, `largest_margin_errors.csv`, `feature_quality.csv`, `feature_family_ablation.csv`, `merged_predictions_actuals.csv` | every slot run | overwritten | none |
| `analysis/market_*.csv` (the remaining market files: ROI, disagreement, calibration, blend, spread, policy) and `analysis/volatility_classifier_*.csv` | every slot run | overwritten | none |
| `evaluation/live_prediction_vs_actual_week_<nn>.csv`, `_summary_week_<nn>.csv`, `_segments_week_<nn>.csv` | every slot run, for every graded week | overwritten (a regrade replaces it) | result-overlays, blog |
| `picks_of_record/picks_of_record_<season>.csv`, `picks_of_record/picks_ledger_<season>.csv` | every slot run | overwritten; locked rows never change; the ledger is append-only | none (informational) |
| `reference/game_venues_<season>.csv` | every slot run | overwritten | venues |

### Results (Tuesday results run, `--results-only`)

| MTB path | Mode | MTB reader |
|---|---|---|
| `evaluation/live_prediction_vs_actual.csv`, `evaluation/live_prediction_vs_actual_summary.csv` (cumulative, every graded week) | overwritten | result-overlays, blog (fallbacks) |
| `evaluation/live_prediction_vs_actual[_summary\|_segments]_week_<nn>.csv` for **every** graded week | overwritten (a regrade replaces it) | result-overlays, blog |
| `picks_of_record/picks_of_record_<season>.csv`, `picks_ledger_<season>.csv` | as above | none |

### Previews (1.1.0, live from Week 14 on Tue 2026-12-08)

| MTB path | Written | Mode | MTB reader |
|---|---|---|---|
| `current/previews.csv` | results run after grading week N, for week N+1; refreshed Wednesday from 12:00 ET, only if the content changed | overwritten | MTB preview pages |
| `previews/<season>/week_<nn>.csv` | same as `current/previews.csv` | overwritten (deliberately **not** under `seasons/`, so the Wednesday refresh and schedule changes can replace it) | MTB preview pages |

## Required columns

### `current/predictions.csv` (one row per game in the run)

- **Identity and schedule:** `game_id`, `season`, `week`, `home_team`, `away_team`, `kickoff`, `kickoff_mt`.
- **Conditions:** `roof`, `weather_temp_f`, `weather_wind_mph`.
- **Model:** `home_win_prob`, `pred_home_margin`, `volatility_prob`, `volatility_label`.
- **Pick:**
  - `home_pick`: the picked team's abbreviation, which may be the away team despite the name;
  - `home_confidence_pct`;
  - `market_moneyline_model_pick_prob`: the model's probability for the picked side, which MTB prefers for "confidence";
  - `market_model_pick`.
- **Model odds:** `home_odds_american`, `home_odds_decimal`. These are the **model's fair odds for the home team**, computed from `home_win_prob`. They are not a sportsbook price, and they are not the price of the pick when the pick is the away team. See "Known disagreements".
- **Text:** `pick_expl`, `decision_explanation`, `decision_narrative`.

### `current/predictions_players_{qb,offense,defense}.csv` (one row per player)

- **Every group:** `season`, `week`, `game_id`, `kickoff`, `team`, `team_side`, `player_rank`, `player_id`, `player_name`, `jersey_number`, `games_sampled`.
- **qb:** `projected_passing_yards`, `projected_passing_tds`, `projected_rushing_tds`, `projected_passing_attempts`, `projected_interceptions`, `projected_completions`, `projected_rushing_yards`.
- **offense:** `projected_rushing_yards`, `projected_rushing_tds`, `projected_carries`, `projected_receiving_yards`, `projected_receiving_tds`, `projected_receptions`, `projected_targets`, `projected_total_yards`, `projected_total_tds`, `penalties_per_game`, `penalty_count`, `penalty_yards`.
- **defense:** `projected_sacks`, `projected_qb_hits`, `projected_tfl`, `projected_blocked_punts`, `projected_loss_yards`, `penalties_per_game`, `penalty_count`, `penalty_yards`.

### `current/player_props_public.csv` (one row per player × market)

MTB maps every one of these columns (import-predictions `mapPlayerProp`):
- **Identity:** `season`, `week`, `game_id`, `team`, `opponent`, `team_side`, `player_id`, `player_name`, `position`, `position_group`, `market`.
- **Projection:** `projection`, `prob_over`, `confidence_tier`, `model_name`, `baseline_projection`, `model_projection`, `sample_size`, `feature_count`.
- **Line and edge:** `line`, `line_type`, `implied_probability`, `prob_edge`, `edge`, `edge_side`, `edge_odds`, `edge_break_even_probability`, `edge_ev`, `edge_probability_edge`.
- **Recommendation policy:** `edge_recommendation_flag`, `edge_recommendation_tier`, `edge_policy_threshold`, `edge_policy_roi`, `edge_policy_n_bets`.
- **Public board:** `public_board_status`, `public_board_reason`, `public_bettable_flag`, `public_sort_rank`, `model_signal_score`, `model_signal_label`, `model_signal_reason`.
- **Book and line spread:** `sportsbook`, `over_odds`, `under_odds`, `line_updated_at`, `line_consensus`, `line_min`, `line_max`, `line_range`, `line_std`, `line_book_count`, `line_options_count`, `line_variance_flag`.
- **Injury:** `injury_status`, `injury_practice_status`, `injury_body_part`, `injury_report_week`, `injury_source`, `injury_last_updated`, `injury_reported`, `injury_exclusion_flag`.
- **Other:** `prop_quality_flag`, `bettable_flag`, `games_sampled`, `player_rank`, `generated_at`.

### `evaluation/player_props/prop_quality_week_<nn>.csv` and `prop_quality_summary_week_<nn>.csv`

`season`, `week`, `scope`, `prediction_group`, `market`, `sportsbook`, `rows`, `rows_with_lines`, `missing_line_rows`, `bettable_rows`, `injury_reported_rows`, `injury_review_rows`, `injury_exclusion_rows`, `line_variance_rows`, `avg_line_range`, `max_line_range`, `avg_abs_edge`, `avg_line_book_count`, `generated_at`.

The blog filters on `scope == "overall"` and `prediction_group == "all"`.

### `evaluation/player_props/prop_line_coverage_week_<nn>.csv`

`season`, `week`, `scope`, `market`, `sportsbook`, `raw_rows`, `raw_unmodeled_market_rows`, `prepared_line_rows`, `offered_game_player_pairs`, `offered_players`, `prediction_rows`, `prediction_players`, `matchable_prediction_rows`, `published_line_rows`, `prediction_rows_without_offered_line`, `offered_pairs_not_in_predictions`, `player_name_overlap_no_game`, `name_key_collision_groups`, `generated_at`.

### `evaluation/live_prediction_vs_actual*.csv` (graded games)

- **Weekly and cumulative game files:** `season`, `week`, `game_id`, `model_pick_team`, `actual_winner`, `home_score`, `away_score`, `home_team`, `away_team`, `home_team_actual`, `away_team_actual`, `margin_error`, `abs_margin_error`, `model_pick_moneyline`, `moneyline_profit_unit`, `moneyline_result`, `selected_spread_result`, `su_correct`, `su_result`.
- **A blank `model_pick_team` means "no pick of record".** MTB doesn't grade such a row.
- **MTB also displays every column whose name contains `clv`**, so CLV columns count as required in meaning.
- **Summary files:** `scope`, `segment_column`, `segment_value`, `n_games`, plus the metric columns.

### `reference/game_venues_<season>.csv`

`game_id`, `home_team`, `neutral_site`, `venue_name`, `venue_city`, `venue_country`, `venue_lat`, `venue_lon`.

- `venue_country` uses the names `USA`, `UK`, `Australia`, `Brazil`, `France`, `Germany`, `Mexico`, `Spain` (2026 file; MTB's country map also accepts `Ireland`, `Canada`, `United Kingdom`). Adding a new country name is a change MTB must confirm.
- `venue_lat` / `venue_lon` of `0,0` mean unknown.

### `analysis/model_metrics_week_<nn>.csv` (one row per metric)

- **Columns:** `season`, `week`, `generated_at`, `metric_key`, `value`, `display_value`, `label`, `description`, `section`, `sort_order`, `sample_size`, `unit`, `source_artifact`.
- **Metric keys MTB looks up by name:** `accuracy`, `evaluation_games`, `evaluation_start_season`, `excluded_from_week`, `clv_moneyline_disagreement_tracked_games`, `clv_moneyline_disagreement_avg_prob_delta`, `clv_moneyline_disagreement_positive_rate`. These keys must stay present with the same meaning.
- **Other keys** (for example `live_published_season_*` and `wf_*`) are informational until an MTB page uses them.

### `analysis/market_benchmark_summary.csv`, `analysis/market_clv_summary.csv`, `analysis/market_edge_policy.csv` (blog)

- The benchmark and CLV summaries need `scope` and `segment`. The blog filters on `scope == "overall"` and `segment == "all"`.
- `market_edge_policy.csv` is read as a whole. Its current header counts as required.

### Previews (1.1.0): `current/previews.csv` and `previews/<season>/week_<nn>.csv`

These are exactly the 18 agreed columns, in this order. A test in nfl-predictions pins the list and refuses anything model-like.

| Column | Meaning |
|---|---|
| `season`, `week`, `game_id` | as in `predictions.csv` |
| `home_team`, `away_team` | team abbreviations |
| `kickoff` | UTC, `YYYY-MM-DDTHH:MM:SSZ` |
| `pick_expected_at` | UTC; when that game's pick run window opens (MTB shows only the day) |
| `roof` | `outdoors`, `dome`, `closed`, `open`, or blank (retractable roof not yet announced) |
| `venue_name`, `venue_city` | as in `game_venues` |
| `neutral_site` | `true` / `false` (lower case) |
| `spread_home` | home handicap; negative means the home team is favored; blank if no line |
| `total` | points; blank if no line |
| `home_ml`, `away_ml` | American, whole numbers; blank if no line |
| `odds_source` | `the-odds-api:us-median` (median across US books, shown as "market median"), `nflverse-schedule`, or blank |
| `odds_captured_at`, `generated_at` | UTC ISO-8601 |

The file contains **no model output** (no probability, pick, margin, edge or confidence). It never reads predictions.

## Formats

- **`game_id`:** `<season>_<week, 2 digits>_<AWAY>_<HOME>`, for example `2026_14_MIN_NE`. This is nflverse's id, and it contains no date or time.
  - It stays stable through flexed games, moved kickoffs and neutral-site games, where the nominal home team is kept.
  - It changes only if a game moves to another week or the home and away teams swap. MTB treats an id that's missing from a later file as gone.
- **Team abbreviations:** nflverse (`LA` for the Rams, `LAC`, `LV`, `WAS`, ...).
- **Kickoff:**
  - `kickoff` in game files is UTC written as `YYYY-MM-DD HH:MM:SS+00:00` (a space, not `T`);
  - previews use `YYYY-MM-DDTHH:MM:SSZ`, and MTB parses both;
  - `kickoff_mt` is display text, `YYYY-MM-DD HH:MM MST|MDT`.
- **Other timestamps** (`generated_at`, `line_updated_at`, `recorded_at`, ...): ISO-8601, UTC unless they carry an offset.
- **Booleans:**
  - game and prop files: `True` / `False`;
  - graded files: `1.0` / `0.0`, or blank when unknown (pandas floats; for example `su_correct`);
  - previews: lower-case `true` / `false`.
  - MTB reads `1`, `true`, `y`, `yes` (in any case) as true. In graded files, `0` and `false` read as false and a blank means unknown.
- **Blanks:** a missing or NaN value is an **empty field**. Files never contain the literal text `nan`, `None` or `NaN`; the result overlays treat `nan` as blank anyway.
- **Numbers:** plain decimal, no thousands separators. Probabilities are 0–1 unless the column name ends in `_pct`, which is 0–100.
- **Odds:**
  - American odds are whole numbers or at most 2 decimals (`-110`, `+125`, `-101.48`); decimal odds are greater than 1.
  - The market columns (`market_home_moneyline`, `market_away_moneyline`, and the preview `home_ml` / `away_ml`) are medians across US books.
  - From nfl-predictions PR #51 (merging Tue 2026-12-01), moneyline medians are taken in decimal odds and rounded to cents, so no published market moneyline lies between −100 and +100. MTB treats any value in that range as missing.
  - Spreads are the home team's handicap: **negative means the home team is favored**.
- **Encoding:** UTF-8 without a byte-order mark, comma-separated, one header row.
- **Line endings:** CRLF for most files (pandas on Windows); LF for `game_venues_<season>.csv`, `picks_of_record/` and previews. Readers must accept both. The line ending isn't part of the contract.

## Pick of record and the transparency rule

**Recording.**
- Every live slot run appends one row per not-yet-started game to the append-only ledger `picks_ledger_<season>.csv`, stamped with the predictions file's write time (`recorded_at`).
- A game's **pick of record** is the latest ledger row recorded at least **2 hours before kickoff** (`lock_deadline_utc`). MTB mirrors this rule in `lib/picks-of-record.ts` (`pickLockLeadMs`), so the graded pick is the pick the site showed.
- If no row made the deadline, the first later row is used, marked `late`.
- Locked rows never change.

**Transparency rule.** Only picks **published on MTB before kickoff** count.
- **Started games:** `predict_upcoming` drops every started game from later runs, so a later run can never add or change a pick for a game in progress.
- **Games first seen after kickoff:** MTB shows "no pick of record". These games are never back-filled, and no substitute pick is used. `model_pick_team` is blank in the graded files, and such rows don't count toward the record.
- **Corrections:** nothing published is rewritten. A later fix to the pipeline (for example PR #51's moneyline median) applies only to runs after it ships.

## Versioning

Versions follow MAJOR.MINOR.PATCH, the same as mlb-predictions' contract.
- **1.x (non-breaking):**
  - adding a file, or an informational column;
  - new metric keys;
  - documentation fixes.
- **2.0.0 (breaking):**
  - renaming or removing a required column, or a file MTB reads;
  - changing a required column's meaning, units, sign or format;
  - changing the `game_id` format, or which runs write a file.
- **Every change, including 1.x additions MTB is meant to use, is confirmed by MTB through the user before it ships.**
  - nfl-predictions bumps `CONTRACT_VERSION` in the same PR.
  - The user sets `contract_confirmed_version` to the version MTB confirmed.
  - Until both match, `publish_mtb_csvs.py` logs the mismatch, alerts, exits non-zero, and copies nothing. *(The version check is a separate PR, built after MTB confirms this document.)*

## Known disagreements (found writing 1.0.0, open with MTB)

These aren't format changes. They're places where MTB's reading and nfl-predictions' meaning differ today.

1. **`home_odds_american` is shown as the sportsbook price for the pick.** MTB's result overlay says "Odds X are the sportsbook price for the pick", and the pick table shows the same value as "Odds".
   - The column is actually the model's fair odds for the **home** team (from `home_win_prob`).
   - Example: 2026_11_CIN_WAS. The pick was CIN, and MTB showed +261, the model's price for WAS. The sportsbook had CIN at −310.
   - In the last 40 runs, 293 game rows had an away-team pick.
   - Proposed fix, non-breaking, 1.2.0: nfl-predictions adds `pick_market_moneyline` (the market median price for the picked side) and `pick_model_odds_american` (the model's fair price for the picked side). MTB switches its "Odds" display to the one it means.
2. **`lib/model-metrics.ts` picks the "latest" metrics file by week number only, across seasons.** Once 2027 starts, `model_metrics_week_18.csv` from 2026 (or a playoff week 19–22) would outrank 2027's `week_01`. Suggested MTB fix: choose by the file's `season` column first, then the week (or by modification time). No format change is needed.
3. **`picks_of_record/*.csv` has no MTB reader.** `lib/picks-of-record.ts` re-applies the 2-hour lock to MTB's own JSON archive instead. The CSVs stay informational, an audit copy.
4. **Week CSV archives can be missing** (see the Files section): 2026 Week 11 has none. The blog falls back to `current/player_props_public.csv` for that week.

## Changelog

- **1.1.0** (proposed 2026-11-25): adds the preview files `current/previews.csv` and `previews/<season>/week_<nn>.csv`, live from Week 14 (Tue 2026-12-08). Code: nfl-predictions PR #50. Schedule: `docs/automation.md` "Game previews".
- **1.0.0** (proposed 2026-11-25): baseline. Records the files, required columns and formats in use as of nfl-predictions `a0304fb`, read against MTB `b07a21c`. No format changes.
