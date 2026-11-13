# Data Sources

This file documents the data sources used by the NFL predictions pipeline, how each source is consumed, and whether the pipeline treats it as required or optional.

The weekly entry point is `python -m tools.weekly_run` ([docs/weekly_run.md](docs/weekly_run.md)), which resolves the week and phase and runs the pipeline with a debug log. The underlying pipeline command is:

```powershell
python -m tools.run_pipeline --start-year 2017 --train-start-year 2017 --max-parallel-data 8 --data-start-delay 1 --use-async --prediction-week <week> --live-run --skip-tomorrow-weather --publish-mtb --mtb-repo-dir C:\Code\mtb
```

## Status Key

| Status | Meaning |
|---|---|
| Required | The normal full pipeline expects this data or a direct fallback. Missing data can stop feature building, training, or prediction. |
| Required for specific outputs | The team model can run without it, but a named output will be empty, degraded, or unavailable. |
| Optional | The pipeline will skip this source when credentials or access are missing. Features may be less rich, but the core team prediction workflow should still run. |
| Diagnostic | Used for analysis, monitoring, or dashboard context, not required for weekly predictions. |

## Source Inventory

| Source | Fetcher | Main Raw Outputs | Used For | Status |
|---|---|---|---|---|
| NFLVerse release assets, with `nfl_data_py` fallback | `src.data.nflverse` | `nfl_schedules.parquet`, `nfl_pbp.parquet`, `nfl_rosters.parquet`, `nfl_injuries.parquet`, `nfl_snap_counts.parquet`, `nfl_depth_charts.parquet`, `nfl_player_stats.parquet`, `nfl_pfr_*.parquet` | Canonical historical schedule/results, play-by-play team efficiency, injuries, roster/depth context, snap counts, player actuals, player identity linking, pressure/red-zone/passing EPA features, novelty TD labels, and defensive event labels | Required for historical feature/model richness and player labels; release files are primary for maintained datasets, with `nfl_data_py` retained as compatibility fallback |
| ESPN schedule fallback | `src.data.nflsdv` and cached `espn_schedule.parquet` | `espn_schedule.parquet`, `espn_schedule_<season>.parquet` | Fallback schedule source if NFLVerse schedule is missing; used by prediction code as a schedule input | Required fallback only |
| NFL.com team stats | `src.data.nflcom` | `nflcom_teamstats_{passing,rushing,receiving,scoring,downs}.parquet` | Team-season offensive/defensive production inputs used in matchup features | Required for full feature richness |
| ESPN player stats | `src.data.espn_players` | `espn_playerstats_{category}_{season}_stype2.parquet` | Aggregated player passing, rushing, and receiving totals into team-season signals | Optional but recommended |
| ESPN player news | `src.data.espn_player_news` | `espn_player_news.parquet` | Player news counts, injury keyword counts, and pre-game news timing features | Optional |
| ESPN team news | `src.data.espn_team_news` | `espn_news.parquet` | Team news counts and injury keyword features | Optional |
| NOAA weather observations | `src.data.noaa` | `processed/noaa_weather.parquet`, station cache JSON | Stadium-adjacent observed weather merged into game weather features | Optional but recommended for weather features |
| Tomorrow.io live weather | `src.data.weather` | `processed/weather_games.parquet` | Live forecast weather for the publish window when enabled; cache is preserved if a live refresh returns no rows | Optional when Visual Crossing/NOAA/static fallback are available; skip with `--skip-tomorrow-weather` |
| Visual Crossing weather | `src.data.visualcrossing` | `processed/visualcrossing_weather.parquet` | Paid hourly game-window weather, precipitation, wind, gust, and condition fields | Optional but preferred for current operations; requires `VISUAL_CROSSING_API_KEY` |
| Static climatology fallback | `src.data.weather_fallback` | No fetched raw file | Last-resort weather estimates when live weather sources miss a game | Built in fallback |
| BallDontLie | `src.data.balldontlie`, `src.features.balldontlie_team_form` | `balldontlie_teams.parquet`, `balldontlie_players.parquet`, `balldontlie_active_players.parquet`, `balldontlie_games_<season>[_wkNN].parquet`, `balldontlie_standings_<season>.parquet`, injury/stat/team-stat files | Commercial supplemental schedule, standings, player directory, active-player, injury, player-stat, and team-stat context. Team game stats are converted into final-only prior/rolling weekly form features for the team model. | Optional, requires `BALLDONTLIE_API_KEY`; current default commercial provider for ALL-STAR-tier feeds |
| PropLine | `src.data.propline` | `propline_player_props_<season>_wkNN.parquet` | Player-prop line snapshots for weekly prop outputs, including O/U rows and milestone ladders such as `70+ Rushing Yards`; defaults to DraftKings, Hard Rock, and FanDuel | Optional for projections, required for populated player-prop line/edge columns; requires `PROPLINE_API_KEY` |
| SportsDataIO | `src.data.sportsdataio` | `sportsdataio_teams.parquet`, `sportsdataio_stadiums.parquet`, `sportsdataio_schedules_<season>.parquet`, `sportsdataio_player_game_projections_<season>_wkNN.parquet`, DFS, futures, draft, free-agent files | Legacy commercial schedule, stadium metadata, betting metadata, DFS/player projections, and supplemental player/team context | Optional fallback, requires `SPORTSDATAIO_API_KEY`; skipped by default when BallDontLie is configured unless `--enable-sportsdataio` is passed |
| Yahoo Fantasy Sports | `src.data.yahoo` | Yahoo game/team/player/injury cache files when authorized | Fantasy player/team metadata and injury context | Optional, requires approved Yahoo Fantasy API access and OAuth tokens |
| Odds API | `src.utils.odds` via `src.predict.predict_upcoming`; `src.data.odds_api_historical` and `src.data.odds_api_backfill` for historical events/event markets/event odds | Upcoming odds have no persistent raw cache by default; historical backfill writes aggregate events/markets/odds files plus `data/processed/oddsapi_historical_player_prop_lines.parquet` | Upcoming sportsbook spreads/prices attached to prediction exports; historical player-prop odds feed `src.player_props.historical_lines` and `src.player_props.over_probability` for calibrated over probabilities and edge-threshold backtests | Optional; non-historical calls require `odds_api_free_key`, historical calls require `odds_api_paid_key`; historical player-prop pulls are manual/backfill because they spend quota per event and market |
| Sportradar | `src.data.sportradar`, `src.data.sportradar_transform` | `data/raw/sportradar/*` and transformed game/team/player parquet files | Advanced team/player metrics and dashboard context when files exist | Optional |
| Reference CSVs | `data/reference/*`, `src.data.reference.update_regular_season` | Team locations, stadium advantages, rivalry list, regular-season reference games | Team normalization, travel/rest/stadium features, schedule/reference joins | Required local static data |
| Historical predictions | `src.predict.predict_history --walk-forward` | `predictions/history/*.csv` | Out-of-sample calibration, evaluation, volatility labels, market ROI, run comparison | Required for calibration/evaluation; regenerated by pipeline |
| MattyTheBookie local publish | `tools.publish_mtb_csvs`, `tools.import_mtb_predictions` | `C:\Code\mtb\data\prediction-csvs\current\*.csv`, `C:\Code\mtb\data\predictions\current.json`, plus `seasons/<season>/week_NN` snapshots | Copies selected prediction/evaluation/analysis CSVs into the local website repo and refreshes the normalized website JSON after a successful pipeline run | Optional operational export; enabled with `--publish-mtb` |
| Locked/live predictions | `analysis.live_tracking` | `predictions_log/*` and live tracking outputs | Audit trail for pre-game locked picks and post-game results | Diagnostic/operational |

## Required Core Path

The team prediction model primarily needs:

1. A schedule source: `nfl_schedules.parquet`, with `espn_schedule.parquet` as fallback.
2. Historical game/team signal: NFLVerse play-by-play plus NFL.com team stats.
3. Roster/injury context: NFLVerse rosters, injuries, snap counts, and depth charts.
4. Weather context: merged weather fields, using NOAA, Visual Crossing, and static fallback as available.
5. Local reference data: team, stadium, and schedule metadata under `data/reference`.

If these are present, the team-level weekly predictions can run even when optional commercial or fantasy APIs are unavailable.

## What NFLVerse Is Used For

NFLVerse is still one of the backbone datasets, but not every weekly output depends on every NFLVerse subfeed. The ingestion path now prefers the maintained `nflverse-data` GitHub release assets for PBP, schedules, rosters, injuries, snap counts, depth charts, and player stats, then falls back to `nfl_data_py` only when a release file is unavailable or empty.

- `nfl_schedules.parquet`: canonical schedule/results spine used by weather, Odds API backfills, feature joins, evaluation, and archive/result matching. ESPN schedule is the fallback.
- `nfl_pbp.parquet`: historical play-by-play used for EPA/pass-efficiency features, pressure/red-zone style labels, defensive event aggregation, player actuals, and novelty touchdown labels. Current-season PBP can lag; BallDontLie/NFL.com now cover some current-season team-form gaps, but they do not fully replace PBP-derived labels.
- `nfl_player_stats.parquet`: weekly player actuals used for player-prop training labels and legacy player projection tables.
- `nfl_rosters.parquet`: GSIS-backed identity bridge for player aliases, ESPN/BallDontLie/Sportradar links, roster validation, and player projection joins.
- `nfl_injuries.parquet`, `nfl_depth_charts.parquet`, and `nfl_snap_counts.parquet`: injury/availability, QB-health, role/depth, and player-quality controls. ESPN and BallDontLie help current-week availability, but NFLVerse remains useful historically.
- `nfl_pfr_*.parquet`: historical supplemental passing/rushing/receiving context. These are optional enrichment files and are allowed to be absent for unsupported seasons.

In short: BallDontLie and NFL.com reduce our dependence on live current-season NFLVerse availability, but NFLVerse still supplies the historical labels and play-level detail that make the team model, player-prop labels, novelty TD labels, and player identity table reproducible.

## Player Projection Path

The player CSVs are separate from the team model:

| Output | Main Inputs | Behavior When Missing |
|---|---|---|
| `predictions.csv` | Team model outputs plus Odds API/market policy artifacts | Writes the team board with `market_public_recommendation` and `market_public_recommendation_reason` so consumers can separate `recommended`, `lean`, `research_only`, `review`, and `no_bet`. Spread rows stay research-only or blocked until cover-probability calibration improves; moneyline disagreement gates can use both probability-edge bins and model-side moneyline buckets. |
| `predictions_players_qb.csv` | NFLVerse weekly player stats plus current NFLVerse roster | Writes a current empty CSV instead of leaving stale rows. |
| `predictions_players_offense.csv` | NFLVerse weekly player stats, current roster, player penalties | Writes a current empty CSV instead of leaving stale rows. |
| `predictions_players_defense.csv` | NFLVerse play-by-play defense events plus current roster | Writes rows when PBP/roster data is available. |
| `player_props_qb.csv`, `player_props_offense.csv`, `player_props_defense.csv`, `player_props_novelty.csv` | Trained player-prop models plus legacy projection CSVs; optional PropLine sportsbook line snapshots fill `line`, odds, edge, and line-consensus/variance fields; nflverse/BallDontLie injuries fill quality-control flags | Full diagnostic exports. Writes model projections without lines when prop odds are unavailable or unmatched. Offered sportsbook rows can expand the prediction universe only when the player joins to identity, active roster availability, and prior pre-week labels for that market. Injury-excluded rows remain visible but are not marked bettable. Each row includes `edge_publish_status`, `edge_reason`, and public-board policy fields so consumer apps can explain rows without recreating backend policy logic. |
| `player_props_public.csv` | Derived from the full player-prop CSVs after line, injury, roster, and fallback-line quality gates | Cleaner website-facing board. Includes exact sportsbook-line rows for active roster players without injury exclusions or roster mismatches. Fallback-only TD split rows stay out of this board; exact novelty rows can remain visible as `research_only` but are not marked bettable. `public_bettable_flag` means the row is clean enough to evaluate; `public_pick_flag` is the only public prop-pick indicator. |
| `player_prop_historical_line_eval.parquet`, `player_prop_over_probability_eval.parquet`, `over_probability_metrics.csv`, `over_probability_bins.csv` | Normalized Odds API historical player-prop lines joined to walk-forward player-prop predictions and final actuals | Empty or stale if the paid historical backfill/join has not been refreshed. First-pass residual-CDF probabilities are diagnostic until they beat sportsbook implied probabilities in backtests. |
| `predictions/evaluation/player_props/prop_quality_week_NN.csv`, `prop_quality_summary_week_NN.csv`, `prop_line_coverage_week_NN.csv`, `prop_roster_line_audit_week_NN.csv`, `prop_line_gap_audit_week_NN.csv`, `edge_policy_diagnostics.csv` | Weekly player-prop CSVs plus PropLine snapshots and historical Odds API line evaluations | Empty reports if prop CSVs are unavailable; otherwise summarizes missing lines, bettable rows, injury flags, line variance, raw PropLine rows, prepared sportsbook lines, matchable rows, roster-line status, prediction rows without offered lines, offered player/game/market pairs outside the prediction universe, and edge-policy blocks split between below-threshold rows and no-positive-EV rows. The line-gap audit classifies offered rows with `offer_gap_category`, so non-player D/ST/no-touchdown outcomes, unsupported position/market offers such as QB rushing yards under the RB rushing model, inactive/IR/practice-squad offers, missing-history offers, identity misses, and true `candidate_expansion_gap` rows are distinguishable. Edge-policy diagnostics keep rejected market/threshold candidates visible, including insufficient sample and non-positive historical ROI reasons, before any market is promoted from research to bettable recommendations. |

NFLVerse weekly player stats may lag the current season. In that case, prediction code logs the latest available stats season and relaxes the recency floor so the output is explicit rather than stale.

BallDontLie and SportsDataIO player/stat/projection feeds are optional supplemental data. Empty cached commercial-provider files are treated as stale and refetched on the next run.

## Optional Credentials

| Credential | Enables | Pipeline Behavior If Missing |
|---|---|---|
| `BALLDONTLIE_API_KEY` | BallDontLie supplemental team/player/stat feeds | BallDontLie job is skipped. If SportsDataIO is configured, it can still run as fallback. |
| `PROPLINE_API_KEY` | PropLine player prop odds snapshots | PropLine job is skipped. Player prop line columns remain empty unless another compatible odds cache exists. |
| `SPORTSDATAIO_API_KEY` | Legacy SportsDataIO feeds | SportsDataIO job is skipped when missing, or when BallDontLie is configured unless `--enable-sportsdataio` is passed. |
| `VISUAL_CROSSING_API_KEY` | Visual Crossing hourly weather | Visual Crossing job is skipped; NOAA/static weather can still fill. |
| `TOMORROW_API_KEY` or `TOMORROWIO_API_KEY` | Tomorrow.io live publish-window forecast refresh | Tomorrow.io refresh is skipped; Visual Crossing/NOAA/static fallback can still fill. |
| `YAHOO_CLIENT_ID`, `YAHOO_CLIENT_SECRET`, `YAHOO_ACCESS_TOKEN`, `YAHOO_REFRESH_TOKEN`, `YAHOO_REDIRECT_URI` | Yahoo Fantasy feeds | Yahoo job is skipped unless credentials and API approval are valid. |
| `odds_api_free_key` or `ODDS_API_FREE_KEY` | Non-historical Odds API live/upcoming odds merge in upcoming predictions | Odds columns are omitted or left empty. This key is not used for historical endpoints. |
| `odds_api_paid_key` or `ODDS_API_PAID_KEY` | Historical Odds API event discovery, event-market discovery, and event/player-prop odds backfills | Historical prop calibration/backtesting remains unavailable without another historical line source. This key is not used for non-historical endpoints. |
| `NOAA_CONTACT_EMAIL` | NOAA User-Agent contact | Defaults to the repo fallback contact if unset. |

## Current Operational Notes

- Yahoo Fantasy is optional until API access is approved. OAuth tokens alone are not sufficient if Yahoo has not granted Fantasy Sports API access.
- BallDontLie is the preferred commercial provider for supplemental player/team context. PropLine is the player-prop line source. When the main pipeline is run with an explicit numeric `--prediction-week` and `PROPLINE_API_KEY` is configured, it requests NFL prop markets from DraftKings, Hard Rock, and FanDuel before weekly prop predictions are scored.
- Weather feature merging keeps the highest-priority weather values, but fills missing venue coordinates from source context first (`vx_latitude`/`vx_longitude` for Visual Crossing), then local team-location reference data. Visual Crossing is the preferred paid weather source for current operations; NOAA and static fallback fill when higher-priority sources miss. The live Tomorrow.io weather builder scores duplicate OSM stadium names against expected team city/state before updating stadium coordinates, so ambiguous names like Highmark Stadium do not pick the wrong venue. Tomorrow.io calls are scoped to the live publish window (`--forecast-days`, default `7`) and can be disabled with `--skip-tomorrow-weather` while rate-limited. If a live Tomorrow.io refresh returns zero rows, the job logs request-status diagnostics and preserves the existing `weather_games.parquet` cache instead of replacing it with an empty file.
- PropLine event names are normalized through sportsbook team aliases such as `ATL Falcons`, `GB Packers`, and `LA Chargers`. Player-line matching uses a stricter first-name-prefix key first and only falls back to initial/last-name matching when the game/market/player key is unique. Line-offered candidate expansion treats sportsbook `LA` and roster `LAR` as the same Rams team so valid Rams prop offers are not dropped by availability matching. Current PropLine requests include first/last/2+/3+ TD scorer novelty keys; those feed a separate novelty prediction path but remain research-only until historical calibration and edge backtests are in place. Stat-specific TD rows may attach an `anytime_td_fallback` line for visibility, but fallback-only rows are `research_only`, `bettable_flag=false`, excluded from `player_props_public.csv`, and should not be displayed as edge picks unless a true sportsbook line exists for the modeled market.
- Weekly prediction candidate expansion uses sportsbook-offered player/game/market lines only when the player identity match is unambiguous, the player is on the active roster for the expected team, and prior pre-week labels exist for that market. This keeps the `prop_line_gap_audit_week_NN.csv` workflow actionable without introducing same-week stat leakage.
- The Odds API historical backfill is intentionally not part of the default weekly pipeline. Use `src.data.odds_api_backfill` for the resumable one-time player-prop backfill: `--mode manifest` writes `data/processed/oddsapi_player_prop_backfill_manifest.parquet`, `--mode discover-events` fills event IDs, `--mode discover-markets` records per-event market availability, `--mode fetch-odds` pulls only available target player-prop markets, and `--mode normalize` writes `data/processed/oddsapi_historical_player_prop_lines.parquet`. Historical API modes require `--confirm-paid-backfill` unless `--dry-run` is used. For novelty TD history, request `player_1st_td`, `player_last_td`, and `player_tds_over`; `player_tds_over` is normalized into 2+ or 3+ TD model markets from the source milestone.
- Odds API key separation is enforced in code. Historical commands use only the paid key name/alias; non-historical live/upcoming odds use only the free key name/alias. Do not repoint either path to the other key without explicit approval.
- SportsDataIO is retained as a fallback while BallDontLie caches are verified. Avoid running it accidentally if cost is a concern.
- Current-season NFLVerse release files can still lag the league/stat-correction schedule, but the fetchers now use release assets before `nfl_data_py` for maintained datasets. If release files and wrapper calls both miss, the pipeline continues with cached historical data where possible and logs the unavailable feed. Optional PFR helper/feed misses are logged as optional enrichment misses, not primary pipeline failures. NFLVerse schedule provider ID columns are normalized to text before parquet writes, and PBP validation treats `wpa` as signed win-probability-added rather than a 0-to-1 probability.
- Yahoo Fantasy authorization failures are expected while access approval is pending. The Yahoo job logs those as optional-source skips rather than hard errors so weekly debug logs stay focused on actionable failures.
- Calibration and volatility artifacts are generated from walk-forward `predictions/history` rows so calibrators learn from games scored by models that did not train on those games. The volatility classifier defaults to RF with log-loss-only labels because it feeds probability shrinkage; margin-error labels are available with `--volatility-include-margin` for analysis runs. These jobs should exclude the target live week using `--live-run` and `--prediction-week`. Player-prop historical line joins, confidence-tier metrics, over-probability calibration, and edge-policy backtests also receive the same live cutoff, so current-week actuals cannot become prop-policy training data during Friday/Saturday reruns.
