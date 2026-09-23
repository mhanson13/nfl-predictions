# Data Sources

This file documents the data sources used by the NFL predictions pipeline, how each source is consumed, and whether the pipeline treats it as required or optional.

The main pipeline entry point is:

```powershell
python -m tools.run_pipeline --start-year 2017 --train-start-year 2016 --max-parallel-data 8 --data-start-delay 1 --use-async --prediction-week <week> --live-run --publish-mtb --mtb-repo-dir C:\Code\mtb
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
| NFLVerse via `nfl_data_py` | `src.data.nflverse` | `nfl_schedules.parquet`, `nfl_pbp.parquet`, `nfl_rosters.parquet`, `nfl_injuries.parquet`, `nfl_snap_counts.parquet`, `nfl_depth_charts.parquet`, `nfl_player_stats.parquet`, `nfl_pfr_*.parquet` | Core schedule, historical results, play-by-play team efficiency, injuries, roster/depth context, snap counts, player projections, player actuals, pressure/red-zone/passing EPA features | Required, with some optional subfeeds |
| ESPN schedule fallback | `src.data.nflsdv` and cached `espn_schedule.parquet` | `espn_schedule.parquet`, `espn_schedule_<season>.parquet` | Fallback schedule source if NFLVerse schedule is missing; used by prediction code as a schedule input | Required fallback only |
| NFL.com team stats | `src.data.nflcom` | `nflcom_teamstats_{passing,rushing,receiving,scoring,downs}.parquet` | Team-season offensive/defensive production inputs used in matchup features | Required for full feature richness |
| ESPN player stats | `src.data.espn_players` | `espn_playerstats_{category}_{season}_stype2.parquet` | Aggregated player passing, rushing, and receiving totals into team-season signals | Optional but recommended |
| ESPN player news | `src.data.espn_player_news` | `espn_player_news.parquet` | Player news counts, injury keyword counts, and pre-game news timing features | Optional |
| ESPN team news | `src.data.espn_team_news` | `espn_news.parquet` | Team news counts and injury keyword features | Optional |
| NOAA weather observations | `src.data.noaa` | `processed/noaa_weather.parquet`, station cache JSON | Stadium-adjacent observed weather merged into game weather features | Optional but recommended for weather features |
| Internal weather merger | `src.data.weather` | Weather-enriched raw/processed outputs from schedule and stadium metadata | Combines schedule, stadium, and weather fields used by feature builder and prediction exports | Required for normal weather columns, but has fallbacks |
| Visual Crossing weather | `src.data.visualcrossing` | `processed/visualcrossing_weather.parquet` | Hourly game-window weather, precipitation, wind, gust, and condition fields | Optional, requires `VISUAL_CROSSING_API_KEY` |
| Static climatology fallback | `src.data.weather_fallback` | No fetched raw file | Last-resort weather estimates when live weather sources miss a game | Built in fallback |
| BallDontLie | `src.data.balldontlie` | `balldontlie_teams.parquet`, `balldontlie_players.parquet`, `balldontlie_active_players.parquet`, `balldontlie_games_<season>[_wkNN].parquet`, `balldontlie_standings_<season>.parquet`, injury/stat/team-stat files | Commercial supplemental schedule, standings, player directory, active-player, injury, player-stat, and team-stat context | Optional, requires `BALLDONTLIE_API_KEY`; current default commercial provider for ALL-STAR-tier feeds |
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

## Player Projection Path

The player CSVs are separate from the team model:

| Output | Main Inputs | Behavior When Missing |
|---|---|---|
| `predictions_players_qb.csv` | NFLVerse weekly player stats plus current NFLVerse roster | Writes a current empty CSV instead of leaving stale rows. |
| `predictions_players_offense.csv` | NFLVerse weekly player stats, current roster, player penalties | Writes a current empty CSV instead of leaving stale rows. |
| `predictions_players_defense.csv` | NFLVerse play-by-play defense events plus current roster | Writes rows when PBP/roster data is available. |
| `player_props_qb.csv`, `player_props_offense.csv`, `player_props_defense.csv` | Trained player-prop models plus legacy projection CSVs; optional PropLine sportsbook line snapshots fill `line`, odds, edge, and line-consensus/variance fields; nflverse/BallDontLie injuries fill quality-control flags | Writes model projections without lines when prop odds are unavailable or unmatched. Injury-excluded rows remain visible but are not marked bettable. |
| `player_prop_historical_line_eval.parquet`, `player_prop_over_probability_eval.parquet`, `over_probability_metrics.csv`, `over_probability_bins.csv` | Normalized Odds API historical player-prop lines joined to walk-forward player-prop predictions and final actuals | Empty or stale if the paid historical backfill/join has not been refreshed. First-pass residual-CDF probabilities are diagnostic until they beat sportsbook implied probabilities in backtests. |
| `predictions/evaluation/player_props/prop_quality_week_NN.csv`, `prop_quality_summary_week_NN.csv`, `prop_line_coverage_week_NN.csv` | Weekly player-prop CSVs plus PropLine snapshots | Empty reports if prop CSVs are unavailable; otherwise summarizes missing lines, bettable rows, injury flags, line variance, raw PropLine rows, prepared sportsbook lines, matchable rows, and unmatched offered pairs. |

NFLVerse weekly player stats may lag the current season. In that case, prediction code logs the latest available stats season and relaxes the recency floor so the output is explicit rather than stale.

BallDontLie and SportsDataIO player/stat/projection feeds are optional supplemental data. Empty cached commercial-provider files are treated as stale and refetched on the next run.

## Optional Credentials

| Credential | Enables | Pipeline Behavior If Missing |
|---|---|---|
| `BALLDONTLIE_API_KEY` | BallDontLie supplemental team/player/stat feeds | BallDontLie job is skipped. If SportsDataIO is configured, it can still run as fallback. |
| `PROPLINE_API_KEY` | PropLine player prop odds snapshots | PropLine job is skipped. Player prop line columns remain empty unless another compatible odds cache exists. |
| `SPORTSDATAIO_API_KEY` | Legacy SportsDataIO feeds | SportsDataIO job is skipped when missing, or when BallDontLie is configured unless `--enable-sportsdataio` is passed. |
| `VISUAL_CROSSING_API_KEY` | Visual Crossing hourly weather | Visual Crossing job is skipped; NOAA/static weather can still fill. |
| `YAHOO_CLIENT_ID`, `YAHOO_CLIENT_SECRET`, `YAHOO_ACCESS_TOKEN`, `YAHOO_REFRESH_TOKEN`, `YAHOO_REDIRECT_URI` | Yahoo Fantasy feeds | Yahoo job is skipped unless credentials and API approval are valid. |
| `odds_api_free_key` or `ODDS_API_FREE_KEY` | Non-historical Odds API live/upcoming odds merge in upcoming predictions | Odds columns are omitted or left empty. This key is not used for historical endpoints. |
| `odds_api_paid_key` or `ODDS_API_PAID_KEY` | Historical Odds API event discovery, event-market discovery, and event/player-prop odds backfills | Historical prop calibration/backtesting remains unavailable without another historical line source. This key is not used for non-historical endpoints. |
| `NOAA_CONTACT_EMAIL` | NOAA User-Agent contact | Defaults to the repo fallback contact if unset. |

## Current Operational Notes

- Yahoo Fantasy is optional until API access is approved. OAuth tokens alone are not sufficient if Yahoo has not granted Fantasy Sports API access.
- BallDontLie is the preferred commercial provider for supplemental player/team context. PropLine is the player-prop line source. When the main pipeline is run with an explicit numeric `--prediction-week` and `PROPLINE_API_KEY` is configured, it requests NFL prop markets from DraftKings, Hard Rock, and FanDuel before weekly prop predictions are scored.
- PropLine event names are normalized through sportsbook team aliases such as `ATL Falcons`, `GB Packers`, and `LA Chargers`. Player-line matching uses a stricter first-name-prefix key first and only falls back to initial/last-name matching when the game/market/player key is unique.
- The Odds API historical backfill is intentionally not part of the default weekly pipeline. Use `src.data.odds_api_backfill` for the resumable one-time player-prop backfill: `--mode manifest` writes `data/processed/oddsapi_player_prop_backfill_manifest.parquet`, `--mode discover-events` fills event IDs, `--mode discover-markets` records per-event market availability, `--mode fetch-odds` pulls only available target player-prop markets, and `--mode normalize` writes `data/processed/oddsapi_historical_player_prop_lines.parquet`. Historical API modes require `--confirm-paid-backfill` unless `--dry-run` is used.
- Odds API key separation is enforced in code. Historical commands use only the paid key name/alias; non-historical live/upcoming odds use only the free key name/alias. Do not repoint either path to the other key without explicit approval.
- SportsDataIO is retained as a fallback while BallDontLie caches are verified. Avoid running it accidentally if cost is a concern.
- Current-season NFLVerse feeds can 404 or lag. The fetchers should continue with cached historical data and log which current-season feeds were unavailable.
- Calibration and volatility artifacts are generated from walk-forward `predictions/history` rows so calibrators learn from games scored by models that did not train on those games. The volatility classifier defaults to RF with log-loss-only labels because it feeds probability shrinkage; margin-error labels are available with `--volatility-include-margin` for analysis runs. These jobs should exclude the target live week using `--live-run` and `--prediction-week`.
