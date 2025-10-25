
# NFL Predictions (SportsDataverse + NFL.com Team Stats)

A reproducible Python project for building **NFL game predictions** using
- [`sportsdataverse-py`](https://sportsdataverse-py.sportsdataverse.org/) for schedules, pbp and live/ESPN endpoints, and
- team-level aggregates scraped from **NFL.com** (passing, rushing, receiving, scoring, downs) for long-run baselines.

It includes data pipelines, feature engineering, model training (win probability, spread, and totals), and a minimal Streamlit app.

## Quickstart (Windows PowerShell)

```pwsh
# 1) Create and activate a virtual environment
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# (Append --debug to any command below if you need verbose diagnostics.)

# 2) Fetch data (examples)
#    - Schedules and play-by-play from SportsDataverse (per-season cache; use --force to refresh cached seasons)
python -m src.data.nflsdv --season 2024 2025 --pbp --schedules
#    - nflverse PBP/schedules and weekly datasets (rosters/injuries/snaps/depth-charts/player-stats). Uses API, with direct-release fallbacks.
python -m src.data.nflverse --season 2024 2025 --pbp --schedules --rosters --injuries --snaps --depthcharts --players
#    - NFL.com team stats (per-year/per-category cache; use --force to refresh)
python -m src.data.nflcom --year 2025 --category passing rushing receiving scoring downs
#    - ESPN player stats (passing / rushing / receiving)
python -m src.data.espn_players --season 2025 --season_type 2 --category passing rushing receiving
python -m src.data.espn_players --season 2004 2005 --season_type 3 --category passing  # postseason examples
#    - ESPN player news (per-player updates/injuries via fantasy API; limit is per player id)
python -m src.data.espn_player_news --season 2025 --limit 8
#    - Odds API spreads (set ODDS_API_KEY in secrets.env or environment to enrich predictions with bookmaker lines)
#      The fetch happens automatically during prediction when the key is present.

# 3) Build features (merges schedule with NFL.com team stats and engineers matchup features)
python -m src.features.build_features --season 2024 2025

# 4) Train models
#    a) Win probability (supports optional probability calibration)
python -m src.models.train --target win_prob            # uncalibrated
python -m src.models.train --target win_prob --calibrate # calibrated (isotonic)
#    Optional GPU acceleration (requires XGBoost with GPU support; falls back to CPU if unavailable)
python -m src.models.train --target win_prob --use-gpu
#    Apply tuned parameters (if tuning/winprob_best.json exists)
python -m src.models.train --target win_prob --calibrate --param-config tuning/winprob_best.json
#    b) Point spread (home margin) regression
#       (Huber loss with a median imputer + quantile sidebands for intervals)
python -m src.models.train --target spread
python -m src.models.train --target spread --use-gpu
#    Apply tuned spread parameters
python -m src.models.train --target spread --param-config tuning/spread_best.json
#    c) Hyperparameter tuning with Optuna (optional)
python -m src.models.tune --target win_prob --trials 50 --metric logloss --save-best tuning/winprob_best.json
python -m src.models.tune --target spread --trials 50 --metric mae --save-best tuning/spread_best.json

# 5) Predict upcoming games (curated by default). Add --dump-all to save a full columns dump as well.
python -m src.predict.predict_upcoming --season 2025 --week auto --save predictions_2025_wN.csv --dump-all predictions_full_2025_wN.csv

# 6) Run the Streamlit UI (optional)
streamlit run streamlit_app.py
```

By default, data/model scripts now run quietly to speed up the pipeline. Append `--debug` to any of them (or run `run_pipeline.bat --debug`) to surface detailed progress logs when you need to inspect the workflow. The batch runner also supports `--start-year YYYY` (defaults to 2023) to automatically include every season through the current year, `--train-start-year YYYY` to focus model fitting on recent history, `--calibration-season-window N` for the win-probability calibrator, and `--use-gpu` to switch the training jobs to GPU-accelerated XGBoost (when available).

## Pipeline automation

- `run_pipeline.bat` is the default entry point on Windows and forwards all flags to the Python orchestrator (`python -m tools.run_pipeline ...`).
- Useful options:
  - `--start-year YYYY` controls which seasons are refreshed before feature building (default 2023).
  - `--dry-run` prints the full command plan without executing anything (great for sanity checks).
  - `--max-parallel-data N` and `--data-start-delay SECONDS` control concurrency and staggering for data fetch jobs.
  - `--sportradar-mode full|nightly` selects between a full Sportradar refresh (static + change feeds, default) and a lightweight nightly run that only pulls change/transaction logs.
  - `--sportradar-args -- ...` passes extra parameters directly to `src.data.sportradar` (for example, `--sportradar-args -- --seasons 2025 --weeks 1 2`).
  - `--skip-sportradar` disables Sportradar entirely when the API key is missing or you want to avoid those calls.
  - `--train-start-year YYYY` trims the training sets to recent seasons while still ingesting earlier data for context.
  - `--calibration-season-window N` constrains the isotonic win-probability calibrator to the last `N` seasons.
  - `--use-gpu` enables GPU-accelerated training when a compatible XGBoost build (e.g., CUDA) is installed (pipeline will fall back to CPU if GPU tree methods are unavailable).
  - `--enable-tuning` runs Optuna sweeps before training; paired files in `tuning/` are then fed back via `--param-config`.
  - `--tuning-trials-winprob/--tuning-trials-spread`, `--tuning-storage`, and related flags customize trial counts, storage, and resume behavior for the built-in tuner.
- When a `SPORTSRADAR_NFL_API_KEY` is present, the pipeline automatically:
  1. Fetches the selected Sportradar feeds (static + change logs).
  2. Normalizes the raw JSON via `python -m src.data.sportradar_transform`.
  3. Generates daily summaries with `python -m tools.sportradar_report`, writing parquet summaries to `predictions/evaluation/`.
  4. Generates backtest history predictions (`predictions/history/wXX_predictions_history_<season>.csv`) so the evaluator can report historical metrics.
- `evaluate_predictions` writes per-week charts/tables and an overall summary to `predictions/evaluation/overall_metrics.csv` (ACC, AUC, Brier, LogLoss, MAE, RMSE).

Example orchestrations:

```pwsh
# Full refresh (all data sources, Sportradar static + change feeds)
python -m tools.run_pipeline --start-year 2002

# Nightly delta run focused on change/transaction feeds
python -m tools.run_pipeline --sportradar-mode nightly --start-year 2025

# Dry-run preview with explicit Sportradar arguments
python -m tools.run_pipeline --dry-run --sportradar-args -- --seasons 2025 --weeks 1 2
```

## Sportradar ingestion & transforms

1. Ensure `SPORTSRADAR_NFL_API_KEY` is set in `secrets.env` or your environment.
2. Fetch desired feeds (examples):
   ```pwsh
   # One-time static setup
   python -m src.data.sportradar --feeds league_hierarchy teams seasons

   # Pull season schedules and derive weekly game IDs
   python -m src.data.sportradar --feeds season_schedule --seasons 2025

   # Grab game-level stats using the saved schedule for IDs
   python -m src.data.sportradar --feeds game_boxscore game_statistics --schedule-json data/raw/sportradar/season_schedule/2025/<file>.json --weeks 1
   ```
3. Normalize to parquet tables:
   ```pwsh
   python -m src.data.sportradar_transform --targets schedule rosters game_stats transactions change_log
   ```
   The transformer writes curated tables to `data/processed/sportradar/` and is invoked automatically by the pipeline.
4. Inspect daily change/transaction summaries:
   ```pwsh
   python -m tools.sportradar_report --dry-run  # or omit --dry-run to write to predictions/evaluation/
   ```
   5. Generate backtest predictions (optional)
   ```pwsh
   python -m src.predict.predict_history --seasons 2023 2024 --weeks 1 2 3
   ```

## Data sources
- SportsDataverse Python: schedules, play-by-play, box scores, EPA/WP support (cached per season in `data/raw/espn_schedule_{season}.parquet` and `data/raw/espn_pbp_{season}.parquet`).  
- nflverse: rich PBP and weekly datasets (snap counts, depth charts, rosters, injuries, player stats) with API-first and release URL fallbacks.  
- NFL.com team stats: season-level team aggregates for **passing**, **rushing**, **receiving**, **scoring**, **downs** (per-year cache in `data/raw/nflcom_teamstats_{category}_{year}.parquet`).  
- Sportradar NFL (trial feeds): supplemental league/team/game data, including league hierarchy, season/weekly schedules, rosters, game box scores/statistics, and daily change/transaction logs. Raw JSON lands under `data/raw/sportradar/...`; normalized parquet tables are written to `data/processed/sportradar/` by `python -m src.data.sportradar_transform`.
## Repo layout

```
src/
  data/         # data acquisition (sportsdataverse + nfl.com team stats)
  features/     # feature engineering for team + matchup
  models/       # training, evaluation
  predict/      # weekly predictions for upcoming games
  utils/        # helpers (io, logging, enums)
data/
  raw/          # raw pulled data
  processed/    # feature tables, model-ready
models/         # persisted sklearn/xgboost models
streamlit_app.py
requirements.txt
```

## Output files
- Processed features: `data/processed/matchup_features.parquet`
- Trained models:
  - Win prob: `models/winprob_gb.pkl` (contains model, feature list, and a calibrated flag)
  - Spread: `models/spread_gb.pkl` (contains the Huber regressor, feature list, a median imputer, and optional quantile models for interval bands)
- Raw player news snapshots: `data/raw/espn_player_news.parquet` (per-player ESPN fantasy news used for roster availability signals)
- Predictions CSVs:
  - Curated: `--save` (default `predictions.csv`) - concise, human-friendly columns for modeling and review
  - Full dump: optional `--dump-all path.csv` - all available columns
- Player projections:
  - Quarterbacks: `--save-players-qb` (default `predictions_players_qb.csv`) - all QBs per team with projected passing/rushing production.
  - Offensive skill players: `--save-players-offense` (default `predictions_players_offense.csv`) - top 10 RB/WR/TE per team with projected rushing and receiving production.
  - Defensive playmakers: `--save-players-defense` (default `predictions_players_defense.csv`) - top defenders per team with projected sacks, QB hits, tackles for loss, and blocked punts.

Curated columns (high-level):
- `home_win_prob`: Model-estimated probability the home team wins (0-1). 0.5 = coin flip; closer to 1 favors home; closer to 0 favors away.
- `pred_home_margin`: Predicted home margin in points (home score - away score). Positive = home favored; negative = away favored.
- `pred_home_margin_lo`, `pred_home_margin_hi`: Optional 20th/80th percentile bounds surrounding the predicted margin (populated when quantile side models are available).
- `home_win_prob_expl`: Human-readable explanation of `home_win_prob`.
- `pred_home_margin_expl`: Human-readable explanation of `pred_home_margin`.
 - `date`, `game_id`: Game metadata when available.
 - `home_pick`: HOME or AWAY based on `home_win_prob >= 0.5`.
 - `home_confidence_pct`: `home_win_prob` as a percent (0-100%).
 - `home_odds_american`, `home_odds_decimal`: Implied odds from `home_win_prob`.
  - `pick_expl`: Short text summary combining pick, probability, and margin.
- Team availability / usage (via nflverse weekly data):
  - Injury counts: `inj_out`, `inj_doubtful`, `inj_questionable` plus `_prev`, `_delta`, `_pct_change`, `_rolling3` suffixes.
  - Snap totals: `snaps_*` (offense/defense/special teams) with the same lag/delta/rolling enrichments.
  - Depth chart / roster / player stat aggregates (`depth_*`, `roster_*`, `pstats_*`) when numeric columns are available, also carrying those lag/delta/rolling metrics.
 - Per-game baselines (season totals / 17) when inputs exist:
   - `pred_home_passing_att`, `pred_away_passing_att`
   - `pred_home_rushing_att`, `pred_away_rushing_att`
   - `pred_home_scoring_tottd`, `pred_away_scoring_tottd` (TDs)
   - `pred_home_passing_int`, `pred_away_passing_int` (INTs thrown)
   - `pred_home_first_downs`, `pred_away_first_downs` (rush+rec 1st downs)
   - `pred_home_downs_4thatt`, `pred_away_downs_4thatt` (4th down att)
   - `pred_home_downs_4thmd`, `pred_away_downs_4thmd` (4th down made)
 - `pred_home_punts`, `pred_away_punts` (placeholder: not in current data)
  - `pred_home_field_goals`, `pred_away_field_goals` (placeholder: not in current data)
  - `odds_<book>_home_spread`, `odds_<book>_home_price`, `odds_<book>_away_spread`, `odds_<book>_away_price` (+ timestamp) when `ODDS_API_KEY` is configured, for every bookmaker returned by The Odds API.

- Curated offense (friendly names; _home/_away/_diff): `pass_att`, `pass_yds`, `pass_td`, `pass_int`, `rush_att`, `rush_yds`, `rush_td`, `recv_rec`, `recv_tgt`, `recv_yds`, `recv_td`.
- Defense allowed (native from PBP or derived from opponent offense): `def_ypp_allowed`, `def_pass_rate_allowed`, `def_pass_ypp_allowed`, `def_rush_ypc_allowed`, `def_epa_per_play_allowed`, plus allowed totals (`def_plays_allowed`, `def_pass_plays_allowed`, `def_rush_plays_allowed`, `def_yards_allowed`, `def_pass_yards_allowed`, `def_rush_yards_allowed`, `def_epa_allowed`).
- Weekly availability (from nflverse aggregates): `starters_*`, `inj_out/doubtful/questionable_*`, `questionable_rate_*`, snap totals for offense/defense/special-teams where available.

The full dump (if `--dump-all` is provided) will include all additional engineered and source-derived columns for advanced analysis.

## Troubleshooting
- If training fails with "Input X contains NaN", the win-probability model uses `HistGradientBoostingClassifier` (native NaN support) and the spread model runs a Huber `GradientBoostingRegressor` paired with a median `SimpleImputer`. Rebuild features and retrain to regenerate the persisted imputer.
- If upcoming predictions are empty, ensure:
  - You trained models first (step 4).
  - `matchup_features.parquet` includes the target season/week.
  - The prediction script selected the intended season/week (use `--season` and optionally `--week`).
- Timezone errors when selecting weeks: the script uses UTC-aware timestamps; keep raw schedule dates intact.
- Cached fetches:
  - `src.data.nflsdv` uses per-season caches for schedules/PBP; pass `--force` to refresh immutable seasons.
  - `src.data.nflcom` uses per-year caches for each category; pass `--force` to refresh.
  - `src.data.nflverse` will fetch via API and then fall back to release URLs; rerun if some resources are temporarily unavailable.

## Notes
- NFL.com HTML structure can change. The scraper uses `pandas.read_html` with graceful fallbacks.
- The Streamlit app is a minimal starter; extend it with charts, filters, and matchup cards.
