# AI coding agent guide for this repo

Purpose and architecture
- Builds NFL game predictions by combining multiple data sources, feature engineering, and two models (win probability classifier, point spread regressor).
- Data ingestion: `src/data/`
  - `nflverse.py` (nfl_data_py): schedules, PBP, rosters/injuries/snaps/depth-charts/player stats, with release-download fallbacks.
  - `nflsdv.py` (sportsdataverse): ESPN schedules + PBP, stored per-season caches.
  - `nflcom.py`: NFL.com team season aggregates (passing/rushing/receiving/scoring/downs) via `pandas.read_html` with per-year caches.
  - `espn_players.py`: ESPN player stats pages scraped and normalized; later aggregated to team-season.
  - `weather.py`: Tomorrow.io hourly forecast near kickoff → game-level `weather_*` features; reads API key from TOMORROW_API_KEY or `secrets.env`.
- Feature engineering: `src/features/build_features.py`
  - Normalizes schedule columns from varied schemas and canonicalizes team codes (e.g., JAC→JAX, WSH→WAS, resolves LA→LAR/LAC using names).
  - Merges team-season features from PBP offense/defense, NFL.com aggregates, and ESPN team totals.
  - Creates matchup rows with `_home/_away` suffixes; downstream derives `*_diff = home − away`.
  - Derives targets `home_win` and `home_margin` from scores (fills from PBP if missing in schedule).
  - Integrates weekly availability (injuries, snaps, depth charts) and merges game-level weather by `game_id`/UID/keys.
  - Builds defensive "allowed" metrics from PBP, with robust fallbacks to opponent offense totals when PBP is missing.
- Modeling: `src/models/train.py` uses sklearn HGB models; auto-discovers numeric `*_diff` features (excludes leakage like score/margin/win).
- Prediction: `src/predict/predict_upcoming.py` recreates diffs, clips to train ranges, and exports curated predictions plus optional full dump.
- Utilities: `src/utils/io.py` centralizes Parquet/CSV IO and sanitization; creates `data/raw`, `data/processed`, `models`.

Key conventions and patterns
- Columns follow: per-team metrics with `_home`/`_away` and derived `*_diff`. Weather columns begin with `weather_`; stadium/roof context via `roof` + `roof_is_dome`.
- Defensive metrics are exported as `def_*_allowed_{home,away,diff}`; prediction renames any discovered `*allowed*` to `def_*` for consistency.
- Stable game identifiers: `game_id` when available, plus `game_uid` and a relaxed `game_uid_relaxed` (collapses LAR/LAC→LA) to improve joins.
- Abbreviation normalization is critical across sources; see `ALT_ABBR_MAP`, `TEAM_NAME_TO_ABBR` in `build_features.py` and helpers in data modules.
- IO writes go through `write_df` which JSON-encodes nested objects; do not `to_parquet` directly.

Developer workflows (Windows PowerShell)
- Setup and deps:
  - python -m venv .venv; . .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
- Typical pipeline (idempotent caches in data/raw):
  - python -m src.data.nflverse --season 2024 2025 --pbp --schedules --rosters --injuries --snaps --depthcharts --players
  - python -m src.data.nflcom --year 2024 2025 --category passing rushing receiving scoring downs
  - python -m src.data.espn_players --season 2025 --season_type 2 --category passing rushing receiving
  - Set Tomorrow.io key in env or `secrets.env` (TOMORROW_API_KEY=...). Then: python -m src.data.weather --season 2025
- python -m src.features.build_features --season 2024 2025
- python -m src.models.train --target win_prob --calibrate
- python -m src.models.train --target spread
- python -m src.predict.predict_upcoming --season 2025 --week auto --save predictions.csv --dump-all predictions_full.csv --save-players-qb predictions_players_qb.csv --save-players-offense predictions_players_offense.csv --save-players-defense predictions_players_defense.csv
- All scripts run quietly by default; append `--debug` (or run `run_pipeline.bat --debug`) when you need verbose diagnostics and intermediate logging.
- Set `ODDS_API_KEY` in `secrets.env` (or environment) so prediction runs can fetch bookmaker spreads from The Odds API and append them to the outputs.
- Player projections leverage season-to-date `nfl_player_stats.parquet` averages; refresh this dataset so `--save-players` produces top-5 per-team projections.
- - `run_pipeline.bat` also accepts `--start_year YYYY` (defaults to 2023) to automatically include every season through the current year.`r`n- Streamlit UI: streamlit run streamlit_app.py (loads predictions.csv by default).

Non-obvious details and gotchas
- Schedules/PBP may come from either nflverse or ESPN; `build_features` handles schema drift and fills scores from PBP when schedule lacks them.
- Weather joins try `game_id`, then `game_uid`, then relaxed UID, then (season, week, home, away); kickoff times treated as UTC.
- LA ambiguity: the relaxed UID collapses LAR/LAC to LA to improve merges with external sources.
- Training selects only numeric `*_diff` and numeric weather fields with sufficient support/variance; targets: `home_win` (clf) and `home_margin` (reg).
- Prediction optionally explains decisions via SHAP (if installed) or simple heuristics; curated export standardizes offense and defense panels.
- Avoid adding features that leak outcomes (scores/margins/wins); the trainer auto-filters by substrings but keep naming clean.

Where to look for examples
- Data fetch: `src/data/nflverse.py:main`, `src/data/nflsdv.py:main`, `src/data/nflcom.py:main`, `src/data/espn_players.py:main`.
- Feature build: `src/features/build_features.py` (schedule normalization, team-season merges, weather integration, def-allowed fallbacks).
- Modeling: `src/models/train.py` (diff creation, feature selection, calibration), artifacts in `models/`.
- Prediction and exports: `src/predict/predict_upcoming.py` (week auto-select, clipping, curated columns, odds conversions).
- Inspectors: `inspect_schedule.py`, `inspect_features.py`, `inspect_weather.py` to debug inputs and merges.
