# NFL Predictions Platform

End-to-end tooling for collecting NFL data, engineering matchup features, training prediction models, calibrating win probabilities, and exploring results through a local conversational assistant. Pipelines cover historical seasons (2002–present) with weather-aware volatility adjustments and market-aligned evaluation.

---

## Highlights

- **Deterministic pipelines** – single entry point (`tools/run_pipeline.py`) for ingestion, feature engineering, training, calibration, and prediction.
- **Comprehensive data coverage** – SportsDataIO, NFL.com, nflverse, ESPN, Visual Crossing, NOAA, and The Odds API with resilient caching.
- **Feature-rich models** – win probability, spread regression, and volatility classification using weather deltas, QB practice deltas, and travel × rest interactions.
- **Probability calibration** – isotonic scaling layered on volatility-based shrinkage for sharper probability estimates.
- **Interactive analytics** – GPT4All + Milvus conversational agent grounded in local predictions, evaluation metrics, and documentation.
- **Caching first** – immutable seasons persist locally; refresh only when underlying data changes.

---

## Repository Layout

```
analysis/              Evaluation tooling, volatility diagnostics, ROI studies
convo_agent/          Conversational assistant (indexer, retriever, FastAPI, CLI)
data/                 Raw and processed datasets (cached per season/year)
models/               Persisted model artifacts (sklearn/XGBoost, calibrators)
predictions/          Historical & upcoming prediction outputs
src/                  Data ingestion, feature engineering, modeling packages
tools/                Pipeline runner and orchestration helpers
```

---

## Getting Started

### 1. Prerequisites

- Python 3.10+
- PowerShell or compatible shell
- NVIDIA GPU with CUDA drivers (optional, recommended)
- Docker (optional, required for Milvus container deployment)

### 2. Environment Setup

```pwsh
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

# Optional: install CUDA-enabled PyTorch
pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 3. Configure Secrets

Populate `secrets.env` (or set environment variables) for external feeds:

| Variable | Purpose |
|----------|---------|
| `SPORTSDATAIO_API_KEY` | SportsDataIO projections, DFS, futures/draft, injuries |
| `NOAA_CONTACT_EMAIL` | NOAA hourly observations (required) |
| `VISUAL_CROSSING_API_KEY` | Historical weather backfill |
| `ODDS_API_KEY` | Market odds enrichment (The Odds API) |
| `THEODDS_API_TIMEOUT` | Optional request tuning |

Load using `python-dotenv` or export manually before running pipelines.

---

## Core Pipelines

### Unified Runner

`tools/run_pipeline.py` orchestrates the full workflow.

```pwsh
python -m tools.run_pipeline `
  --debug `
  --start-year 2002 `
  --train-start-year 2016 `
  --max-parallel-data 16 `
  --data-start-delay 0.5 `
  --use-gpu
```

Key flags:
- `--debug` propagates verbose logging (all subcommands now accept `--debug`).
- `--skip-*` toggles optional jobs (see `--help` for full list).
- `--max-parallel-data` protects API quotas; adjust based on provider limits.

### Standalone Commands

#### Data Ingestion

| Command | Purpose | Notes |
|---------|---------|-------|
| `python -m src.data.nflsdv` | Schedules & play-by-play via SportsDataverse | Per-season cache; use `--force` to refresh |
| `python -m src.data.nflverse` | nflverse weekly datasets (pbp, injuries, snaps, depth charts, rosters) | API with release fallback, throttled |
| `python -m src.data.nflcom` | NFL.com team stats (passing/rushing/etc.) | Cached per category/year |
| `python -m src.data.espn_players` | ESPN player stats | Immutable seasons skipped unless `--force-refresh` |
| `python -m src.data.espn_player_news` | ESPN player news (injuries/updates) | Refresh limited to every 4h unless forced |
| `python -m src.data.noaa` | NOAA hourly observations | Historical weather near kickoff windows |
| `python -m src.data.visualcrossing` | Visual Crossing backfill | Only for gaps left by NOAA |
| `python -m src.data.sportsdataio` | SportsDataIO projections, DFS, futures/draft, injuries | Replaces Sportradar static feeds |

#### Feature Engineering & Modeling

```pwsh
# Weather, travel, rest, injury, and market features
python -m src.features.build_features --season 2023 2024 2025

# Train win probability (supports GPU)
python -m src.models.train --target win_prob --use-gpu

# Train home-margin regression
python -m src.models.train --target spread

# Hyperparameter tuning (optional)
python -m src.models.tune --target win_prob --metric logloss --trials 50
```

`matchup_features.parquet` now includes weather deltas (`wx_temp_delta`, `wx_wind_delta`, `wx_rain_delta`), QB practice delta features, and travel × rest interactions that power both the classifier and the primary model set.

#### Volatility & Calibration

```pwsh
python -m analysis.volatility_classifier `
  --model logreg `
  --percentile 0.6 `
  --decision-threshold 0.8 `
  --disable-season-split `
  --calibrate `
  --debug

python -m src.evaluation.calibrate_winprob `
  --history-dir predictions/history `
  --season-window 3 `
  --volatility-dataset analysis/volatility_classifier_dataset.csv `
  --volatility-threshold 0.60 `
  --volatility-strength 0.35 `
  --apply-isotonic `
  --save-calibrator models/isotonic_calibrator.pkl
```

Artifacts:
- `analysis/volatility_classifier_dataset.csv` – per-game probabilities & labels.
- `analysis/volatility_classifier_metrics.json` – threshold sweep metadata.
- `analysis/reliability_curve.png` – win probability calibration plot.
- `models/isotonic_calibrator.pkl` – shrink + isotonic calibrator.
- `predictions/evaluation/overall_metrics.csv` – baseline vs calibrated metrics.

#### Predictions

```pwsh
# Historical reconstruction
python -m src.predict.predict_history --seasons 2002 2025 --overwrite --debug

# Upcoming week (saves team & player outputs)
python -m src.predict.predict_upcoming --season 2025 --week 9 --save-players-offense --save-players-defense --debug
```

Outputs include:
- `predictions/history/*.csv` – backtests with volatility columns.
- `predictions/upcoming/*.csv` – latest picks, player leaders, odds snapshot.

---

## Conversational Agent

Local Retrieval-Augmented Generation over predictions, metrics, and documentation.

1. **Run Milvus** (docker-compose or existing deployment) on `127.0.0.1:19530`.
2. **Build the index**:
   ```pwsh
   python -m convo_agent.data_indexer --reset
   ```
   Embeds predictions, evaluation snapshots, volatility metrics, README sections, and calibrator metadata.
3. **Query options**:
   ```pwsh
   # CLI chat
   python -m convo_agent.cli --show-context

   # REST API
   uvicorn convo_agent.api:app --reload
   ```

The assistant performs structured Pandas filtering before vector search, keeping answers grounded in local data. GPU acceleration is automatically selected when CUDA is available; otherwise the stack falls back to CPU.

---

## Evaluation Snapshot (2023–2025 Focus Window)

| Metric | Value | Notes |
|--------|-------|-------|
| Accuracy | 0.579 | Full-history win probability |
| AUC | 0.761 | Calibrated + volatility shrink |
| Brier Score | 0.195 | Post-calibration |
| LogLoss | 0.654 | Historical sample |
| MAE (margin) | 10.63 | Spread regression |
| RMSE (margin) | 13.68 | Spread regression |
| Volatility coverage | 15% | Threshold 0.60, strength 0.35 |

Forward holdout (2024–2025):
- Accuracy 0.714, AUC 0.774, Brier 0.215, MAE 10.04.

Volatility classifier (2024–2025 holdout):
- AUC 0.98, Precision 1.00, Recall 0.85 at threshold 0.8.

---

## Caching Strategy & Data Freshness

- Historical seasons are cached under `data/` and `predictions/history/`. Immutable years are not re-fetched unless `--force` is supplied.
- ESPN player news enforces a 4-hour minimum interval (`--min-interval-minutes` adjustable); use `--force-refresh` for immediate updates.
- `src.data.reference.update_regular_season` keeps `nfl_regular_season_games.csv` current. Leverages curated Wikipedia season references and lands in cache unless explicitly refreshed.
- Visual Crossing serves as a fallback when NOAA lacks coverage; NOAA fetcher only revisits recent windows.
- SportsDataIO replaces Sportradar static feeds, avoiding rate-limit downtime and additional licensing burden.

---

## Troubleshooting

| Issue | Resolution |
|-------|------------|
| `Torch not compiled with CUDA` | Install CUDA-enabled PyTorch via `pip install --force-reinstall ... cu124`. |
| Milvus collection missing | Start Milvus, then run `python -m convo_agent.data_indexer --reset`. |
| NOAA fetch revisits early seasons | Caching now locks finalized years; ensure you are not passing `--force`. |
| Pipeline command rejects `--debug` | All modules have `--debug`; pull latest branch if missing. |
| Sklearn version warning for calibrator | Re-run calibration to regenerate `models/isotonic_calibrator.pkl` under current sklearn. |
| ESPN player news too frequent | Default throttle is 4 hours; override via `--min-interval-minutes`. |

---

## Roadmap

- Season-by-season metric table in this README.
- Change log documenting major data/model updates.
- Automated metric snapshot after each feature addition.
- Feature enrichment focused on weather/QB/travel precision (target precision ≥ 0.70 without recall loss).
- Replace legacy Streamlit prototype with the conversational experience.

---

## License & Usage

This repository is for research and personal use. Respect data provider terms (SportsDataIO, nflverse, ESPN, NOAA, Visual Crossing, The Odds API). Obtain appropriate licenses before redistributing or commercialising outputs.

---

## Acknowledgements

- [SportsDataverse](https://sportsdataverse.org/) for schedules, play-by-play, and ESPN endpoints.
- [nflverse](https://nflverse.github.io/) for extensive open NFL datasets.
- [Visual Crossing](https://www.visualcrossing.com/) and [NOAA](https://www.weather.gov/documentation/services-web-api) for weather archives.
- [GPT4All](https://gpt4all.io/) and [Milvus](https://milvus.io/) for enabling local conversational analytics.
