# Installation Guide

## Overview

This document explains how to get the NFL Predictions Platform running locally so you can:

- build matchup features from raw league, weather, and betting data (`src/features/build_features.py`)
- train the win-probability, spread, and volatility models (`src/models/train.py`, `analysis/volatility_classifier.py`)
- generate historical and upcoming predictions (`src/predict/predict_history.py`, `src/predict/predict_upcoming.py`)
- evaluate outputs and produce post-run analytics (`src/evaluation/*.py`, `analysis/*`)

Successful installs write artifacts under:

- `data/raw/` – cached ingestion parquet files (nflverse, ESPN, NOAA, BallDontLie, etc.)
- `data/processed/` – derived matchup matrices (e.g., `matchup_features.parquet`)
- `models/` – serialized model weights and calibrators
- `predictions/` and `results/` – weekly CSVs, evaluation reports, and dashboard assets

## Prerequisites

- **Operating systems**
  - Tested on Windows 11 (PowerShell) and Ubuntu 22.04 (GitHub Actions).
  - Expected to work on other modern Linux distributions and macOS for CPU workloads; ensure compatible compilers are installed.
- **Python**: 3.11.x (matches `.github/workflows/ci.yml`). Use the same interpreter for tooling and the pipelines.
- **Git**: required to clone and pull updates.
- **Build tools**: `build-essential`/`gcc`/`gfortran` on Linux and the “Desktop development with C++” workload on Windows (needed for packages such as `shap`).
- **Pip utilities**: `pip`, `venv`, or `conda`.
- **Disk space**: ~5 GB for the repo + dependencies and 20–40 GB for historic data, models, and predictions (full 2002–present seasons).

## Quickstart (CPU-only)

```bash
# 1. Clone the repository
git clone https://github.com/<your-org>/nfl-predictions.git
cd nfl-predictions

# 2. Create and activate a virtual environment
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 4. Run the lightweight smoke test
python scripts/smoke_test.py
```

The smoke test verifies that Python can import core dependencies and project entry points. If it fails, resolve the reported imports or missing directories before proceeding.

## Quickstart (Conda)

```bash
conda create -n nfl-preds python=3.11 -y
conda activate nfl-preds
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/smoke_test.py
```

Conda handles native dependencies well on Windows/Linux. Prefer `venv` if you want lighter-weight environments or are integrating with system Python. Avoid mixing `pip install` from outside the activated environment.

## Optional: GPU Setup

GPU acceleration is optional but speeds up `src/models/train.py` and SHAP explainability:

1. Install the latest NVIDIA Game Ready or Studio driver with CUDA 12.x support.
2. Ensure the CUDA runtime matches your driver (CUDA toolkit is not strictly required, but `nvcc --version` is useful for debugging).
3. Install dependencies inside your environment (already covered in `requirements.txt`). The bundled `xgboost>=2.0` automatically enables GPU if CUDA is detected.
4. Verify PyTorch/XGBoost can see the GPU:
   ```bash
   python -c "import xgboost; import sys; print('xgboost', xgboost.__version__);"
   ```
5. Pass `--use-gpu` to the relevant commands (for example, `python -m src.models.train --target win_prob --use-gpu`). CPU-only mode is the default fallback.

If CUDA is missing, leave `--use-gpu` off; the same commands run with CPU-backed estimators.

## Configuration

Secrets and API tokens are loaded from environment variables or `secrets.env` at the repo root (see `src/utils/secrets.py`). Create the file manually and **never commit it**.

```ini
# secrets.env (example)
BALLDONTLIE_API_KEY=...
SPORTSDATAIO_API_KEY=...
VISUAL_CROSSING_API_KEY=...
TOMORROW_API_KEY=...           # or TOMORROWIO_API_KEY
ODDS_API_KEY=...
SPORTSRADAR_NFL_API_KEY=...    # optional; only needed for Sportradar fetches
NOAA_CONTACT_EMAIL=you@example.com
YAHOO_APP_ID=...
YAHOO_CLIENT_ID=...
YAHOO_CLIENT_SECRET=...
YAHOO_ACCESS_TOKEN=...
YAHOO_ACCESS_TOKEN_SECRET=...
```

- Use `setx`/`$Env:VAR=value`/`export VAR=value` if you prefer shell variables.
- Weather fetchers fall back to Visual Crossing/Tomorrow.io if NOAA is rate-limited.
- Data is staged under `data/raw/` (ingestion) and `data/processed/` (engineered features). Models output to `models/`, and predictions/evaluation artifacts live in `predictions/` and `results/`.

## Data Setup

Ingestion scripts write parquet files under `data/raw`. Each script accepts `--help` for full options. Typical commands:

```bash
# ESPN schedules + play-by-play via SportsDataverse cache
python -m src.data.nflsdv --season 2024 2025 --schedules --pbp

# nflverse bundles (rosters, injuries, snaps, participation, players, PFR tables)
python -m src.data.nflverse --season 2024 2025 \
  --schedules --rosters --injuries --snaps --participation --players \
  --pfr passing rushing receiving

# NFL.com team stats
python -m src.data.nflcom --year 2024 2025

# BallDontLie commercial supplement (requires BALLDONTLIE_API_KEY; ALL-STAR tier)
python -m src.data.balldontlie --feeds teams players active_players games standings injuries stats season_stats team_stats team_season_stats \
  --seasons 2024 2025 --weeks 1 2 3

# Legacy SportsDataIO fallback (requires SPORTSDATAIO_API_KEY; not run by default when BallDontLie is configured)
python -m src.data.sportsdataio --feeds teams schedules player_season_projections dfs_slates \
  --seasons 2024 2025 --weeks 1 2 3

# Weather (NOAA first, then Visual Crossing for gaps)
python -m src.data.noaa --seasons 2024 2025 --sleep 0.25
python -m src.data.visualcrossing --season 2024 2025 --only-missing

# Tomorrow.io fallback for live weather snapshots
python -m src.data.weather --season 2025 --api-key $Env:TOMORROW_API_KEY
```

Notes:

- Each fetcher caches per-season parquet files. Rerun with `--force`/`--force-refresh`/`--overwrite` when the upstream provider updates data.
- Safe cleanup: delete stale files under `data/raw/` to trigger a fresh download; do not delete `data/reference/` unless you plan to rebuild curated tables.
- Outputs from ingestion feed the feature builder (`data/processed/matchup_features.parquet`). If this file looks stale, delete it before re-running feature engineering.

## Running the Pipeline

Use `tools/run_pipeline.py` for end-to-end runs or call stages individually.

### Full pipeline

```bash
python -m tools.run_pipeline \
  --start-year 2002 \
  --train-start-year 2016 \
  --max-parallel-data 8 \
  --data-start-delay 1 \
  --use-async \
  --use-gpu \
  --skip-logit \
  --prediction-week 1 \
  --live-run \
  --debug
```

Outputs:

- Historical and upcoming predictions under `predictions/`
- Evaluation CSVs/plots under `predictions/evaluation/`
- SHAP summaries and run-comparison reports under `analysis/`
- Use `--prediction-week 1 --live-run` for an explicit Week 1 run that excludes that slate from training/calibration/evaluation, or leave the default `auto` for the current/next slate.

### Stage-by-stage workflow

```bash
# Build engineered matchup matrix for recent seasons
python -m src.features.build_features --season 2023 2024 2025

# Train models
python -m src.models.train --target win_prob --use-gpu --calibrate
python -m src.models.train --target spread

# Optional: tune volatility classifier / SHAP explainability
python -m analysis.volatility_classifier --model logreg --percentile 0.6 --decision-threshold 0.8 --calibrate
python -m src.analysis.shap_gpu

# Generate predictions
python -m src.predict.predict_history --seasons 2002 2024 --overwrite
python -m src.predict.predict_upcoming --season 2025 --week auto --debug

# Evaluate results
python -m src.evaluation.evaluate_predictions \
  --pred-dir predictions \
  --features-path data/processed/matchup_features.parquet \
  --output-dir predictions/evaluation
```

The commands above mirror the README and `tools/run_pipeline.py`. Adjust season ranges based on the data you downloaded.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `ImportError: No module named ...` | Re-run `python scripts/smoke_test.py`; confirm the virtual environment is activated before installing `requirements.txt`. |
| `BALLDONTLIE_API_KEY not configured` (or similar) | Add the missing key to `secrets.env` or export it in your shell. The pipeline skips BallDontLie when the key is absent. |
| `SPORTSDATAIO_API_KEY not configured` (or similar) | Add the missing key to `secrets.env` or export it in your shell. |
| `pyarrow.lib.ArrowInvalid: Repetition level histogram size mismatch` | Delete the problematic parquet file from `data/raw/` and rerun the corresponding fetcher; `src.utils.io` automatically retries with `fastparquet`. |
| `CUDA driver version is insufficient` | Upgrade NVIDIA drivers or run without `--use-gpu`. |
| Windows path or permission errors | Run PowerShell as Administrator once to allow long paths (`git config --system core.longpaths true`) and keep the repo outside OneDrive-protected folders. |
| NOAA rate limit or 403 | Set `NOAA_CONTACT_EMAIL` in `secrets.env` and increase `--sleep` for `src.data.noaa`. |

**Cache reset**: Remove stale artifacts using safe deletes such as `rm -rf data/raw/*.parquet data/processed/matchup_features.parquet predictions/*.csv`. The next pipeline run will regenerate them.

## Developer Extras

- **Tests**: `python -m pytest` runs the unit tests under `tests/`. Use `PYTEST_ADDOPTS="-m 'not slow'"` if needed.
- **Linting**: `ruff check .` (matches `.github/workflows/ci.yml`).
- **Doc agent**: `python scripts/doc_agent.py --base origin/main --head HEAD` issues warnings when README/docstrings drift.
- **Streamlit dashboard**: `streamlit run streamlit_app.py`.

## Uninstall / Clean

```bash
# Remove virtual environment
deactivate  # if active
rm -rf .venv  # or conda env remove -n nfl-preds

# Clear caches/artifacts (optional but safe)
rm -rf data/raw/*.parquet data/processed/matchup_features.parquet
rm -rf models/*.pkl predictions/*.csv predictions/evaluation/*.*
rm -rf results/*
```

Delete `secrets.env` if the machine is being decommissioned. Your git clone can then be removed with `cd .. && rm -rf nfl-predictions`.
