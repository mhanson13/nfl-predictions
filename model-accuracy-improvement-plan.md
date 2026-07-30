# NFL Predictions — Model Accuracy Improvement Plan

## Overview

A full audit of the ML pipeline was conducted. The current best model (as of 2026-02-11, run 148 on 2,751 games) achieves:

| Metric | Current | Goal (GOALS.md) | Notes |
|--------|---------|-----------------|-------|
| Accuracy | **83.28%** | 70%+ | ✅ Exceeds goal |
| AUC | **0.9568** | 0.75+ | ✅ Exceeds goal |
| Brier (raw) | **0.1311** | — | Strong |
| Brier (calibrated) | **0.0867** | — | Excellent post-shrinkage |
| MAE (margin) | **3.95 pts** | — | Down from ~10 pts historically |
| RMSE (margin) | **5.08 pts** | — | Down from ~13 pts historically |

The model is performing well above baseline. The audit identified **six high-leverage improvement opportunities** that range from low-effort/high-impact to medium-effort/medium-impact. No changes will be made to production logic, calibration thresholds, or existing validated features unless explicitly called out.

---

## Decisions Made

- Focus on improvements that are **grounded in the existing pipeline** — no rewrites, no framework changes
- **k-fold cross-validation** is the single highest-priority model quality fix (currently hardcoded 60/20/20 single split)
- **Drive-level EPA momentum** is the most impactful new feature (documented in GOALS.md, not yet implemented)
- **Altitude × rest interaction** is a small, targeted feature addition that closes a known gap
- **Injury recovery gradient** is a medium-effort, measurable improvement to QB health features
- **Data pipeline resilience** (weather fallback + API retry) is a reliability fix, not an accuracy fix — but data gaps directly cause NaN feature imputation, which degrades model quality
- **Feature quality reporting** is a pre-training diagnostic that catches silent quality regressions

---

## Sub-Task 1 — Add k-Fold Cross-Validation to Model Training

**Status:** `[x] done`

### Intent
The current training pipeline uses a single hardcoded 60/20/20 stratified split (`random_state=42`). This means performance estimates are based on one particular partition of the data, which can mask overfitting or produce high-variance metric estimates. Adding controllable k-fold cross-validation — defaulting to 3 folds — produces more reliable estimates and gives the Optuna tuner a better signal, while keeping runtime manageable.

The number of folds is exposed as a new CLI flag (`--cv-folds`) on both `src/models/train.py` and `src/models/tune.py`, defaulting to **3**. Setting `--cv-folds 1` restores the original single-split behaviour. End-of-season full retrains can use `--cv-folds 5` for maximum reliability.

**Timing impact at default (3 folds, 30 tuning trials):** ~90 total XGBoost fits vs 60 today — approximately **1.5x current tuning time**.

This is the highest-leverage **model quality** change with relatively low implementation risk — it does not change the feature set or prediction logic.

### Expected Outcomes
- `train_win_prob()` and `train_spread()` in `src/models/train.py` accept a `cv_folds: int = 3` parameter
- When `cv_folds > 1`, training reports mean ± std metrics across folds before fitting the final model
- Final production model artifact is always trained on the **full training split** (folds used only for evaluation/tuning signal — never for the saved model)
- Overfitting bias becomes quantifiable (if CV AUC < holdout AUC by >0.03, overfitting is confirmed)
- `src/models/tune.py` Optuna objective uses the mean CV score instead of a single-split score when `cv_folds > 1`
- New CLI flags added:
  - `src/models/train.py`: `--cv-folds` (int, default 3)
  - `src/models/tune.py`: `--cv-folds` (int, default 3)
  - `tools/run_pipeline.py`: `--cv-folds` (int, default 3, passed through to train and tune stages)

### Todo List
1. Read `src/models/train.py` fully — understand the current split logic (lines 174-372 for win prob, 374-540 for spread) and the existing CLI argument parser
2. Read `src/models/tune.py` fully — understand the current single-fold objective function (lines 127-133 for split logic, 179-242 for win_prob_objective) and existing CLI argument parser
3. Read `tools/run_pipeline.py` lines 100-160 — understand how `--tuning-trials-winprob` and train/tune flags are currently passed through
4. In `src/models/train.py`:
   - Add `--cv-folds` CLI argument (int, default 3, help: "Number of CV folds for evaluation. Use 1 to disable CV and use a single split.")
   - Add `cv_folds: int = 3` parameter to `train_win_prob()` and `train_spread()`
   - When `cv_folds > 1`: use `StratifiedKFold(n_splits=cv_folds, shuffle=False)` for win prob and `KFold(n_splits=cv_folds, shuffle=False)` for spread
   - Collect per-fold AUC, Brier, LogLoss, accuracy; log mean ± std after the loop
   - When `cv_folds == 1`: fall back to the existing single 80/20 split (no behaviour change)
   - Final model fit on full training data is unchanged regardless of `cv_folds`
5. In `src/models/tune.py`:
   - Add `--cv-folds` CLI argument (int, default 3)
   - Add `cv_folds: int = 3` parameter to `win_prob_objective()` and `spread_objective()`
   - When `cv_folds > 1`: replace single-split objective score with mean CV score across folds
   - When `cv_folds == 1`: existing single-split logic unchanged
6. In `tools/run_pipeline.py`:
   - Add `--cv-folds` CLI argument (int, default 3)
   - Pass the value through to the `tune_win_prob`, `tune_spread`, `train_win_prob`, and `train_spread` subprocess/function calls
7. Ensure the final saved model artifacts (`models/winprob_gb.pkl`, `models/spread_gb.pkl`) are still trained on the full training split regardless of fold count
8. Run `pytest tests/` to confirm no regressions

### Relevant Context
- `src/models/train.py` — `train_win_prob()` lines 174-372, `train_spread()` lines 374-540
- `src/models/tune.py` — Optuna objective functions
- Current split: lines 213-219 in `train.py`
- `models/winprob_gb.pkl`, `models/spread_gb.pkl` — serialized model artifacts

---

## Sub-Task 2 — Add Drive-Level EPA Momentum Features

**Status:** `[x] done`

### Intent
The current passing EPA features operate at the season-rolling-average level (3-game and 5-game windows). They do not capture **within-game momentum or game-flow dynamics** — e.g., a team that consistently scores on opening drives, or consistently collapses in the 4th quarter. Drive-level EPA from NFL play-by-play data captures this.

This is the **highest-expected-lift new feature** identified in `GOALS.md`. The nflverse play-by-play data (already ingested into `data/raw/`) contains all the required columns: `drive`, `epa`, `posteam`, `defteam`, `game_id`, `down`, `quarter_seconds_remaining`.

Expected AUC lift: **+0.01 to +0.02** based on academic benchmarks for drive-level features.

### Expected Outcomes
- New feature file `src/features/drive_epa_features.py` created
- At minimum 6 new features added per team per game:
  - `drive_epa_mean_rolling3` — average EPA per drive over last 3 games
  - `drive_epa_first_drive_rolling3` — first-drive EPA (opening efficiency)
  - `drive_epa_q4_rolling3` — 4th quarter drive EPA (late-game execution)
  - `drive_epa_red_zone_rolling3` — drive EPA on red zone possessions
  - `drive_completion_rate_rolling3` — % of drives ending in score
  - `drives_per_game_rolling3` — pace proxy
- Features integrated into `src/features/build_features.py` merge pipeline
- Corresponding `_diff` columns auto-generated by existing `make_feature_diffs()` mechanism
- Leakage prevention: all rolling windows use `.shift(1)` (same pattern as `passing_epa_features.py`)

### Todo List
1. Read `src/features/passing_epa_features.py` in full to understand the pattern to follow (rolling shift, join keys, league normalization)
2. Read the nflverse play-by-play schema — check `data/raw/nflverse_pbp.parquet` columns to confirm `drive`, `epa`, `posteam`, `game_id`, `quarter_seconds_remaining` are present
3. Create `src/features/drive_epa_features.py` following the same module pattern as `passing_epa_features.py`:
   - Aggregate EPA by `(game_id, posteam, drive)` → drive-level summary
   - Compute per-team-week rolling averages (3-game, 5-game) with `.shift(1)`
   - Normalize against league average for that season
4. Add first-drive EPA and Q4 EPA filters using `quarter` or `quarter_seconds_remaining` column
5. Import and call the new module in `src/features/build_features.py` in the feature merge pipeline (follow the same pattern used for `passing_epa_features`)
6. Verify new `_diff` columns are picked up by `make_feature_diffs()` / `select_feature_columns()` in `src/models/feature_columns.py`
7. Run the feature build pipeline on a single season to verify no NaN explosion and correct row count
8. Run `pytest tests/` to confirm no regressions

### Relevant Context
- `src/features/passing_epa_features.py` — template to follow
- `src/features/build_features.py` — integration point (feature merge pipeline)
- `src/models/feature_columns.py` — `make_feature_diffs()`, `select_feature_columns()`
- `data/raw/nflverse_pbp.parquet` — source data
- `GOALS.md` — documents drive-level EPA as desired improvement

---

## Sub-Task 3 — Add Altitude × Rest Interaction Feature

**Status:** `[x] done`

### Intent
The current volatility features track travel distance and rest days as **independent signals**. But a known gap documented in the audit is that the **altitude + short rest combination** (e.g., a sea-level team playing in Denver on 3 days rest) is not modeled. This interaction is well-documented in sports science as a compounding fatigue multiplier.

This is a **small, targeted change** to `src/features/volatility.py` — adding a single interaction term that multiplies stadium altitude delta by an inverse rest-days factor.

### Expected Outcomes
- One new interaction feature: `altitude_rest_interaction_diff`
  - Formula: `(altitude_delta / 1000) * (1 / log(rest_days + 1))`
  - Zero for dome/sea-level games; amplified for altitude + short rest combinations
- Feature added to the volatility feature family in `src/features/volatility.py`
- Picked up automatically by `make_feature_diffs()` as a `_diff` column
- Verified non-zero for at least Denver home games (altitude ~1609m) on short rest schedules

### Todo List
1. Read `src/features/volatility.py` in full to understand existing feature patterns and available column names
2. Read `data/reference/` directory to find stadium altitude data — confirm `altitude_ft` or `altitude_m` column exists in the stadium reference file
3. Add the interaction feature computation to the volatility feature function:
   - Compute `altitude_delta` = home stadium altitude - away stadium altitude (already may exist as a column)
   - Compute `rest_interaction` = `altitude_delta / (log(rest_days + 1) + 1e-6)`
   - Name: `altitude_rest_interaction`
4. Verify the feature appears in the merged matchup frame after `build_features.py` runs
5. Spot-check: Denver home games (altitude 1609m) on 3 days rest should have meaningfully higher values than neutral-site same-rest games
6. Run `pytest tests/` to confirm no regressions

### Relevant Context
- `src/features/volatility.py` — integration point
- `src/features/build_features.py` — volatility feature merge call
- `data/reference/` — stadium reference data (check for altitude column)
- `src/models/feature_columns.py` — confirm new column picked up by feature selection

---

## Sub-Task 4 — Improve QB Injury Recovery Gradient

**Status:** `[x] done`

### Intent
The current QB health features use **binary/discrete status flags** (OUT=2.0, IR=2.0, DOUBT=1.5, etc.) defined in `src/features/qb_health.py`. This captures week-of status but not the **trajectory of recovery** — a QB who was OUT two weeks ago and is now FULL is very different from a QB who is in week 1 of a return.

Adding a **decay-weighted recovery score** that considers the last 3 weeks of QB status creates a richer signal for post-injury game-quality estimation.

### Expected Outcomes
- New feature: `qb_recovery_score_rolling3` — exponentially weighted average of the last 3 QB status values (most recent weighted highest)
  - Week N weight: 0.6, Week N-1 weight: 0.3, Week N-2 weight: 0.1
- New feature: `qb_weeks_since_injury` — number of weeks since QB was last flagged as OUT or DOUBT (capped at 8)
- Both features added to `src/features/qb_health.py`
- Leakage-safe: uses `.shift(1)` before rolling, same as existing QB health features
- All existing status severity weights preserved (no changes to existing feature logic)

### Todo List
1. Read `src/features/qb_health.py` in full to understand the current feature computation and available input columns
2. Add `qb_recovery_score_rolling3`:
   - Use the existing numeric severity score (already computed per game)
   - Apply exponential weighted moving average with `span=3` and `adjust=False`
   - Apply `.shift(1)` before the EWMA to prevent leakage
3. Add `qb_weeks_since_injury`:
   - For each QB/team row, compute weeks elapsed since last status was OUT (severity ≥ 2.0)
   - Cap at 8 (beyond 8 weeks, injury history is irrelevant)
   - Apply `.shift(1)` before the lookback
4. Ensure both features flow into the team-level matchup frame via the existing QB health merge in `build_features.py`
5. Verify non-null values for recent seasons (2020+) where QB injury data is dense
6. Run `pytest tests/` to confirm no regressions

### Relevant Context
- `src/features/qb_health.py` — full integration point
- `src/features/build_features.py` — QB health feature merge call
- Existing severity weights: OUT=2.0, IR=2.0, DOUBT=1.5, QUESTION=1.0, LIMITED=0.5, DNP=0.7

---

## Sub-Task 5 — Add Pre-Training Feature Quality Report

**Status:** `[x] done`

### Intent
Currently, training runs silently on whatever features are present in `matchup_features.parquet`. If a data source fails (e.g., weather API), features become NaN-filled and are silently imputed, potentially degrading model quality without any warning.

Adding a **feature quality report** that runs before model training — checking NaN rates, variance, and observation counts per feature — catches these silent regressions and gives the developer a clear picture of data health before a training run.

This is a **diagnostic tool**, not a model change. It does not alter predictions or training logic.

### Expected Outcomes
- New function `report_feature_quality(df, feature_cols)` in `src/models/train.py` (or a new file `src/models/feature_quality.py`)
- Report prints (and optionally saves to `predictions/evaluation/feature_quality.csv`):
  - Feature name
  - % NaN (flag if > 10%)
  - Variance (flag if < 1e-6, i.e., near-constant)
  - Number of non-null observations
  - Min / Max / Mean
- Report called at the start of `train_win_prob()` before the train/test split
- Any feature with >20% NaN or zero variance logs a WARNING (does not halt training)

### Todo List
1. Read `src/models/train.py` lines 174-230 to identify the right insertion point (before the train/test split)
2. Create the `report_feature_quality()` function:
   - Accept `df` (the full feature dataframe) and `feature_cols` (list of columns being used)
   - Compute per-column: null rate, variance, n_obs, min, max, mean
   - Log a WARNING for any column with null_rate > 0.20 or variance < 1e-9
   - Return a summary DataFrame
3. Call `report_feature_quality()` at the start of both `train_win_prob()` and `train_spread()`
4. Optionally write the summary DataFrame to `predictions/evaluation/feature_quality.csv` (append with run timestamp)
5. Run `pytest tests/` to confirm no regressions

### Relevant Context
- `src/models/train.py` — integration point (before train/test split)
- `src/models/feature_columns.py` — `select_feature_columns()` produces the feature list
- `predictions/evaluation/` — output directory for existing metric CSVs

---

## Sub-Task 6 — Strengthen Data Pipeline Resilience

**Status:** `[x] done`

### Intent
The weather feature pipeline has three sources (Visual Crossing → NOAA → Tomorrow.io) but **no static fallback** if all three fail. When weather features are NaN, the imputer substitutes the training mean — this silently degrades game-specific predictions for weather-sensitive matchups (rain, wind, extreme cold).

Similarly, API rate-limiting failures in SportsDataIO and Yahoo are currently logged but not retried, causing intermittent NaN bursts in injury and odds features.

This sub-task adds:
1. A **static stadium weather lookup table** as a last-resort fallback (seasonal averages per stadium location and month)
2. **Exponential backoff retry** on API fetch failures in the async fetcher

These are reliability changes that directly protect model accuracy from data-gap-induced degradation.

### Expected Outcomes
- `src/data/weather_fallback.py` created: a static lookup of average monthly weather per NFL stadium (temperature, wind, precipitation probability) sourced from climatological norms
- `build_features.py` weather merge uses the static fallback if all three live sources return NaN for a game
- `src/utils/async_fetcher.py` (or equivalent fetch utilities) adds retry with exponential backoff (max 3 retries, base delay 2s, jitter ±0.5s) for HTTP 429 and 5xx responses
- NaN rate for weather features drops measurably (verifiable via feature quality report from Sub-Task 5)

### Todo List
1. Read `src/data/visualcrossing.py`, `src/data/noaa.py`, and `src/data/weather.py` to understand the current fallback chain and where NaN values propagate
2. Read `src/utils/async_fetcher.py` (or equivalent) to understand the current fetch/retry logic
3. Create `src/data/weather_fallback.py`:
   - Hardcode a dictionary mapping `(stadium_id, month)` → `{temp_f, wind_mph, precip_prob}` using public climatological data
   - Use the existing `data/reference/` stadium list for stadium IDs and locations
   - Expose a single function: `get_fallback_weather(stadium_id, game_date) -> dict`
4. In `build_features.py`, after the weather merge, add a fill step: for rows where weather features are still NaN, call `get_fallback_weather()` and fill
5. In the async fetcher, wrap HTTP calls with retry logic:
   - Catch `HTTPError` with status 429 or 500-599
   - Retry up to 3 times with delays: 2s, 4s, 8s (+ random jitter)
   - Log each retry attempt at WARNING level
6. Run the feature build for one season and verify weather NaN rate is reduced
7. Run `pytest tests/` to confirm no regressions

### Relevant Context
- `src/data/visualcrossing.py`, `src/data/noaa.py`, `src/data/weather.py` — weather source chain
- `src/utils/async_fetcher.py` — HTTP fetch utility
- `src/features/build_features.py` — weather merge pipeline (lines 500-778)
- `data/reference/` — stadium reference data (locations for fallback climate lookup)

---

## Implementation Order & Expected Impact

Sub-Tasks can be started after the previous one is verified, or run in parallel where noted:

| Order | Sub-Task | Type | Expected Impact | Can Parallelize With |
|-------|----------|------|-----------------|----------------------|
| 1st | **ST1 — k-Fold CV** | Model quality | Quantifies overfitting; improves tuner signal | — |
| 2nd | **ST5 — Feature Quality Report** | Diagnostic | Catches data gaps silently degrading accuracy | ST1 |
| 3rd | **ST6 — Pipeline Resilience** | Data quality | Eliminates NaN degradation in weather/injury features | ST5 |
| 4th | **ST2 — Drive-Level EPA** | New feature | +0.01–0.02 AUC (highest new-feature lift) | — |
| 5th | **ST3 — Altitude × Rest** | New feature | Targeted lift in ~30 games/season | ST2 |
| 6th | **ST4 — QB Recovery Gradient** | Feature enrichment | Better post-injury game modeling | ST3 |

## Notes for Implementor

- **Do not change** existing feature severity weights, calibration thresholds, or isotonic recalibration logic — these are tuned and production-validated
- All new rolling features must use `.shift(1)` before the rolling window to prevent leakage (see `passing_epa_features.py` as the canonical pattern)
- New features will be **automatically included** in training via `make_feature_diffs()` and `select_feature_columns()` — no manual column list edits required
- After each sub-task, verify with `pytest tests/` before proceeding
- After ST2-ST4 (new features), retrain the model and compare run metrics in `predictions/evaluation/overall_metrics.csv` against the current baseline (run 148: accuracy=0.8328, AUC=0.9568, Brier=0.1311)
