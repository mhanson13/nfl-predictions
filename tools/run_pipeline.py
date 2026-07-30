# Copyright (c) 2025 Matt Hanson
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Orchestrate the end-to-end pipeline: ingest data, build features, train, predict, and analyze."""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence

from src.analysis.compare_runs import main as compare_runs_main
from src.analysis.shap_gpu import main as shap_gpu_main
from src.analysis.update_readme_metrics import main as update_readme_main
from src.utils.secrets import get_secret

# Ensure subprocesses use the same interpreter as the pipeline itself.
PYTHON_EXECUTABLE = sys.executable or "python"

STACK_OVERFLOW_CODES = {0xC0000409, 3221226505}
PREDICT_OUTPUTS = [
    Path("predictions/predictions.csv"),
    Path("predictions/predictions_full.csv"),
    Path("predictions/predictions_players_qb.csv"),
    Path("predictions/predictions_players_offense.csv"),
    Path("predictions/predictions_players_defense.csv"),
]
# Pipeline stages overview:
#   1. Concurrent data ingestion (NFLverse/ESPN/weather/etc.)
#   2. Sequential transforms: reference refresh, features, training, predictions, calibration, evaluation.
#   3. Post-run analysis: run comparisons, GPU SHAP explainability, README metric refresh.


@dataclass(frozen=True)
class Job:
    name: str
    command: Sequence[str]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """CLI parser covering data years, concurrency, GPU flags, and dry-run mode."""
    parser = argparse.ArgumentParser(description="Run the NFL predictions pipeline with concurrency.")
    parser.add_argument("--start-year", type=int, default=2023, help="First season to include in data refresh.")
    parser.add_argument("--debug", action="store_true", help="Enable verbose logging in downstream scripts.")
    parser.add_argument(
        "--train-start-year",
        type=int,
        default=2015,
        help="First season to include when training models (still ingest data earlier for history).",
    )
    parser.add_argument(
        "--max-parallel-data",
        type=int,
        default=1,
        help="Maximum concurrent data ingestion jobs (increase carefully to avoid rate limiting).",
    )
    parser.add_argument(
        "--data-start-delay",
        type=float,
        default=1.5,
        help="Seconds to wait between launching each data job (helps stagger API requests).",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip running the evaluation module at the end of the pipeline.",
    )
    parser.add_argument(
        "--skip-market-roi",
        action="store_true",
        help="Skip market ROI analysis after evaluation.",
    )
    parser.add_argument(
        "--skip-sportsdataio",
        action="store_true",
        help="Skip SportsDataIO data fetch even if an API key is configured.",
    )
    parser.add_argument(
        "--skip-visualcrossing",
        action="store_true",
        help="Skip Visual Crossing weather fetch even if an API key is configured.",
    )
    parser.add_argument(
        "--skip-yahoo",
        action="store_true",
        help="Skip Yahoo Sports API fetch even if credentials are configured.",
    )
    parser.add_argument(
        "--skip-volatility",
        action="store_true",
        help="Skip volatility classifier and shrinkage adjustments before calibration.",
    )
    parser.add_argument(
        "--skip-logit",
        action="store_true",
        help="Skip logistic companion model during win probability training.",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Enable GPU-accelerated model training (requires XGBoost with GPU support).",
    )
    parser.add_argument(
        "--enable-tuning",
        action="store_true",
        help="Run Optuna hyperparameter tuning before training models.",
    )
    parser.add_argument(
        "--tuning-trials-winprob",
        type=int,
        default=60,
        help="Number of Optuna trials for win probability tuning (when enabled).",
    )
    parser.add_argument(
        "--tuning-trials-spread",
        type=int,
        default=60,
        help="Number of Optuna trials for spread tuning (when enabled).",
    )
    parser.add_argument(
        "--tuning-winprob-timeout",
        type=float,
        default=None,
        help="Optional timeout (seconds) for win probability tuning.",
    )
    parser.add_argument(
        "--tuning-spread-timeout",
        type=float,
        default=None,
        help="Optional timeout (seconds) for spread tuning.",
    )
    parser.add_argument(
        "--tuning-storage",
        type=str,
        default=None,
        help="Optuna storage URI for pipeline tuning (e.g., sqlite:///tuning/tuning.db).",
    )
    parser.add_argument(
        "--tuning-study-prefix",
        type=str,
        default="pipeline",
        help="Prefix for Optuna study names created by the pipeline.",
    )
    parser.add_argument(
        "--volatility-model",
        choices=["logreg", "xgb", "rf"],
        default="logreg",
        help="Model to use for volatility classifier.",
    )
    parser.add_argument(
        "--volatility-percentile",
        type=float,
        default=0.6,
        help="Percentile for labeling high-error (volatile) games.",
    )
    parser.add_argument(
        "--volatility-random-state",
        type=int,
        default=42,
        help="Random seed for volatility classifier splits.",
    )
    parser.add_argument(
        "--volatility-threshold",
        type=float,
        default=0.55,
        help="Volatility probability threshold triggering win-prob shrinkage.",
    )
    parser.add_argument(
        "--volatility-strength",
        type=float,
        default=0.35,
        help="Shrinkage strength toward 0.5 for volatile games (0-1).",
    )
    parser.add_argument(
        "--calibrate-winprob",
        action="store_true",
        help="Calibrate the win probability model during training (CalibratedClassifierCV).",
    )
    parser.add_argument(
        "--calibration-season-window",
        type=int,
        default=3,
        help="Number of most recent seasons to use when fitting win probability calibration.",
    )
    parser.add_argument(
        "--sportsdataio-feeds",
        nargs="+",
        default=None,
        help="Override SportsDataIO feeds (default: teams stadiums schedules standings projections dfs_slates betting_futures draft_picks free_agents).",
    )
    parser.add_argument(
        "--yahoo-feeds",
        nargs="+",
        default=None,
        help="Override Yahoo feeds (default: game teams players injuries).",
    )
    parser.add_argument(
        "--yahoo-season",
        type=int,
        default=None,
        help="Optional season value for Yahoo scoreboard feed.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=3,
        help="Number of CV folds passed to tune and train stages (1=single split, 3=default, 5=full retrain quality).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log the planned commands without executing them.",
    )
    return parser.parse_args(argv)


def build_years(start_year: int) -> list[int]:
    """Return list of seasons from `start_year` through the current year inclusive."""
    current_year = datetime.now().year
    if start_year > current_year:
        raise ValueError(f"start_year {start_year} exceeds current year {current_year}")
    return list(range(start_year, current_year + 1))


def ensure_predictions_dir(pred_dir: Path) -> None:
    """Make sure prediction output directory exists and current artifacts are removed."""
    pred_dir.mkdir(parents=True, exist_ok=True)
    for file_path in Path(".").glob("predictions*.csv*"):
        if pred_dir.resolve() in file_path.resolve().parents:
            continue
        if not file_path.is_file():
            continue
        move_target = pred_dir / file_path.name
        if not move_target.exists():
            file_path.rename(move_target)
            continue
        counter = 1
        while True:
            candidate = pred_dir / f"{file_path.stem}.migration_{counter}{file_path.suffix}"
            if not candidate.exists():
                file_path.rename(candidate)
                break
            counter += 1


def format_command(cmd: Sequence[str]) -> str:
    """Render a subprocess command into a readable string for logging."""
    return " ".join(f'"{c}"' if " " in c else c for c in cmd)

def run_job(job: Job, env: dict[str, str] | None = None, dry_run: bool = False) -> None:
    """Execute a single pipeline job with logging and error propagation."""
    command = list(job.command)
    if command and command[0] == "python":
        command[0] = PYTHON_EXECUTABLE
    print(f"[pipeline] Starting {job.name}: {format_command(command)}")
    if dry_run:
        print(f"[pipeline] (dry-run) Skipping execution of {job.name}")
        return
    result = subprocess.run(command, env=env, check=False)
    if result.returncode != 0:
        if (
            job.name == "predict_upcoming"
            and result.returncode in STACK_OVERFLOW_CODES
            and all(path.exists() for path in PREDICT_OUTPUTS)
        ):
            print(
                "[pipeline] Warning: predict_upcoming exited with 0xC0000409 but outputs exist; "
                "treating as success (likely Windows fastparquet shutdown issue)."
            )
            return
        raise RuntimeError(f"Job {job.name} failed with exit code {result.returncode}")
    print(f"[pipeline] Completed {job.name}")


def run_jobs_sequential(jobs: Iterable[Job], env: dict[str, str] | None, dry_run: bool) -> None:
    """Run a list of jobs one after another, failing fast when a stage errors."""
    for job in jobs:
        run_job(job, env=env, dry_run=dry_run)


def run_jobs_parallel(
    jobs: Sequence[Job],
    max_workers: int,
    launch_delay: float,
    env: dict[str, str] | None,
    dry_run: bool,
) -> None:
    """Launch IO-heavy ingestion jobs in parallel threads with optional staggering."""
    optional_jobs = {"yahoo_static"}
    if max_workers <= 1 or len(jobs) <= 1:
        run_jobs_sequential(jobs, env, dry_run)
        return

    launch_delay = max(launch_delay, 0.0)
    futures: dict[concurrent.futures.Future[None], Job] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, job in enumerate(jobs):
            if idx > 0 and launch_delay > 0:
                time.sleep(launch_delay)
            futures[executor.submit(run_job, job, env, dry_run)] = job
        for future in concurrent.futures.as_completed(futures):
            job = futures[future]
            try:
                future.result()
            except Exception as exc:  # pragma: no cover - propagate with context
                if job.name in optional_jobs:
                    print(f"[pipeline] Warning: optional job {job.name} failed ({exc}); continuing.")
                    continue
                raise RuntimeError(f"Data job {job.name} failed") from exc


def build_data_jobs(
    years: list[int],
    current_year: int,
    debug_flag: list[str],
    *,
    include_sportsdataio: bool,
    sportsdataio_feeds: list[str] | None,
    include_yahoo: bool,
    yahoo_feeds: list[str] | None,
    yahoo_season: int | None,
) -> list[Job]:
    """Construct the list of ingestion jobs (NFLverse/ESPN/etc.) for the requested seasons."""
    season_args = [str(y) for y in years]
    debug = debug_flag.copy()
    jobs: list[Job] = []

    jobs.append(
        Job(
            "nflverse",
            ["python", "-m", "src.data.nflverse", "--season", *season_args, "--pbp", "--schedules", "--rosters", "--injuries", "--snaps", "--depthcharts", "--players", "--pfr", "passing", "rushing", "receiving", *debug],
        )
    )
    jobs.append(
        Job(
            "nflcom",
            ["python", "-m", "src.data.nflcom", "--year", *season_args, "--category", "passing", "rushing", "receiving", "scoring", "downs", *debug],
        )
    )
    jobs.append(
        Job(
            "espn_players",
            ["python", "-m", "src.data.espn_players", "--season", *season_args, "--season_type", "2", "--category", "passing", "rushing", "receiving", *debug],
        )
    )
    jobs.append(
        Job(
            "espn_player_news",
            ["python", "-m", "src.data.espn_player_news", "--season", str(current_year), "--limit", "4", *debug],
        )
    )
    jobs.append(
        Job(
            "espn_team_news",
            ["python", "-m", "src.data.espn_team_news", "--season", str(current_year), "--limit", "50", *debug],
        )
    )
    jobs.append(
        Job(
            "noaa_weather",
            ["python", "-m", "src.data.noaa", "--seasons", *season_args, *debug],
        )
    )
    if include_sportsdataio:
        sportsdataio_cmd = [
            "python",
            "-m",
            "src.data.sportsdataio",
            "--seasons",
            str(current_year),
            *debug,
        ]
        if sportsdataio_feeds:
            sportsdataio_cmd.extend(["--feeds", *sportsdataio_feeds])
        jobs.append(Job("sportsdataio", sportsdataio_cmd))
    if include_yahoo:
        yahoo_cmd = ["python", "-m", "src.data.yahoo", *debug]
        if yahoo_feeds:
            yahoo_cmd.extend(["--feeds", *yahoo_feeds])
        if yahoo_season is not None:
            yahoo_cmd.extend(["--season", str(yahoo_season)])
        jobs.append(Job("yahoo_static", yahoo_cmd))
    return jobs


def build_sequential_jobs(
    years: list[int],
    current_year: int,
    pred_dir: Path,
    debug_flag: list[str],
    skip_evaluation: bool,
    include_visualcrossing: bool,
    train_start_year: int,
    calibration_season_window: int,
    use_gpu: bool,
    calibrate_winprob: bool,
    *,
    enable_tuning: bool,
    tuning_dir: Path,
    tuning_options: dict[str, object],
    skip_logit: bool,
    skip_market_roi: bool,
    volatility_options: dict[str, object] | None = None,
    cv_folds: int = 3,
) -> list[Job]:
    """Build the ordered list of feature/model/evaluation jobs that must run serially."""
    season_args = [str(y) for y in years]
    debug = debug_flag.copy()
    jobs: list[Job] = []
    vol_cfg = volatility_options or {}
    volatility_dataset_path = Path(vol_cfg.get("dataset_path", Path("analysis/volatility_classifier_dataset.csv")))
    skip_volatility = vol_cfg.get("skip", False)

    jobs.append(Job("nfl_reference_refresh", ["python", "-m", "src.data.reference.update_regular_season", *debug]))
    winprob_config_path = tuning_dir / "winprob_best.json"
    spread_config_path = tuning_dir / "spread_best.json"
    apply_winprob_config = enable_tuning or winprob_config_path.exists()
    apply_spread_config = enable_tuning or spread_config_path.exists()

    jobs.append(Job("weather", ["python", "-m", "src.data.weather", "--season", *season_args, *debug]))
    if include_visualcrossing:
        jobs.append(
            Job(
                "visualcrossing_weather",
                ["python", "-m", "src.data.visualcrossing", "--seasons", *season_args, "--only-missing", *debug],
            )
        )
    jobs.append(
        Job(
            "player_actuals",
            ["python", "-m", "src.data.player_actuals", "--seasons", *season_args, *debug],
        )
    )
    jobs.append(Job("build_features", ["python", "-m", "src.features.build_features", "--season", *season_args, *debug]))

    if enable_tuning:
        tune_storage = tuning_options.get("storage_uri")
        tune_prefix = tuning_options.get("study_prefix", "pipeline")
        load_existing = tuning_options.get("load_if_exists", True)
        tune_win_cmd = [
            "python",
            "-m",
            "src.models.tune",
            "--target",
            "win_prob",
            "--metric",
            "logloss",
            "--trials",
            str(tuning_options.get("trials_winprob", 60)),
            "--save-best",
            str(winprob_config_path),
            *debug,
        ]
        if train_start_year is not None:
            tune_win_cmd.extend(["--train-start-year", str(train_start_year)])
        if use_gpu:
            tune_win_cmd.append("--use-gpu")
        if tune_storage:
            tune_win_cmd.extend(["--storage", tune_storage])
        if tune_prefix:
            tune_win_cmd.extend(["--study-name", f"{tune_prefix}_win_prob"])
        if load_existing:
            tune_win_cmd.append("--load-if-exists")
        timeout_win = tuning_options.get("timeout_winprob")
        if timeout_win is not None:
            tune_win_cmd.extend(["--timeout", str(timeout_win)])
        tune_win_cmd.extend(["--cv-folds", str(cv_folds)])
        jobs.append(Job("tune_win_prob", tune_win_cmd))

        tune_spread_cmd = [
            "python",
            "-m",
            "src.models.tune",
            "--target",
            "spread",
            "--metric",
            "mae",
            "--trials",
            str(tuning_options.get("trials_spread", 60)),
            "--save-best",
            str(spread_config_path),
            *debug,
        ]
        if train_start_year is not None:
            tune_spread_cmd.extend(["--train-start-year", str(train_start_year)])
        if use_gpu:
            tune_spread_cmd.append("--use-gpu")
        if tune_storage:
            tune_spread_cmd.extend(["--storage", tune_storage])
        if tune_prefix:
            tune_spread_cmd.extend(["--study-name", f"{tune_prefix}_spread"])
        if load_existing:
            tune_spread_cmd.append("--load-if-exists")
        timeout_spread = tuning_options.get("timeout_spread")
        if timeout_spread is not None:
            tune_spread_cmd.extend(["--timeout", str(timeout_spread)])
        tune_spread_cmd.extend(["--cv-folds", str(cv_folds)])
        jobs.append(Job("tune_spread", tune_spread_cmd))

    train_win_cmd = ["python", "-m", "src.models.train", "--target", "win_prob", *debug]
    if train_start_year is not None:
        train_win_cmd.extend(["--train-start-year", str(train_start_year)])
    if use_gpu:
        train_win_cmd.append("--use-gpu")
    if skip_logit:
        train_win_cmd.append("--skip-logit")
    if calibrate_winprob:
        train_win_cmd.append("--calibrate")
    if apply_winprob_config:
        train_win_cmd.extend(["--param-config", str(winprob_config_path)])
    train_win_cmd.extend(["--cv-folds", str(cv_folds)])
    jobs.append(Job("train_win_prob", train_win_cmd))

    train_spread_cmd = ["python", "-m", "src.models.train", "--target", "spread", "--calibrate", *debug]
    if train_start_year is not None:
        train_spread_cmd.extend(["--train-start-year", str(train_start_year)])
    if use_gpu:
        train_spread_cmd.append("--use-gpu")
    if apply_spread_config:
        train_spread_cmd.extend(["--param-config", str(spread_config_path)])
    train_spread_cmd.extend(["--cv-folds", str(cv_folds)])
    jobs.append(Job("train_spread", train_spread_cmd))

    pred_save = str(pred_dir / "predictions.csv")
    pred_full = str(pred_dir / "predictions_full.csv")
    qb_path = str(pred_dir / "predictions_players_qb.csv")
    off_path = str(pred_dir / "predictions_players_offense.csv")
    def_path = str(pred_dir / "predictions_players_defense.csv")

    jobs.append(
        Job(
            "predict_upcoming",
            [
                "python",
                "-m",
                "src.predict.predict_upcoming",
                "--season",
                str(current_year),
                "--week",
                "auto",
                "--save",
                pred_save,
                "--dump-all",
                pred_full,
                "--save-players-qb",
                qb_path,
                "--save-players-offense",
                off_path,
                "--save-players-defense",
                def_path,
                *debug,
            ],
        )
    )
    history_args = [
        "python",
        "-m",
        "src.predict.predict_history",
        "--seasons",
        *season_args,
        "--output-dir",
        str(Path("predictions/history")),
        "--overwrite",
        *debug,
    ]
    jobs.append(Job("predict_history", history_args))

    if not skip_volatility:
        volatility_dataset_path.parent.mkdir(parents=True, exist_ok=True)
        volatility_cmd = [
            "python",
            "-m",
            "analysis.volatility_classifier",
            "--model",
            str(vol_cfg.get("model", "logreg")),
            "--percentile",
            str(vol_cfg.get("percentile", 0.6)),
            "--random-state",
            str(vol_cfg.get("random_state", 42)),
            "--disable-season-split",
            "--calibrate",
            *debug,
        ]
        jobs.append(Job("volatility_classifier", volatility_cmd))

    calibrate_args = [
        "python",
        "-m",
        "src.evaluation.calibrate_winprob",
        "--history-dir",
        str(Path("predictions/history")),
        "--season-window",
        str(calibration_season_window),
        *debug,
    ]
    if not skip_volatility:
        calibrate_args.extend(
            [
                "--volatility-dataset",
                str(volatility_dataset_path),
                "--volatility-threshold",
                str(vol_cfg.get("threshold", 0.55)),
                "--volatility-strength",
                str(vol_cfg.get("strength", 0.35)),
            ]
        )
    else:
        calibrate_args.append("--disable-volatility")
    jobs.append(Job("calibrate_winprob", calibrate_args))

    if not skip_evaluation:
        jobs.append(
            Job(
                "evaluate_predictions",
                [
                    "python",
                    "-m",
                    "src.evaluation.evaluate_predictions",
                    "--min-games",
                    "1",
                    "--skip-plots",
                    *(["--start-season", str(train_start_year)] if train_start_year is not None else []),
                    *debug,
                ],
            )
        )

    if not skip_evaluation and not skip_market_roi:
        roi_cmd = [
            "python",
            "-m",
            "analysis.market_roi",
            *(["--start-season", str(train_start_year)] if train_start_year is not None else []),
            *debug,
        ]
        jobs.append(Job("market_roi", roi_cmd))

    return jobs


def run_post_run_analysis() -> None:
    """Run post-training analysis hooks (leaderboards, SHAP, README refresh) in sequence."""
    # Order matters: leaderboard consumes latest metrics, SHAP saves plots used by README text.
    steps = [
        ("compare_runs", compare_runs_main),
        ("shap_gpu", shap_gpu_main),
        ("update_readme_metrics", update_readme_main),
    ]
    for name, func in steps:
        try:
            print(f"[pipeline] Running post-run analysis: {name}")
            func()
        except Exception as exc:  # pragma: no cover - best effort
            print(f"[pipeline] Warning: post-run analysis '{name}' failed ({exc}).")


def main(argv: Sequence[str] | None = None) -> int:
    """Entrypoint for the multi-stage pipeline; returns process exit code."""
    args = parse_args(argv)
    years = build_years(args.start_year)
    current_year = datetime.now().year
    debug_flag = ["--debug"] if args.debug else []

    pred_dir = Path("predictions")
    ensure_predictions_dir(pred_dir)

    tuning_dir = Path("tuning")
    tuning_storage: str | None
    if args.enable_tuning:
        tuning_dir.mkdir(parents=True, exist_ok=True)
        if args.tuning_storage:
            tuning_storage = args.tuning_storage
        else:
            default_storage_path = tuning_dir / "tuning.db"
            tuning_storage = f"sqlite:///{default_storage_path.resolve().as_posix()}"
    else:
        tuning_storage = args.tuning_storage

    tuning_options = {
        "storage_uri": tuning_storage,
        "study_prefix": args.tuning_study_prefix,
        "trials_winprob": args.tuning_trials_winprob,
        "trials_spread": args.tuning_trials_spread,
        "timeout_winprob": args.tuning_winprob_timeout,
        "timeout_spread": args.tuning_spread_timeout,
        "load_if_exists": True,
    }

    volatility_options = {
        "skip": args.skip_volatility,
        "model": args.volatility_model,
        "percentile": args.volatility_percentile,
        "random_state": args.volatility_random_state,
        "threshold": args.volatility_threshold,
        "strength": args.volatility_strength,
        "dataset_path": Path("analysis/volatility_classifier_dataset.csv"),
    }

    env = os.environ.copy()
    sportsdataio_key = get_secret("SPORTSDATAIO_API_KEY")
    include_sportsdataio = bool(sportsdataio_key) and not args.skip_sportsdataio
    if args.skip_sportsdataio:
        print("[pipeline] SportsDataIO fetch skipped via flag.")
    elif not sportsdataio_key:
        print("[pipeline] SportsDataIO API key not found; skipping SportsDataIO job.")

    visualcrossing_key = get_secret("VISUAL_CROSSING_API_KEY")
    include_visualcrossing = bool(visualcrossing_key) and not args.skip_visualcrossing
    if args.skip_visualcrossing:
        print("[pipeline] Visual Crossing fetch skipped via flag.")
    elif not visualcrossing_key:
        print("[pipeline] Visual Crossing API key not found; skipping Visual Crossing weather job.")

    yahoo_client_id = get_secret("YAHOO_CLIENT_ID")
    yahoo_client_secret = get_secret("YAHOO_CLIENT_SECRET")
    yahoo_access_token = get_secret("YAHOO_ACCESS_TOKEN")
    yahoo_access_secret = get_secret("YAHOO_ACCESS_TOKEN_SECRET")
    yahoo_refresh_token = yahoo_access_secret  # backwards compatibility
    if not yahoo_refresh_token:
        yahoo_refresh_token = get_secret("YAHOO_REFRESH_TOKEN")
    include_yahoo = bool(yahoo_client_id and yahoo_client_secret and yahoo_access_token and yahoo_refresh_token) and not args.skip_yahoo
    if args.skip_yahoo:
        print("[pipeline] Yahoo fetch skipped via flag.")
    elif not include_yahoo:
        print("[pipeline] Yahoo API credentials not found; skipping Yahoo job.")

    data_jobs = build_data_jobs(
        years,
        current_year,
        debug_flag,
        include_sportsdataio=include_sportsdataio,
        sportsdataio_feeds=args.sportsdataio_feeds,
        include_yahoo=include_yahoo,
        yahoo_feeds=args.yahoo_feeds,
        yahoo_season=args.yahoo_season,
    )
    print(
        f"[pipeline] Running {len(data_jobs)} data jobs with max_parallel={args.max_parallel_data}, "
        f"launch_delay={args.data_start_delay:.1f}s"
    )
    run_jobs_parallel(
        data_jobs,
        max_workers=max(1, args.max_parallel_data),
        launch_delay=args.data_start_delay,
        env=env,
        dry_run=args.dry_run,
    )

    sequential_jobs = build_sequential_jobs(
        years,
        current_year,
        pred_dir,
        debug_flag,
        args.skip_evaluation,
        include_visualcrossing,
        args.train_start_year,
        args.calibration_season_window,
        args.use_gpu,
        args.calibrate_winprob,
        enable_tuning=args.enable_tuning,
        tuning_dir=tuning_dir,
        tuning_options=tuning_options,
        skip_logit=args.skip_logit,
        skip_market_roi=args.skip_market_roi,
        volatility_options=volatility_options,
        cv_folds=args.cv_folds,
    )
    print(f"[pipeline] Running {len(sequential_jobs)} sequential jobs")
    run_jobs_sequential(sequential_jobs, env=env, dry_run=args.dry_run)

    if not args.dry_run:
        run_post_run_analysis()

    print("[pipeline] Pipeline complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
