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
from typing import Iterable, Sequence

from src.utils.secrets import get_secret


@dataclass(frozen=True)
class Job:
    name: str
    command: Sequence[str]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
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
        "--skip-sportradar",
        action="store_true",
        help="Skip Sportradar data fetch even if an API key is configured.",
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
        "--calibration-season-window",
        type=int,
        default=3,
        help="Number of most recent seasons to use when fitting win probability calibration.",
    )
    parser.add_argument(
        "--sportradar-feeds",
        nargs="+",
        default=None,
        help="Override the list of Sportradar feeds (Phase A static feeds only).",
    )
    parser.add_argument(
        "--sportradar-args",
        nargs=argparse.REMAINDER,
        default=None,
        help="Additional arguments forwarded to the Sportradar data module.",
    )
    parser.add_argument(
        "--sportradar-mode",
        choices=["full", "nightly"],
        default="full",
        help="Sportradar fetch mode: 'full' runs static + change feeds, 'nightly' focuses on change feeds.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log the planned commands without executing them.",
    )
    return parser.parse_args(argv)


def build_years(start_year: int) -> list[int]:
    current_year = datetime.now().year
    if start_year > current_year:
        raise ValueError(f"start_year {start_year} exceeds current year {current_year}")
    return list(range(start_year, current_year + 1))


def ensure_predictions_dir(pred_dir: Path) -> None:
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
    return " ".join(f'"{c}"' if " " in c else c for c in cmd)


def run_job(job: Job, env: dict[str, str] | None = None, dry_run: bool = False) -> None:
    print(f"[pipeline] Starting {job.name}: {format_command(job.command)}")
    if dry_run:
        print(f"[pipeline] (dry-run) Skipping execution of {job.name}")
        return
    result = subprocess.run(job.command, env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Job {job.name} failed with exit code {result.returncode}")
    print(f"[pipeline] Completed {job.name}")


def run_jobs_sequential(jobs: Iterable[Job], env: dict[str, str] | None, dry_run: bool) -> None:
    for job in jobs:
        run_job(job, env=env, dry_run=dry_run)


def run_jobs_parallel(
    jobs: Sequence[Job],
    max_workers: int,
    launch_delay: float,
    env: dict[str, str] | None,
    dry_run: bool,
) -> None:
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
                raise RuntimeError(f"Data job {job.name} failed") from exc


def build_data_jobs(
    years: list[int],
    current_year: int,
    debug_flag: list[str],
    *,
    include_sportradar: bool,
    sportradar_feeds: list[str] | None,
    sportradar_args: list[str] | None,
    sportradar_mode: str,
) -> list[Job]:
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
    if include_sportradar:
        if sportradar_feeds:
            feeds = list(sportradar_feeds)
        elif sportradar_mode == "nightly":
            feeds = []
        else:
            feeds = ["league_hierarchy", "teams", "seasons", "season_schedule", "team_roster"]
        for change_feed in ("daily_change_log", "daily_transactions"):
            if change_feed not in feeds:
                feeds.append(change_feed)
        season_params: list[str] = []
        needs_seasons = {"season_schedule", "weekly_schedule", "weekly_depth_charts", "seasonal_statistics"}
        if any(feed in needs_seasons for feed in feeds):
            has_custom_seasons = bool(sportradar_args) and "--seasons" in sportradar_args
            if not has_custom_seasons:
                season_params = ["--seasons", str(current_year)]
        command = [
            "python",
            "-m",
            "src.data.sportradar",
            "--feeds",
            *feeds,
            *debug,
            *season_params,
        ]
        if sportradar_args:
            command.extend(sportradar_args)
        jobs.append(
            Job(
                "sportradar_static",
                command,
            )
        )
    return jobs


def build_sequential_jobs(
    years: list[int],
    current_year: int,
    pred_dir: Path,
    debug_flag: list[str],
    skip_evaluation: bool,
    include_sportradar: bool,
    train_start_year: int,
    calibration_season_window: int,
    use_gpu: bool,
    *,
    enable_tuning: bool,
    tuning_dir: Path,
    tuning_options: dict[str, object],
) -> list[Job]:
    season_args = [str(y) for y in years]
    debug = debug_flag.copy()
    jobs: list[Job] = []

    winprob_config_path = tuning_dir / "winprob_best.json"
    spread_config_path = tuning_dir / "spread_best.json"

    if include_sportradar:
        jobs.append(
            Job(
                "sportradar_transform",
                ["python", "-m", "src.data.sportradar_transform", *debug],
            )
        )
        jobs.append(
            Job(
                "sportradar_report",
                ["python", "-m", "tools.sportradar_report", *debug],
        )
        )
    jobs.append(Job("weather", ["python", "-m", "src.data.weather", "--season", *season_args, *debug]))
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
        jobs.append(Job("tune_spread", tune_spread_cmd))

    train_win_cmd = ["python", "-m", "src.models.train", "--target", "win_prob", "--calibrate", *debug]
    if train_start_year is not None:
        train_win_cmd.extend(["--train-start-year", str(train_start_year)])
    if use_gpu:
        train_win_cmd.append("--use-gpu")
    if enable_tuning:
        train_win_cmd.extend(["--param-config", str(winprob_config_path)])
    jobs.append(Job("train_win_prob", train_win_cmd))

    train_spread_cmd = ["python", "-m", "src.models.train", "--target", "spread", "--calibrate", *debug]
    if train_start_year is not None:
        train_spread_cmd.extend(["--train-start-year", str(train_start_year)])
    if use_gpu:
        train_spread_cmd.append("--use-gpu")
    if enable_tuning:
        train_spread_cmd.extend(["--param-config", str(spread_config_path)])
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
                    *(["--start-season", str(train_start_year)] if train_start_year is not None else []),
                    *debug,
                ],
            )
        )

    return jobs


def main(argv: Sequence[str] | None = None) -> int:
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

    env = os.environ.copy()

    sportradar_key = get_secret("SPORTSRADAR_NFL_API_KEY")
    include_sportradar = bool(sportradar_key) and not args.skip_sportradar
    if args.skip_sportradar:
        print("[pipeline] Sportradar fetch skipped via flag.")
    elif not sportradar_key:
        print("[pipeline] Sportradar API key not found; skipping Sportradar jobs.")

    data_jobs = build_data_jobs(
        years,
        current_year,
        debug_flag,
        include_sportradar=include_sportradar,
        sportradar_feeds=args.sportradar_feeds,
        sportradar_args=args.sportradar_args,
        sportradar_mode=args.sportradar_mode,
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
        include_sportradar,
        args.train_start_year,
        args.calibration_season_window,
        args.use_gpu,
        enable_tuning=args.enable_tuning,
        tuning_dir=tuning_dir,
        tuning_options=tuning_options,
    )
    print(f"[pipeline] Running {len(sequential_jobs)} sequential jobs")
    run_jobs_sequential(sequential_jobs, env=env, dry_run=args.dry_run)

    print("[pipeline] Pipeline complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
