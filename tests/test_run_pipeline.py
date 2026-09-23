"""
Unit tests for tools/run_pipeline.py — focused on the async ingestion path.

We do NOT import the module at collection time because it transitively pulls in
matplotlib (compare_runs) which may not be installed.  All tests mock or
work-around that import at runtime.
"""
from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers to import run_pipeline without needing matplotlib / shap / etc.
# ---------------------------------------------------------------------------

def _import_run_pipeline():
    """Import tools.run_pipeline, stubbing optional heavy deps if missing."""
    stubs = {
        "matplotlib": types.ModuleType("matplotlib"),
        "matplotlib.pyplot": types.ModuleType("matplotlib.pyplot"),
        "shap": types.ModuleType("shap"),
    }
    # Only add stubs for modules that are not already importable
    for name, stub in stubs.items():
        if name not in sys.modules:
            sys.modules[name] = stub

    # matplotlib.pyplot.switch_backend is called at import
    sys.modules["matplotlib.pyplot"].switch_backend = lambda *a, **kw: None  # type: ignore[attr-defined]

    # Remove cached copy so we get a fresh import with stubs in place
    if "tools.run_pipeline" in sys.modules:
        del sys.modules["tools.run_pipeline"]

    import importlib
    return importlib.import_module("tools.run_pipeline")


# ---------------------------------------------------------------------------
# parse_args — flag wiring
# ---------------------------------------------------------------------------

class TestParseArgs:
    def setup_method(self):
        self.rp = _import_run_pipeline()

    def test_use_async_defaults_false(self):
        args = self.rp.parse_args([])
        assert args.use_async is False

    def test_use_async_flag_sets_true(self):
        args = self.rp.parse_args(["--use-async"])
        assert args.use_async is True

    def test_use_async_combined_with_max_parallel(self):
        args = self.rp.parse_args(["--use-async", "--max-parallel-data", "6"])
        assert args.use_async is True
        assert args.max_parallel_data == 6

    def test_prediction_week_defaults_to_auto(self):
        args = self.rp.parse_args([])
        assert args.prediction_week == "auto"

    def test_volatility_defaults_target_logloss_rf(self):
        args = self.rp.parse_args([])
        assert args.volatility_model == "rf"
        assert args.volatility_percentile == 0.5
        assert args.volatility_include_margin is False

    def test_prediction_week_accepts_explicit_week(self):
        args = self.rp.parse_args(["--prediction-week", "1"])
        assert args.prediction_week == "1"

    def test_live_run_defaults_false(self):
        args = self.rp.parse_args([])
        assert args.live_run is False

    def test_live_run_flag_sets_true(self):
        args = self.rp.parse_args(["--prediction-week", "1", "--live-run"])
        assert args.live_run is True

    def test_balldontlie_flags_parse(self):
        args = self.rp.parse_args([
            "--skip-balldontlie",
            "--balldontlie-feeds",
            "teams",
            "games",
            "--enable-sportsdataio",
        ])

        assert args.skip_balldontlie is True
        assert args.enable_sportsdataio is True
        assert args.balldontlie_feeds == ["teams", "games"]

    def test_propline_flags_parse(self):
        args = self.rp.parse_args([
            "--skip-propline",
            "--propline-bookmakers",
            "draftkings",
            "hardrock",
            "fanduel",
            "--propline-markets",
            "player_pass_yds",
            "player_sacks",
        ])

        assert args.skip_propline is True
        assert args.propline_bookmakers == ["draftkings", "hardrock", "fanduel"]
        assert args.propline_markets == ["player_pass_yds", "player_sacks"]

    def test_mtb_publish_flags_parse(self):
        args = self.rp.parse_args([
            "--publish-mtb",
            "--mtb-repo-dir",
            "C:/Code/mtb",
            "--mtb-csv-dir",
            "data/prediction-csvs",
            "--mtb-current-only",
            "--mtb-publish-strict",
            "--skip-mtb-import",
        ])

        assert args.publish_mtb is True
        assert args.mtb_repo_dir == Path("C:/Code/mtb")
        assert args.mtb_csv_dir == Path("data/prediction-csvs")
        assert args.mtb_current_only is True
        assert args.mtb_publish_strict is True
        assert args.skip_mtb_import is True


class TestRuntimeDependencies:
    def setup_method(self):
        self.rp = _import_run_pipeline()

    def test_missing_nfl_data_py_raises_clear_error(self):
        with patch.object(self.rp.importlib.util, "find_spec", return_value=None):
            with pytest.raises(RuntimeError, match="nfl-data-py"):
                self.rp.verify_runtime_dependencies()


class TestResolveLiveExclude:
    def setup_method(self):
        self.rp = _import_run_pipeline()

    def test_live_run_requires_explicit_prediction_week(self):
        with pytest.raises(ValueError, match="explicit --prediction-week"):
            self.rp.resolve_live_exclude("auto", True, 2026)

    def test_live_run_requires_numeric_prediction_week(self):
        with pytest.raises(ValueError, match="numeric --prediction-week"):
            self.rp.resolve_live_exclude("wk2", True, 2026)

    def test_live_run_returns_cutoff(self, capsys):
        cutoff = self.rp.resolve_live_exclude("2", True, 2026)

        assert cutoff == (2026, 2)
        assert "excluding rows at/after 2026 Week 2" in capsys.readouterr().out

    def test_explicit_prediction_week_without_live_run_warns(self, capsys):
        cutoff = self.rp.resolve_live_exclude("2", False, 2026)

        assert cutoff is None
        assert "--prediction-week was set without --live-run" in capsys.readouterr().out


class TestBuildDataJobs:
    def setup_method(self):
        self.rp = _import_run_pipeline()

    def test_propline_weekly_job_fetches_configured_player_props(self):
        jobs = self.rp.build_data_jobs(
            years=[2026],
            current_year=2026,
            debug_flag=[],
            include_balldontlie=True,
            balldontlie_feeds=None,
            include_propline=True,
            propline_bookmakers=["draftkings", "hardrock", "fanduel"],
            propline_markets=["player_pass_yds", "player_sacks"],
            include_sportsdataio=False,
            sportsdataio_feeds=None,
            include_yahoo=False,
            yahoo_feeds=None,
            yahoo_season=None,
            prediction_week="3",
        )

        balldontlie = next(job for job in jobs if job.name == "balldontlie")
        assert "--feeds" not in balldontlie.command

        propline = next(job for job in jobs if job.name == "propline_player_props")
        assert propline.command[propline.command.index("--season") + 1] == "2026"
        assert propline.command[propline.command.index("--week") + 1] == "3"
        assert propline.command[propline.command.index("--bookmakers") + 1 : propline.command.index("--markets")] == [
            "draftkings",
            "hardrock",
            "fanduel",
        ]
        assert propline.command[propline.command.index("--markets") + 1 :] == ["player_pass_yds", "player_sacks"]

    def test_propline_auto_week_does_not_fetch_player_props(self):
        jobs = self.rp.build_data_jobs(
            years=[2026],
            current_year=2026,
            debug_flag=[],
            include_balldontlie=True,
            balldontlie_feeds=None,
            include_propline=True,
            propline_bookmakers=["draftkings"],
            propline_markets=["player_pass_yds"],
            include_sportsdataio=False,
            sportsdataio_feeds=None,
            include_yahoo=False,
            yahoo_feeds=None,
            yahoo_season=None,
            prediction_week="auto",
        )

        job = next(job for job in jobs if job.name == "balldontlie")
        assert "--feeds" not in job.command
        assert not any(job.name == "propline_player_props" for job in jobs)
        assert not any(job.name == "espn_rosters" for job in jobs)

    def test_explicit_prediction_week_fetches_espn_rosters(self):
        jobs = self.rp.build_data_jobs(
            years=[2026],
            current_year=2026,
            debug_flag=[],
            include_balldontlie=True,
            balldontlie_feeds=None,
            include_propline=False,
            propline_bookmakers=None,
            propline_markets=None,
            include_sportsdataio=False,
            sportsdataio_feeds=None,
            include_yahoo=False,
            yahoo_feeds=None,
            yahoo_season=None,
            prediction_week="3",
        )

        espn_rosters = next(job for job in jobs if job.name == "espn_rosters")
        assert espn_rosters.command[espn_rosters.command.index("--season") + 1] == "2026"
        assert espn_rosters.command[espn_rosters.command.index("--week") + 1] == "3"


# ---------------------------------------------------------------------------
# build_sequential_jobs — predict_upcoming command wiring
# ---------------------------------------------------------------------------

class TestBuildSequentialJobs:
    def setup_method(self):
        self.rp = _import_run_pipeline()

    def test_prediction_week_is_passed_to_predict_upcoming(self, tmp_path):
        jobs = self.rp.build_sequential_jobs(
            years=[2026],
            current_year=2026,
            pred_dir=tmp_path / "predictions",
            debug_flag=[],
            skip_evaluation=True,
            include_visualcrossing=False,
            train_start_year=2016,
            calibration_season_window=3,
            use_gpu=False,
            calibrate_winprob=False,
            enable_tuning=False,
            tuning_dir=tmp_path / "tuning",
            tuning_options={},
            skip_logit=True,
            skip_market_roi=True,
            volatility_options={"skip": True},
            prediction_week="1",
        )

        predict_job = next(job for job in jobs if job.name == "predict_upcoming")
        week_arg_index = predict_job.command.index("--week")
        assert predict_job.command[week_arg_index + 1] == "1"

    def test_live_exclude_is_passed_to_training_and_analysis_jobs(self, tmp_path):
        jobs = self.rp.build_sequential_jobs(
            years=[2024, 2025, 2026],
            current_year=2026,
            pred_dir=tmp_path / "predictions",
            debug_flag=[],
            skip_evaluation=False,
            include_visualcrossing=False,
            train_start_year=2016,
            calibration_season_window=3,
            use_gpu=False,
            calibrate_winprob=False,
            enable_tuning=False,
            tuning_dir=tmp_path / "tuning",
            tuning_options={},
            skip_logit=True,
            skip_market_roi=False,
            volatility_options={"skip": False},
            prediction_week="1",
            live_exclude=(2026, 1),
        )

        for name in [
            "train_win_prob",
            "train_spread",
            "volatility_classifier",
            "calibrate_winprob",
            "evaluate_predictions",
            "market_roi",
            "model_metrics_export",
            "player_prop_labels",
            "player_prop_features",
            "evaluate_player_prop_baselines",
            "train_player_prop_models",
        ]:
            command = next(job.command for job in jobs if job.name == name)
            assert "--exclude-from-season" in command
            assert command[command.index("--exclude-from-season") + 1] == "2026"
            assert "--exclude-from-week" in command
            assert command[command.index("--exclude-from-week") + 1] == "1"

        predict_job = next(job for job in jobs if job.name == "predict_upcoming")
        assert "--exclude-from-season" not in predict_job.command

        calibrate_job = next(job for job in jobs if job.name == "calibrate_winprob")
        assert "--volatility-threshold" not in calibrate_job.command

        volatility_job = next(job for job in jobs if job.name == "volatility_classifier")
        assert volatility_job.command[volatility_job.command.index("--model") + 1] == "rf"
        assert volatility_job.command[volatility_job.command.index("--percentile") + 1] == "0.5"
        assert "--disable-margin" in volatility_job.command

    def test_predict_upcoming_runs_after_fresh_calibration(self, tmp_path):
        jobs = self.rp.build_sequential_jobs(
            years=[2024, 2025, 2026],
            current_year=2026,
            pred_dir=tmp_path / "predictions",
            debug_flag=[],
            skip_evaluation=False,
            include_visualcrossing=False,
            train_start_year=2016,
            calibration_season_window=3,
            use_gpu=False,
            calibrate_winprob=False,
            enable_tuning=False,
            tuning_dir=tmp_path / "tuning",
            tuning_options={},
            skip_logit=True,
            skip_market_roi=False,
            volatility_options={"skip": False},
            prediction_week="2",
            live_exclude=(2026, 2),
        )

        names = [job.name for job in jobs]
        assert names.index("predict_history") < names.index("volatility_classifier")
        assert names.index("volatility_classifier") < names.index("calibrate_winprob")
        assert names.index("calibrate_winprob") < names.index("predict_upcoming")
        assert names.index("predict_upcoming") < names.index("evaluate_player_prop_baselines")
        assert names.index("evaluate_player_prop_baselines") < names.index("train_player_prop_models")
        assert names.index("train_player_prop_models") < names.index("join_historical_player_prop_lines")
        assert names.index("join_historical_player_prop_lines") < names.index("evaluate_player_prop_over_probability")
        assert names.index("evaluate_player_prop_over_probability") < names.index("predict_player_props")
        assert names.index("train_player_prop_models") < names.index("predict_player_props")
        assert names.index("predict_player_props") < names.index("player_prop_quality_report")
        assert names.index("player_prop_quality_report") < names.index("evaluate_predictions")
        assert names.index("predict_player_props") < names.index("evaluate_predictions")
        assert names.index("evaluate_player_prop_baselines") < names.index("evaluate_predictions")
        assert names.index("predict_upcoming") < names.index("evaluate_predictions")
        assert names.index("market_roi") < names.index("model_metrics_export")
        assert names.index("player_actuals") < names.index("player_prop_labels")
        assert names.index("player_actuals") < names.index("player_availability")
        assert names.index("player_availability") < names.index("predict_player_props")
        assert names.index("player_prop_labels") < names.index("player_prop_features")
        assert names.index("player_prop_features") < names.index("build_features")

        metrics_job = next(job for job in jobs if job.name == "model_metrics_export")
        assert metrics_job.command[metrics_job.command.index("--season") + 1] == "2026"
        assert metrics_job.command[metrics_job.command.index("--week") + 1] == "2"

        player_props_job = next(job for job in jobs if job.name == "predict_player_props")
        assert player_props_job.command[player_props_job.command.index("--season") + 1] == "2026"
        assert player_props_job.command[player_props_job.command.index("--week") + 1] == "2"
        odds_vendor_index = player_props_job.command.index("--odds-vendor")
        assert player_props_job.command[odds_vendor_index + 1 : odds_vendor_index + 4] == [
            "draftkings",
            "hardrock",
            "fanduel",
        ]
        availability_index = player_props_job.command.index("--availability")
        assert Path(player_props_job.command[availability_index + 1]) == Path(
            "data/processed/current_player_availability.parquet"
        )
        assert str(tmp_path / "predictions" / "player_props_qb.csv") in player_props_job.command

        prop_quality_job = next(job for job in jobs if job.name == "player_prop_quality_report")
        assert prop_quality_job.command[prop_quality_job.command.index("--season") + 1] == "2026"
        assert prop_quality_job.command[prop_quality_job.command.index("--week") + 1] == "2"
        assert str(tmp_path / "predictions" / "evaluation" / "player_props") in prop_quality_job.command

        history_job = next(job for job in jobs if job.name == "predict_history")
        assert "--walk-forward" in history_job.command
        assert "--train-start-year" in history_job.command
        assert history_job.command[history_job.command.index("--train-start-year") + 1] == "2016"
        assert "--skip-logit" in history_job.command

    def test_walk_forward_history_starts_after_train_start_year(self, tmp_path):
        jobs = self.rp.build_sequential_jobs(
            years=[2015, 2016, 2017, 2018],
            current_year=2018,
            pred_dir=tmp_path / "predictions",
            debug_flag=[],
            skip_evaluation=True,
            include_visualcrossing=False,
            train_start_year=2016,
            calibration_season_window=3,
            use_gpu=False,
            calibrate_winprob=False,
            enable_tuning=False,
            tuning_dir=tmp_path / "tuning",
            tuning_options={},
            skip_logit=True,
            skip_market_roi=True,
            volatility_options={"skip": True},
            prediction_week="2",
            live_exclude=(2018, 2),
        )

        history_job = next(job for job in jobs if job.name == "predict_history")
        seasons_start = history_job.command.index("--seasons") + 1
        seasons_end = history_job.command.index("--output-dir")
        assert history_job.command[seasons_start:seasons_end] == ["2017", "2018"]

    def test_analysis_jobs_start_at_first_history_season(self, tmp_path):
        jobs = self.rp.build_sequential_jobs(
            years=[2015, 2016, 2017, 2018],
            current_year=2018,
            pred_dir=tmp_path / "predictions",
            debug_flag=[],
            skip_evaluation=False,
            include_visualcrossing=False,
            train_start_year=2016,
            calibration_season_window=3,
            use_gpu=False,
            calibrate_winprob=False,
            enable_tuning=False,
            tuning_dir=tmp_path / "tuning",
            tuning_options={},
            skip_logit=True,
            skip_market_roi=False,
            volatility_options={"skip": True},
            prediction_week="2",
            live_exclude=(2018, 2),
        )

        for name in [
            "evaluate_player_prop_baselines",
            "train_player_prop_models",
            "evaluate_predictions",
            "market_roi",
            "model_metrics_export",
        ]:
            command = next(job.command for job in jobs if job.name == name)
            assert "--start-season" in command
            assert command[command.index("--start-season") + 1] == "2017"


class TestBuildMtbPublishJob:
    def setup_method(self):
        self.rp = _import_run_pipeline()

    def test_build_mtb_publish_job_archives_numeric_week(self):
        job = self.rp.build_mtb_publish_job(
            repo_dir=Path("C:/Code/mtb"),
            dest_dir=Path("data/prediction-csvs"),
            current_year=2026,
            prediction_week="2",
            current_only=True,
            strict=True,
        )

        assert job.name == "publish_mtb_csvs"
        assert job.command[:3] == ["python", "-m", "tools.publish_mtb_csvs"]
        assert "--repo-dir" in job.command
        assert Path(job.command[job.command.index("--repo-dir") + 1]) == Path("C:/Code/mtb")
        assert "--season" in job.command
        assert job.command[job.command.index("--season") + 1] == "2026"
        assert "--week" in job.command
        assert job.command[job.command.index("--week") + 1] == "2"
        assert "--current-only" in job.command
        assert "--strict" in job.command

    def test_build_mtb_publish_job_skips_auto_week_archive(self):
        job = self.rp.build_mtb_publish_job(
            repo_dir=Path("../mtb"),
            dest_dir=Path("data/prediction-csvs"),
            current_year=2026,
            prediction_week="auto",
            current_only=False,
            strict=False,
        )

        assert "--week" not in job.command
        assert "--current-only" not in job.command
        assert "--strict" not in job.command

    def test_build_mtb_import_job(self):
        job = self.rp.build_mtb_import_job(repo_dir=Path("C:/Code/mtb"))

        assert job.name == "import_mtb_predictions"
        assert job.command[:3] == ["python", "-m", "tools.import_mtb_predictions"]
        assert "--repo-dir" in job.command
        assert Path(job.command[job.command.index("--repo-dir") + 1]) == Path("C:/Code/mtb")


# ---------------------------------------------------------------------------
# run_jobs_parallel — async branch routing
# ---------------------------------------------------------------------------

class TestRunJobsParallel:
    def setup_method(self):
        self.rp = _import_run_pipeline()

    def test_sequential_when_max_workers_one(self):
        """max_workers=1 always uses the sequential path even with use_async."""
        Job = self.rp.Job
        jobs = [Job("j1", ["echo", "hello"]), Job("j2", ["echo", "world"])]

        with patch.object(self.rp, "run_jobs_sequential") as mock_seq:
            self.rp.run_jobs_parallel(
                jobs, max_workers=1, launch_delay=0, env=None, dry_run=False, use_async=True
            )
            mock_seq.assert_called_once()

    def test_sequential_when_single_job(self):
        """Single job never goes through async path."""
        Job = self.rp.Job
        jobs = [Job("j1", ["echo", "hello"])]

        with patch.object(self.rp, "run_jobs_sequential") as mock_seq:
            self.rp.run_jobs_parallel(
                jobs, max_workers=4, launch_delay=0, env=None, dry_run=False, use_async=True
            )
            mock_seq.assert_called_once()

    def test_async_path_invoked_when_conditions_met(self):
        """use_async=True + max_workers>1 + len(jobs)>1 → _async_run_jobs_parallel called."""
        Job = self.rp.Job
        jobs = [Job("j1", ["echo", "a"]), Job("j2", ["echo", "b"])]

        # Patch the coroutine function to avoid an unawaited-coroutine warning
        async def _noop(*a, **kw):
            pass

        with patch.object(self.rp, "_async_run_jobs_parallel", side_effect=_noop) as mock_coro:
            self.rp.run_jobs_parallel(
                jobs, max_workers=4, launch_delay=0, env=None, dry_run=False, use_async=True
            )
            mock_coro.assert_called_once()

    def test_dry_run_skips_async_path(self):
        """dry_run=True must NOT enter the async subprocess path."""
        Job = self.rp.Job
        jobs = [Job("j1", ["echo", "a"]), Job("j2", ["echo", "b"])]

        async def _noop(*a, **kw):
            pass

        with patch.object(self.rp, "_async_run_jobs_parallel", side_effect=_noop) as mock_coro:
            with patch.object(self.rp, "run_jobs_sequential"):
                self.rp.run_jobs_parallel(
                    jobs, max_workers=4, launch_delay=0, env=None, dry_run=True, use_async=True
                )
            mock_coro.assert_not_called()
