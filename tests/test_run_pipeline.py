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
        assert names.index("predict_upcoming") < names.index("evaluate_predictions")


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
