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
