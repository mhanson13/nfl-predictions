import sys
from argparse import Namespace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import run_log
from src.utils.nfl_calendar import EASTERN
from tools import run_pipeline


def test_debug_log_paths_use_the_nfl_week_number():
    archive, latest = run_log.debug_log_paths(2026, 10, "early_week", log_dir=Path("logs/pipeline"), stamp="20261110-0900")
    assert latest == Path("output_wk10_debug.txt")
    assert archive == Path("logs/pipeline/2026/output_wk10_early_week_20261110-0900_debug.txt")
    # postseason weeks keep nflverse numbering (wild card = 19)
    _, latest = run_log.debug_log_paths(2026, 19, "early_week")
    assert latest == Path("output_wk19_debug.txt")
    _, latest = run_log.debug_log_paths(2026, 3, "in_progress")
    assert latest == Path("output_wk3_debug.txt")


def test_run_logged_writes_every_log_live_with_footer(tmp_path):
    paths = [tmp_path / "a" / "run.txt", tmp_path / "latest.txt"]
    cmd = [sys.executable, "-c", "print('[pipeline] Starting x'); print('[espn] wrote injury rows=0'); print('[pipeline] Completed x')"]
    rc = run_log.run_logged(cmd, "HEADER\n", paths)
    assert rc == 0
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert text.startswith("HEADER") and "[pipeline] Completed x" in text
        assert "issues found in output: 1" in text and "exit code   : 0 (OK)" in text


def test_auto_week_resolves_from_schedule(tmp_path, monkeypatch):
    sched = pd.DataFrame(
        {
            "season": [2026, 2026], "week": [10, 11], "game_type": "REG",
            "gameday": ["2026-11-15", "2026-11-22"], "gametime": ["13:00", "13:00"],
            "home_score": [np.nan, np.nan], "away_score": [np.nan, np.nan],
        }
    )
    path = tmp_path / "sched.parquet"
    sched.to_parquet(path)
    import src.utils.nfl_calendar as cal

    monkeypatch.setattr(cal, "DEFAULT_SCHEDULE_PATH", path)
    monkeypatch.setattr(cal.load_week_context, "__defaults__", (path, None))
    args = Namespace(prediction_week="auto", live_run=True, season=None)
    phase = run_pipeline.resolve_auto_week(args, now=datetime(2026, 11, 10, 10, tzinfo=EASTERN))
    assert (args.season, args.prediction_week, phase) == (2026, "10", "early_week")

    explicit = Namespace(prediction_week="7", live_run=True, season=None)
    assert run_pipeline.resolve_auto_week(explicit) == "manual" and explicit.prediction_week == "7"


def test_debug_run_reexecs_under_logger_only_once(monkeypatch):
    calls = []
    monkeypatch.setattr(run_pipeline, "rerun_with_debug_log", lambda args, argv, phase: calls.append(argv) or 0)
    monkeypatch.delenv(run_log.CHILD_ENV_FLAG, raising=False)
    assert run_pipeline.main(["--debug", "--prediction-week", "10", "--live-run", "--season", "2026"]) == 0
    assert calls == [["--debug", "--prediction-week", "10", "--live-run", "--season", "2026"]]
