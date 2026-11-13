from datetime import datetime

import numpy as np
import pandas as pd

from src.utils.nfl_calendar import EASTERN
from tools import weekly_run


def _write_schedule(tmp_path):
    rows = [
        (2026, 9, "2026-11-08", "13:00", 17.0, 34.0),
        (2026, 10, "2026-11-12", "20:15", np.nan, np.nan),
        (2026, 10, "2026-11-15", "13:00", np.nan, np.nan),
    ]
    frame = pd.DataFrame(rows, columns=["season", "week", "gameday", "gametime", "home_score", "away_score"])
    frame["game_type"] = "REG"
    path = tmp_path / "sched.parquet"
    frame.to_parquet(path)
    return path


def _plan(tmp_path, when, *argv):
    args, extra = weekly_run.parse_args(["--schedule", str(_write_schedule(tmp_path)), "--log-dir", str(tmp_path), *argv])
    return weekly_run.build_plan(args, extra, now=datetime.fromisoformat(when).replace(tzinfo=EASTERN))


def test_early_week_plan_overwrites_archive_and_sets_week(tmp_path):
    plan = _plan(tmp_path, "2026-11-10T10:00")
    cmd = plan.command
    assert plan.phase == "early_week" and plan.overwrite_archive
    assert cmd[cmd.index("--prediction-week") + 1] == "10"
    assert cmd[cmd.index("--season") + 1] == "2026"
    assert "--live-run" in cmd and "--debug" in cmd and "--skip-logit" in cmd
    assert "--mtb-overwrite-week-archive" in cmd
    assert plan.log_path.name.startswith("output_wk10_early_week_") and plan.log_path.name.endswith("_debug.txt")


def test_in_progress_plan_keeps_archive_and_passes_extra_flags(tmp_path):
    plan = _plan(tmp_path, "2026-11-13T07:00", "--", "--skip-tomorrow-weather")
    assert plan.phase == "in_progress" and not plan.overwrite_archive
    assert "--mtb-overwrite-week-archive" not in plan.command
    assert plan.command[-1] == "--skip-tomorrow-weather"


def test_no_publish_and_overrides(tmp_path):
    plan = _plan(tmp_path, "2026-11-10T10:00", "--no-publish", "--week", "11", "--phase", "in_progress")
    assert "--publish-mtb" not in plan.command
    assert plan.week == 11 and plan.phase == "in_progress"


def test_issue_scanner_flags_log_lines_and_rate_limits_but_not_tables():
    flagged = [
        "[espn_rosters] wrote injury rows=0 -> data/raw/espn_injuries_2026_wk10.parquet",
        "[pipeline] Warning: post-run analysis 'shap_gpu' failed (boom).",
        "   hourly          429 Too Many Requests         2     0        2",
        "Traceback (most recent call last):",
        "ValueError: bad input",
    ]
    ignored = [
        "2026    10   market   all   anytime_td   all   429   344   344   0",
        "[pipeline] Starting predict_upcoming: python -m src.predict.predict_upcoming --exclude-missing",
        "[weather] window rows=14 missing(lat,lon,dt)=(0,0,0)",
        "[market_roi] Prepared dataset with 2354 games (0 missing odds filtered out)",
        "[win_prob][cv=3] AUC=0.7278  Brier=0.2095",
    ]
    assert all(weekly_run.is_issue_line(line) for line in flagged)
    assert not any(weekly_run.is_issue_line(line) for line in ignored)


def test_offseason_plan_has_no_command(tmp_path):
    plan = _plan(tmp_path, "2026-12-01T10:00")
    assert plan.phase == "offseason" and plan.command == []
