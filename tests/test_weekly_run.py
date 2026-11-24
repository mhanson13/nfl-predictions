from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd

from src.utils.nfl_calendar import EASTERN
from tools import dispatcher, weekly_run


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


# --- MTB publish lock for manual runs -------------------------------------------------------------

COPY = ["py", "-m", "tools.publish_mtb_csvs", "--repo-dir", "mtb"]
IMPORT = ["py", "-m", "tools.import_mtb_predictions", "--repo-dir", "mtb"]


def _write_plan(path, *cmds):
    path.write_text(json.dumps({"jobs": [{"name": c[2], "command": c} for c in cmds]}), encoding="utf-8")
    return path


def _lock(repo):
    return repo / ".publish.lock"


def _recording_runner(repo, calls, rc_for=lambda cmd: 0):
    def run(cmd, **kw):
        held = json.loads(_lock(repo).read_text()) if _lock(repo).exists() else None
        token = (kw.get("env") or {}).get(dispatcher.PUBLISH_LOCK_TOKEN_ENV)
        calls.append((cmd[2], held is not None and held["token"] == token and held["sport"] == "nfl"))
        return rc_for(cmd)
    return run


def test_manual_publishing_run_defers_mtb_jobs_to_a_plan(tmp_path):
    plan = _plan(tmp_path, "2026-11-10T10:00")
    cmd = plan.command
    assert "--publish-mtb" in cmd
    assert plan.publish_plan is not None and plan.publish_plan.name.endswith("_mtb_publish_plan.json")
    assert cmd[cmd.index("--mtb-publish-plan") + 1] == str(plan.publish_plan)
    assert plan.publish_plan.parent == plan.log_path.parent.resolve()


def test_dispatcher_and_no_publish_runs_are_unchanged(tmp_path):
    dispatched = _plan(tmp_path, "2026-11-10T10:00", "--", "--mtb-publish-plan", "slot_plan.json")
    assert dispatched.publish_plan is None
    assert dispatched.command.count("--mtb-publish-plan") == 1 and dispatched.command[-1] == "slot_plan.json"
    no_publish = _plan(tmp_path, "2026-11-10T10:00", "--no-publish")
    assert no_publish.publish_plan is None and "--mtb-publish-plan" not in no_publish.command


def test_publish_locked_runs_every_step_inside_the_lock(tmp_path, monkeypatch):
    monkeypatch.delenv(dispatcher.PUBLISH_LOCK_PATH_ENV, raising=False)
    plan = _write_plan(tmp_path / "plan.json", COPY, IMPORT)
    calls = []
    rc = weekly_run.publish_locked(plan, tmp_path, runner=_recording_runner(tmp_path, calls))
    assert rc == 0
    assert calls == [("tools.publish_mtb_csvs", True), ("tools.import_mtb_predictions", True)]
    assert not _lock(tmp_path).exists()


def test_busy_lock_copies_nothing_and_exits_75(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv(dispatcher.PUBLISH_LOCK_PATH_ENV, raising=False)
    other = {"owner": "mlb-predictions", "pid": 1, "sport": "mlb", "stale_after_minutes": 90,
             "started_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "token": "mlbtoken"}
    _lock(tmp_path).write_text(json.dumps(other))
    plan = _write_plan(tmp_path / "plan.json", COPY, IMPORT)
    calls = []
    rc = weekly_run.publish_locked(plan, tmp_path, wait_seconds=0, runner=_recording_runner(tmp_path, calls))
    assert rc == 75 and calls == []
    assert json.loads(_lock(tmp_path).read_text())["token"] == "mlbtoken"  # left alone
    assert f"--publish-only {plan}" in capsys.readouterr().out


def test_failed_step_and_missing_plan(tmp_path, monkeypatch):
    monkeypatch.delenv(dispatcher.PUBLISH_LOCK_PATH_ENV, raising=False)
    plan = _write_plan(tmp_path / "plan.json", COPY, IMPORT)
    calls = []
    rc = weekly_run.publish_locked(plan, tmp_path, runner=_recording_runner(
        tmp_path, calls, rc_for=lambda cmd: 3 if "tools.publish_mtb_csvs" in cmd else 0))
    assert rc == 3 and [c[0] for c in calls] == ["tools.publish_mtb_csvs"]
    assert not _lock(tmp_path).exists()
    assert weekly_run.publish_locked(tmp_path / "missing.json", tmp_path, runner=_recording_runner(tmp_path, [])) == 1


def test_main_publishes_after_a_successful_pipeline_only(tmp_path, monkeypatch):
    published = []
    fixed_plan = _plan(tmp_path, "2026-11-10T10:00")
    monkeypatch.setattr(weekly_run, "publish_locked", lambda plan, repo, **kw: published.append(plan) or 0)
    monkeypatch.setattr(weekly_run, "preflight", lambda schedule: [])
    monkeypatch.setattr(weekly_run, "build_plan", lambda args, extra: fixed_plan)
    for pipeline_rc, expect_publish in [(0, True), (1, False)]:
        published.clear()
        monkeypatch.setattr(weekly_run, "run_logged", lambda cmd, header, paths, rc=pipeline_rc: rc)
        assert weekly_run.main(["--mtb-repo-dir", str(tmp_path)]) == pipeline_rc
        assert bool(published) == expect_publish


def test_publish_only_skips_the_pipeline(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(weekly_run, "publish_locked",
                        lambda plan, repo, **kw: seen.update(plan=plan, repo=repo, **kw) or 75)
    monkeypatch.setattr(weekly_run, "run_logged", lambda *a, **k: (_ for _ in ()).throw(AssertionError("ran")))
    rc = weekly_run.main(["--publish-only", str(tmp_path / "p.json"), "--mtb-repo-dir", str(tmp_path),
                          "--publish-lock-wait-sec", "5"])
    assert rc == 75 and seen["plan"] == tmp_path / "p.json" and seen["wait_seconds"] == 5.0
