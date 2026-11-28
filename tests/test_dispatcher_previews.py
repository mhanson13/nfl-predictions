"""Dispatcher game previews: built after grading, published from PREVIEWS_LIVE_FROM, Wednesday refresh."""

from datetime import timedelta
import json
from pathlib import Path

import pandas as pd
import pytest
from test_dispatcher import _args, _utc

from src.data import game_previews as gp
from tools import dispatcher as d
from tools import publish_mtb_csvs as pub


def _schedule():
    rows = [
        (12, "2026-11-26", "20:20"), (12, "2026-11-29", "13:00"), (12, "2026-11-30", "20:15"),
        (13, "2026-12-03", "20:15"), (13, "2026-12-06", "13:00"), (13, "2026-12-07", "20:15"),
        (14, "2026-12-10", "20:15"), (14, "2026-12-13", "13:00"), (14, "2026-12-14", "20:15"),
    ]
    return pd.DataFrame({"season": 2026, "week": [w for w, _, _ in rows],
                         "game_id": [f"2026_{w}_{i}" for i, (w, _, _) in enumerate(rows)],
                         "gameday": [g for _, g, _ in rows], "gametime": [t for _, _, t in rows]})


def _write_preview(week, spread=-3.0):
    frame = pd.DataFrame([{
        "season": 2026, "week": week, "game_id": f"2026_{week}_X_Y", "home_team": "Y", "away_team": "X",
        "kickoff": "2026-12-11T01:15:00Z", "pick_expected_at": "2026-12-10T21:45:00Z", "roof": "dome",
        "venue_name": "V", "venue_city": "C", "neutral_site": False, "spread_home": spread, "total": 44.5,
        "home_ml": -150, "away_ml": 130, "odds_source": gp.ODDS_SOURCE_API,
        "odds_captured_at": "2026-12-08T10:00:00Z", "generated_at": "2026-12-08T10:00:00Z",
    }], columns=gp.PREVIEW_COLUMNS)
    gp.write_previews(frame, 2026, week)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Run in tmp_path (preview files are relative paths) with a schedule parquet and a quiet state dir."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(d, "STATE_DIR", tmp_path / "logs")
    (tmp_path / "logs").mkdir()
    (tmp_path / "mtb").mkdir()
    sched = tmp_path / "schedule.parquet"
    _schedule().to_parquet(sched)
    return tmp_path, sched


def _grade_args(tmp_path, sched, **kw):
    return _args(tmp_path / "mtb", schedule=sched, publish_lock_wait_sec=0, **kw)


def _runner(calls, preview_rc=0, spread=-3.0):
    def run(cmd, **kw):
        cmd = [str(c) for c in cmd]
        calls.append(cmd)
        if "src.evaluation.live_results" in cmd:
            Path(cmd[cmd.index("--report-json") + 1]).write_text(json.dumps({"report": "graded"}))
            return 0
        if "src.data.game_previews" in cmd:
            if preview_rc == 0:
                _write_preview(int(cmd[cmd.index("--week") + 1]), spread)
            return preview_rc
        return 0
    return run


def _copy_cmd(calls):
    return next(c for c in calls if "tools.publish_mtb_csvs" in c)


def test_previews_before_go_live_are_built_locally_only(env):
    tmp_path, sched = env
    calls, alerts = [], []
    state = {}
    rec = d.handle_grading(2026, 12, _grade_args(tmp_path, sched), state, _utc("2026-12-01T10:00:00"),
                           runner=_runner(calls), alert=lambda *a, **kw: alerts.append(a))
    assert rec["status"] == "success"
    build = next(c for c in calls if "src.data.game_previews" in c)
    assert build[build.index("--week") + 1] == "13" and build[build.index("--max-age-hours") + 1] == "6"
    assert "--preview-week" not in _copy_cmd(calls)  # week 13 is before PREVIEWS_LIVE_FROM
    assert Path("predictions/previews/previews_2026_week_13.csv").exists()
    assert "locally only" in rec["preview"]["note"]
    assert state[d.preview_key(2026, 13)]["build_status"] == "built"
    assert "published_key" not in state[d.preview_key(2026, 13)]
    # The build runs before the publish lock is taken, i.e. before the copy step.
    assert calls.index(build) < calls.index(_copy_cmd(calls))


def test_live_week_previews_ride_along_with_the_results(env):
    tmp_path, sched = env
    calls, state = [], {}
    rec = d.handle_grading(2026, 13, _grade_args(tmp_path, sched), state, _utc("2026-12-08T10:00:00"),
                           runner=_runner(calls))
    assert rec["status"] == "success"
    copy = _copy_cmd(calls)
    assert "--results-only" in copy and copy[copy.index("--preview-week") + 1] == "14"
    assert rec["preview"]["status"] == "published"
    prec = state[d.preview_key(2026, 14)]
    assert prec["published_key"] == gp.content_key(pd.read_csv(gp.week_paths(2026, 14)[1]))


@pytest.mark.parametrize("failure", ["rc", "raise", "no_schedule"])
def test_a_preview_failure_never_blocks_the_results_publish(env, failure):
    tmp_path, sched = env
    calls, alerts = [], []
    runner = _runner(calls, preview_rc=1)
    if failure == "raise":
        def runner(cmd, _inner=_runner(calls), **kw):
            if "src.data.game_previews" in [str(c) for c in cmd]:
                raise OSError("disk full")
            return _inner(cmd, **kw)
    args = _grade_args(tmp_path, tmp_path / "missing.parquet" if failure == "no_schedule" else sched)
    rec = d.handle_grading(2026, 13, args, {}, _utc("2026-12-08T10:00:00"), runner=runner,
                           alert=lambda *a, **kw: alerts.append(a))
    assert rec["status"] == "success" and rec["preview"]["status"] == "failed"
    assert "--preview-week" not in _copy_cmd(calls)
    assert any("scripts/publish_site.py" in " ".join(c) for c in calls)
    assert len(alerts) == 1


def test_grading_alert_carries_the_preview_note(monkeypatch):
    sent = []
    monkeypatch.setattr("src.utils.alerts.send_alert", lambda subject, body: sent.append(body))
    d.send_grading_alert(2026, 13, {"status": "success", "grade_rc": 0, "publish_rc": 0, "report": "",
                                    "preview": {"note": "week 14 previews failed (exit 1); results published without them"}},
                         final=False)
    assert "Previews: week 14 previews failed (exit 1)" in sent[0]


def test_no_previews_flag_skips_the_build(env):
    tmp_path, sched = env
    calls = []
    rec = d.handle_grading(2026, 13, _grade_args(tmp_path, sched, no_previews=True), {}, _utc("2026-12-08T10:00:00"),
                           runner=_runner(calls))
    assert rec["status"] == "success" and "preview" not in rec
    assert not any("src.data.game_previews" in c for c in calls)


# --- Wednesday refresh -------------------------------------------------------------------------

WED_NOON_ET = _utc("2026-12-09T17:00:00")  # 12:00 EST


def test_refresh_due_wednesday_from_noon_et_for_the_upcoming_live_week():
    slots = d.build_slots(_schedule())
    assert d.due_preview_refresh(slots, {}, WED_NOON_ET - timedelta(minutes=1)) is None
    assert d.due_preview_refresh(slots, {}, WED_NOON_ET) == (2026, 14)
    assert d.due_preview_refresh(slots, {}, WED_NOON_ET - timedelta(days=1)) is None  # Tuesday
    # Week 13 is before go-live: no refresh on Wed 12-02.
    assert d.due_preview_refresh(slots, {}, _utc("2026-12-02T17:00:00")) is None


def test_refresh_skips_while_a_game_day_window_is_open():
    sched = _schedule()
    sched.loc[sched.week == 14, "gameday"] = ["2026-12-09", "2026-12-13", "2026-12-14"]  # a Wednesday game
    sched.loc[sched.week == 14, "gametime"] = ["16:30", "13:00", "20:15"]
    slots = d.build_slots(sched)
    assert d.due_preview_refresh(slots, {}, WED_NOON_ET + timedelta(hours=1)) is None  # window opened 13:00 ET


def test_refresh_done_for_the_day_once_published_or_unchanged_and_rechecks_failures():
    slots = d.build_slots(_schedule())
    key = d.preview_key(2026, 14)
    for status in ("published", "unchanged"):
        assert d.due_preview_refresh(slots, {key: {"refresh_day": "2026-12-09", "refresh_status": status}},
                                     WED_NOON_ET + timedelta(hours=2)) is None
    failed = {key: {"refresh_day": "2026-12-09", "refresh_status": "failed", "refresh_failures": 1,
                    "refresh_attempt": WED_NOON_ET.isoformat()}}
    assert d.due_preview_refresh(slots, failed, WED_NOON_ET + timedelta(minutes=30)) is None
    assert d.due_preview_refresh(slots, failed, WED_NOON_ET + timedelta(hours=1)) == (2026, 14)
    failed[key]["refresh_failures"] = d.PREVIEW_REFRESH_MAX_FAILURES
    assert d.due_preview_refresh(slots, failed, WED_NOON_ET + timedelta(hours=3)) is None
    pending = {key: {"refresh_day": "2026-12-09", "refresh_status": d.PUBLISH_PENDING,
                     "refresh_attempt": WED_NOON_ET.isoformat()}}
    assert d.due_preview_refresh(slots, pending, WED_NOON_ET + timedelta(minutes=15)) == (2026, 14)


def test_refresh_publishes_only_on_a_change(env):
    tmp_path, sched = env
    args = _grade_args(tmp_path, sched)
    calls, state = [], {}
    _write_preview(14)
    state[d.preview_key(2026, 14)] = {"published_key": gp.content_key(pd.read_csv(gp.week_paths(2026, 14)[1]))}

    rec = d.handle_preview_refresh(2026, 14, args, state, WED_NOON_ET, runner=_runner(calls))
    assert rec["refresh_status"] == "unchanged"
    assert not any("tools.publish_mtb_csvs" in c for c in calls)

    calls.clear()
    rec = d.handle_preview_refresh(2026, 14, args, state, WED_NOON_ET + timedelta(minutes=15),
                                   runner=_runner(calls, spread=-4.5))
    assert rec["refresh_status"] == "published"
    build = next(c for c in calls if "src.data.game_previews" in c)
    assert "--max-age-hours" not in build  # fresh odds on Wednesday
    copy = _copy_cmd(calls)
    assert "--previews-only" in copy and copy[copy.index("--preview-week") + 1] == "14"
    site = next(c for c in calls if "scripts/publish_site.py" in " ".join(c))
    assert "Refresh NFL previews 2026 week 14" in site
    assert rec["published_key"] == gp.content_key(pd.read_csv(gp.week_paths(2026, 14)[1]))


def test_refresh_failure_alerts_once_and_keeps_the_published_previews(env):
    tmp_path, sched = env
    args = _grade_args(tmp_path, sched)
    alerts, calls, state = [], [], {}
    for i in range(2):
        rec = d.handle_preview_refresh(2026, 14, args, state, WED_NOON_ET + timedelta(hours=i),
                                       runner=_runner(calls, preview_rc=1), alert=lambda *a: alerts.append(a))
    assert rec["refresh_status"] == "failed" and rec["refresh_failures"] == 2 and len(alerts) == 1
    assert not any("tools.publish_mtb_csvs" in c for c in calls)


def test_refresh_waits_for_a_busy_publish_lock(env):
    tmp_path, sched = env
    from src.utils import publish_lock
    publish_lock.acquire(tmp_path / "mtb" / publish_lock.LOCK_FILE_NAME, owner="mtb", sport="mlb",
                         stale_minutes=90, wait_seconds=0)
    calls, state = [], {}
    rec = d.handle_preview_refresh(2026, 14, _grade_args(tmp_path, sched), state, WED_NOON_ET, runner=_runner(calls))
    assert rec["refresh_status"] == d.PUBLISH_PENDING and not rec.get("refresh_failures")
    assert not any("tools.publish_mtb_csvs" in c for c in calls)


# --- publish_mtb_csvs --------------------------------------------------------------------------

def _publish(tmp_path, **kw):
    repo = tmp_path / "mtb"
    (repo / "data" / "prediction-csvs").mkdir(parents=True, exist_ok=True)
    return pub.publish_csvs(source_root=tmp_path, repo_dir=repo, season=2026, **kw), repo / "data" / "prediction-csvs"


def test_results_copy_includes_previews_at_the_agreed_paths(env):
    tmp_path, _ = env
    _write_preview(14)
    published, dest = _publish(tmp_path, results_only=True, preview_week=14)
    assert (dest / "current" / "previews.csv").read_text() == (dest / "previews" / "2026" / "week_14.csv").read_text()
    assert (dest / "previews" / "2026" / "week_14.csv").read_text() == gp.week_paths(2026, 14)[1].read_text()
    assert not (dest / "seasons").exists()


def test_results_copy_without_a_preview_file_still_succeeds(env):
    tmp_path, _ = env
    published, dest = _publish(tmp_path, results_only=True, preview_week=14)
    assert not (dest / "current" / "previews.csv").exists()


def test_previews_only_copies_just_the_previews_and_requires_them(env):
    tmp_path, _ = env
    with pytest.raises(FileNotFoundError):
        _publish(tmp_path, previews_only=True, preview_week=14)
    _write_preview(14)
    published, dest = _publish(tmp_path, previews_only=True, preview_week=14)
    assert sorted(p.destination.relative_to(dest).as_posix() for p in published) == [
        "current/previews.csv", "previews/2026/week_14.csv"]


def test_previews_are_overwritten_not_preserved_like_week_archives(env):
    tmp_path, _ = env
    _write_preview(14, spread=-3.0)
    _publish(tmp_path, previews_only=True, preview_week=14)
    _write_preview(14, spread=-4.5)
    _, dest = _publish(tmp_path, previews_only=True, preview_week=14)
    assert "-4.5" in (dest / "previews" / "2026" / "week_14.csv").read_text()


def test_preview_flags_need_a_season():
    with pytest.raises(ValueError):
        pub.build_artifact_plan(results_only=True, preview_week=14)
    with pytest.raises(ValueError):
        pub.build_artifact_plan(season=2026, previews_only=True)
