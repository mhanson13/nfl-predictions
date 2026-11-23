import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.utils import alerts
from tools import dispatcher as d
from tools import review_run


def _schedule():
    return pd.DataFrame(
        {
            "season": [2026] * 5,
            "week": [10] * 5,
            "game_id": ["thu", "lon", "sun1", "snf", "mnf"],
            "gameday": ["2026-11-12", "2026-11-15", "2026-11-15", "2026-11-15", "2026-11-16"],
            "gametime": ["20:15", "09:30", "13:00", "20:20", "20:15"],
        }
    )


def _utc(s):
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def test_one_slot_per_game_day_with_separate_early_international_slot():
    slots = d.build_slots(_schedule())
    assert [s.slot_id for s in slots] == [
        "2026_wk10_Thu_2026-11-12", "2026_wk10_Sun-early_2026-11-15",
        "2026_wk10_Sun_2026-11-15", "2026_wk10_Mon_2026-11-16",
    ]
    early, sun = slots[1], slots[2]
    assert early.game_ids == ("lon",) and early.first_kickoff == _utc("2026-11-15T14:30:00")  # 09:30 ET
    assert sun.game_ids == ("sun1", "snf") and sun.first_kickoff == _utc("2026-11-15T18:00:00")


def test_no_early_split_without_morning_games():
    sched = _schedule()
    sched = sched[sched.game_id != "lon"]
    assert [s.slot_id for s in d.build_slots(sched)][1] == "2026_wk10_Sun_2026-11-15"


def test_window_opens_before_first_kick_and_catches_up_for_later_games():
    early, sun = d.build_slots(_schedule())[1:3]
    s, l = d.START_LEAD, d.LAST_START_LEAD
    assert not early.window_open(_utc("2026-11-15T10:59:00"), s, l)  # 05:59 ET, before 3h30 lead
    assert early.window_open(_utc("2026-11-15T11:00:00"), s, l)      # 06:00 ET
    assert not early.window_open(_utc("2026-11-15T12:00:00"), s, l)  # London < 2h45 away
    assert not sun.window_open(_utc("2026-11-15T14:29:00"), s, l)    # main slot opens 09:30 ET
    assert sun.window_open(_utc("2026-11-15T14:30:00"), s, l)
    assert sun.window_open(_utc("2026-11-15T21:00:00"), s, l)        # SNF still >= 2h45 away
    assert not sun.window_open(_utc("2026-11-15T22:40:00"), s, l)    # every game < 2h45 away


def test_due_slot_skips_success_and_exhausted_retries():
    slots = d.build_slots(_schedule())
    now = _utc("2026-11-15T15:00:00")
    sun = slots[2].slot_id
    assert d.due_slot(slots, {}, now).slot_id == sun
    assert d.due_slot(slots, {sun: {"status": "failed", "attempts": 1}}, now).slot_id == sun  # retries
    assert d.due_slot(slots, {sun: {"status": "failed", "attempts": 2}}, now).slot_id == sun
    assert d.due_slot(slots, {sun: {"status": "failed", "attempts": d.MAX_ATTEMPTS}}, now) is None
    assert d.due_slot(slots, {sun: {"status": "success", "attempts": 1}}, now) is None


def test_lock_blocks_overlap_but_expires(tmp_path):
    lock = tmp_path / "running.lock"
    now = _utc("2026-11-15T12:00:00")
    assert d.acquire_lock(now, lock)
    assert not d.acquire_lock(now + timedelta(minutes=30), lock)
    assert d.acquire_lock(now + d.STALE_LOCK + timedelta(minutes=1), lock)


def _args(tmp_path, **kw):
    base = dict(mtb_repo_dir=tmp_path, no_publish=False, no_site_deploy=False, no_review=True,
                pipeline_timeout_min=1, publish_timeout_min=1, review_timeout_min=1)
    base.update(kw)
    return argparse.Namespace(**base)


COPY_CMD = ["py", "-m", "tools.publish_mtb_csvs", "--repo-dir", "mtb", "--season", "2026", "--week", "10"]
IMPORT_CMD = ["py", "-m", "tools.import_mtb_predictions", "--repo-dir", "mtb"]


def _write_plan(cmd):
    """What run_pipeline --mtb-publish-plan leaves behind after a successful run."""
    plan = Path(cmd[cmd.index("--mtb-publish-plan") + 1])
    plan.write_text(json.dumps({"jobs": [{"name": "publish_mtb_csvs", "command": COPY_CMD},
                                         {"name": "import_mtb_predictions", "command": IMPORT_CMD}]}))


def _pipeline_ok(cmd):
    if "tools.weekly_run" in cmd:
        _write_plan(cmd)
    return 0


def test_run_slot_publishes_only_after_success(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "STATE_DIR", tmp_path)
    slot = d.build_slots(_schedule())[2]
    calls = []

    def runner(cmd, **kw):
        calls.append(cmd)
        if "tools.weekly_run" in cmd:
            assert kw["env"][d.SLOT_ENV] == slot.slot_id
            assert cmd[cmd.index("--week") + 1] == "10"
            return 1  # pipeline fails
        return 0

    res = d.run_slot(slot, _args(tmp_path), runner=runner)
    assert res["status"] == "failed"
    assert len(calls) == 1  # nothing copied or published after a failure

    calls.clear()
    res = d.run_slot(slot, _args(tmp_path), runner=lambda cmd, **kw: calls.append(cmd) or _pipeline_ok(cmd))
    assert res["status"] == "success" and res["publish_rc"] == 0
    assert calls[1] == COPY_CMD and calls[2] == IMPORT_CMD  # the plan's copy/import, in order
    assert "scripts/publish_site.py" in calls[3]


def test_run_slot_fails_when_the_pipeline_left_no_publish_plan(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "STATE_DIR", tmp_path)
    slot = d.build_slots(_schedule())[2]
    calls = []
    res = d.run_slot(slot, _args(tmp_path), runner=lambda cmd, **kw: calls.append(cmd) or 0)
    assert res["status"] == "failed" and "no MTB publish plan" in res["publish_note"]
    assert len(calls) == 1 and not (tmp_path / ".publish.lock").exists()


def test_no_publish_runs_the_pipeline_only(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "STATE_DIR", tmp_path)
    slot = d.build_slots(_schedule())[2]
    calls = []
    res = d.run_slot(slot, _args(tmp_path, no_publish=True), runner=lambda cmd, **kw: calls.append(cmd) or 0)
    assert res["status"] == "success" and len(calls) == 1
    assert "--no-publish" in calls[0] and "--mtb-publish-plan" not in calls[0]


def test_publish_failure_marks_slot_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "STATE_DIR", tmp_path)
    slot = d.build_slots(_schedule())[2]
    res = d.run_slot(slot, _args(tmp_path),
                     runner=lambda cmd, **kw: 1 if "scripts/publish_site.py" in cmd else _pipeline_ok(cmd))
    assert res["status"] == "failed" and res["pipeline_rc"] == 0 and res["publish_rc"] == 1


def test_alert_outbox_only_when_email_not_configured(tmp_path):
    def boom(*a):
        raise AssertionError("must not send")

    secrets = {"BIRD_API_KEY": "k", "ALERT_EMAIL_FROM": "alerts@example.com"}
    res = alerts.send_alert("Run OK", "body", outbox_dir=tmp_path, secret=secrets.get, bird_send=boom, sendgrid_send=boom)
    assert not res.emailed and "ALERT_EMAIL_TO" in res.detail
    assert "body" in res.outbox_path.read_text(encoding="utf-8")
    res = alerts.send_alert("Run OK", "body", outbox_dir=tmp_path, bird_send=boom, sendgrid_send=boom,
                            secret={"ALERT_EMAIL_FROM": "a@x.com", "ALERT_EMAIL_TO": "b@x.com"}.get)
    assert not res.emailed and "no BIRD_ACCESS_KEY" in res.detail


def test_alert_prefers_bird_smtp_and_records_failures(tmp_path):
    secrets = {"BIRD_ACCESS_KEY": "bk", "SENDGRID_API_KEY": "sg", "ALERT_EMAIL_FROM": "alerts@mattythebookie.com",
               "ALERT_EMAIL_TO": "a@x.com, b@x.com"}
    sent = {}

    def bird(key, msg):
        sent.update(key=key, msg=msg)
        return "bird smtp accepted"

    res = alerts.send_alert("Run OK", "body", outbox_dir=tmp_path, secret=secrets.get, bird_send=bird,
                            sendgrid_send=lambda *a: (_ for _ in ()).throw(AssertionError("bird first")))
    assert res.emailed and sent["key"] == "bk"
    assert sent["msg"]["To"] == "a@x.com, b@x.com"
    assert sent["msg"]["From"] == "MattyTheBookie Alerts <alerts@mattythebookie.com>"
    assert "mattythebookie.com" in sent["msg"]["Message-ID"]

    def auth_fail(key, msg):
        raise RuntimeError("535 authentication failed")

    res = alerts.send_alert("Run OK", "body", outbox_dir=tmp_path, secret=secrets.get, bird_send=auth_fail)
    assert not res.emailed and "535" in res.outbox_path.read_text(encoding="utf-8")


def test_alert_sendgrid_fallback(tmp_path):
    secrets = {"SENDGRID_API_KEY": "sg", "ALERT_EMAIL_FROM": "alerts@x.com", "ALERT_EMAIL_TO": "a@x.com"}
    got = {}
    res = alerts.send_alert("Run OK", "body", outbox_dir=tmp_path, secret=secrets.get,
                            sendgrid_send=lambda key, payload: got.update(payload) or "sendgrid HTTP 202")
    assert res.emailed and got["tracking_settings"]["click_tracking"]["enable"] is False


def test_review_probability_moves_and_facts(tmp_path):
    ledger = pd.DataFrame(
        {
            "game_id": ["g", "g", "h", "h"],
            "slot": ["thu", "sun", "thu", "sun"],
            "recorded_at": ["2026-11-12T20:00:00Z", "2026-11-15T10:00:00Z"] * 2,
            "home_win_prob": [0.50, 0.58, 0.40, 0.41],
            "market_home_implied_prob_novig": [0.50, 0.57, 0.40, 0.40],
        }
    )
    moves = review_run.probability_moves(ledger, "sun")
    assert len(moves) == 1 and moves[0].startswith("g:")
    log = tmp_path / "log.txt"
    log.write_text("[picks_of_record] games=3\nRun summary\nexit code   : 0 (OK)\n  - [nflverse] warn\n", encoding="utf-8")
    facts = review_run.build_facts("2026_x", "success", log, None)
    assert "[picks_of_record] games=3" in facts and "exit code" in facts and "\n- [nflverse] warn" in facts


def test_review_without_agent_writes_facts(tmp_path):
    out = tmp_path / "r.md"
    review_run.main(["--slot", "2026_x", "--status", "failed", "--out", str(out), "--no-agent"])
    text = out.read_text(encoding="utf-8")
    assert "FAILED" in text and "not run" in text


# ---- grading completed weeks ----

def _grade_args(tmp_path, **kw):
    return _args(tmp_path, **kw)


def test_grading_due_after_last_game_until_window_closes():
    slots = d.build_slots(_schedule())  # week 10 ends with MNF 2026-11-17 01:15 UTC
    mnf = _utc("2026-11-17T01:15:00")
    assert d.due_grading(slots, {}, mnf + timedelta(hours=7)) is None
    assert d.due_grading(slots, {}, mnf + d.GRADE_DELAY) == (2026, 10)
    key = d.grade_key(2026, 10)
    recent = {key: {"status": "pending", "last_attempt": (mnf + d.GRADE_DELAY).isoformat()}}
    assert d.due_grading(slots, recent, mnf + d.GRADE_DELAY + timedelta(minutes=30)) is None  # hourly recheck
    assert d.due_grading(slots, recent, mnf + d.GRADE_DELAY + timedelta(hours=1)) == (2026, 10)
    assert d.due_grading(slots, {key: {"status": "success"}}, mnf + timedelta(hours=10)) is None
    assert d.due_grading(slots, {key: {"failures": d.GRADE_MAX_FAILURES}}, mnf + timedelta(hours=10)) is None
    assert d.due_grading(slots, {}, mnf + d.GRADE_WINDOW + timedelta(minutes=1)) is None


def test_grading_waits_for_final_scores_then_publishes(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "STATE_DIR", tmp_path)
    now = _utc("2026-11-17T10:00:00")
    calls, alerts_sent = [], []

    def runner(rc_grade):
        def run(cmd, **kw):
            calls.append([str(c) for c in cmd])
            if "src.evaluation.live_results" in cmd:
                report = Path(cmd[cmd.index("--report-json") + 1])
                lines = ["week 10: graded 14 games, SU 9-5", "2026 season straight-up: 80-50"]
                report.write_text(json.dumps({"report": "\n".join(lines)}))
                return rc_grade
            return 0
        return run

    state = {}
    alert = lambda *a, **kw: alerts_sent.append(a)
    rec = d.handle_grading(2026, 10, _grade_args(tmp_path), state, now, runner=runner(d.GRADE_EXIT_NOT_FINAL), alert=alert)
    assert rec["status"] == "pending" and not alerts_sent
    assert not any("tools.publish_mtb_csvs" in c for c in calls)  # nothing published before final

    calls.clear()
    rec = d.handle_grading(2026, 10, _grade_args(tmp_path), state, now, runner=runner(0), alert=alert)
    assert rec["status"] == "success" and len(alerts_sent) == 1
    flat = [" ".join(c) for c in calls]
    assert "src.evaluation.picks_of_record --resolve" in flat[0]  # resolve before grading
    assert any("tools.publish_mtb_csvs" in c and "--results-only" in c for c in flat)
    assert any("scripts/publish_site.py" in c for c in flat)
    assert "SU 9-5" in rec["report"]
    assert "80-50" in d.grading_summary(state)


def test_grading_failure_alerts_once_and_skips_publish(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "STATE_DIR", tmp_path)
    alerts_sent, calls = [], []
    state = {}
    for _ in range(2):
        rec = d.handle_grading(2026, 10, _grade_args(tmp_path), state, _utc("2026-11-17T10:00:00"),
                               runner=lambda cmd, **kw: calls.append(cmd) or (1 if "src.evaluation.live_results" in cmd else 0),
                               alert=lambda *a, **kw: alerts_sent.append(a))
    assert rec["status"] == "failed" and rec["failures"] == 2 and len(alerts_sent) == 1
    assert not any("tools.publish_mtb_csvs" in c for c in calls)


class _Clock:
    def __init__(self, start):
        self.now = start

    def __call__(self):
        return self.now


def _attempts(tmp_path, monkeypatch, slot, clock, outcomes, run_minutes=0.2):
    """Run ``run_slot_attempts`` with a fake pipeline returning ``outcomes`` in turn."""
    monkeypatch.setattr(d, "STATE_DIR", tmp_path)
    monkeypatch.setattr(d, "save_state", lambda state, path=None: None)
    rcs = iter(outcomes)
    runs = []

    def runner(cmd, **kw):
        if "tools.weekly_run" in cmd:
            runs.append(clock.now)
            clock.now += timedelta(minutes=run_minutes)
            rc = next(rcs)
            if rc == 0:
                _write_plan(cmd)
            return rc
        return 0

    def sleep(seconds):
        clock.now += timedelta(seconds=seconds)

    state = {}
    res = d.run_slot_attempts(slot, _args(tmp_path), state, runner=runner, clock=clock, sleep=sleep)
    return res, state[slot.slot_id], runs


def test_quick_failure_is_retried_in_the_same_tick(tmp_path, monkeypatch):
    early = d.build_slots(_schedule())[1]  # London 9:30 ET: window 6:00-6:45 ET
    clock = _Clock(_utc("2026-11-15T11:00:00"))
    res, rec, runs = _attempts(tmp_path, monkeypatch, early, clock, [1, 0])
    assert res["status"] == "success" and rec["attempts"] == 2 and len(runs) == 2


def test_quick_retries_stop_at_max_attempts(tmp_path, monkeypatch):
    early = d.build_slots(_schedule())[1]
    clock = _Clock(_utc("2026-11-15T11:00:00"))
    res, rec, runs = _attempts(tmp_path, monkeypatch, early, clock, [1, 1, 1, 1])
    assert res["status"] == "failed" and rec["attempts"] == d.MAX_ATTEMPTS == len(runs)


def test_slow_failure_waits_for_the_next_tick(tmp_path, monkeypatch):
    early = d.build_slots(_schedule())[1]
    clock = _Clock(_utc("2026-11-15T11:00:00"))
    res, rec, runs = _attempts(tmp_path, monkeypatch, early, clock, [1, 0], run_minutes=25)
    assert res["status"] == "failed" and rec["attempts"] == 1 and len(runs) == 1


def test_no_quick_retry_after_the_window_closes(tmp_path, monkeypatch):
    early = d.build_slots(_schedule())[1]
    clock = _Clock(_utc("2026-11-15T11:44:30"))  # 30 s before the last start time
    res, rec, runs = _attempts(tmp_path, monkeypatch, early, clock, [1, 0])
    assert res["status"] == "failed" and len(runs) == 1


# ---- MTB shared publish lock ----

class _FakeLockTime:
    """Stands in for the dispatcher's lock-wait clock/sleep: sleeping only advances the clock."""

    def __init__(self):
        self.start = self.now = time.time()
        self.on_sleep = None  # optional hook, e.g. the other holder releasing the lock

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds
        if self.on_sleep is not None:
            self.on_sleep(self.now - self.start)

    @property
    def waited(self):
        return self.now - self.start


@pytest.fixture(autouse=True)
def lock_time(monkeypatch):
    fake = _FakeLockTime()
    monkeypatch.setattr(d, "_lock_clock", fake.clock)
    monkeypatch.setattr(d, "_lock_sleep", fake.sleep)
    return fake


def _lock(tmp_path):
    return tmp_path / ".publish.lock"


def _other_holder(tmp_path, **kw):
    """Another publish (e.g. MLB) holding MTB's lock."""
    record = {"owner": "mlb-predictions mtb_csvs", "pid": 1, "sport": "mlb", "stale_after_minutes": 90,
              "started_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "token": "mlbtoken"}
    record.update(kw)
    _lock(tmp_path).write_text(json.dumps(record))
    return record


def _locked_runner(tmp_path, calls, rc_for=lambda cmd: 0):
    """Fake runner that records, for every step, whether the lock was held with the token it got."""
    def run(cmd, **kw):
        if "tools.weekly_run" in cmd:
            calls.append(("weekly_run", None))
            return _pipeline_ok(cmd)
        held = json.loads(_lock(tmp_path).read_text()) if _lock(tmp_path).exists() else None
        token = kw.get("env", {}).get(d.PUBLISH_LOCK_TOKEN_ENV)
        calls.append((d._step_name(cmd), held is not None and held["token"] == token and held["sport"] == "nfl"))
        return rc_for(cmd)
    return run


def test_lock_held_from_copy_until_publish_site_finishes(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "STATE_DIR", tmp_path)
    monkeypatch.delenv(d.PUBLISH_LOCK_PATH_ENV, raising=False)
    slot = d.build_slots(_schedule())[2]
    calls = []
    res = d.run_slot(slot, _args(tmp_path), runner=_locked_runner(tmp_path, calls))
    assert res["status"] == "success"
    # every copy/import/publish step ran inside our lock, with its token in MTB_PUBLISH_LOCK_TOKEN
    assert calls == [("weekly_run", None), ("tools.publish_mtb_csvs", True),
                     ("tools.import_mtb_predictions", True), ("scripts/publish_site.py", True)]
    assert not _lock(tmp_path).exists()  # released


def test_lock_record_format_matches_mtb(tmp_path):
    seen = {}

    def run(cmd, **kw):
        seen.update(json.loads(_lock(tmp_path).read_text()))
        return 0

    d.run_locked_publish([{"cmd": ["py", "scripts/publish_site.py"]}], tmp_path, runner=run)
    assert seen["sport"] == "nfl" and seen["stale_after_minutes"] == 90
    assert seen["owner"] == d.PUBLISH_LOCK_OWNER and seen["started_at"].endswith("Z") and seen["token"]


def test_lock_released_when_a_step_fails_or_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "STATE_DIR", tmp_path)
    slot = d.build_slots(_schedule())[2]
    calls = []
    fail_copy = lambda cmd: 2 if "tools.publish_mtb_csvs" in cmd else 0  # noqa: E731
    res = d.run_slot(slot, _args(tmp_path), runner=_locked_runner(tmp_path, calls, rc_for=fail_copy))
    assert res["status"] == "failed" and res["publish_rc"] == 2
    assert "tools.publish_mtb_csvs exited 2" in res["publish_note"]
    assert [c[0] for c in calls] == ["weekly_run", "tools.publish_mtb_csvs"]  # import/publish skipped
    assert not _lock(tmp_path).exists()

    def boom(cmd, **kw):
        assert _lock(tmp_path).exists()
        raise RuntimeError("runner crashed")

    try:
        d.run_locked_publish([{"cmd": COPY_CMD}], tmp_path, runner=boom)
        raise AssertionError("expected the step's exception to propagate")
    except RuntimeError as exc:
        assert "runner crashed" in str(exc)
    assert not _lock(tmp_path).exists()


def test_lock_released_after_a_publish_timeout(tmp_path):
    # _run turns subprocess.TimeoutExpired into 124
    res = d.run_locked_publish([{"cmd": COPY_CMD}, {"cmd": ["py", "scripts/publish_site.py"]}], tmp_path,
                               runner=lambda cmd, **kw: 124 if "scripts/publish_site.py" in cmd else 0)
    assert res["outcome"] == "failed" and res["rc"] == 124 and not _lock(tmp_path).exists()


def test_busy_lock_skips_the_copy_and_keeps_the_other_holders_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "STATE_DIR", tmp_path)
    slot = d.build_slots(_schedule())[2]
    other = _other_holder(tmp_path)
    calls = []
    res = d.run_slot(slot, _args(tmp_path), runner=_locked_runner(tmp_path, calls))
    assert res["status"] == d.PUBLISH_PENDING and "publish lock busy" in res["publish_note"]
    assert calls == [("weekly_run", None)]  # nothing copied into MTB, nothing published
    assert json.loads(_lock(tmp_path).read_text())["token"] == other["token"]


def test_stale_lock_is_taken_over(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "STATE_DIR", tmp_path)
    slot = d.build_slots(_schedule())[2]
    _other_holder(tmp_path, started_at="2026-01-01T00:00:00Z")  # long dead holder
    calls = []
    res = d.run_slot(slot, _args(tmp_path), runner=_locked_runner(tmp_path, calls))
    assert res["status"] == "success" and all(ok for _, ok in calls[1:])
    assert not _lock(tmp_path).exists()


def _pending_then(tmp_path, monkeypatch, clock, publish_rcs):
    """run_slot_attempts on the Sunday slot; publish_site.py returns ``publish_rcs`` in turn."""
    monkeypatch.setattr(d, "STATE_DIR", tmp_path)
    monkeypatch.setattr(d, "save_state", lambda state, path=None: None)
    slot = d.build_slots(_schedule())[2]
    rcs = iter(publish_rcs)
    calls, alerts_sent = [], []
    rc_for = lambda cmd: next(rcs) if "scripts/publish_site.py" in cmd else 0  # noqa: E731
    runner = _locked_runner(tmp_path, calls, rc_for=rc_for)
    state = {}
    alert = lambda *a: alerts_sent.append(a)  # noqa: E731
    first = d.run_slot_attempts(slot, _args(tmp_path), state, runner=runner, clock=clock, sleep=lambda s: None,
                                alert=alert)
    return slot, state, calls, alerts_sent, first, runner, alert


def test_exit_75_defers_the_publish_without_using_an_attempt(tmp_path, monkeypatch):
    clock = _Clock(_utc("2026-11-15T15:00:00"))  # Sunday slot window open, first kick 18:00 UTC
    slot, state, calls, alerts_sent, first, runner, alert = _pending_then(tmp_path, monkeypatch, clock, [75, 0])
    rec = state[slot.slot_id]
    assert first["status"] == d.PUBLISH_PENDING and rec["status"] == d.PUBLISH_PENDING
    assert rec["attempts"] == 0 and not alerts_sent  # no attempt used, no failure email
    assert not _lock(tmp_path).exists()
    assert d.due_slot(d.build_slots(_schedule()), state, clock.now).slot_id == slot.slot_id

    calls.clear()
    clock.now += timedelta(minutes=15)  # next tick
    second = d.run_slot_attempts(slot, _args(tmp_path), state, runner=runner, clock=clock, sleep=lambda s: None,
                                 alert=alert)
    assert second["status"] == "success" and second["publish_retry"]
    assert [c[0] for c in calls] == ["tools.publish_mtb_csvs", "tools.import_mtb_predictions",
                                     "scripts/publish_site.py"]  # pipeline not re-run
    assert state[slot.slot_id]["attempts"] == 0 and len(alerts_sent) == 1  # the success alert


def test_deferred_publish_reruns_the_pipeline_after_a_kickoff(tmp_path, monkeypatch):
    clock = _Clock(_utc("2026-11-15T17:50:00"))  # 10 min before the 1 pm ET game
    slot, state, calls, _, first, runner, alert = _pending_then(tmp_path, monkeypatch, clock, [75, 0])
    assert first["status"] == d.PUBLISH_PENDING
    calls.clear()
    clock.now = _utc("2026-11-15T18:05:00")  # that game has kicked off; SNF still ahead
    res = d.run_slot_attempts(slot, _args(tmp_path), state, runner=runner, clock=clock, sleep=lambda s: None,
                              alert=alert)
    assert res["status"] == "success" and calls[0] == ("weekly_run", None)
    assert state[slot.slot_id]["attempts"] == 1


def test_exit_77_is_a_failure_with_an_alert(tmp_path, monkeypatch):
    clock = _Clock(_utc("2026-11-15T15:00:00"))
    slot, state, calls, alerts_sent, first, _, _ = _pending_then(tmp_path, monkeypatch, clock, [77, 77, 77])
    rec = state[slot.slot_id]
    assert first["status"] == "failed" and rec["publish_rc"] == 77 and "exit 77" in rec["publish_note"]
    assert rec["attempts"] >= 1 and alerts_sent and not _lock(tmp_path).exists()


def test_deferred_publish_that_outlives_its_window_fails_with_an_alert(monkeypatch):
    monkeypatch.setattr(d, "save_state", lambda state, path=None: None)
    slots = d.build_slots(_schedule())
    sun = slots[2]
    state = {sun.slot_id: {"status": d.PUBLISH_PENDING, "attempts": 0, "publish_note": "publish lock busy"}}
    assert d.expired_deferred_publishes(slots, state, _utc("2026-11-15T21:00:00")) == []  # window still open
    late = _utc("2026-11-15T23:00:00")  # every Sunday game < 2h45 away
    expired = d.expired_deferred_publishes(slots, state, late)
    assert expired == [sun]
    alerts_sent = []
    d.fail_expired_publishes(expired, state, alert=lambda *a: alerts_sent.append(a))
    assert state[sun.slot_id]["status"] == "failed" and "never published" in state[sun.slot_id]["publish_note"]
    assert len(alerts_sent) == 1


def _grade_runner(calls, rc_for=lambda cmd: 0):
    def run(cmd, **kw):
        calls.append([str(c) for c in cmd])
        if "src.evaluation.live_results" in cmd:
            Path(cmd[cmd.index("--report-json") + 1]).write_text(json.dumps({"report": "week 10: graded"}))
            return 0
        return rc_for(cmd)
    return run


def test_grading_busy_lock_retries_next_tick_without_a_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "STATE_DIR", tmp_path)
    now = _utc("2026-11-17T10:00:00")
    _other_holder(tmp_path)
    calls, alerts_sent = [], []
    alert = lambda *a, **kw: alerts_sent.append(a)  # noqa: E731

    state = {}
    rec = d.handle_grading(2026, 10, _grade_args(tmp_path), state, now, runner=_grade_runner(calls), alert=alert)
    assert rec["status"] == d.PUBLISH_PENDING and not rec.get("failures") and not alerts_sent
    assert not any("tools.publish_mtb_csvs" in " ".join(c) for c in calls)
    slots = d.build_slots(_schedule())
    assert d.due_grading(slots, state, now + timedelta(minutes=15)) == (2026, 10)  # no hourly wait

    _lock(tmp_path).unlink()
    calls.clear()
    rec = d.handle_grading(2026, 10, _grade_args(tmp_path), state, now + timedelta(minutes=15),
                           runner=_grade_runner(calls), alert=alert)
    assert rec["status"] == "success" and "publish_note" not in rec and len(alerts_sent) == 1
    flat = [" ".join(c) for c in calls]
    assert any("tools.publish_mtb_csvs" in c and "--results-only" in c for c in flat)
    assert not _lock(tmp_path).exists()


def test_grading_exit_77_fails_with_an_alert(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "STATE_DIR", tmp_path)
    calls, alerts_sent = [], []
    runner = _grade_runner(calls, rc_for=lambda cmd: 77 if "scripts/publish_site.py" in cmd else 0)
    rec = d.handle_grading(2026, 10, _grade_args(tmp_path), {}, _utc("2026-11-17T10:00:00"), runner=runner,
                           alert=lambda *a, **kw: alerts_sent.append(a))
    assert rec["status"] == "failed" and rec["failures"] == 1 and rec["publish_rc"] == 77
    assert len(alerts_sent) == 1 and not _lock(tmp_path).exists()


def test_grading_exit_75_is_pending_not_a_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "STATE_DIR", tmp_path)
    calls, alerts_sent = [], []
    runner = _grade_runner(calls, rc_for=lambda cmd: 75 if "scripts/publish_site.py" in cmd else 0)
    rec = d.handle_grading(2026, 10, _grade_args(tmp_path), {}, _utc("2026-11-17T10:00:00"), runner=runner,
                           alert=lambda *a, **kw: alerts_sent.append(a))
    assert rec["status"] == d.PUBLISH_PENDING and not rec.get("failures") and not alerts_sent


def test_lock_wait_defaults_to_300s_and_is_configurable():
    assert d.PUBLISH_LOCK_WAIT_SEC == 300
    assert d.parse_args([]).publish_lock_wait_sec == 300
    assert d.parse_args(["--publish-lock-wait-sec", "60"]).publish_lock_wait_sec == 60


def test_busy_lock_that_frees_within_the_wait_publishes_in_the_same_tick(tmp_path, monkeypatch, lock_time):
    monkeypatch.setattr(d, "STATE_DIR", tmp_path)
    slot = d.build_slots(_schedule())[2]
    _other_holder(tmp_path)
    # MLB's site build finishes ~2 minutes into our wait
    lock_time.on_sleep = lambda waited: _lock(tmp_path).unlink() if waited >= 120 and _lock(tmp_path).exists() else None
    calls = []
    res = d.run_slot(slot, _args(tmp_path), runner=_locked_runner(tmp_path, calls))
    assert res["status"] == "success" and "publish_note" not in res
    assert calls == [("weekly_run", None), ("tools.publish_mtb_csvs", True),
                     ("tools.import_mtb_predictions", True), ("scripts/publish_site.py", True)]
    assert 120 <= lock_time.waited < d.PUBLISH_LOCK_WAIT_SEC
    assert not _lock(tmp_path).exists()


def test_lock_busy_longer_than_the_wait_defers_without_copying(tmp_path, monkeypatch, lock_time):
    monkeypatch.setattr(d, "STATE_DIR", tmp_path)
    monkeypatch.setattr(d, "save_state", lambda state, path=None: None)
    slot = d.build_slots(_schedule())[2]
    other = _other_holder(tmp_path)
    calls, alerts_sent, state = [], [], {}
    res = d.run_slot_attempts(slot, _args(tmp_path), state, runner=_locked_runner(tmp_path, calls),
                              clock=_Clock(_utc("2026-11-15T15:00:00")), sleep=lambda s: None,
                              alert=lambda *a: alerts_sent.append(a))
    assert res["status"] == d.PUBLISH_PENDING and "publish lock busy" in res["publish_note"]
    assert state[slot.slot_id]["attempts"] == 0 and not alerts_sent
    assert calls == [("weekly_run", None)]  # nothing copied into MTB
    assert json.loads(_lock(tmp_path).read_text()) == other  # other holder's lock untouched
    assert d.PUBLISH_LOCK_WAIT_SEC <= lock_time.waited < d.PUBLISH_LOCK_WAIT_SEC + 10


def test_lock_wait_comes_from_args(tmp_path, monkeypatch, lock_time):
    monkeypatch.setattr(d, "STATE_DIR", tmp_path)
    slot = d.build_slots(_schedule())[2]
    _other_holder(tmp_path)
    res = d.run_slot(slot, _args(tmp_path, publish_lock_wait_sec=0), runner=_locked_runner(tmp_path, []))
    assert res["status"] == d.PUBLISH_PENDING and lock_time.waited == 0


def test_grading_waits_for_the_lock_too(tmp_path, monkeypatch, lock_time):
    monkeypatch.setattr(d, "STATE_DIR", tmp_path)
    _other_holder(tmp_path)
    lock_time.on_sleep = lambda waited: _lock(tmp_path).unlink() if waited >= 60 and _lock(tmp_path).exists() else None
    calls = []
    rec = d.handle_grading(2026, 10, _grade_args(tmp_path), {}, _utc("2026-11-17T10:00:00"),
                           runner=_grade_runner(calls), alert=lambda *a, **kw: None)
    assert rec["status"] == "success" and 60 <= lock_time.waited < d.PUBLISH_LOCK_WAIT_SEC
    assert any("tools.publish_mtb_csvs" in " ".join(c) for c in calls) and not _lock(tmp_path).exists()
