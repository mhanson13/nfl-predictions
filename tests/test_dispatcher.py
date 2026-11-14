import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

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
    assert d.due_slot(slots, {sun: {"status": "failed", "attempts": 1}}, now).slot_id == sun  # one retry
    assert d.due_slot(slots, {sun: {"status": "failed", "attempts": 2}}, now) is None
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
    assert not any("publish_site.py" in " ".join(map(str, c)) for c in calls)

    calls.clear()
    res = d.run_slot(slot, _args(tmp_path), runner=lambda cmd, **kw: calls.append(cmd) or 0)
    assert res["status"] == "success" and res["publish_rc"] == 0
    assert any("scripts/publish_site.py" in c for c in calls[1])


def test_publish_failure_marks_slot_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "STATE_DIR", tmp_path)
    slot = d.build_slots(_schedule())[2]
    res = d.run_slot(slot, _args(tmp_path), runner=lambda cmd, **kw: 1 if "scripts/publish_site.py" in cmd else 0)
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
