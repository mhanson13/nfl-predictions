"""A blog post that fails to publish is a heads-up in the alert, never a run failure."""
from datetime import datetime, timezone

from test_dispatcher import _args, _schedule
from tools import dispatcher as d
from tools import review_run

# logs/dispatcher/grade_2026_wk10_publish.txt lines 11-17 (Week 10 grading, 2026-11-17).
GRADE_WK10_PUBLISH = """\
> mattythebookie@0.1.0 blog:auto
> tsx scripts/blog-run.ts auto

[blog] due: recap 2026 week 10 (14 of 14 games graded)
[blog] writing recap for 2026 week 10: 14 games, 20 news items
[blog] recap week 10 failed: terminated

> mattythebookie@0.1.0 lint
"""
# logs/dispatcher/grade_2026_wk10_recap_republish.txt: the successful retry.
REPUBLISH = """\
[blog] due: recap 2026 week 10 (14 of 14 games graded)
[blog] attempt 1 needed fixes: percentage not in the data: 68%
[blog] published "Week 10 NFL Model Recap" -> /blog/2026-week-10-nfl-picks-recap (2 attempt(s), 106s)
"""
NOT_PUBLISHED = "[blog] preview week 11 NOT published; failed checks (1200 in / 800 out tokens):\n  - spread not in the data\n"


def _log(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_finds_failed_and_not_published_lines_only(tmp_path):
    assert review_run.blog_failures(_log(tmp_path, "g.txt", GRADE_WK10_PUBLISH)) == [
        "[blog] recap week 10 failed: terminated"]
    assert review_run.blog_failures(_log(tmp_path, "r.txt", REPUBLISH)) == []
    assert review_run.blog_failures(_log(tmp_path, "n.txt", NOT_PUBLISHED)) == [
        "[blog] preview week 11 NOT published; failed checks (1200 in / 800 out tokens):"]
    assert review_run.blog_failures(tmp_path / "missing.txt") == [] and review_run.blog_failures(None) == []


def _capture(monkeypatch):
    sent = []
    monkeypatch.setattr("src.utils.alerts.send_alert", lambda subject, body, **kw: sent.append((subject, body)))
    monkeypatch.setattr(d, "load_state", lambda path=None: {})
    return sent


def test_grading_alert_flags_the_week10_recap_failure(tmp_path, monkeypatch):
    sent = _capture(monkeypatch)
    log = _log(tmp_path, "grade_2026_wk10_publish.txt", GRADE_WK10_PUBLISH)
    d.send_grading_alert(2026, 10, {"status": "success", "grade_rc": 0, "publish_rc": 0, "publish_log": str(log)},
                         final=False)
    subject, body = sent[0]
    assert subject == "[NFL pipeline] RESULTS (blog post failed): 2026 week 10"
    assert "[blog] recap week 10 failed: terminated" in body and "heads-up only" in body


def test_slot_alert_flags_blog_failure_and_clean_runs_stay_plain(tmp_path, monkeypatch):
    sent = _capture(monkeypatch)
    slot = d.build_slots(_schedule())[2]
    bad = _log(tmp_path, "bad.txt", NOT_PUBLISHED)
    d.send_slot_alert(slot, {"status": "success", "pipeline_rc": 0, "publish_rc": 0, "publish_log": str(bad)}, None)
    good = _log(tmp_path, "good.txt", REPUBLISH)
    d.send_slot_alert(slot, {"status": "success", "pipeline_rc": 0, "publish_rc": 0, "publish_log": str(good)}, None)
    assert sent[0][0] == f"[NFL pipeline] SUCCESS (blog post failed): {slot.slot_id}"
    assert "NOT published; failed checks" in sent[0][1]
    assert sent[1][0] == f"[NFL pipeline] SUCCESS: {slot.slot_id}" and "Blog post problem" not in sent[1][1]


def test_blog_failure_does_not_change_status_or_retries(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "STATE_DIR", tmp_path)
    alerts_sent = []

    def runner(cmd, **kw):
        if "scripts/publish_site.py" in cmd:
            kw["log"].write_text(GRADE_WK10_PUBLISH, encoding="utf-8")  # MTB exits 0 after a blog failure
        return 0

    state = {}
    rec = d.handle_grading(2026, 10, _args(tmp_path), state, datetime(2026, 11, 17, 10, tzinfo=timezone.utc),
                           runner=runner, alert=lambda *a, **kw: alerts_sent.append(a))
    assert rec["status"] == "success" and not rec.get("failures")
    assert d.due_grading(d.build_slots(_schedule()), state, datetime(2026, 11, 17, 11, tzinfo=timezone.utc)) is None
    assert len(alerts_sent) == 1


def test_review_facts_list_the_blog_failure(tmp_path):
    log = _log(tmp_path, "pub.txt", GRADE_WK10_PUBLISH)
    facts = review_run.build_facts("2026_wk10_Mon_2026-11-16", "success", None, log)
    assert "## Blog post failed" in facts and "recap week 10 failed: terminated" in facts
