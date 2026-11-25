"""ESPN player news fetch: retries, failure reasons and the run-log summary line (no network)."""

from collections import Counter

import requests

from src.data import espn_player_news as news
from src.utils.run_log import is_issue_line


class _Resp:
    def __init__(self, status, payload=None, bad_json=False):
        self.status_code = status
        self._payload = payload
        self._bad_json = bad_json

    def json(self):
        if self._bad_json:
            raise ValueError("not json")
        return self._payload


class _Session:
    """Returns (or raises) the queued outcomes in order and records each call's params."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append(params)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


FEED = {"feed": [{"id": 11, "headline": "Limited in practice", "published": "2026-11-20T18:00:00Z"}]}


def _fetch(session):
    return news._fetch_player_news_result(123, 4, session=session, retry_delay=0)


def test_success_returns_rows_and_no_error():
    session = _Session(_Resp(200, FEED))
    rows, error = _fetch(session)
    assert error is None
    assert [r["news_id"] for r in rows] == [11]
    assert rows[0]["player_id"] == 123
    assert session.calls == [{"limit": 4, "playerId": "123"}]


def test_empty_feed_is_a_success_not_a_failure():
    rows, error = _fetch(_Session(_Resp(200, {"feed": []})))
    assert (rows, error) == ([], None)


def test_rate_limit_is_retried_once_then_succeeds():
    session = _Session(_Resp(429), _Resp(200, FEED))
    rows, error = _fetch(session)
    assert error is None and len(rows) == 1
    assert len(session.calls) == 2


def test_timeout_is_retried_once_then_reported():
    session = _Session(requests.Timeout("slow"), requests.Timeout("slow"))
    rows, error = _fetch(session)
    assert (rows, error) == ([], "Timeout")
    assert len(session.calls) == 2


def test_persistent_server_error_reports_the_status():
    session = _Session(_Resp(503), _Resp(503))
    assert _fetch(session) == ([], "HTTP 503")


def test_client_error_is_not_retried():
    session = _Session(_Resp(404))
    assert _fetch(session) == ([], "HTTP 404")
    assert len(session.calls) == 1


def test_bad_json_is_a_failure():
    assert _fetch(_Session(_Resp(200, bad_json=True))) == ([], "invalid JSON")


def test_legacy_wrapper_still_returns_rows(monkeypatch):
    monkeypatch.setattr(news, "_thread_session", lambda: _Session(_Resp(200, FEED)))
    assert [r["news_id"] for r in news._fetch_player_news(123, 4)] == [11]


def test_summary_without_failures_is_not_a_run_issue():
    line = news._failure_summary(2026, 2154, Counter(), 95.0, 8)
    assert "all 2154 player requests succeeded" in line
    assert not is_issue_line(line)


def test_summary_with_failures_is_a_run_issue():
    line = news._failure_summary(2026, 2154, Counter({"HTTP 429": 3, "Timeout": 1}), 95.0, 8)
    assert "4 of 2154 player requests failed (HTTP 429: 3, Timeout: 1)" in line
    assert is_issue_line(line)
