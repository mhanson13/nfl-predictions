from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.data.balldontlie import (
    FEEDS,
    _build_params,
    _cached_feed_has_rows,
    _fetch_payload_records,
)
from src.utils.io import write_df


def test_cached_feed_has_rows_rejects_empty_cache(tmp_path):
    path = tmp_path / "empty.parquet"
    write_df(pd.DataFrame(), path)

    assert _cached_feed_has_rows(path) is False


def test_cached_feed_has_rows_accepts_nonempty_cache(tmp_path):
    path = tmp_path / "nonempty.parquet"
    write_df(pd.DataFrame({"row": [1]}), path)

    assert _cached_feed_has_rows(path) is True


def test_build_params_uses_balldontlie_array_names():
    params = _build_params(
        FEEDS["games"],
        season=2026,
        week=2,
        season_types=[2],
        team_ids=[16, 17],
    )

    assert ("seasons[]", 2026) in params
    assert ("weeks[]", 2) in params
    assert ("season_types[]", 2) in params
    assert ("team_ids[]", 16) in params
    assert ("team_ids[]", 17) in params


@dataclass
class _FakeResponse:
    payload: dict
    url: str = "https://api.balldontlie.io/nfl/v1/players"
    status_code: int = 200
    text: str = "{}"

    def json(self):
        return self.payload

    def raise_for_status(self):
        return None


class _FakeSession:
    def __init__(self):
        self.calls = 0

    def get(self, url, params, timeout):
        self.calls += 1
        if self.calls == 1:
            return _FakeResponse({"data": [{"id": 1}], "meta": {"next_cursor": 2}})
        return _FakeResponse({"data": [{"id": 2}], "meta": {"next_cursor": None}})


def test_fetch_payload_records_follows_next_cursor():
    session = _FakeSession()

    records = _fetch_payload_records(
        session,
        FEEDS["players"],
        params=[],
        per_page=100,
        retries=0,
        sleep=0,
        timeout=30,
    )

    assert records == [{"id": 1}, {"id": 2}]
    assert session.calls == 2
