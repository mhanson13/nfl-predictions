from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.data.balldontlie import (
    FEEDS,
    _build_params,
    _cached_feed_has_rows,
    _fetch_player_prop_records,
    _fetch_payload_records,
    _resolve_jobs,
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


def test_build_params_uses_team_stats_array_names():
    params = _build_params(
        FEEDS["team_stats"],
        season=2026,
        week=None,
        season_types=[2],
        team_ids=[16],
    )

    assert ("seasons[]", 2026) in params
    assert ("season_types[]", 2) in params
    assert ("team_ids[]", 16) in params
    assert ("seasons", 2026) not in params
    assert ("team_ids", 16) not in params


def test_build_params_uses_team_season_stats_team_id_array_name():
    params = _build_params(
        FEEDS["team_season_stats"],
        season=2026,
        week=None,
        season_types=[2],
        team_ids=[16],
    )

    assert ("season", 2026) in params
    assert ("season_types[]", 2) in params
    assert ("team_ids[]", 16) in params
    assert ("team_ids", 16) not in params


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


class _PropsSession:
    def __init__(self):
        self.params = []

    def get(self, url, params, timeout):
        self.params.append(list(params))
        return _FakeResponse(
            {
                "data": [
                    {
                        "game_id": 101,
                        "player_id": 55,
                        "vendor": "draftkings",
                        "prop_type": "passing_yards",
                        "line_value": "245.5",
                    }
                ],
                "meta": {"per_page": 1},
            },
            url=url,
        )


def test_fetch_player_prop_records_uses_game_id_and_vendor_array():
    session = _PropsSession()

    records = _fetch_player_prop_records(
        session,
        game_ids=[101],
        vendors=["draftkings"],
        prop_types=[],
        retries=0,
        sleep=0,
        timeout=30,
    )

    assert records[0]["prop_type"] == "passing_yards"
    assert ("game_id", 101) in session.params[0]
    assert ("vendors[]", "draftkings") in session.params[0]


def test_resolve_jobs_keeps_week_for_player_props_feed():
    jobs = _resolve_jobs(["player_props"], [2026], [3])

    assert jobs == [("player_props", 2026, 3)]
