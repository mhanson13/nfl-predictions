from __future__ import annotations

import pytest

from src.data.odds_api_historical import (
    OddsApiError,
    _build_event_markets_output_path,
    _build_event_odds_output_path,
    _build_events_output_path,
    _error_hint,
    _flatten_event_markets,
    _flatten_event_odds,
    _flatten_historical_events,
    _request_with_retry,
    _redact_url,
)


def test_build_events_output_path_uses_snapshot_label():
    path = _build_events_output_path("2024-09-08T16:55:00Z")

    assert path.name == "oddsapi_historical_events_20240908T165500Z.parquet"


def test_build_event_odds_output_path_uses_week_label_when_available():
    path = _build_event_odds_output_path(
        event_id="evt-abc-123",
        date="2024-09-08T16:55:00Z",
        season=2024,
        week=1,
    )

    assert path.name == "oddsapi_historical_player_props_2024_wk01_evtabc123_20240908T165500Z.parquet"


def test_build_event_markets_output_path_uses_week_label_when_available():
    path = _build_event_markets_output_path(
        event_id="evt-abc-123",
        date="2024-09-08T16:55:00Z",
        season=2024,
        week=1,
    )

    assert path.name == "oddsapi_historical_event_markets_2024_wk01_evtabc123_20240908T165500Z.parquet"


def test_flatten_historical_events_preserves_snapshot_metadata():
    payload = {
        "timestamp": "2024-09-08T16:50:00Z",
        "previous_timestamp": "2024-09-08T16:45:00Z",
        "next_timestamp": "2024-09-08T16:55:00Z",
        "data": [
            {
                "id": "evt-1",
                "sport_key": "americanfootball_nfl",
                "sport_title": "NFL",
                "commence_time": "2024-09-08T17:00:00Z",
                "home_team": "Atlanta Falcons",
                "away_team": "Carolina Panthers",
            }
        ],
    }

    frame = _flatten_historical_events(payload, requested_date="2024-09-08T16:55:00Z")

    assert len(frame) == 1
    assert frame["snapshot_timestamp"].iloc[0] == "2024-09-08T16:50:00Z"
    assert frame["event_id"].iloc[0] == "evt-1"
    assert frame["home_team"].iloc[0] == "Atlanta Falcons"


def test_flatten_event_odds_matches_player_prop_shape():
    payload = {
        "timestamp": "2024-09-08T16:50:00Z",
        "previous_timestamp": "2024-09-08T16:45:00Z",
        "next_timestamp": "2024-09-08T16:55:00Z",
        "data": {
            "id": "evt-1",
            "sport_key": "americanfootball_nfl",
            "sport_title": "NFL",
            "commence_time": "2024-09-08T17:00:00Z",
            "home_team": "Atlanta Falcons",
            "away_team": "Carolina Panthers",
            "bookmakers": [
                {
                    "key": "draftkings",
                    "title": "DraftKings",
                    "last_update": "2024-09-08T16:50:00Z",
                    "markets": [
                        {
                            "key": "player_pass_yds",
                            "last_update": "2024-09-08T16:50:00Z",
                            "outcomes": [
                                {
                                    "name": "Over",
                                    "description": "QB One",
                                    "price": -110,
                                    "point": 245.5,
                                },
                                {
                                    "name": "Under",
                                    "description": "QB One",
                                    "price": -110,
                                    "point": 245.5,
                                },
                            ],
                        }
                    ],
                }
            ],
        },
    }

    frame = _flatten_event_odds(
        payload,
        requested_date="2024-09-08T16:55:00Z",
        requested_markets=["player_pass_yds"],
        requested_bookmakers=["draftkings"],
        requested_regions=["us"],
        quota_headers={
            "x_requests_remaining": "100",
            "x_requests_used": "20",
            "x_requests_last": "10",
        },
    )

    assert len(frame) == 2
    assert set(frame["outcome_name"]) == {"Over", "Under"}
    assert frame["event_id"].iloc[0] == "evt-1"
    assert frame["bookmaker_key"].iloc[0] == "draftkings"
    assert frame["market_key"].iloc[0] == "player_pass_yds"
    assert frame["player_name"].iloc[0] == "QB One"
    assert frame["requested_markets"].iloc[0] == "player_pass_yds"
    assert frame["x_requests_last"].iloc[0] == "10"


def test_flatten_event_markets_matches_market_discovery_shape():
    payload = {
        "timestamp": "2024-09-08T16:50:00Z",
        "previous_timestamp": "2024-09-08T16:45:00Z",
        "next_timestamp": "2024-09-08T16:55:00Z",
        "data": {
            "id": "evt-1",
            "sport_key": "americanfootball_nfl",
            "sport_title": "NFL",
            "commence_time": "2024-09-08T17:00:00Z",
            "home_team": "Atlanta Falcons",
            "away_team": "Carolina Panthers",
            "bookmakers": [
                {
                    "key": "draftkings",
                    "title": "DraftKings",
                    "markets": [
                        {"key": "player_pass_yds", "last_update": "2024-09-08T16:50:00Z"},
                        {"key": "player_rush_yds", "last_update": "2024-09-08T16:50:00Z"},
                    ],
                }
            ],
        },
    }

    frame = _flatten_event_markets(
        payload,
        requested_date="2024-09-08T16:55:00Z",
        requested_bookmakers=["draftkings"],
        requested_regions=["us"],
        quota_headers={
            "x_requests_remaining": "100",
            "x_requests_used": "20",
            "x_requests_last": "1",
        },
    )

    assert len(frame) == 2
    assert set(frame["market_key"]) == {"player_pass_yds", "player_rush_yds"}
    assert frame["event_id"].iloc[0] == "evt-1"
    assert frame["bookmaker_key"].iloc[0] == "draftkings"
    assert frame["requested_bookmakers"].iloc[0] == "draftkings"
    assert frame["x_requests_last"].iloc[0] == "1"


def test_error_helpers_redact_api_key_and_explain_paid_plan():
    url = "https://api.the-odds-api.com/v4/historical/sports/nfl/events?apiKey=secret123&date=2024"
    body = '{"error_code":"HISTORICAL_UNAVAILABLE_ON_FREE_USAGE_PLAN"}'

    assert "secret123" not in _redact_url(url)
    assert "apiKey=***" in _redact_url(url)
    assert "paid Odds API usage plan" in _error_hint(body)


class _FakeResponse:
    status_code = 401
    reason = "Unauthorized"
    url = "https://api.the-odds-api.com/v4/historical/events?apiKey=secret123&date=2024"
    text = '{"error_code":"HISTORICAL_UNAVAILABLE_ON_FREE_USAGE_PLAN"}'
    headers: dict[str, str] = {}


class _FakeSession:
    def get(self, *_args, **_kwargs):
        return _FakeResponse()


def test_request_with_retry_does_not_leak_api_key_on_http_error():
    with pytest.raises(OddsApiError) as exc_info:
        _request_with_retry(
            _FakeSession(),
            "/historical/events",
            params={"apiKey": "secret123"},
            retries=0,
            sleep=0,
            timeout=1,
        )

    message = str(exc_info.value)
    assert "secret123" not in message
    assert "apiKey=***" in message
    assert "paid Odds API usage plan" in message
