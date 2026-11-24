from __future__ import annotations

import sys

import pandas as pd
import pytest

import src.data.weather as weather_module


def test_apply_osm_coords_prefers_matching_city_state_for_duplicate_stadium_names(monkeypatch):
    catalog = pd.DataFrame(
        [
            {
                "name": "Highmark Stadium",
                "city": None,
                "state": None,
                "country": None,
                "latitude": 40.436238,
                "longitude": -80.009596,
            },
            {
                "name": "Highmark Stadium",
                "city": "Orchard Park",
                "state": "NY",
                "country": None,
                "latitude": 42.773750,
                "longitude": -78.786978,
                "wikidata": "Q1475645",
            },
        ]
    )
    stadiums = {
        "BUF": {"city": "Orchard Park, NY", "lat": 42.0, "lon": -78.0},
    }

    monkeypatch.setattr(weather_module, "_load_osm_stadium_catalog", lambda refresh=False: catalog)

    weather_module._apply_osm_coords_to_stadiums(stadiums)

    assert stadiums["BUF"]["lat"] == 42.773750
    assert stadiums["BUF"]["lon"] == -78.786978


def test_apply_osm_coords_scores_all_aliases_before_selecting(monkeypatch):
    catalog = pd.DataFrame(
        [
            {
                "name": "FirstEnergy Stadium",
                "city": "Reading",
                "state": "PA",
                "country": None,
                "latitude": 40.365519,
                "longitude": -75.933722,
            },
            {
                "name": "Huntington Bank Field",
                "city": "Cleveland",
                "state": "OH",
                "country": None,
                "latitude": 41.506052,
                "longitude": -81.699609,
            },
        ]
    )
    stadiums = {
        "CLE": {"city": "Cleveland, OH", "lat": 41.0, "lon": -81.0},
    }

    monkeypatch.setattr(weather_module, "_load_osm_stadium_catalog", lambda refresh=False: catalog)

    weather_module._apply_osm_coords_to_stadiums(stadiums)

    assert stadiums["CLE"]["lat"] == 41.506052
    assert stadiums["CLE"]["lon"] == -81.699609


def test_tomorrow_forecast_hourly_records_non_retryable_error(monkeypatch):
    class FakeResponse:
        status_code = 401
        reason = "Unauthorized"
        text = '{"message":"bad key","apikey":"secret"}'

    weather_module.WEATHER_REQUEST_DIAGNOSTICS.clear()
    monkeypatch.setattr(weather_module.requests, "get", lambda *args, **kwargs: FakeResponse())

    result = weather_module._tomorrow_forecast_hourly(
        42.7738,
        -78.7868,
        "secret",
        retries=1,
        context="2026 wk8 BAL@BUF 2026_08_BAL_BUF",
    )

    assert result is None
    assert weather_module.WEATHER_REQUEST_DIAGNOSTICS == [
        {
            "timesteps": "hourly",
            "context": "2026 wk8 BAL@BUF 2026_08_BAL_BUF",
            "attempt": 1,
            "status_code": 401,
            "reason": "Unauthorized",
            "rows": 0,
            "retried": False,
            "message": '{"message":"bad key","apikey":"***"}',
        }
    ]


def test_filter_forecast_window_limits_tomorrow_live_lookahead():
    now = pd.Timestamp("2026-10-31T19:00:00Z")
    sched = pd.DataFrame(
        [
            {"game_id": "recent", "_kickoff": now - pd.Timedelta(hours=5, minutes=59)},
            {"game_id": "too_old", "_kickoff": now - pd.Timedelta(hours=6, minutes=1)},
            {"game_id": "current_week", "_kickoff": now + pd.Timedelta(days=2)},
            {"game_id": "too_far", "_kickoff": now + pd.Timedelta(days=8)},
            {"game_id": "missing", "_kickoff": pd.NaT},
        ]
    )

    filtered = weather_module._filter_forecast_window(
        sched,
        now=now,
        lookback_hours=6,
        forecast_days=7,
    )

    assert filtered["game_id"].tolist() == ["recent", "current_week"]


def test_main_preserves_existing_weather_cache_when_live_refresh_is_empty(tmp_path, monkeypatch, capsys):
    proc_dir = tmp_path / "processed"
    proc_dir.mkdir()
    cache_path = proc_dir / "weather_games.parquet"
    cached = pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 8,
                "game_id": "2026_08_BAL_BUF",
                "home_team": "BUF",
                "away_team": "BAL",
            }
        ]
    )
    cached.to_parquet(cache_path, index=False)

    monkeypatch.setattr(weather_module, "PROC_DIR", proc_dir)
    monkeypatch.setattr(
        weather_module,
        "build_game_weather",
        lambda seasons, api_key=None, debug=False, forecast_days=7.0, lookback_hours=6.0: pd.DataFrame(),
    )
    monkeypatch.setattr(sys, "argv", ["weather.py", "--season", "2026", "--debug"])

    weather_module.main()

    output = capsys.readouterr().out
    assert "No weather rows built from live Tomorrow.io refresh." in output
    assert "preserved existing cache rows=1" in output
    assert cache_path.exists()
    assert len(pd.read_parquet(cache_path)) == 1


class _Resp:
    def __init__(self, status_code, headers=None, payload=None):
        self.status_code = status_code
        self.reason = {200: "OK", 429: "Too Many Requests"}.get(status_code, "")
        self.headers = headers or {}
        self.text = '{"code":429001}'
        self._payload = payload

    def json(self):
        return self._payload


def test_tomorrow_calls_are_spaced_out(monkeypatch):
    clock = {"t": 100.0}
    sleeps = []
    monkeypatch.setattr(weather_module.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(weather_module.time, "sleep", lambda s: (sleeps.append(s), clock.__setitem__("t", clock["t"] + s)))
    monkeypatch.setattr(weather_module.requests, "get", lambda *a, **k: _Resp(200))
    monkeypatch.setattr(weather_module, "_tomorrow_last_call", 0.0)
    weather_module._tomorrow_get("u", {})
    weather_module._tomorrow_get("u", {})  # immediately after: waits out the interval
    assert sleeps == [pytest.approx(weather_module.TOMORROW_MIN_INTERVAL_S)]


def test_retry_delay_honours_retry_after_then_backs_off():
    assert weather_module._retry_delay(_Resp(429, {"Retry-After": "3"}), 0, 0.6) == 3.0
    assert weather_module._retry_delay(_Resp(429, {"Retry-After": "120"}), 0, 0.6) == weather_module.TOMORROW_429_MAX_WAIT_S
    assert [weather_module._retry_delay(_Resp(429), i, 0.6) for i in range(3)] == [1.0, 2.0, 4.0]
    assert weather_module._retry_delay(_Resp(503), 1, 0.6) == pytest.approx(1.2)


def test_hourly_recovers_after_repeated_429s(monkeypatch):
    responses = iter([_Resp(429), _Resp(429), _Resp(429),
                      _Resp(200, payload={"timelines": {"hourly": [{"time": "2026-11-30T18:00:00Z",
                                                                    "values": {"temperature": 40}}]}})])
    waits = []
    monkeypatch.setattr(weather_module, "_tomorrow_get", lambda url, params: next(responses))
    monkeypatch.setattr(weather_module.time, "sleep", lambda s: waits.append(s))
    weather_module.WEATHER_REQUEST_DIAGNOSTICS.clear()
    out = weather_module._tomorrow_forecast_hourly(40.0, -80.0, "k", context="g")
    assert out is not None and len(out) == 1  # 4th attempt: the 11-23 failure gave up after 3 quick tries
    assert waits == [1.0, 2.0, 4.0]


def test_hourly_skip_threshold_matches_pick_tolerance():
    # beyond horizon + tolerance the hourly forecast can never give a row within tolerance
    assert weather_module.TOMORROW_HOURLY_SKIP_AFTER_H > (weather_module.TOMORROW_HOURLY_HORIZON_H
                                                          + weather_module.HOURLY_MATCH_TOLERANCE_H)
