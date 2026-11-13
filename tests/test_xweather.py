from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.data import xweather

KICK = datetime(2024, 12, 22, 18, 0, tzinfo=timezone.utc)


def _payload():
    periods = [
        {"timestamp": (KICK + timedelta(hours=h)).timestamp(), "tempF": 30 + h, "windSpeedMPH": 10 + h,
         "windGustMPH": 20, "precipIN": 0.1 if h == 0 else 0.0, "snowIN": 0.0, "humidity": 80, "sky": 90,
         "weatherPrimary": "Light Snow"}
        for h in range(-4, 5)
    ]
    return {"success": True, "error": None, "response": [{"loc": {}, "periods": periods}]}


def test_aggregate_window_matches_visualcrossing_style_columns():
    agg = xweather.aggregate_window(xweather.periods_from_payload(_payload()), KICK, window_hours=3)
    assert agg["xw_hours"] == 7  # -3..+3
    assert agg["xw_temp_kickoff_f"] == 30
    assert agg["xw_temp_f"] == pytest.approx(30.0)
    assert agg["xw_wind_mph"] == pytest.approx(10.0)
    assert agg["xw_precip_in_total"] == pytest.approx(0.1)
    assert agg["xw_weather"] == "Light Snow"


def test_credentials_are_found_case_insensitively(monkeypatch):
    from src.utils import secrets as secrets_mod

    monkeypatch.setattr(secrets_mod, "_load_env_file", lambda: {"XWeather_Client_ID": "abc", "XWeather_Client_Secret": "xyz"})
    monkeypatch.delenv("XWEATHER_CLIENT_ID", raising=False)
    assert xweather.get_xweather_credentials() == ("abc", "xyz")
    monkeypatch.setattr(
        secrets_mod, "_load_env_file", lambda: {"XWEATHER_CLIENT_API_KEY": "k1", "XWEATHER_CLIENT_API_SECRET": "s1"}
    )
    assert xweather.get_xweather_credentials() == ("k1", "s1")


class _Resp:
    def __init__(self, status, payload):
        self.status_code, self._payload = status, payload
        self.headers = {"X-RateLimit-Remaining-Period": "14999"}
        self.text = ""

    def json(self):
        return self._payload


class _Session:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def get(self, url, params, timeout):
        self.calls.append((url, params))
        return self.responses.pop(0)


def test_client_caches_responses_and_enforces_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(xweather, "MIN_INTERVAL_S", 0)
    session = _Session([_Resp(200, _payload())])
    client = xweather.XweatherClient("id", "secret", max_requests=1, cache_dir=tmp_path, session=session)
    first = xweather.fetch_game_window(client, "2024_16_X_Y", 41.5, -81.7, KICK)
    again = xweather.fetch_game_window(client, "2024_16_X_Y", 41.5, -81.7, KICK)  # served from cache
    assert first["xw_temp_kickoff_f"] == again["xw_temp_kickoff_f"] == 30
    assert len(session.calls) == 1 and client.requests_made == 1
    url, params = session.calls[0]
    assert url.endswith("/conditions/41.5000,-81.7000") and params["from"].isdigit()
    assert client.last_rate_headers == {"X-RateLimit-Remaining-Period": "14999"}
    with pytest.raises(xweather.BudgetExhausted):
        xweather.fetch_game_window(client, "another_game", 41.5, -81.7, KICK)


def test_score_snapshots_by_source_and_lead(tmp_path, monkeypatch):
    from src.evaluation import weather_source_trial as trial

    monkeypatch.setattr(trial, "OUT_DIR", tmp_path)
    monkeypatch.setattr(trial, "SNAPSHOTS", tmp_path / "snap.csv")
    pd.DataFrame(
        [
            {"game_id": "g1", "source": "xweather", "lead_hours": 50, "temp_kickoff_f": 41, "wind_mph": 6},
            {"game_id": "g1", "source": "visualcrossing", "lead_hours": 50, "temp_kickoff_f": 54, "wind_mph": 7},
        ]
    ).to_csv(tmp_path / "snap.csv", index=False)
    obs = pd.DataFrame({"game_id": ["g1"], "vx_data_source": ["obs"], "weather_temp_kickoff_f": [40.0], "weather_wind_mph": [5.0]})
    monkeypatch.setattr(trial, "read_df", lambda path: obs)
    skill = trial.score_snapshots().set_index(["source", "metric"])
    assert skill.loc[("xweather", "temp_kickoff_f"), "mae"] == 1.0
    assert skill.loc[("visualcrossing", "temp_kickoff_f"), "mae"] == 14.0
    assert set(skill["lead_bucket"]) == {"1-3d"}
