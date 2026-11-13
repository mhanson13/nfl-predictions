from datetime import datetime, timedelta, timezone

import pandas as pd

from src.data import visualcrossing as vx

NOW = datetime(2026, 11, 13, 15, 0, tzinfo=timezone.utc)


def _schedule():
    return pd.DataFrame(
        {
            "game_id": ["played_obs", "played_fcst", "upcoming", "far_future", "played_uncached"],
            "start_utc": [
                NOW - timedelta(days=10),
                NOW - timedelta(days=3),
                NOW + timedelta(days=2),
                NOW + timedelta(days=20),
                NOW - timedelta(days=1),
            ],
        }
    )


def _cache():
    return pd.DataFrame(
        {
            "game_id": ["played_obs", "played_fcst", "upcoming", "far_future"],
            "vx_data_source": ["obs", "fcst", "fcst", "fcst"],
        }
    )


def test_select_refresh_games_classifies_stale_upcoming_and_beyond():
    stale, upcoming, beyond = vx.select_refresh_games(_cache(), _schedule(), now=NOW, horizon_days=15)
    assert stale == {"played_fcst"}  # observed rows are final; uncached games are handled as missing
    assert upcoming == {"upcoming"}
    assert beyond == {"far_future"}


def test_game_in_progress_is_neither_stale_nor_upcoming():
    sched = pd.DataFrame({"game_id": ["live"], "start_utc": [NOW - timedelta(hours=1)]})
    cache = pd.DataFrame({"game_id": ["live"], "vx_data_source": ["fcst"]})
    assert vx.select_refresh_games(cache, sched, now=NOW) == (set(), set(), set())


def test_build_refetches_stale_and_upcoming_and_drops_far_future(tmp_path, monkeypatch):
    cache_path = tmp_path / "vx.parquet"
    cached = _cache().assign(
        season=2026, week=1, home_team="KC", away_team="BUF", weather_temp_f=50.0, weather_source="visualcrossing"
    )
    cached.to_parquet(cache_path)
    sched = _schedule().assign(season=2026, week=1, home_team="KC", away_team="BUF", home_lat=39.0, home_lon=-94.5)

    monkeypatch.setattr(vx, "CACHE_PATH", cache_path)
    monkeypatch.setattr(vx, "get_secret", lambda name: "key")
    def fake_read(path):
        if str(path).endswith("vx.parquet"):
            return pd.read_parquet(path)
        if "noaa" in str(path):
            return pd.DataFrame()
        return sched

    monkeypatch.setattr(vx, "read_df", fake_read)
    monkeypatch.setattr(vx, "_resolve_game_times", lambda schedule, meta: sched)
    monkeypatch.setattr(vx.pd, "read_csv", lambda path: pd.DataFrame())
    monkeypatch.setattr(vx.Path, "exists", lambda self: True)
    monkeypatch.setattr(vx, "validate_dataframe", lambda *a, **k: "ok")
    monkeypatch.setattr(vx, "write_df", lambda df, path: df.to_parquet(path))
    monkeypatch.setattr(vx.time, "sleep", lambda s: None)
    fetched = []

    def fake_fetch(session, lat, lon, start, end, api_key, **kwargs):
        kickoff = start + timedelta(hours=3)
        fetched.append(kickoff)
        source = "obs" if kickoff < NOW else "fcst"
        return {"days": [{"hours": [{"datetimeEpoch": kickoff.timestamp(), "temp": 40.0, "windspeed": 5.0, "source": source}]}]}

    monkeypatch.setattr(vx, "_fetch_visualcrossing", fake_fetch)
    out = vx.build_visualcrossing_weather(
        [2026], sleep=0, overwrite=False, window_hours=3, only_missing=True, now=NOW
    ).set_index("game_id")

    assert len(fetched) == 3  # played_fcst, upcoming, played_uncached
    assert "far_future" not in out.index
    assert out.loc["played_fcst", "vx_data_source"] == "obs" and out.loc["played_fcst", "weather_temp_f"] == 40.0
    assert out.loc["played_obs", "weather_temp_f"] == 50.0  # untouched
    assert out.loc["upcoming", "vx_data_source"] == "fcst" and pd.notna(out.loc["upcoming", "vx_fetched_at"])
