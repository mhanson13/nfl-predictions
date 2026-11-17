import pandas as pd

from src.data import noaa


def _schedule():
    return pd.DataFrame(
        {
            "game_id": ["2026_04_IND_WAS", "2026_10_MIN_GB", "2026_10_HOU_CLE"],
            "season": [2026, 2026, 2026],
            "week": [4, 10, 10],
            "gameday": ["2026-10-04", "2026-11-15", "2026-11-15"],
            "gametime": ["09:30", "13:00", "13:00"],
            "home_team": ["WAS", "GB", "CLE"],
            "away_team": ["IND", "MIN", "HOU"],
            "stadium_id": ["LON02", "GNB00", "CLE00"],
            "stadium": ["Tottenham Hotspur Stadium", "Lambeau Field", "Huntington Bank Field"],
            "roof": ["outdoors", "outdoors", "outdoors"],
            "location": ["Neutral", "Home", "Home"],
        }
    )


def test_noaa_skips_international_venues_and_survives_lookup_errors(tmp_path, monkeypatch, capsys):
    sched = _schedule()
    monkeypatch.setattr(noaa, "OBS_CACHE_PATH", tmp_path / "noaa.parquet")
    monkeypatch.setattr(noaa, "read_df", lambda path: sched.copy())
    monkeypatch.setattr(noaa, "_load_station_cache", lambda: {})
    monkeypatch.setattr(noaa, "_save_station_cache", lambda cache: None)
    monkeypatch.setattr(noaa, "validate_dataframe", lambda *a, **k: "ok")
    monkeypatch.setattr(noaa, "write_df", lambda df, path: None)
    looked_up = []

    def fake_stations(session, lat, lon, cache, sleep):
        looked_up.append((round(lat, 2), round(lon, 2)))
        if round(lat, 2) == 44.50:  # Green Bay lookup fails (e.g. HTTP 404)
            raise RuntimeError("404 Client Error")
        return ["KCLE"]

    monkeypatch.setattr(noaa, "_nearest_stations", fake_stations)
    monkeypatch.setattr(noaa, "_collect_observations", lambda *a, **k: [])

    noaa.build_noaa_weather([2026], sleep=0, overwrite=True)

    assert (51.6, -0.07) not in looked_up  # London never sent to weather.gov
    assert (44.5, -88.06) in looked_up and (41.51, -81.7) in looked_up  # US venues still processed
    out = capsys.readouterr().out
    assert "station lookup failed for 2026_10_MIN_GB" in out
    assert "skipped 1 games at non-US venues" in out


def test_noaa_job_is_optional_in_pipeline():
    from tools import run_pipeline

    # One constant drives both the threaded and async data-job runners.
    assert {"yahoo_static", "noaa_weather"} <= run_pipeline.OPTIONAL_DATA_JOBS


def test_public_venue_columns_for_neutral_and_home_games():
    from src.utils.venues import PUBLIC_VENUE_COLUMNS, attach_venue_columns, game_venue_table

    sched = _schedule().assign(stadium=["Tottenham Hotspur Stadium", "Lambeau Field", "Huntington Bank Field"],
                               stadium_id=["LON02", "GNB00", "CLE00"])
    sched.loc[sched.game_id == "2026_10_MIN_GB", ["stadium", "stadium_id", "location"]] = [
        "FC Bayern Munich Stadium", "MUN01", "Neutral"]  # a Munich game nflverse files under GB
    table = game_venue_table(sched).set_index("game_id")
    munich = table.loc["2026_10_MIN_GB"]
    assert munich["venue_city"] == "Munich" and munich["venue_country"] == "Germany" and bool(munich["neutral_site"])
    assert round(munich["venue_lat"], 2) == 48.22 and round(munich["venue_lon"], 2) == 11.62
    assert bool(table.loc["2026_04_IND_WAS", "neutral_site"]) and table.loc["2026_04_IND_WAS", "venue_city"] == "London"
    assert not bool(table.loc["2026_10_HOU_CLE", "neutral_site"]) and table.loc["2026_10_HOU_CLE", "venue_country"] == "USA"

    preds = pd.DataFrame({"game_id": ["2026_10_MIN_GB"], "home_win_prob": [0.6], "venue_lat": [44.5], "venue_lon": [-88.06]})
    out = attach_venue_columns(preds, sched)
    assert set(PUBLIC_VENUE_COLUMNS) <= set(out.columns)
    assert out.loc[0, "home_win_prob"] == 0.6 and round(out.loc[0, "venue_lat"], 2) == 48.22


def test_game_venues_file_published_with_predictions_only():
    from tools import publish_mtb_csvs as pub

    dests = [str(a.destination).replace("\\", "/") for a in pub.build_artifact_plan(season=2026, week=11)]
    assert "reference/game_venues_2026.csv" in dests
    graded = pub.build_artifact_plan(season=2026, week=10, results_only=True)
    assert not any("game_venues" in str(a.destination) for a in graded)
