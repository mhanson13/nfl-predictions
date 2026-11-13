from datetime import datetime, timezone

import pandas as pd

from src.data import visualcrossing as vx
from src.utils.venues import apply_venue_roof, kickoff_utc, load_venues, resolve_game_venues


def _sched(rows):
    cols = ["game_id", "season", "week", "gameday", "gametime", "home_team", "away_team", "stadium_id", "stadium", "roof", "location"]
    return pd.DataFrame(rows, columns=cols)


def test_venue_table_is_complete_and_sane():
    venues = load_venues()
    assert venues["stadium_id"].is_unique
    assert venues["latitude"].between(-60, 70).all() and venues["longitude"].between(-180, 180).all()
    assert set(venues["roof_type"]) <= {"outdoor", "dome", "retractable"}


def test_resolves_actual_venue_not_home_team_stadium():
    sched = _sched(
        [
            # London game tagged with Jacksonville's stadium_id: the name wins
            ("2026_05_PHI_JAX", 2026, 5, "2026-10-11", "09:30", "JAX", "PHI", "JAX00", "Tottenham Hotspur Stadium", "outdoors", "Neutral"),
            # Chargers at StubHub Center, Rams at the Coliseum, Raiders in Oakland
            ("2017_02_MIA_LAC", 2017, 2, "2017-09-17", "16:05", "LAC", "MIA", "LAX97", "StubHub Center", "outdoors", "Home"),
            ("2018_01_ARI_LA", 2018, 1, "2018-09-16", "16:05", "LA", "ARI", "LAX99", "Los Angeles Memorial Coliseum", "outdoors", "Home"),
            ("2018_01_LA_OAK", 2018, 1, "2018-09-10", "22:20", "OAK", "LA", "OAK00", "Oakland-Alameda County Coliseum", "outdoors", "Home"),
            # unknown venue -> home team's current stadium (LA alias)
            ("2030_01_SF_LA", 2030, 1, "2030-09-15", "16:25", "LA", "SF", "ZZZ99", "Future Field", "dome", "Home"),
        ]
    )
    out = resolve_game_venues(sched).set_index("game_id")
    assert out.loc["2026_05_PHI_JAX", "venue_id"] == "LON02" and out.loc["2026_05_PHI_JAX", "venue_source"] == "name"
    assert round(out.loc["2026_05_PHI_JAX", "venue_lat"], 2) == 51.60
    assert out.loc["2017_02_MIA_LAC", "venue_id"] == "LAX97"
    assert out.loc["2018_01_ARI_LA", "venue_id"] == "LAX99"
    assert round(out.loc["2018_01_LA_OAK", "venue_lat"], 2) == 37.75  # Oakland, not Las Vegas
    assert out.loc["2030_01_SF_LA", "venue_source"] == "home_team"
    assert round(out.loc["2030_01_SF_LA", "venue_lat"], 2) == 33.95


def test_kickoff_is_eastern_24h_with_dst():
    sched = _sched(
        [
            ("a", 2024, 6, "2024-10-13", "13:00", "ATL", "CAR", "ATL97", "Mercedes-Benz Stadium", "closed", "Home"),
            ("b", 2024, 16, "2024-12-22", "13:00", "ATL", "NYG", "ATL97", "Mercedes-Benz Stadium", "closed", "Home"),
            ("c", 2026, 10, "2026-11-15", "09:30", "DET", "NE", "MUN01", "FC Bayern Munich Stadium", "dome", "Neutral"),
        ]
    )
    k = kickoff_utc(sched)
    assert k.iloc[0] == pd.Timestamp("2024-10-13 17:00", tz="UTC")  # EDT
    assert k.iloc[1] == pd.Timestamp("2024-12-22 18:00", tz="UTC")  # EST
    assert k.iloc[2] == pd.Timestamp("2026-11-15 14:30", tz="UTC")


def test_roof_corrected_for_fixed_venues_only():
    sched = _sched(
        [
            ("m", 2026, 1, "2026-09-10", "20:35", "LA", "SF", "MEL00", "Melbourne Cricket Ground", "dome", "Neutral"),
            ("h", 2024, 3, "2024-09-22", "13:00", "HOU", "MIN", "HOU00", "NRG Stadium", "open", "Home"),
            ("d", 2024, 3, "2024-09-22", "13:00", "DET", "ARI", "DET00", "Ford Field", "", "Home"),
        ]
    )
    roof = apply_venue_roof(sched).set_index("game_id")["roof"]
    assert roof["m"] == "outdoors"  # open-air cricket ground mislabelled as dome
    assert roof["h"] == "open"  # retractable keeps the per-game state
    assert roof["d"] == "dome"


def test_visualcrossing_audit_flags_wrong_place_and_wrong_time():
    sched = pd.DataFrame(
        {
            "game_id": ["ok", "moved", "late"],
            "home_lat": [42.09, 51.556, 39.0],
            "home_lon": [-71.26, -0.28, -94.5],
            "start_utc": pd.to_datetime(["2024-10-13 17:00", "2024-10-13 13:30", "2024-10-13 17:00"], utc=True),
        }
    )
    cached = pd.DataFrame(
        {
            "game_id": ["ok", "moved", "late"],
            "vx_latitude": [42.09, 30.32, 39.0],  # London game fetched in Jacksonville
            "vx_longitude": [-71.26, -81.64, -94.5],
            "start_utc": pd.to_datetime(["2024-10-13 17:00", "2024-10-13 13:30", "2024-10-13 22:00"], utc=True),
        }
    )
    moved, misaligned = vx.select_location_time_fixes(cached, sched)
    assert moved == {"moved"} and misaligned == {"late"}


def test_visualcrossing_resolver_uses_venue_and_eastern_kickoff():
    sched = _sched(
        [("2023_10_IND_NE", 2023, 10, "2023-11-12", "09:30", "NE", "IND", "FRA00", "Deutsche Bank Park", "outdoors", "Neutral")]
    )
    out = vx._resolve_game_times(sched).iloc[0]
    assert round(out["home_lat"], 2) == 50.07  # Frankfurt, not Foxborough
    assert out["start_utc"] == datetime(2023, 11, 12, 14, 30, tzinfo=timezone.utc)
