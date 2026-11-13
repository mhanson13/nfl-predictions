"""Where and when each game is actually played.

Weather must be looked up at the venue the game is played at, not at the home
team's current stadium: relocated seasons (Rams at the Coliseum 2016-19,
Chargers at StubHub Center 2017-19, Raiders in Oakland through 2019),
international games (London, Munich, Frankfurt, Mexico City, Sao Paulo,
Madrid, Paris, Melbourne, Rio) and one-off moves all differ.

Resolution order per game (``venue_source`` records which applied):

1. ``name``       - the schedule's ``stadium`` name matches a venue alias
                    (wins over the id: nflverse has at least one London game
                    tagged with Jacksonville's stadium_id);
2. ``stadium_id`` - the schedule's ``stadium_id``;
3. ``home_team``  - the home team's current stadium (``team_locations.csv``,
                    with LA -> LAR and OAK/SD/STL historical aliases).

Kickoff: nflverse ``gametime`` is US/Eastern wall-clock (24h), so
``kickoff_utc`` = ``gameday`` + ``gametime`` localized to America/New_York.

Roof: ``venues.csv`` holds each venue's roof type. Fixed venues override the
schedule label (``outdoor`` -> "outdoors", ``dome`` -> "dome"); retractable
venues keep the schedule's per-game "open"/"closed".
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

VENUES_PATH = Path("data/reference/venues.csv")
TEAM_LOCATIONS_PATH = Path("data/reference/team_locations.csv")
EASTERN = ZoneInfo("America/New_York")
# historical / alternate home-team codes -> team_locations.csv code
TEAM_ALIASES = {"LA": "LAR", "STL": "LAR", "OAK": "LV", "SD": "LAC", "JAC": "JAX", "WSH": "WAS"}
# historical codes whose *venue* is not the current team stadium (used only when
# the schedule has neither a known stadium name nor id)
HISTORICAL_HOME_VENUE = {"OAK": "OAK00", "SD": "SDG00", "STL": "STL00"}


def _norm(text: object) -> str:
    return " ".join(str(text).lower().replace("'", "").replace(".", "").split())


@lru_cache(maxsize=4)
def load_venues(path: str = str(VENUES_PATH)) -> pd.DataFrame:
    venues = pd.read_csv(path, dtype={"stadium_id": str})
    venues["roof_type"] = venues["roof_type"].str.lower().str.strip()
    return venues


def _alias_map(venues: pd.DataFrame) -> dict[str, str]:
    out: dict[str, str] = {}
    for _, row in venues.iterrows():
        for alias in str(row["aliases"]).split("|") + [row["venue_name"]]:
            if alias and alias.strip():
                out[_norm(alias)] = row["stadium_id"]
    return out


def kickoff_utc(schedule: pd.DataFrame) -> pd.Series:
    """UTC kickoff from nflverse ``gameday`` + ``gametime`` (US/Eastern, 24h)."""
    day = pd.to_datetime(schedule["gameday"], errors="coerce").dt.strftime("%Y-%m-%d")
    clock = schedule.get("gametime", pd.Series("13:00", index=schedule.index)).astype(str).str.strip()
    clock = clock.where(clock.str.match(r"^\d{1,2}:\d{2}$"), "13:00")
    local = pd.to_datetime(day + " " + clock, errors="coerce", format="%Y-%m-%d %H:%M")
    return local.dt.tz_localize(EASTERN, ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert("UTC")


def resolve_game_venues(schedule: pd.DataFrame, venues: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return ``schedule`` with venue_id, venue_name, venue_lat, venue_lon, venue_roof,
    venue_roof_type, venue_country, venue_source and kickoff_utc columns."""
    venues = load_venues() if venues is None else venues
    by_id = venues.set_index("stadium_id")
    aliases = _alias_map(venues)
    out = schedule.copy()

    name_id = out.get("stadium", pd.Series(np.nan, index=out.index)).map(
        lambda s: aliases.get(_norm(s)) if isinstance(s, str) and s.strip() else None
    )
    sched_id = out.get("stadium_id", pd.Series(np.nan, index=out.index)).astype("string")
    sched_id = sched_id.where(sched_id.isin(by_id.index))
    home = out.get("home_team", pd.Series("", index=out.index)).astype(str).str.upper().str.strip()
    hist_id = home.map(HISTORICAL_HOME_VENUE)

    venue_id = pd.Series(name_id, index=out.index, dtype="object")
    source = pd.Series(np.where(venue_id.notna(), "name", None), index=out.index, dtype="object")
    use_id = venue_id.isna() & sched_id.notna()
    venue_id[use_id] = sched_id[use_id]
    source[use_id] = "stadium_id"
    use_hist = venue_id.isna() & hist_id.notna()
    venue_id[use_hist] = hist_id[use_hist]
    source[use_hist] = "home_team"

    out["venue_id"] = venue_id
    known = venue_id.notna()
    out["venue_name"] = venue_id.map(by_id["venue_name"])
    out["venue_lat"] = venue_id.map(by_id["latitude"]).astype(float)
    out["venue_lon"] = venue_id.map(by_id["longitude"]).astype(float)
    out["venue_roof_type"] = venue_id.map(by_id["roof_type"])
    out["venue_country"] = venue_id.map(by_id["country"])

    # last resort: home team's current stadium
    teams = pd.read_csv(TEAM_LOCATIONS_PATH)
    teams["team_abbr"] = teams["team_abbr"].astype(str).str.upper()
    team_lat = teams.set_index("team_abbr")["latitude"]
    team_lon = teams.set_index("team_abbr")["longitude"]
    team_key = home.map(lambda t: TEAM_ALIASES.get(t, t))
    fallback = ~known
    out.loc[fallback, "venue_lat"] = team_key[fallback].map(team_lat)
    out.loc[fallback, "venue_lon"] = team_key[fallback].map(team_lon)
    source[fallback & out["venue_lat"].notna()] = "home_team"
    out["venue_source"] = source

    sched_roof = out.get("roof", pd.Series("", index=out.index)).astype("string").str.lower().str.strip()
    roof = sched_roof.copy()
    roof[out["venue_roof_type"].eq("outdoor")] = "outdoors"
    roof[out["venue_roof_type"].eq("dome")] = "dome"
    out["venue_roof"] = roof

    if "gameday" in out.columns:
        out["kickoff_utc"] = kickoff_utc(out)
    return out


def apply_venue_roof(schedule: pd.DataFrame) -> pd.DataFrame:
    """Overwrite the schedule ``roof`` label with the venue-corrected value."""
    if "roof" not in schedule.columns:
        return schedule
    resolved = resolve_game_venues(schedule)
    out = schedule.copy()
    out["roof"] = resolved["venue_roof"].fillna(out["roof"])
    return out


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype=float)) for x in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))
