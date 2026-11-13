"""Where and when each game is actually played.

Weather must be looked up at the venue the game is played at, not at the home
team's current stadium: relocated seasons (Rams at the Coliseum 2016-19,
Chargers at StubHub Center 2017-19, Raiders in Oakland through 2019),
international games (London, Munich, Frankfurt, Mexico City, Sao Paulo,
Madrid, Paris, Melbourne, Rio) and one-off moves all differ.

Resolution order per game (``venue_source`` records which applied):

0. ``override``   - ``venue_overrides.csv`` by game_id, for games nflverse
                    files under the wrong stadium (e.g. the 2025 international
                    games carry the home team's US stadium);
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
OVERRIDES_PATH = Path("data/reference/venue_overrides.csv")
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


@lru_cache(maxsize=4)
def load_overrides(path: str = str(OVERRIDES_PATH)) -> dict[str, str]:
    if not Path(path).exists():
        return {}
    frame = pd.read_csv(path, dtype=str)
    return dict(zip(frame["game_id"].str.strip(), frame["stadium_id"].str.strip()))


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

    overrides = load_overrides()
    override_id = out.get("game_id", pd.Series("", index=out.index)).astype(str).map(overrides)
    override_id = override_id.where(override_id.isin(by_id.index))

    venue_id = pd.Series(override_id, index=out.index, dtype="object")
    source = pd.Series(np.where(venue_id.notna(), "override", None), index=out.index, dtype="object")
    use_name = venue_id.isna() & pd.Series(name_id, index=out.index).notna()
    venue_id[use_name] = pd.Series(name_id, index=out.index)[use_name]
    source[use_name] = "name"
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


# Home time zone of each franchise (including historical codes) for body-clock shifts.
TEAM_TIMEZONES = {
    "ARI": "America/Phoenix", "ATL": "America/New_York", "BAL": "America/New_York", "BUF": "America/New_York",
    "CAR": "America/New_York", "CHI": "America/Chicago", "CIN": "America/New_York", "CLE": "America/New_York",
    "DAL": "America/Chicago", "DEN": "America/Denver", "DET": "America/Detroit", "GB": "America/Chicago",
    "HOU": "America/Chicago", "IND": "America/Indiana/Indianapolis", "JAX": "America/New_York",
    "KC": "America/Chicago", "LA": "America/Los_Angeles", "LAR": "America/Los_Angeles", "LAC": "America/Los_Angeles",
    "LV": "America/Los_Angeles", "OAK": "America/Los_Angeles", "MIA": "America/New_York", "MIN": "America/Chicago",
    "NE": "America/New_York", "NO": "America/Chicago", "NYG": "America/New_York", "NYJ": "America/New_York",
    "PHI": "America/New_York", "PIT": "America/New_York", "SD": "America/Los_Angeles", "SEA": "America/Los_Angeles",
    "SF": "America/Los_Angeles", "STL": "America/Chicago", "TB": "America/New_York", "TEN": "America/Chicago",
    "WAS": "America/New_York",
}


def _utc_offset_hours(tz_name: object, when: pd.Timestamp) -> float:
    if not isinstance(tz_name, str) or pd.isna(when):
        return np.nan
    return ZoneInfo(tz_name).utcoffset(when.to_pydatetime().replace(tzinfo=None)).total_seconds() / 3600.0


def travel_context(schedule: pd.DataFrame) -> pd.DataFrame:
    """Per-game travel from each team's home base to the venue actually played at.

    Returns one row per ``game_id`` with venue coordinates, ``is_international_game``,
    ``is_neutral_site_game``, ``travel_km_home``/``travel_km_away`` (great-circle km
    from the team's home base) and ``travel_tz_shift_home``/``travel_tz_shift_away``
    (absolute hours between the team's home clock and the venue clock at kickoff,
    daylight-saving aware). Paired ``_home``/``_away`` columns become ``*_diff``
    model features.
    """
    resolved = resolve_game_venues(schedule)
    venues = load_venues().set_index("stadium_id")
    teams = pd.read_csv(TEAM_LOCATIONS_PATH)
    teams["team_abbr"] = teams["team_abbr"].astype(str).str.upper()
    base = teams.set_index("team_abbr")[["latitude", "longitude"]]
    # historical franchises' home base = their home venue then
    for code, venue_id in HISTORICAL_HOME_VENUE.items():
        if venue_id in venues.index:
            base.loc[code] = venues.loc[venue_id, ["latitude", "longitude"]].to_numpy()
    out = pd.DataFrame({"game_id": resolved["game_id"].astype(str)})
    out["venue_lat"] = resolved["venue_lat"].to_numpy()
    out["venue_lon"] = resolved["venue_lon"].to_numpy()
    out["venue_id"] = resolved["venue_id"].to_numpy()
    country = resolved["venue_country"].fillna("USA")
    out["is_international_game"] = (~country.isin(["USA"])).astype(int).to_numpy()
    location = resolved.get("location", pd.Series("", index=resolved.index)).astype(str).str.lower()
    out["is_neutral_site_game"] = location.eq("neutral").astype(int).to_numpy()
    venue_tz = resolved["venue_id"].map(venues["timezone"])
    kick = resolved.get("kickoff_utc", pd.Series(pd.NaT, index=resolved.index))
    for side in ("home", "away"):
        team = resolved[f"{side}_team"].astype(str).str.upper().str.strip()
        key = team.map(lambda t: t if t in base.index else TEAM_ALIASES.get(t, t))
        lat = key.map(base["latitude"])
        lon = key.map(base["longitude"])
        out[f"travel_km_{side}"] = haversine_km(lat, lon, resolved["venue_lat"], resolved["venue_lon"])
        team_off = [_utc_offset_hours(TEAM_TIMEZONES.get(t), k) for t, k in zip(team, kick)]
        venue_off = [_utc_offset_hours(z, k) for z, k in zip(venue_tz, kick)]
        out[f"travel_tz_shift_{side}"] = np.abs(np.asarray(venue_off, float) - np.asarray(team_off, float))
    return out.drop_duplicates("game_id")


def audit_neutral_venues(schedule: pd.DataFrame) -> pd.DataFrame:
    """Neutral-site regular-season games that resolve to the home team's usual
    stadium - usually an international game nflverse filed under the home team's
    US stadium. Add those to ``venue_overrides.csv``."""
    resolved = resolve_game_venues(schedule)
    if "location" not in resolved.columns:
        return resolved.iloc[0:0]
    neutral = resolved["location"].astype(str).str.lower().eq("neutral")
    regular = resolved.get("game_type", pd.Series("REG", index=resolved.index)).astype(str).eq("REG")
    not_overridden = resolved["venue_source"].ne("override")
    usual = resolved[~neutral].groupby(["season", "home_team"])["venue_id"].agg(
        lambda x: x.mode().iloc[0] if len(x.dropna()) else None
    )
    home_usual = np.asarray([usual.get((s, t)) for s, t in zip(resolved["season"], resolved["home_team"])], dtype=object)
    suspicious = neutral & regular & not_overridden & (resolved["venue_id"].to_numpy(dtype=object) == home_usual)
    cols = [c for c in ("game_id", "gameday", "gametime", "stadium", "venue_id") if c in resolved.columns]
    return resolved.loc[suspicious, cols]
