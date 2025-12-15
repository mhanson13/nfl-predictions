# Copyright (c) 2025 Matt Hanson
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Weather ingestion utilities: stadium metadata, API fusion, and feature exports."""

from __future__ import annotations
import os
import argparse
import time
import re
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import requests

from src.utils.io import RAW_DIR, PROC_DIR, read_df, write_df
from src.utils.logging import configure as configure_logging
from pathlib import Path
from datetime import timedelta
import uuid


# ----------------------- Stadium location fallbacks -----------------------

# Fallback coordinates (lat, lon) and roof type for each team. Coordinates are approximate.
# Source: public stadium info; sufficient for weather queries. Roof: "dome" (incl. retractable/covered) or "outdoor".
STADIUMS: Dict[str, Dict[str, Any]] = {
    # NFC
    "ARI": {"city": "Glendale, AZ", "lat": 33.5276, "lon": -112.2626, "roof": "dome"},
    "ATL": {"city": "Atlanta, GA", "lat": 33.7554, "lon": -84.4009, "roof": "dome"},
    "CAR": {"city": "Charlotte, NC", "lat": 35.2250, "lon": -80.8526, "roof": "outdoor"},
    "CHI": {"city": "Chicago, IL", "lat": 41.8623, "lon": -87.6167, "roof": "outdoor"},
    "DAL": {"city": "Arlington, TX", "lat": 32.7473, "lon": -97.0945, "roof": "dome"},
    "DET": {"city": "Detroit, MI", "lat": 42.3390, "lon": -83.0456, "roof": "dome"},
    "GB":  {"city": "Green Bay, WI", "lat": 44.5013, "lon": -88.0622, "roof": "outdoor"},
    "LAR": {"city": "Inglewood, CA", "lat": 33.9535, "lon": -118.3387, "roof": "dome"},
    "MIN": {"city": "Minneapolis, MN", "lat": 44.9738, "lon": -93.2581, "roof": "dome"},
    "NO":  {"city": "New Orleans, LA", "lat": 29.9511, "lon": -90.0812, "roof": "dome"},
    "NYG": {"city": "East Rutherford, NJ", "lat": 40.8136, "lon": -74.0745, "roof": "outdoor"},
    "PHI": {"city": "Philadelphia, PA", "lat": 39.9008, "lon": -75.1675, "roof": "outdoor"},
    "SF":  {"city": "Santa Clara, CA", "lat": 37.4030, "lon": -121.9700, "roof": "outdoor"},
    "SEA": {"city": "Seattle, WA", "lat": 47.5952, "lon": -122.3316, "roof": "outdoor"},
    "TB":  {"city": "Tampa, FL", "lat": 27.9759, "lon": -82.5033, "roof": "outdoor"},
    "WAS": {"city": "Landover, MD", "lat": 38.9077, "lon": -76.8645, "roof": "outdoor"},
    # AFC
    "BAL": {"city": "Baltimore, MD", "lat": 39.2779, "lon": -76.6227, "roof": "outdoor"},
    "BUF": {"city": "Orchard Park, NY", "lat": 42.7738, "lon": -78.7868, "roof": "outdoor"},
    "CIN": {"city": "Cincinnati, OH", "lat": 39.0954, "lon": -84.5161, "roof": "outdoor"},
    "CLE": {"city": "Cleveland, OH", "lat": 41.5061, "lon": -81.6995, "roof": "outdoor"},
    "DEN": {"city": "Denver, CO", "lat": 39.7439, "lon": -105.0201, "roof": "outdoor"},
    "HOU": {"city": "Houston, TX", "lat": 29.6847, "lon": -95.4107, "roof": "dome"},
    "IND": {"city": "Indianapolis, IN", "lat": 39.7601, "lon": -86.1639, "roof": "dome"},
    "JAX": {"city": "Jacksonville, FL", "lat": 30.3239, "lon": -81.6373, "roof": "outdoor"},
    "KC":  {"city": "Kansas City, MO", "lat": 39.0490, "lon": -94.4839, "roof": "outdoor"},
    "LV":  {"city": "Las Vegas, NV", "lat": 36.0909, "lon": -115.1830, "roof": "dome"},
    "LAC": {"city": "Inglewood, CA", "lat": 33.9535, "lon": -118.3387, "roof": "dome"},
    "MIA": {"city": "Miami Gardens, FL", "lat": 25.9580, "lon": -80.2389, "roof": "outdoor"},
    "NE":  {"city": "Foxborough, MA", "lat": 42.0909, "lon": -71.2643, "roof": "outdoor"},
    "NYJ": {"city": "East Rutherford, NJ", "lat": 40.8136, "lon": -74.0745, "roof": "outdoor"},
    "PIT": {"city": "Pittsburgh, PA", "lat": 40.4468, "lon": -80.0158, "roof": "outdoor"},
    "TEN": {"city": "Nashville, TN", "lat": 36.1665, "lon": -86.7713, "roof": "outdoor"},
}

# Ensure we include all current teams (mapping above duplicates LAR; acceptable)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
# Cache file for stadium coordinates discovered via Overpass
OSM_CACHE_PATH = RAW_DIR / "stadium_locations_osm.csv"

# Primary + alternate stadium names per team to aid Overpass matching
TEAM_STADIUM_NAMES: Dict[str, List[str]] = {
    "ARI": ["State Farm Stadium"],
    "ATL": ["Mercedes-Benz Stadium"],
    "BAL": ["M&T Bank Stadium"],
    "BUF": ["Highmark Stadium", "New Era Field", "Ralph Wilson Stadium"],
    "CAR": ["Bank of America Stadium"],
    "CHI": ["Soldier Field"],
    "CIN": ["Paycor Stadium", "Paul Brown Stadium"],
    "CLE": ["Cleveland Browns Stadium", "FirstEnergy Stadium", "Huntington Bank Field"],
    "DAL": ["AT&T Stadium"],
    "DEN": ["Empower Field at Mile High", "Mile High Stadium"],
    "DET": ["Ford Field"],
    "GB": ["Lambeau Field"],
    "HOU": ["NRG Stadium", "Reliant Stadium"],
    "IND": ["Lucas Oil Stadium"],
    "JAX": ["EverBank Stadium", "EverBank Field", "TIAA Bank Field"],
    "KC": ["Arrowhead Stadium", "GEHA Field at Arrowhead Stadium"],
    "LV": ["Allegiant Stadium"],
    "LAC": ["SoFi Stadium"],
    "LAR": ["SoFi Stadium"],
    "MIA": ["Hard Rock Stadium", "Sun Life Stadium"],
    "MIN": ["U.S. Bank Stadium"],
    "NE": ["Gillette Stadium"],
    "NO": ["Caesars Superdome", "Mercedes-Benz Superdome", "Louisiana Superdome"],
    "NYG": ["MetLife Stadium"],
    "NYJ": ["MetLife Stadium"],
    "PHI": ["Lincoln Financial Field"],
    "PIT": ["Acrisure Stadium", "Heinz Field"],
    "SF": ["Levi's Stadium"],
    "SEA": ["Lumen Field", "CenturyLink Field", "Qwest Field"],
    "TB": ["Raymond James Stadium"],
    "TEN": ["Nissan Stadium"],
    "WAS": ["Commanders Field", "FedExField"],
}

INTERNATIONAL_STADIUM_NAMES: List[str] = [
    # United Kingdom
    "Tottenham Hotspur Stadium",
    "Wembley Stadium",
    # Germany
    "Allianz Arena",
    "Deutsche Bank Park",
    # Brazil
    "Arena Corinthians",
    "Neo Química Arena",
    # Mexico
    "Estadio Azteca",
]

# Ensure the fallback STADIUMS map includes the canonical stadium name
for _abbr, _names in TEAM_STADIUM_NAMES.items():
    primary = _names[0] if _names else None
    if primary:
        STADIUMS.setdefault(_abbr, {})["name"] = STADIUMS.get(_abbr, {}).get("name", primary)

# Team full-name to abbreviation mapping for normalization
TEAM_NAME_TO_ABBR: Dict[str, str] = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC", "Los Angeles Rams": "LAR", "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
    # Common nicknames
    "Rams": "LAR", "Chargers": "LAC", "Raiders": "LV", "49ers": "SF", "Niners": "SF",
    "Seahawks": "SEA", "Patriots": "NE", "Dolphins": "MIA", "Jets": "NYJ", "Giants": "NYG",
    "Eagles": "PHI", "Steelers": "PIT", "Browns": "CLE", "Bengals": "CIN", "Ravens": "BAL",
    "Bills": "BUF", "Cowboys": "DAL", "Broncos": "DEN", "Lions": "DET", "Packers": "GB",
    "Bears": "CHI", "Vikings": "MIN", "Saints": "NO", "Falcons": "ATL", "Panthers": "CAR",
    "Buccaneers": "TB", "Bucs": "TB", "Jaguars": "JAX", "Jags": "JAX", "Titans": "TEN",
    "Texans": "HOU", "Colts": "IND", "Chiefs": "KC", "Commanders": "WAS", "Cardinals": "ARI",
    # LA variants
    "LA Rams": "LAR", "LA Chargers": "LAC",
}


def _normalize_stadium_label(val: Optional[str]) -> str:
    if not isinstance(val, str):
        return ""
    return re.sub(r"[^a-z0-9]", "", val.lower())


def _load_osm_stadium_catalog(refresh: bool = False) -> pd.DataFrame:
    """
    Fetch stadium coordinates from the Overpass API (or cached file) for the known NFL venues.
    Returns an empty DataFrame if the request fails or data is unavailable.
    """
    if not refresh and OSM_CACHE_PATH.exists():
        try:
            cached = pd.read_csv(OSM_CACHE_PATH)
            if not cached.empty:
                return cached
        except Exception:
            pass
    names = sorted({
        name
        for values in TEAM_STADIUM_NAMES.values()
        for name in values
        if isinstance(name, str) and name.strip()
    } | {nm for nm in INTERNATIONAL_STADIUM_NAMES if isinstance(nm, str) and nm.strip()})
    if not names:
        return pd.DataFrame()
    records: List[Dict[str, Any]] = []
    chunk_size = 12
    for chunk_start in range(0, len(names), chunk_size):
        chunk = names[chunk_start: chunk_start + chunk_size]
        clauses: List[str] = []
        for nm in chunk:
            if not nm:
                continue
            safe_name = nm.replace('"', '\\"')
            clauses.append(f'  nwr["leisure"="stadium"]["name"="{safe_name}"];')
            clauses.append(f'  nwr["building"="stadium"]["name"="{safe_name}"];')
        if not clauses:
            continue
        filters = "\n".join(clauses)
        query = f"""[out:json][timeout:180];
(
{filters}
);
out center tags;
"""
        try:
            resp = requests.get(OVERPASS_URL, params={"data": query}, timeout=120)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            print(f"[weather] OSM Overpass request failed: {exc}")
            continue
        elements = (payload or {}).get("elements") or []
        for el in elements:
            if not isinstance(el, dict):
                continue
            tags = el.get("tags") or {}
            center = el.get("center") or {}
            lat = el.get("lat", center.get("lat"))
            lon = el.get("lon", center.get("lon"))
            name = tags.get("name")
            if name is None or lat is None or lon is None:
                continue
            records.append({
                "name": name,
                "city": tags.get("addr:city") or tags.get("is_in:city") or tags.get("addr:city:towndistrict"),
                "state": tags.get("addr:state"),
                "country": tags.get("addr:country"),
                "latitude": float(lat),
                "longitude": float(lon),
                "osm_id": el.get("id"),
                "wikidata": tags.get("wikidata"),
                "wikipedia": tags.get("wikipedia"),
            })
        time.sleep(1.0)
    df = pd.DataFrame(records)
    if not df.empty:
        try:
            df.to_csv(OSM_CACHE_PATH, index=False)
        except Exception:
            pass
    return df


def _apply_osm_coords_to_stadiums(stadiums: Dict[str, Dict[str, Any]], refresh: bool = False) -> None:
    """
    Update the stadium lookup table with Overpass-derived latitude/longitude
    for each team when available.
    """
    try:
        catalog = _load_osm_stadium_catalog(refresh=refresh)
    except Exception as exc:
        print(f"[weather] OSM catalog load failed: {exc}")
        return
    if catalog is None or catalog.empty:
        return
    df = catalog.dropna(subset=["name", "latitude", "longitude"]).copy()
    df["name_norm"] = df["name"].map(_normalize_stadium_label)
    df = df.sort_values(["name_norm", "latitude"]).drop_duplicates(subset=["name_norm"], keep="first")
    name_to_row = df.set_index("name_norm", drop=False).to_dict(orient="index")
    for abbr, info in stadiums.items():
        candidates = TEAM_STADIUM_NAMES.get(abbr, [])
        # Include any explicit name already stored on the mapping
        if info.get("name"):
            candidates = list(dict.fromkeys([info["name"], *candidates]))
        match_row = None
        for cand in candidates:
            key = _normalize_stadium_label(cand)
            if key in name_to_row:
                match_row = name_to_row[key]
                break
        if match_row is None:
            continue
        lat = match_row.get("latitude")
        lon = match_row.get("longitude")
        if pd.isna(lat) or pd.isna(lon):
            continue
        info["lat"] = float(lat)
        info["lon"] = float(lon)
        # Preserve existing location metadata, but fill if missing
        if not info.get("city") and match_row.get("city"):
            info["city"] = match_row.get("city")
        if match_row.get("country"):
            info["country"] = match_row.get("country")
        info["osm_id"] = match_row.get("osm_id")
        info["osm_name"] = match_row.get("name")
        info["source"] = info.get("source") or "osm"


def _get_api_key(explicit: Optional[str] = None) -> str:
    # 1) direct arg or environment
    key = explicit or os.getenv("TOMORROW_API_KEY") or os.getenv("TOMORROWIO_API_KEY")
    if key:
        return key
    # 2) load from secrets.env at project root if present
    try:
        root = Path(__file__).resolve().parents[2]
        env_path = root / "secrets.env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k and v:
                        os.environ.setdefault(k, v)
        key = os.getenv("TOMORROW_API_KEY") or os.getenv("TOMORROWIO_API_KEY")
        if key:
            return key
    except Exception:
        pass
    raise RuntimeError("Set TOMORROW_API_KEY in environment or secrets.env, or pass --api-key")


def _coalesce(cols: List[str], df: pd.DataFrame) -> Optional[pd.Series]:
    for c in cols:
        if c in df.columns:
            return df[c]
    return None


def _try_parse_jsonlike(v: Any) -> Any:
    if isinstance(v, str) and v and v[0] in "[{":
        try:
            import json
            return json.loads(v)
        except Exception:
            return v
    return v


def _extract_venue_latlon(row: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    # Try common flattened columns first
    cand_lat = [
        "venue.latitude", "competitions.venue.latitude", "competitions.venue.address.latitude",
        "venue.address.latitude", "latitude"
    ]
    cand_lon = [
        "venue.longitude", "competitions.venue.longitude", "competitions.venue.address.longitude",
        "venue.address.longitude", "longitude"
    ]
    for la, lo in zip(cand_lat, cand_lon):
        if la in row.index and lo in row.index:
            try:
                lat = float(row[la])
                lon = float(row[lo])
                if pd.notna(lat) and pd.notna(lon):
                    return lat, lon
            except Exception:
                pass

    # Search inside nested JSON in known container columns
    for container_col in ["competitions", "venue"]:
        if container_col in row.index:
            obj = _try_parse_jsonlike(row[container_col])
            try:
                if isinstance(obj, list) and obj:
                    obj = obj[0]
                if isinstance(obj, dict):
                    # Common paths
                    v = obj.get("venue", obj)
                    addr = v.get("address", {}) if isinstance(v, dict) else {}
                    lat = addr.get("latitude") or v.get("latitude")
                    lon = addr.get("longitude") or v.get("longitude")
                    if lat is not None and lon is not None:
                        return float(lat), float(lon)
            except Exception:
                pass
    return None, None


def _kickoff_time_utc(row: pd.Series) -> Optional[pd.Timestamp]:
    # ESPN-style direct timestamps
    for c in ["date", "start_date", "startTime", "game_date"]:
        if c in row.index:
            try:
                dt = pd.to_datetime(row[c], errors="coerce", utc=True)
                if pd.notna(dt):
                    return dt
            except Exception:
                continue
    # nflverse import_schedules uses 'gameday' (YYYY-MM-DD) and 'gametime' (HH:MM in ET)
    if "gameday" in row.index and "gametime" in row.index:
        try:
            day = str(row["gameday"]).strip()
            tme = str(row["gametime"]).strip()
            if day and tme:
                dt_local = pd.to_datetime(f"{day} {tme}", errors="coerce")
                if pd.notna(dt_local):
                    # Assume Eastern Time for schedule times, convert to UTC
                    dt_et = dt_local.tz_localize("America/New_York", nonexistent="shift_forward", ambiguous="NaT")
                    return dt_et.tz_convert("UTC")
        except Exception:
            pass
    return None


def _pick_nearest_hourly(df: pd.DataFrame, target_ts: pd.Timestamp) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    try:
        df = df.copy()
        df["ts"] = pd.to_datetime(df["startTime"], utc=True, errors="coerce")
        df = df.dropna(subset=["ts"]).copy()
        if df.empty:
            return None
        # Within +/- 6 hours of kickoff
        # Compute absolute time difference to target in seconds
        target_ns = pd.to_datetime(target_ts, utc=True).value
        df["abs_diff"] = (df["ts"].astype("int64") - target_ns).abs() / 1e9
        row = df.sort_values("abs_diff").iloc[0]
        if float(row["abs_diff"]) > 12 * 3600:
            return None
        return row
    except Exception:
        return None


def _tomorrow_forecast_hourly(lat: float, lon: float, api_key: str, retries: int = 3, pause: float = 0.6) -> Optional[pd.DataFrame]:
    base = "https://api.tomorrow.io/v4/weather/forecast"
    params = {
        "location": f"{lat},{lon}",
        "timesteps": "hourly",
        "units": "imperial",
        "fields": ",".join([
            "temperature","temperatureApparent","dewPoint","humidity","windSpeed","windGust","windDirection",
            "precipitationIntensity","precipitationType","precipitationProbability","visibility","cloudCover","weatherCode"
        ]),
        "apikey": api_key,
    }
    for i in range(retries):
        try:
            r = requests.get(base, params=params, timeout=20)
            if r.status_code == 200:
                data = r.json()
                hourly = (((data or {}).get("timelines") or {}).get("hourly"))
                if not hourly:
                    return None
                rows = []
                for it in hourly:
                    rows.append({"startTime": it.get("time"), **(it.get("values") or {})})
                return pd.DataFrame(rows)
            elif r.status_code in (429, 500, 502, 503):
                time.sleep(pause * (i + 1))
                continue
            else:
                # Non-retryable error
                return None
        except Exception:
            time.sleep(pause * (i + 1))
    return None


def _tomorrow_forecast_daily(lat: float, lon: float, api_key: str, retries: int = 3, pause: float = 0.6) -> Optional[pd.DataFrame]:
    """Daily forecast fallback with extended horizon (~14–15 days).
    Returns DataFrame with one row per day and 'startTime' timestamps.
    """
    base = "https://api.tomorrow.io/v4/weather/forecast"
    params = {
        "location": f"{lat},{lon}",
        "timesteps": "daily",
        "units": "imperial",
        "fields": ",".join([
            "temperatureAvg","temperatureMax","temperatureMin",
            "dewPointAvg","humidityAvg",
            "windSpeedAvg","windGustAvg","windDirectionAvg",
            "precipitationIntensityAvg","precipitationType","precipitationProbabilityAvg",
            "visibilityAvg","cloudCoverAvg","weatherCodeMax",
        ]),
        "apikey": api_key,
    }
    for i in range(retries):
        try:
            r = requests.get(base, params=params, timeout=20)
            if r.status_code == 200:
                data = r.json()
                daily = (((data or {}).get("timelines") or {}).get("daily"))
                if not daily:
                    return None
                rows = []
                for it in daily:
                    rows.append({"startTime": it.get("time"), **(it.get("values") or {})})
                return pd.DataFrame(rows)
            elif r.status_code in (429, 500, 502, 503):
                time.sleep(pause * (i + 1))
                continue
            else:
                return None
        except Exception:
            time.sleep(pause * (i + 1))
    return None


def _pick_daily_for_date(df: pd.DataFrame, target_ts: pd.Timestamp) -> Optional[pd.Series]:
    """Pick daily row matching the target date (UTC); else closest within 15 days."""
    if df is None or df.empty:
        return None
    try:
        d = df.copy()
        d["ts"] = pd.to_datetime(d["startTime"], utc=True, errors="coerce")
        d = d.dropna(subset=["ts"]).copy()
        if d.empty:
            return None
        target_day = pd.to_datetime(target_ts, utc=True).normalize()
        d["day"] = d["ts"].dt.normalize()
        exact = d.loc[d["day"] == target_day]
        if not exact.empty:
            return exact.iloc[0]
        # Compute absolute day difference numerically for type-checker friendliness
        denom = float(pd.Timedelta(days=1).value)
        d["abs_days"] = ((d["day"].astype("int64") - int(target_day.value)).abs()) / denom
        row = d.sort_values("abs_days").iloc[0]
        if float(row["abs_days"]) <= 15:
            return row
        return None
    except Exception:
        return None


def build_game_weather(seasons: List[int], api_key: Optional[str] = None) -> pd.DataFrame:
    api_key = _get_api_key(api_key)
    # Load schedules (prefer nflverse for consistent keys; fallback to ESPN)
    sched_path = RAW_DIR / "nfl_schedules.parquet"
    if not sched_path.exists():
        alt = RAW_DIR / "espn_schedule.parquet"
        sched_path = alt if alt.exists() else sched_path
    if not sched_path.exists():
        raise FileNotFoundError(f"No schedule found at {RAW_DIR}/espn_schedule.parquet or nfl_schedules.parquet")
    sched = read_df(sched_path)
    try:
        print(f"[weather] using schedule: {sched_path.name} rows={len(sched)} cols={len(sched.columns)}")
    except Exception:
        pass

    # Narrow to requested seasons
    sched = sched[sched["season"].isin(seasons)].copy() if "season" in sched.columns else sched

    # Key identifiers
    # Try to retain home/away abbreviations and game_id for merges
    # We'll reuse the normalization logic from features by lightweight inference here
    def _norm_abbr(x: Any) -> Any:
        if not isinstance(x, str):
            return x
        s = x.strip()
        if not s:
            return s
        # If already an abbr, normalize alternates
        up = s.upper()
        alt = {"JAC": "JAX", "WSH": "WAS", "ARZ": "ARI", "KAN": "KC", "NOR": "NO", "TAM": "TB", "GNB": "GB", "SFO": "SF", "NWE": "NE", "SD": "LAC", "STL": "LAR", "OAK": "LV"}
        if 2 <= len(up) <= 4 and up.isalpha():
            return alt.get(up, up)
        # Try full-name or nickname match
        if s in TEAM_NAME_TO_ABBR:
            return TEAM_NAME_TO_ABBR[s]
        # Partial contains match
        for name, ab in TEAM_NAME_TO_ABBR.items():
            if s.lower() in name.lower() or name.lower() in s.lower():
                return ab
        return up

    def _pick_series(names: List[str]) -> Optional[pd.Series]:
        for n in names:
            if n in sched.columns:
                return sched[n]
        return None

    home = _pick_series(["home_team", "homeTeam", "home", "home_abbr", "homeTeam.abbreviation", "home_name", "homeTeam.displayName"])
    away = _pick_series(["away_team", "awayTeam", "away", "away_abbr", "awayTeam.abbreviation", "away_name", "awayTeam.displayName"])
    if home is not None:
        sched["home_team"] = home.astype(str).map(_norm_abbr)
    if away is not None:
        sched["away_team"] = away.astype(str).map(_norm_abbr)
    if "game_id" not in sched.columns:
        for gid in ["id", "gameId", "event_id", "competitions.id"]:
            if gid in sched.columns:
                sched.rename(columns={gid: "game_id"}, inplace=True)
                break

    # Extract kickoff datetime and venue lat/lon
    # Kickoff time (vectorized best-effort)
    kickoff_col = None
    for c in ["date", "start_date", "startTime", "game_date"]:
        if c in sched.columns:
            kickoff_col = c
            break
    if kickoff_col is not None:
        sched["_kickoff"] = pd.to_datetime(sched[kickoff_col], errors="coerce", utc=True)
    else:
        # nflverse schedules: 'gameday' (YYYY-MM-DD) + 'gametime' (HH:MM ET)
        if ("gameday" in sched.columns) and ("gametime" in sched.columns):
            try:
                dt_local = pd.to_datetime(
                    sched["gameday"].astype(str).str.strip() + " " + sched["gametime"].astype(str).str.strip(),
                    errors="coerce",
                )
                # Localize to Eastern Time, then convert to UTC
                dt_et = dt_local.dt.tz_localize(
                    "America/New_York", nonexistent="shift_forward", ambiguous="NaT"
                )
                sched["_kickoff"] = dt_et.dt.tz_convert("UTC")
            except Exception:
                sched["_kickoff"] = pd.NaT
        else:
            # No known kickoff column; leave as NaT
            sched["_kickoff"] = pd.NaT
    # Optionally refresh stadium coordinates from Overpass API (set REFRESH_OSM_STADIUMS=1 to force refresh)
    refresh_osm = str(os.getenv("REFRESH_OSM_STADIUMS", "")).lower() in ("1", "true", "yes", "y")
    _apply_osm_coords_to_stadiums(STADIUMS, refresh=refresh_osm)

    lats, lons = [], []
    for _, r in sched.iterrows():
        la, lo = _extract_venue_latlon(r)
        lats.append(la)
        lons.append(lo)
    sched["_venue_lat"] = lats
    sched["_venue_lon"] = lons

    # Fallback to team stadium mapping
    # Vectorized fallback from home team -> stadium map
    try:
        map_lat = pd.Series({k: v["lat"] for k, v in STADIUMS.items()})
        map_lon = pd.Series({k: v["lon"] for k, v in STADIUMS.items()})
        if "home_team" in sched.columns:
            mask = sched["_venue_lat"].isna()
            sched.loc[mask, "_venue_lat"] = sched.loc[mask, "home_team"].map(map_lat)
            mask2 = sched["_venue_lon"].isna()
            sched.loc[mask2, "_venue_lon"] = sched.loc[mask2, "home_team"].map(map_lon)
    except Exception:
        pass

    # Limit to games within forecast horizon to avoid empty results (extend to ~15 days with daily fallback)
    try:
        now = pd.Timestamp.now(tz="UTC")
        # Keep games from 6 hours ago to 15 days ahead
        mask_time = (sched["_kickoff"].notna()) & (sched["_kickoff"] >= (now - pd.Timedelta(hours=6))) & (sched["_kickoff"] <= (now + pd.Timedelta(days=15)))
        sched = sched.loc[mask_time].copy()
    except Exception:
        pass

    # Additional fallback: ambiguous 'LA' code -> SoFi coords (same for Rams/Chargers)
    try:
        if "home_team" in sched.columns:
            mask_la = (sched["_venue_lat"].isna()) & (sched["_venue_lon"].isna()) & (sched["home_team"].astype(str).str.upper() == "LA")
            sched.loc[mask_la, "_venue_lat"] = 33.9535
            sched.loc[mask_la, "_venue_lon"] = -118.3387
    except Exception:
        pass

    # Diagnostics
    try:
        total = len(sched)
        missing_lat = int(sched["_venue_lat"].isna().sum())
        missing_lon = int(sched["_venue_lon"].isna().sum())
        missing_dt = int(sched["_kickoff"].isna().sum())
        # Show upcoming window sample
        print(f"[weather] window rows={total} missing(lat,lon,dt)=({missing_lat},{missing_lon},{missing_dt})")
        try:
            head_cols = [c for c in ["season","week","home_team","away_team","game_id","gameday","gametime","date","_kickoff","_venue_lat","_venue_lon"] if c in sched.columns]
            print("[weather] window head:\n", sched[head_cols].head())
        except Exception:
            pass
    except Exception:
        pass

    # Build per-game weather rows
    out_rows: List[Dict[str, Any]] = []
    for _, r in sched.iterrows():
        season = r.get("season")
        week = r.get("week")
        home_t = r.get("home_team")
        away_t = r.get("away_team")
        game_id = r.get("game_id")
        dt = r.get("_kickoff")
        lat = r.get("_venue_lat")
        lon = r.get("_venue_lon")

        if pd.isna(lat) or pd.isna(lon) or pd.isna(dt):
            continue

        # Fetch hourly forecast and pick nearest to kickoff; fallback to daily
        pick = None
        hourly = _tomorrow_forecast_hourly(float(lat), float(lon), api_key)
        if hourly is not None and not hourly.empty:
            pick = _pick_nearest_hourly(hourly, pd.to_datetime(dt, utc=True))
        from_hourly = pick is not None
        if pick is None:
            daily = _tomorrow_forecast_daily(float(lat), float(lon), api_key)
            if daily is not None and not daily.empty:
                pick = _pick_daily_for_date(daily, pd.to_datetime(dt, utc=True))
        if pick is None:
            continue

        # Compose feature row
        roof = None
        if isinstance(home_t, str) and home_t in STADIUMS:
            roof = STADIUMS[home_t].get("roof")
        # Stable game UUID based on season/week/home/away
        try:
            s_val = int(season) if pd.notna(season) else -1
            w_val = int(week) if pd.notna(week) else -1
            ht = (str(home_t) or "").upper()
            at = (str(away_t) or "").upper()
            game_key = f"{s_val}|{w_val}|{ht}|{at}"
            game_uid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"nfl-predictions:{game_key}"))
            # relaxed variant: collapse LAR/LAC -> LA
            def _relax(ab: str) -> str:
                if ab in ("LAR", "LAC", "LA"):
                    return "LA"
                return ab
            key_relax = f"{s_val}|{w_val}|{_relax(ht)}|{_relax(at)}"
            game_uid_relaxed = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"nfl-predictions:{key_relax}"))
        except Exception:
            game_uid = None
            game_uid_relaxed = None
        row: Dict[str, Any] = {
            "season": season,
            "week": week,
            "home_team": home_t,
            "away_team": away_t,
            "game_id": game_id,
            "game_uid": game_uid,
            "kickoff": pd.to_datetime(dt, utc=True),
            "venue_lat": float(lat),
            "venue_lon": float(lon),
            "roof": roof,
            "game_uid_relaxed": game_uid_relaxed,
        }
        # Map fields
        def _get(k: str) -> Any:
            try:
                return pick[k] if isinstance(pick, pd.Series) and (k in pick.index) else None
            except Exception:
                return None
        if from_hourly:
            row.update({
                "weather_temp_f": _get("temperature"),
                "weather_feelslike_f": _get("temperatureApparent"),
                "weather_dewpoint_f": _get("dewPoint"),
                "weather_humidity_pct": _get("humidity"),
                "weather_wind_mph": _get("windSpeed"),
                "weather_windgust_mph": _get("windGust"),
                "weather_wind_dir_deg": _get("windDirection"),
                "weather_precip_intensity_inph": _get("precipitationIntensity"),
                "weather_precip_type": _get("precipitationType"),
                "weather_precip_prob_pct": _get("precipitationProbability"),
                "weather_visibility_mi": _get("visibility"),
                "weather_cloud_cover_pct": _get("cloudCover"),
                "weather_code": _get("weatherCode"),
            })
        else:
            # Daily aggregates: use *Avg when present; tempAvg else mean(max,min)
            def _dget(*names: str):
                for nm in names:
                    v = _get(nm)
                    if v is not None:
                        return v
                return None
            tavg = _dget("temperatureAvg")
            if tavg is None:
                tmax = _dget("temperatureMax")
                tmin = _dget("temperatureMin")
                try:
                    if tmax is not None and tmin is not None:
                        tavg = (float(tmax) + float(tmin)) / 2.0
                    elif tmax is not None:
                        tavg = float(tmax)
                    elif tmin is not None:
                        tavg = float(tmin)
                except Exception:
                    tavg = None
            row.update({
                "weather_temp_f": tavg,
                "weather_feelslike_f": None,
                "weather_dewpoint_f": _dget("dewPointAvg"),
                "weather_humidity_pct": _dget("humidityAvg"),
                "weather_wind_mph": _dget("windSpeedAvg"),
                "weather_windgust_mph": _dget("windGustAvg"),
                "weather_wind_dir_deg": _dget("windDirectionAvg"),
                "weather_precip_intensity_inph": _dget("precipitationIntensityAvg"),
                "weather_precip_type": _dget("precipitationType"),
                "weather_precip_prob_pct": _dget("precipitationProbabilityAvg"),
                "weather_visibility_mi": _dget("visibilityAvg"),
                "weather_cloud_cover_pct": _dget("cloudCoverAvg"),
                "weather_code": _dget("weatherCodeMax"),
            })
        # Quarter temperature snapshots: kickoff and every 45 minutes (approx per quarter)
        try:
            targets = [
                ("kickoff", pd.to_datetime(dt, utc=True)),
                ("q1", pd.to_datetime(dt, utc=True) + timedelta(minutes=45)),
                ("q2", pd.to_datetime(dt, utc=True) + timedelta(minutes=90)),
                ("q3", pd.to_datetime(dt, utc=True) + timedelta(minutes=135)),
                ("q4", pd.to_datetime(dt, utc=True) + timedelta(minutes=180)),
            ]
            if from_hourly and hourly is not None and not hourly.empty:
                for label, tts in targets:
                    p = _pick_nearest_hourly(hourly, tts)
                    temp = None
                    if p is not None:
                        try:
                            temp = p.get("temperature") if isinstance(p, pd.Series) else None
                        except Exception:
                            temp = None
                    row[f"weather_temp_{label}_f"] = temp
            else:
                # Approximate quarter temps by daily average
                for label, _tts in targets:
                    row[f"weather_temp_{label}_f"] = row.get("weather_temp_f")
            # Backward-compatible alias: keep weather_temp_f as kickoff if not already set
            if row.get("weather_temp_f") is None:
                row["weather_temp_f"] = row.get("weather_temp_kickoff_f")
        except Exception:
            pass
        # Simple derived flags
        try:
            wind = float(row.get("weather_wind_mph") or 0)
            row["weather_is_windy"] = int(wind >= 15.0)
        except Exception:
            row["weather_is_windy"] = None
        try:
            temp = float(row.get("weather_temp_f") or 0)
            row["weather_is_cold"] = int(temp <= 32.0)
            row["weather_is_hot"] = int(temp >= 85.0)
        except Exception:
            row["weather_is_cold"] = None
            row["weather_is_hot"] = None
        try:
            pprob = float(row.get("weather_precip_prob_pct") or 0)
            row["weather_is_precip"] = int(pprob >= 50.0)
        except Exception:
            row["weather_is_precip"] = None

        out_rows.append(row)

    return pd.DataFrame(out_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, nargs="+", required=True, help="Season year(s) to fetch weather for")
    ap.add_argument("--api-key", dest="api_key", type=str, default=None, help="Tomorrow.io API key (or set TOMORROW_API_KEY)")
    ap.add_argument("--debug", action="store_true", help="Enable verbose debug output")
    args = ap.parse_args()

    configure_logging(args.debug)

    seasons = [int(s) for s in args.season]
    df = build_game_weather(seasons, api_key=args.api_key)
    if df is None or df.empty:
        print("[weather] No weather rows built (outside forecast window or missing schedule lat/lon)")
        return
    out_path = PROC_DIR / "weather_games.parquet"
    write_df(df, out_path)
    print(f"[weather] saved {len(df)} rows -> {out_path}")


if __name__ == "__main__":
    main()
