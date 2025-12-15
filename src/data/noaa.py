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

from __future__ import annotations

import argparse
import json
import math
import time
import logging
from collections import defaultdict
from numbers import Integral
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import uuid

import pandas as pd
import requests

from src.utils.io import RAW_DIR, PROC_DIR, REF_DIR, read_df, write_df
from src.utils.secrets import get_secret


API_BASE = "https://api.weather.gov"
OBS_CACHE_PATH = PROC_DIR / "noaa_weather.parquet"
STATION_CACHE_PATH = PROC_DIR / "noaa_station_cache.json"

DEFAULT_EMAIL = "mhanson13@gmail.com"
USER_AGENT = get_secret("NOAA_CONTACT_EMAIL", DEFAULT_EMAIL) or DEFAULT_EMAIL
HEADERS = {
    "User-Agent": f"nfl-predictions ({USER_AGENT})",
    "Accept": "application/geo+json",
}

REQUEST_TIMEOUT = 30


def _load_station_cache() -> Dict[str, List[str]]:
    if STATION_CACHE_PATH.exists():
        try:
            return json.loads(STATION_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_station_cache(cache: Dict[str, List[str]]) -> None:
    STATION_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATION_CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _normalize_key(lat: float, lon: float) -> str:
    return f"{round(lat, 3)}_{round(lon, 3)}"


def _nearest_stations(session: requests.Session, lat: float, lon: float, cache: Dict[str, List[str]], sleep: float) -> List[str]:
    key = _normalize_key(lat, lon)
    if key in cache:
        return cache[key]

    url = f"{API_BASE}/points/{lat:.4f},{lon:.4f}"
    resp = session.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    if resp.status_code >= 500:
        time.sleep(sleep)
        resp = session.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    stations_url = data.get("properties", {}).get("observationStations")
    if not stations_url:
        cache[key] = []
        return []

    time.sleep(sleep)
    resp = session.get(stations_url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    if resp.status_code >= 500:
        time.sleep(sleep)
        resp = session.get(stations_url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    stations = [
        feat.get("properties", {}).get("stationIdentifier")
        for feat in (resp.json().get("features") or [])
        if isinstance(feat, dict)
    ]
    stations = [s for s in stations if s]
    cache[key] = stations
    return stations


def _convert_c_to_f(value: Optional[float]) -> Optional[float]:
    if value is None or math.isnan(value):
        return None
    return value * 9.0 / 5.0 + 32.0


def _convert_ms_to_mph(value: Optional[float]) -> Optional[float]:
    if value is None or math.isnan(value):
        return None
    return value * 2.23693629


def _coerce_int(value: object, default: int = -1) -> int:
    """Convert common numeric representations to int while handling null-like values."""
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, float):
        if math.isnan(value):
            return default
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return int(text)
        except ValueError:
            try:
                float_val = float(text)
            except ValueError:
                return default
            if math.isnan(float_val):
                return default
            return int(float_val)
    return default


def _collect_observations(
    session: requests.Session,
    station: str,
    start_utc: datetime,
    end_utc: datetime,
    sleep: float,
) -> List[dict]:
    url = f"{API_BASE}/stations/{station}/observations"
    params = {
        "start": start_utc.isoformat().replace("+00:00", "Z"),
        "end": end_utc.isoformat().replace("+00:00", "Z"),
    }
    resp = session.get(url, headers=HEADERS, params=params, timeout=REQUEST_TIMEOUT)
    if resp.status_code >= 500:
        time.sleep(sleep)
        resp = session.get(url, headers=HEADERS, params=params, timeout=REQUEST_TIMEOUT)
    if resp.status_code == 404:
        return []
    resp.raise_for_status()
    data = resp.json()
    return data.get("features") or []


def _aggregate_observations(observations: List[dict]) -> Dict[str, Optional[float]]:
    if not observations:
        return {}

    speeds: List[Optional[float]] = []
    gusts: List[Optional[float]] = []
    temps: List[Optional[float]] = []
    humidities: List[Optional[float]] = []
    precip: List[Optional[float]] = []

    for obs in observations:
        props = obs.get("properties") or {}

        speed = props.get("windSpeed", {}).get("value")
        if speed is not None:
            speeds.append(_convert_ms_to_mph(speed))

        gust = props.get("windGust", {}).get("value")
        if gust is not None:
            gusts.append(_convert_ms_to_mph(gust))

        temp = props.get("temperature", {}).get("value")
        if temp is not None:
            temps.append(_convert_c_to_f(temp))

        rh = props.get("relativeHumidity", {}).get("value")
        if rh is not None:
            humidities.append(rh)

        precip_last = props.get("precipitationLastHour", {}).get("value")
        if precip_last is not None:
            precip.append(precip_last)

    def _mean(values: List[Optional[float]]) -> Optional[float]:
        clean = [v for v in values if v is not None and not math.isnan(v)]
        if not clean:
            return None
        return float(sum(clean) / len(clean))

    result = {
        "noaa_wind_mph": _mean(speeds),
        "noaa_wind_gust_mph": _mean(gusts),
        "noaa_temp_f": _mean(temps),
        "noaa_relative_humidity_pct": _mean(humidities),
        "noaa_precip_mm": _mean(precip),
    }
    if precip:
        result["noaa_precip_mm_total"] = float(sum(v for v in precip if v is not None))
    return result


def _resolve_game_times(schedule: pd.DataFrame, team_meta: pd.DataFrame) -> pd.DataFrame:
    df = schedule.copy()
    season_source: pd.Series[Any]
    if "season" in df.columns:
        season_source = df["season"]
    else:
        season_source = pd.Series(pd.NA, index=df.index, dtype="Int64")
    df["season"] = pd.to_numeric(season_source, errors="coerce").astype("Int64")
    df = df.dropna(subset=["season"])
    df["season"] = df["season"].astype(int)

    for col in ["gameday", "game_start", "gameDate", "game_date", "start_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=False)

    if "gametime" in df.columns:
        df["gametime"] = df["gametime"].astype(str)

    team_meta = team_meta[["team_abbr", "latitude", "longitude", "timezone_offset_hours"]].copy()
    team_meta["team_abbr"] = team_meta["team_abbr"].astype(str).str.upper()
    df = df.merge(
        team_meta.rename(columns={"team_abbr": "home_team", "latitude": "home_lat", "longitude": "home_lon", "timezone_offset_hours": "home_tz"}),
        on="home_team",
        how="left",
    )

    def compute_start(row: pd.Series) -> datetime:
        season_year = int(row["season"])
        fallback_start = datetime(season_year, 9, 1, 18, 0, tzinfo=None)

        for col in ("game_start", "start_date", "gameDate", "game_date"):
            val = row.get(col)
            if val is None:
                continue
            dt = pd.to_datetime(val, errors="coerce", utc=True)
            if pd.notna(dt):
                return dt.to_pydatetime()

        date_val_raw = row.get("gameday")
        if date_val_raw is None or pd.isna(date_val_raw):
            return fallback_start

        date_val = pd.to_datetime(date_val_raw, errors="coerce")
        if pd.isna(date_val):
            return fallback_start

        time_str = str(row.get("gametime") or "18:00")
        try:
            time_part = pd.to_datetime(time_str, format="%I:%M %p", errors="coerce")
        except Exception:
            time_part = pd.NaT

        if pd.notna(time_part):
            hour = int(time_part.hour)
            minute = int(time_part.minute)
        else:
            hour = 18
            minute = 0

        local_naive = datetime(date_val.year, date_val.month, date_val.day, hour, minute)
        tz_offset = row.get("home_tz")
        if tz_offset is None:
            tz_hours = -5.0
        else:
            try:
                tz_hours = float(tz_offset)
            except (TypeError, ValueError):
                tz_hours = -5.0
        try:
            tzinfo = timezone(timedelta(hours=tz_hours))
        except Exception:
            tzinfo = timezone.utc
        local_dt = datetime(
            date_val.year,
            date_val.month,
            date_val.day,
            hour,
            minute,
            tzinfo=tzinfo,
        )
        return local_dt.astimezone(timezone.utc)

    df["start_utc"] = df.apply(compute_start, axis=1)
    return df


def build_noaa_weather(
    seasons: Iterable[int],
    sleep: float,
    overwrite: bool,
) -> pd.DataFrame:
    seasons = sorted(set(int(s) for s in seasons))
    if not seasons:
        raise ValueError("No seasons specified for NOAA fetch.")

    schedule_path = RAW_DIR / "nfl_schedules.parquet"
    if not schedule_path.exists():
        raise FileNotFoundError("NFL schedules parquet not found; run data ingestion first.")
    schedule = read_df(schedule_path)
    schedule_season: pd.Series[Any]
    if "season" in schedule.columns:
        schedule_season = schedule["season"]
    else:
        schedule_season = pd.Series(pd.NA, index=schedule.index, dtype="Int64")
    schedule["season"] = pd.to_numeric(schedule_season, errors="coerce").astype("Int64")
    schedule = schedule.dropna(subset=["season"])
    schedule["season"] = schedule["season"].astype(int)
    schedule = schedule[schedule["season"].isin(seasons)].copy()

    if schedule.empty:
        print("[noaa] No schedule rows for requested seasons.")
        return pd.DataFrame()

    team_locations_path = REF_DIR / "team_locations.csv"
    if not team_locations_path.exists():
        raise FileNotFoundError(f"Team locations file missing: {team_locations_path}")
    team_meta = pd.read_csv(team_locations_path)

    schedule = _resolve_game_times(schedule, team_meta)
    required_cols = ["game_id", "season", "week", "home_team", "start_utc", "home_lat", "home_lon"]
    for col in required_cols:
        if col not in schedule.columns:
            raise ValueError(f"Schedule missing required column: {col}")

    existing_games: set[str] = set()
    existing_df = pd.DataFrame()
    if OBS_CACHE_PATH.exists() and not overwrite:
        existing_df = read_df(OBS_CACHE_PATH)
        if "game_id" in existing_df.columns and not existing_df.empty:
            existing_df = existing_df.copy()
            existing_df["game_id"] = existing_df["game_id"].astype(str)
            existing_games = set(existing_df["game_id"])

    rows: List[Dict[str, object]] = []
    station_cache = _load_station_cache()
    session = requests.Session()

    for processed_idx, (_, row) in enumerate(schedule.iterrows(), start=1):
        game_id = str(row["game_id"])
        if game_id in existing_games:
            continue

        lat = row.get("home_lat")
        lon = row.get("home_lon")
        if pd.isna(lat) or pd.isna(lon):
            continue
        lat = float(lat)
        lon = float(lon)

        stations = _nearest_stations(session, lat, lon, station_cache, sleep)
        if not stations:
            continue

        start_utc = row["start_utc"]
        if not isinstance(start_utc, datetime):
            continue
        start_window = start_utc - timedelta(hours=3)
        end_window = start_utc + timedelta(hours=3)

        observations: List[dict] = []
        fetch_status = "ok"
        primary_station: Optional[str] = None
        for station in stations[:3]:
            if primary_station is None:
                primary_station = station
            try:
                obs = _collect_observations(session, station, start_window, end_window, sleep)
            except Exception as exc:
                print(f"[noaa] station fetch failed for {station}: {exc}")
                continue
            if obs:
                observations = obs
                primary_station = station
                break
            time.sleep(sleep)

        if not observations:
            aggregated: Dict[str, Optional[float]] = {}
            fetch_status = "missing"
        else:
            aggregated = _aggregate_observations(observations)
            if not aggregated:
                fetch_status = "empty"

        start_utc_value = row.get("start_utc")
        if isinstance(start_utc_value, datetime):
            kickoff_utc = start_utc_value
        elif start_utc_value is not None and not pd.isna(start_utc_value):
            kickoff_utc = pd.to_datetime(start_utc_value, utc=True, errors="coerce")
        else:
            kickoff_utc = pd.NaT

        record = {
            "game_id": game_id,
            "season": int(row["season"]),
            "week": int(row.get("week", 0)) if not pd.isna(row.get("week")) else None,
            "home_team": row.get("home_team"),
            "away_team": row.get("away_team"),
            "start_utc": kickoff_utc,
            "kickoff": kickoff_utc,
            "noaa_station": primary_station or stations[0],
            "noaa_fetch_status": fetch_status,
            **aggregated,
        }
        wind = record.get("noaa_wind_mph")
        gust = record.get("noaa_wind_gust_mph")
        temp_f = record.get("noaa_temp_f")
        rain_mm = record.get("noaa_precip_mm_total", record.get("noaa_precip_mm"))

        if wind is not None:
            record["noaa_is_windy"] = int(wind >= 15.0)
        if gust is not None:
            record["noaa_is_gusty"] = int(gust >= 25.0)
        if temp_f is not None:
            record["noaa_is_cold"] = int(temp_f <= 32.0)
            record["noaa_is_hot"] = int(temp_f >= 85.0)
        if rain_mm is not None:
            record["noaa_is_precip"] = int(rain_mm > 0.0)

        try:
            season_val = _coerce_int(record.get("season"), default=-1)
            week_val = _coerce_int(record.get("week"), default=-1)
            home_abbr = str(record.get("home_team") or "").upper()
            away_abbr = str(record.get("away_team") or "").upper()
            base_key = f"{season_val}|{week_val}|{home_abbr}|{away_abbr}"
            record["game_uid"] = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"nfl-predictions:{base_key}"))

            def _relax(abbr: str) -> str:
                abbr = (abbr or "").upper()
                return "LA" if abbr in {"LA", "LAR", "LAC"} else abbr

            relaxed_key = f"{season_val}|{week_val}|{_relax(home_abbr)}|{_relax(away_abbr)}"
            record["game_uid_relaxed"] = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"nfl-predictions:{relaxed_key}"))
        except Exception:
            pass

        rows.append(record)
        if processed_idx % 100 == 0:
            print(f"[noaa] processed {processed_idx} games...")
        time.sleep(sleep)

    _save_station_cache(station_cache)

    new_df = pd.DataFrame(rows)
    if existing_df is not None and not existing_df.empty:
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["game_id"], keep="last")
    else:
        combined = new_df

    write_df(combined, OBS_CACHE_PATH)
    print(f"[noaa] saved {len(combined)} rows -> {OBS_CACHE_PATH}")
    return combined


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch historical weather from NOAA (weather.gov).")
    parser.add_argument("--seasons", type=int, nargs="+", help="Specific seasons to fetch (e.g. --seasons 2024 2025).")
    parser.add_argument("--start-season", type=int, help="Lower bound season (inclusive) if --seasons not provided.")
    parser.add_argument("--end-season", type=int, help="Upper bound season (inclusive) if --seasons not provided.")
    parser.add_argument("--sleep", type=float, default=0.25, help="Delay between API calls to avoid rate limits.")
    parser.add_argument("--overwrite", action="store_true", help="Ignore existing cache and refetch all games.")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging.")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s:%(name)s:%(message)s")
        logging.getLogger("urllib3").setLevel(logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

    if args.seasons:
        seasons = args.seasons
    else:
        if args.start_season is None and args.end_season is None:
            raise ValueError("Provide --seasons or both --start-season and --end-season.")
        start = args.start_season if args.start_season is not None else 2002
        end = args.end_season if args.end_season is not None else datetime.utcnow().year
        seasons = list(range(int(start), int(end) + 1))

    build_noaa_weather(seasons=seasons, sleep=float(args.sleep), overwrite=args.overwrite)


if __name__ == "__main__":
    main()
