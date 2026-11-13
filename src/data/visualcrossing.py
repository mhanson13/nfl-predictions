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
import math
import logging
import time
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Sequence
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

from src.utils.io import PROC_DIR, RAW_DIR, REF_DIR, read_df, write_df
from src.utils.pydantic_schemas import validate_dataframe
from src.utils.secrets import get_secret


API_BASE = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
CACHE_PATH = PROC_DIR / "visualcrossing_weather.parquet"

DEFAULT_ELEMENTS = ",".join(
    [
        "datetimeEpoch",
        "temp",
        "feelslike",
        "dew",
        "humidity",
        "windspeed",
        "windgust",
        "winddir",
        "precip",
        "preciptype",
        "snow",
        "cloudcover",
        "visibility",
        "severerisk",
        "conditions",
        "icon",
        "source",
        "stations",
    ]
)


def _resolve_game_times(schedule: pd.DataFrame, team_meta: pd.DataFrame | None = None) -> pd.DataFrame:
    """Attach the actual venue's coordinates and the UTC kickoff to each game.

    ``home_lat``/``home_lon`` are the coordinates of the venue the game is played
    at (see ``src.utils.venues``), not the home team's current stadium, and
    ``start_utc`` is ``gameday`` + ``gametime`` read as US/Eastern. ``team_meta``
    is accepted for backwards compatibility and unused.
    """
    from src.utils.venues import resolve_game_venues

    df = schedule.copy()
    df["season"] = pd.to_numeric(df.get("season"), errors="coerce").astype("Int64")
    df = df.dropna(subset=["season"])
    df["season"] = df["season"].astype(int)
    df = resolve_game_venues(df)
    df["home_lat"] = df["venue_lat"]
    df["home_lon"] = df["venue_lon"]
    fallback = pd.to_datetime(df["season"].astype(str) + "-09-01 18:00", utc=True)
    start = df["kickoff_utc"].fillna(fallback)
    df["start_utc"] = [ts.to_pydatetime() for ts in start]
    return df


def _make_uid(season: int, week: int, home: str, away: str) -> str:
    key = f"{season}|{week}|{home}|{away}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"nfl-predictions:{key}"))


def _make_uid_relaxed(season: int, week: int, home: str, away: str) -> str:
    def _relax(abbr: str) -> str:
        abbr = (abbr or "").upper()
        return "LA" if abbr in {"LA", "LAR", "LAC"} else abbr

    key = f"{season}|{week}|{_relax(home)}|{_relax(away)}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"nfl-predictions:{key}"))


def _quote_path_segment(value: str) -> str:
    return quote(value, safe="")


def _format_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S") + "Z"


def _coerce_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        if isinstance(value, str) and not value.strip():
            return None
        return float(value)
    except Exception:
        return None


def _extract_hours(payload: dict) -> List[dict]:
    hours: List[dict] = []
    if isinstance(payload.get("days"), Sequence):
        for day in payload["days"]:
            if isinstance(day, dict) and isinstance(day.get("hours"), Sequence):
                hours.extend([h for h in day["hours"] if isinstance(h, dict)])
    if not hours and isinstance(payload.get("hours"), Sequence):
        hours.extend([h for h in payload["hours"] if isinstance(h, dict)])
    return hours


def _select_window(hours: List[dict], start_utc: datetime, window_hours: float) -> List[tuple[datetime, dict]]:
    if not hours:
        return []
    start = start_utc - timedelta(hours=window_hours)
    end = start_utc + timedelta(hours=window_hours)
    selected: List[tuple[datetime, dict]] = []
    for hour in hours:
        epoch = _coerce_float(hour.get("datetimeEpoch"))
        if epoch is None:
            continue
        ts = datetime.fromtimestamp(epoch, tz=timezone.utc)
        if start <= ts <= end:
            selected.append((ts, hour))
    return selected


def _aggregate_hours(
    hours: List[tuple[datetime, dict]],
    start_utc: datetime,
) -> dict:
    if not hours:
        return {}

    records = []
    for ts, data in hours:
        record = {"ts": ts}
        for key in [
            "temp",
            "feelslike",
            "dew",
            "humidity",
            "windspeed",
            "windgust",
            "winddir",
            "precip",
            "snow",
            "cloudcover",
            "visibility",
            "severerisk",
            "conditions",
            "icon",
            "source",
            "stations",
            "preciptype",
        ]:
            record[key] = data.get(key)
        records.append(record)
    frame = pd.DataFrame(records)

    def _mean(col: str) -> Optional[float]:
        if col not in frame:
            return None
        series = pd.to_numeric(frame[col], errors="coerce")
        if series.notna().any():
            return float(series.mean())
        return None

    def _nearest(col: str) -> Optional[float]:
        if col not in frame:
            return None
        diffs = frame["ts"].apply(lambda ts: abs((ts - start_utc).total_seconds()))
        idx = diffs.idxmin() if len(diffs) else None
        if idx is None:
            return None
        value = frame.at[idx, col]
        return _coerce_float(value)

    precip_in = _mean("precip")
    precip_total_in = None
    if "precip" in frame:
        precip_total_in = _coerce_float(pd.to_numeric(frame["precip"], errors="coerce").sum())

    precip_types: set[str] = set()
    if "preciptype" in frame:
        for item in frame["preciptype"]:
            if isinstance(item, list):
                precip_types.update(str(x).lower() for x in item if x)
            elif isinstance(item, str):
                precip_types.add(item.lower())

    stations: set[str] = set()
    if "stations" in frame:
        for entry in frame["stations"]:
            if isinstance(entry, list):
                stations.update(str(x) for x in entry if x)
            elif isinstance(entry, str):
                for part in entry.split(","):
                    if part:
                        stations.add(part.strip())

    sources_counter: Counter[str] = Counter()
    if "source" in frame:
        for entry in frame["source"]:
            if entry:
                sources_counter[str(entry).lower()] += 1
    primary_source = sources_counter.most_common(1)
    primary_source_val = primary_source[0][0] if primary_source else None

    def _max(col: str) -> Optional[float]:
        if col not in frame:
            return None
        series = pd.to_numeric(frame[col], errors="coerce")
        if series.notna().any():
            return float(series.max())
        return None

    agg = {
        "weather_temp_f": _mean("temp"),
        "weather_feelslike_f": _mean("feelslike"),
        "weather_dewpoint_f": _mean("dew"),
        "weather_humidity_pct": _mean("humidity"),
        "weather_wind_mph": _mean("windspeed"),
        "weather_windgust_mph": _mean("windgust"),
        "weather_wind_dir_deg": _mean("winddir"),
        "weather_visibility_mi": _mean("visibility"),
        "weather_cloud_cover_pct": _mean("cloudcover"),
        "weather_severe_risk_pct": _max("severerisk"),
        "weather_temp_kickoff_f": _nearest("temp"),
        "weather_precip_intensity_inph": precip_in,
        "weather_precip_in_total": precip_total_in,
        "weather_precip_mm": precip_in * 25.4 if precip_in is not None else None,
        "weather_precip_mm_total": precip_total_in * 25.4 if precip_total_in is not None else None,
        "weather_precip_type": "|".join(sorted(precip_types)) if precip_types else None,
        "weather_conditions": _nearest("conditions"),
        "weather_icon": _nearest("icon"),
        "weather_source_detail": primary_source_val,
        "weather_station_list": "|".join(sorted(stations)) if stations else None,
        "weather_hours_observed": len(frame),
    }

    wind = agg.get("weather_wind_mph")
    if wind is not None:
        agg["weather_is_windy"] = int(wind >= 15.0)
    gust = agg.get("weather_windgust_mph")
    if gust is not None:
        agg["weather_is_gusty"] = int(gust >= 25.0)
    temp_f = agg.get("weather_temp_kickoff_f") or agg.get("weather_temp_f")
    if temp_f is not None:
        agg["weather_is_cold"] = int(temp_f <= 32.0)
        agg["weather_is_hot"] = int(temp_f >= 85.0)
    precip_mm_total = agg.get("weather_precip_mm_total")
    if precip_mm_total is not None:
        agg["weather_is_precip"] = int(precip_mm_total > 0.0)

    return {k: v for k, v in agg.items() if v is not None}


def _fetch_visualcrossing(
    session: requests.Session,
    lat: float,
    lon: float,
    start: datetime,
    end: datetime,
    api_key: str,
    *,
    elements: str = DEFAULT_ELEMENTS,
    unit_group: str = "us",
    retries: int = 2,
    timeout: float = 30.0,
) -> Optional[dict]:
    location = _quote_path_segment(f"{lat:.4f},{lon:.4f}")
    url = f"{API_BASE}/{location}/{_quote_path_segment(_format_iso(start))}/{_quote_path_segment(_format_iso(end))}"
    params = {
        "unitGroup": unit_group,
        "include": "hours",
        "elements": elements,
        "key": api_key,
        "contentType": "json",
    }

    backoff = 1.5
    attempt = 0
    while True:
        response = session.get(url, params=params, timeout=timeout)
        if response.status_code == 429 and attempt < retries:
            wait = backoff * (attempt + 1)
            print(f"[visualcrossing] rate limited (429) for {lat},{lon}; retrying in {wait:.1f}s")
            time.sleep(wait)
            attempt += 1
            continue
        if response.status_code >= 500 and attempt < retries:
            wait = backoff * (attempt + 1)
            print(f"[visualcrossing] server error {response.status_code}; retrying in {wait:.1f}s")
            time.sleep(wait)
            attempt += 1
            continue
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = ""
            try:
                detail = f" ({response.text[:256]})"
            except Exception:
                detail = ""
            print(f"[visualcrossing] request failed for {lat},{lon}: {exc}{detail}")
            return None
        break

    try:
        return response.json()
    except ValueError as exc:
        print(f"[visualcrossing] invalid JSON for {lat},{lon}: {exc}")
        return None


FORECAST_HORIZON_DAYS = 15  # Visual Crossing returns statistical normals beyond ~15 days
GAME_DURATION = timedelta(hours=4)
FINAL_DATA_SOURCES = {"obs"}  # values that are observations and never need refreshing


def select_refresh_games(
    existing: pd.DataFrame,
    schedule: pd.DataFrame,
    *,
    now: Optional[datetime] = None,
    horizon_days: float = FORECAST_HORIZON_DAYS,
) -> tuple[set[str], set[str], set[str]]:
    """Classify schedule games for a refresh run.

    Returns ``(stale_played, upcoming_in_horizon, beyond_horizon)``:

    * stale_played - cached, kickoff more than 4h ago, but the cached row is not an
      observation (a forecast or normal pulled before the game) -> refetch for obs;
    * upcoming_in_horizon - kickoff in the future within ``horizon_days`` -> refetch
      every run so the forecast is current;
    * beyond_horizon - kickoff further out -> do not fetch (would be climatology).
    """
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    kick = schedule[["game_id", "start_utc"]].copy()
    kick["game_id"] = kick["game_id"].astype(str)
    kick["start_utc"] = pd.to_datetime(kick["start_utc"], utc=True, errors="coerce")
    kick = kick.dropna(subset=["start_utc"]).drop_duplicates("game_id")
    played = kick["start_utc"] + GAME_DURATION <= now
    future = kick["start_utc"] > now
    in_horizon = kick["start_utc"] <= now + timedelta(days=horizon_days)
    upcoming = set(kick.loc[future & in_horizon, "game_id"])
    beyond = set(kick.loc[future & ~in_horizon, "game_id"])

    stale: set[str] = set()
    if existing is not None and not existing.empty and "game_id" in existing.columns:
        source = existing.get("vx_data_source")
        if source is None:
            source = pd.Series(None, index=existing.index)
        not_obs = ~source.astype(str).str.lower().isin(FINAL_DATA_SOURCES)
        cached_not_obs = set(existing.loc[not_obs, "game_id"].astype(str))
        stale = cached_not_obs & set(kick.loc[played, "game_id"])
    return stale, upcoming, beyond


def select_location_time_fixes(
    existing: pd.DataFrame,
    schedule: pd.DataFrame,
    *,
    max_km: float = 5.0,
    max_hours: float = 1.0,
) -> tuple[set[str], set[str]]:
    """Cached rows fetched at the wrong place or time.

    Returns ``(moved, misaligned)``: rows whose fetched coordinates are more than
    ``max_km`` from the actual venue, and rows whose stored kickoff differs from
    the actual kickoff by ``max_hours`` or more (older fetches parsed the 24h
    ``gametime`` incorrectly and centred the window on 18:00 home-standard time).
    """
    from src.utils.venues import haversine_km

    if existing is None or existing.empty or "game_id" not in existing.columns:
        return set(), set()
    cols = ["game_id", "home_lat", "home_lon", "start_utc"]
    sched = schedule[[c for c in cols if c in schedule.columns]].drop_duplicates("game_id").copy()
    sched["game_id"] = sched["game_id"].astype(str)
    cached = existing.drop_duplicates("game_id", keep="last").copy()
    cached["game_id"] = cached["game_id"].astype(str)
    merged = cached.merge(sched, on="game_id", how="inner", suffixes=("_cached", ""))
    moved: set[str] = set()
    if {"vx_latitude", "vx_longitude", "home_lat", "home_lon"}.issubset(merged.columns):
        dist = haversine_km(merged["vx_latitude"], merged["vx_longitude"], merged["home_lat"], merged["home_lon"])
        moved = set(merged.loc[np.nan_to_num(dist, nan=0.0) > max_km, "game_id"])
    misaligned: set[str] = set()
    if {"start_utc_cached", "start_utc"}.issubset(merged.columns):
        stored = pd.to_datetime(merged["start_utc_cached"], utc=True, errors="coerce")
        actual = pd.to_datetime(merged["start_utc"], utc=True, errors="coerce")
        gap = (stored - actual).abs().dt.total_seconds() / 3600.0
        misaligned = set(merged.loc[gap >= max_hours, "game_id"])
    return moved, misaligned


def build_visualcrossing_weather(
    seasons: Iterable[int],
    *,
    sleep: float,
    overwrite: bool,
    window_hours: float,
    only_missing: bool,
    max_games: Optional[int] = None,
    refresh: bool = True,
    horizon_days: float = FORECAST_HORIZON_DAYS,
    now: Optional[datetime] = None,
    refetch_misaligned: bool = False,
) -> pd.DataFrame:
    api_key = get_secret("VISUAL_CROSSING_API_KEY")
    if not api_key:
        raise RuntimeError("VISUAL_CROSSING_API_KEY not configured in environment or secrets.env")

    seasons = sorted(set(int(s) for s in seasons))
    if not seasons:
        raise ValueError("No seasons specified for Visual Crossing fetch.")

    schedule_path = RAW_DIR / "nfl_schedules.parquet"
    if not schedule_path.exists():
        raise FileNotFoundError("NFL schedules parquet not found; run schedule ingestion first.")
    schedule = read_df(schedule_path)
    schedule["season"] = pd.to_numeric(schedule.get("season"), errors="coerce").astype("Int64")
    schedule = schedule.dropna(subset=["season"])
    schedule["season"] = schedule["season"].astype(int)
    schedule = schedule[schedule["season"].isin(seasons)].copy()
    if schedule.empty:
        print("[visualcrossing] no schedule rows for requested seasons.")
        return pd.DataFrame()

    team_locations_path = REF_DIR / "team_locations.csv"
    if not team_locations_path.exists():
        raise FileNotFoundError(f"Team locations file missing: {team_locations_path}")
    team_meta = pd.read_csv(team_locations_path)

    schedule = _resolve_game_times(schedule, team_meta)
    required_cols = ["game_id", "season", "week", "home_team", "away_team", "start_utc", "home_lat", "home_lon"]
    for col in required_cols:
        if col not in schedule.columns:
            raise ValueError(f"Schedule missing required column: {col}")

    schedule["game_id"] = schedule["game_id"].astype(str)
    existing_df = pd.DataFrame()
    existing_games: set[str] = set()
    if CACHE_PATH.exists() and not overwrite:
        existing_df = read_df(CACHE_PATH)
        if "game_id" in existing_df.columns:
            existing_games = set(existing_df["game_id"].astype(str))

    refresh_games: set[str] = set()
    skip_games: set[str] = set()
    if refresh:
        stale, upcoming, beyond = select_refresh_games(existing_df, schedule, now=now, horizon_days=horizon_days)
        moved, misaligned = select_location_time_fixes(existing_df, schedule)
        refresh_games = stale | upcoming | moved
        if refetch_misaligned:
            refresh_games |= misaligned
        refresh_games -= beyond
        skip_games = beyond
        print(
            f"[visualcrossing] venue/time audit: {len(moved)} cached games fetched away from the actual venue -> refetch; "
            f"{len(misaligned)} cached games centred on the wrong kickoff time -> "
            + ("refetch" if refetch_misaligned else "kept (pass --refetch-misaligned-kickoff to refetch)")
        )
        print(
            f"[visualcrossing] refresh: {len(stale)} played games still holding forecasts -> refetch observations; "
            f"{len(upcoming)} upcoming games within {horizon_days:g} days -> refresh forecast; "
            f"{len(beyond)} games beyond the forecast horizon skipped"
        )
        # Rows for games beyond the horizon are climatology, not forecasts: drop them.
        if not existing_df.empty and beyond:
            dropped = existing_df["game_id"].astype(str).isin(beyond)
            if dropped.any():
                print(f"[visualcrossing] removing {int(dropped.sum())} cached rows beyond the forecast horizon")
                existing_df = existing_df.loc[~dropped].copy()
                existing_games -= beyond

    if only_missing:
        missing_games = set(schedule["game_id"].astype(str))
        noaa_path = PROC_DIR / "noaa_weather.parquet"
        if noaa_path.exists():
            noaa_df = read_df(noaa_path)
            if not noaa_df.empty and "game_id" in noaa_df.columns:
                noaa_df = noaa_df.copy()
                noaa_df["game_id"] = noaa_df["game_id"].astype(str)
                coverage_cols = [
                    col
                    for col in [
                        "noaa_temp_f",
                        "noaa_wind_mph",
                        "noaa_wind_gust_mph",
                        "noaa_precip_mm_total",
                        "noaa_precip_mm",
                    ]
                    if col in noaa_df.columns
                ]
                if coverage_cols:
                    covered = noaa_df.loc[
                        noaa_df[coverage_cols].apply(lambda row: row.notna().any(), axis=1),
                        "game_id",
                    ].astype(str)
                    missing_games.difference_update(set(covered))
                else:
                    missing_games.difference_update(set(noaa_df["game_id"]))
        if existing_df is not None and not existing_df.empty and "game_id" in existing_df.columns:
            missing_games.difference_update(set(existing_df["game_id"].astype(str)))
        missing_games |= refresh_games & set(schedule["game_id"].astype(str))
        missing_games -= skip_games
        if not missing_games:
            print("[visualcrossing] all requested games already covered by NOAA/VisualCrossing cache; nothing to fetch.")
            if not existing_df.empty:
                return existing_df
            return pd.DataFrame()
        schedule = schedule[schedule["game_id"].isin(missing_games)].copy()
        if schedule.empty:
            print("[visualcrossing] no remaining games require Visual Crossing backfill.")
            if not existing_df.empty:
                return existing_df
            return pd.DataFrame()

    session = requests.Session()
    rows: List[dict] = []
    processed = 0

    for idx, row in schedule.iterrows():
        if max_games is not None and processed >= max_games:
            break

        game_id = str(row["game_id"])
        if game_id in skip_games:
            continue
        if game_id in existing_games and game_id not in refresh_games:
            continue
        lat = row.get("home_lat")
        lon = row.get("home_lon")
        if pd.isna(lat) or pd.isna(lon):
            continue
        lat = float(lat)
        lon = float(lon)
        start_utc = row.get("start_utc")
        if not isinstance(start_utc, datetime):
            continue
        if start_utc.tzinfo is None:
            start_utc = start_utc.replace(tzinfo=timezone.utc)

        start_window = start_utc - timedelta(hours=window_hours)
        end_window = start_utc + timedelta(hours=window_hours)

        payload = _fetch_visualcrossing(session, lat, lon, start_window, end_window, api_key)
        if not payload:
            time.sleep(sleep)
            continue

        hours = _extract_hours(payload)
        selected = _select_window(hours, start_utc, window_hours)
        agg = _aggregate_hours(selected, start_utc)
        if not agg:
            time.sleep(sleep)
            continue

        season = int(row["season"])
        week = int(row.get("week", 0)) if not pd.isna(row.get("week")) else -1
        home_team = str(row.get("home_team") or "").upper()
        away_team = str(row.get("away_team") or "").upper()

        record = {
            "game_id": game_id,
            "season": season,
            "week": week,
            "home_team": home_team,
            "away_team": away_team,
            "start_utc": start_utc,
            "kickoff": start_utc,
            "weather_source": "visualcrossing",
            "vx_resolved_address": payload.get("resolvedAddress"),
            "vx_timezone": payload.get("timezone"),
            "vx_tzoffset": payload.get("tzoffset"),
            "vx_latitude": payload.get("latitude"),
            "vx_longitude": payload.get("longitude"),
            "vx_data_source": agg.get("weather_source_detail"),
            "vx_fetched_at": datetime.now(timezone.utc),
            "venue_id": row.get("venue_id"),
            "venue_name": row.get("venue_name"),
        }
        record.update(agg)
        record["game_uid"] = _make_uid(season, week, home_team, away_team)
        record["game_uid_relaxed"] = _make_uid_relaxed(season, week, home_team, away_team)

        rows.append(record)
        processed += 1

        if idx % 100 == 0:
            print(f"[visualcrossing] processed {idx} games...")
        time.sleep(sleep)

    new_df = pd.DataFrame(rows)
    if not new_df.empty:
        new_df["weather_precip_mm"] = pd.to_numeric(new_df.get("weather_precip_mm"), errors="coerce")
        new_df["weather_precip_mm_total"] = pd.to_numeric(new_df.get("weather_precip_mm_total"), errors="coerce")

    if existing_df is not None and not existing_df.empty:
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["game_id"], keep="last")
    else:
        combined = new_df

    if not new_df.empty:
        report = validate_dataframe(new_df, "visualcrossing_weather", log_errors=False)
        print(f"[visualcrossing] schema validation: {report}")

    write_df(combined, CACHE_PATH)
    print(f"[visualcrossing] saved {len(combined)} rows -> {CACHE_PATH}")
    return combined


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch historical weather from Visual Crossing timeline API.")
    parser.add_argument("--seasons", type=int, nargs="+", help="Specific seasons to fetch (e.g. --seasons 2024 2025).")
    parser.add_argument("--start-season", type=int, help="Lower bound season (inclusive) if --seasons not provided.")
    parser.add_argument("--end-season", type=int, help="Upper bound season (inclusive) if --seasons not provided.")
    parser.add_argument("--sleep", type=float, default=0.2, help="Seconds to wait between API calls.")
    parser.add_argument("--window-hours", type=float, default=3.0, help="Hours before/after kickoff to capture observations.")
    parser.add_argument("--overwrite", action="store_true", help="Refetch all games and ignore existing cache.")
    parser.add_argument("--only-missing", action="store_true", help="Only request games without NOAA or existing Visual Crossing coverage.")
    parser.add_argument("--max-games", type=int, help="Optional limit on number of games to fetch (for testing).")
    parser.add_argument(
        "--no-refresh",
        action="store_true",
        help="Do not refetch played games that still hold forecasts or upcoming forecasts (old behaviour).",
    )
    parser.add_argument(
        "--refetch-misaligned-kickoff",
        action="store_true",
        help="Also refetch cached games whose weather window was centred on the wrong kickoff time (one-time backfill).",
    )
    parser.add_argument(
        "--forecast-horizon-days",
        type=float,
        default=FORECAST_HORIZON_DAYS,
        help="Refresh upcoming games within this many days; games further out are not fetched (climatology).",
    )
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s:%(name)s:%(message)s")
        logging.getLogger("urllib3").setLevel(logging.WARNING)
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

    build_visualcrossing_weather(
        seasons=seasons,
        sleep=float(args.sleep),
        overwrite=args.overwrite,
        window_hours=float(args.window_hours),
        only_missing=args.only_missing,
        max_games=args.max_games,
        refresh=not args.no_refresh,
        horizon_days=float(args.forecast_horizon_days),
        refetch_misaligned=args.refetch_misaligned_kickoff,
    )


if __name__ == "__main__":
    main()
