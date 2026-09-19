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


def _resolve_game_times(schedule: pd.DataFrame, team_meta: pd.DataFrame) -> pd.DataFrame:
    df = schedule.copy()
    df["season"] = pd.to_numeric(df.get("season"), errors="coerce").astype("Int64")
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
        team_meta.rename(
            columns={
                "team_abbr": "home_team",
                "latitude": "home_lat",
                "longitude": "home_lon",
                "timezone_offset_hours": "home_tz",
            }
        ),
        on="home_team",
        how="left",
    )

    def compute_start(row: pd.Series) -> datetime:
        for col in ("game_start", "start_date", "gameDate", "game_date"):
            val = row.get(col)
            if pd.notna(val):
                dt = pd.to_datetime(val, errors="coerce", utc=True)
                if pd.notna(dt):
                    return dt.to_pydatetime()

        date_val = row.get("gameday")
        if pd.isna(date_val):
            return datetime(row["season"], 9, 1, 18, 0, tzinfo=timezone.utc)

        date_val = pd.to_datetime(date_val, errors="coerce")
        if pd.isna(date_val):
            return datetime(row["season"], 9, 1, 18, 0, tzinfo=timezone.utc)

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

        tz_offset = row.get("home_tz")
        try:
            tz_hours = float(tz_offset)
        except Exception:
            tz_hours = -5.0
        try:
            tzinfo = timezone(timedelta(hours=tz_hours))
        except Exception:
            tzinfo = timezone.utc

        local_dt = datetime(date_val.year, date_val.month, date_val.day, hour, minute, tzinfo=tzinfo)
        return local_dt.astimezone(timezone.utc)

    df["start_utc"] = df.apply(compute_start, axis=1)
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


def build_visualcrossing_weather(
    seasons: Iterable[int],
    *,
    sleep: float,
    overwrite: bool,
    window_hours: float,
    only_missing: bool,
    max_games: Optional[int] = None,
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
        if game_id in existing_games:
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
    )


if __name__ == "__main__":
    main()
