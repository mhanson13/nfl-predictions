from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
from typing import Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.data.odds_api_historical import (
    DEFAULT_BOOKMAKERS,
    DEFAULT_PLAYER_PROP_MARKETS,
    DEFAULT_REGIONS,
    DEFAULT_RETRIES,
    DEFAULT_SLEEP,
    DEFAULT_SPORT_KEY,
    REQUEST_TIMEOUT,
    fetch_historical_event_markets,
    fetch_historical_event_odds,
    fetch_historical_events,
)
from src.predict.utils import moneyline_to_prob
from src.utils.io import PROC_DIR, RAW_DIR, read_df, write_df
from src.utils.odds import ODDS_API_PAID_KEY_SECRET, get_odds_api_paid_key
from src.utils.teams import get_team_abbr_from_name, normalize_team_abbr


DEFAULT_SCHEDULE_PATH = RAW_DIR / "nfl_schedules.parquet"
DEFAULT_MANIFEST_PATH = PROC_DIR / "oddsapi_player_prop_backfill_manifest.parquet"
DEFAULT_EVENTS_OUTPUT_PATH = RAW_DIR / "oddsapi_historical_events_backfill.parquet"
DEFAULT_MARKETS_OUTPUT_PATH = RAW_DIR / "oddsapi_historical_event_markets_backfill.parquet"
DEFAULT_ODDS_OUTPUT_PATH = RAW_DIR / "oddsapi_historical_player_props_backfill.parquet"
DEFAULT_NORMALIZED_OUTPUT_PATH = PROC_DIR / "oddsapi_historical_player_prop_lines.parquet"
DEFAULT_KICKOFF_TZ = "America/New_York"
DEFAULT_SNAPSHOT_MINUTES_BEFORE = 10
DEFAULT_START_SEASON = 2023
LINE_BALANCE_DISTANCE_LIMIT = 0.15

API_MODES = {"discover-events", "discover-markets", "fetch-odds", "all"}

ODDSAPI_MARKET_TO_MODEL_MARKET = {
    "player_pass_yds": "qb_passing_yards",
    "player_rush_yds": "rb_rushing_yards",
    "player_reception_yds": "wrte_receiving_yards",
    "player_sacks": "def_sacks",
}

MANIFEST_COLUMNS = [
    "season",
    "week",
    "game_type",
    "game_id",
    "gameday",
    "gametime",
    "away_team",
    "home_team",
    "kickoff_time_utc",
    "snapshot_time_utc",
    "events_snapshot_time_utc",
    "commence_time_from_utc",
    "commence_time_to_utc",
    "target_markets",
    "target_bookmakers",
    "oddsapi_event_id",
    "events_status",
    "event_markets_status",
    "event_odds_status",
]

NORMALIZED_LINE_COLUMNS = [
    "line_source",
    "season",
    "week",
    "game_id",
    "oddsapi_event_id",
    "requested_date",
    "snapshot_timestamp",
    "commence_time",
    "home_team",
    "away_team",
    "market",
    "market_key",
    "player_name",
    "player_name_key",
    "player_name_prefix_key",
    "line",
    "implied_probability",
    "sportsbook",
    "over_odds",
    "under_odds",
    "line_balance_distance",
    "line_is_comparable",
    "line_updated_at",
]


def _utc_iso(value: pd.Timestamp | datetime) -> str:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize(timezone.utc)
    return ts.tz_convert(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _timestamp_naive(value: str | None) -> pd.Timestamp:
    if value:
        ts = pd.Timestamp(value)
        if ts.tzinfo is not None:
            return ts.tz_convert(None)
        return ts.tz_localize(None)
    return pd.Timestamp(datetime.now(timezone.utc).date())


def _kickoff_utc(gameday: object, gametime: object, *, kickoff_tz: str) -> pd.Timestamp:
    date_text = str(gameday).strip()
    time_text = str(gametime).strip() if gametime is not None and not pd.isna(gametime) else "12:00"
    naive = pd.Timestamp(f"{date_text} {time_text}")
    localized = naive.to_pydatetime().replace(tzinfo=ZoneInfo(kickoff_tz))
    return pd.Timestamp(localized).tz_convert(timezone.utc)


def _load_schedule(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Schedule file not found: {path}")
    return read_df(path)


def _read_optional_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return read_df(path)


def _json_list(values: object) -> list[str]:
    if values is None:
        return []
    if isinstance(values, (list, tuple, set)):
        return [str(value).strip() for value in values if str(value).strip()]
    try:
        if pd.isna(values):
            return []
    except (TypeError, ValueError):
        pass
    text = str(values).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = []
        if isinstance(parsed, list):
            return [str(value).strip() for value in parsed if str(value).strip()]
    return [value.strip() for value in text.split(",") if value.strip()]


def _clean_team(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().upper()
    if not text or text in {"NONE", "NAN", "NA", "<NA>"}:
        return None
    try:
        return normalize_team_abbr(text)
    except Exception:
        return text


def _event_team_to_schedule_abbr(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    abbr = get_team_abbr_from_name(text)
    if abbr == "LAR":
        return "LA"
    return _clean_team(abbr or text)


def _player_name_parts(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    if "," in text:
        parts = [part.strip() for part in text.split(",", 1)]
        if len(parts) == 2 and parts[0] and parts[1]:
            text = f"{parts[1]} {parts[0]}"
    text = pd.Series([text]).str.replace(r"\b(jr|sr|ii|iii|iv|v)\.?\b", "", regex=True).iloc[0]
    text = pd.Series([text]).str.replace(r"[^A-Za-z0-9]+", " ", regex=True).iloc[0]
    return str(text).strip().lower().split()


def _player_name_key(value: object) -> str | None:
    parts = _player_name_parts(value)
    if not parts:
        return None
    first_initial = parts[0][0]
    last = "".join(parts[1:]) if len(parts) > 1 else parts[0]
    return f"{first_initial}:{last}"


def _player_name_prefix_key(value: object, *, prefix_len: int = 2) -> str | None:
    parts = _player_name_parts(value)
    if not parts:
        return None
    first = parts[0][: max(1, int(prefix_len))]
    last = "".join(parts[1:]) if len(parts) > 1 else parts[0]
    return f"{first}:{last}"


def _line_balance_distance(frame: pd.DataFrame) -> pd.Series:
    over_prob = moneyline_to_prob(frame["over_odds"])
    under_prob = moneyline_to_prob(frame["under_odds"])
    total_prob = over_prob + under_prob
    implied = np.where(total_prob > 0, over_prob / total_prob, np.nan)
    has_two_way = pd.to_numeric(frame["over_odds"], errors="coerce").notna() & pd.to_numeric(
        frame["under_odds"], errors="coerce"
    ).notna()
    return pd.Series(np.where(has_two_way, np.abs(implied - 0.5), np.inf), index=frame.index)


def _first_present(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    result = pd.Series(pd.NA, index=frame.index, dtype="object")
    for col in columns:
        if col in frame.columns:
            result = result.where(result.notna(), frame[col])
    return result


def _append_and_write(
    existing: pd.DataFrame,
    new_rows: pd.DataFrame,
    output: Path,
    *,
    subset: Sequence[str],
) -> pd.DataFrame:
    frames = [frame for frame in [existing, new_rows] if frame is not None and not frame.empty]
    combined = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if not combined.empty:
        keep_subset = [col for col in subset if col in combined.columns]
        if keep_subset:
            combined = combined.drop_duplicates(subset=keep_subset, keep="last")
    write_df(combined, output)
    return combined


def build_backfill_manifest(
    schedule: pd.DataFrame,
    *,
    start_season: int = DEFAULT_START_SEASON,
    end_season: int | None = None,
    current_date: str | None = None,
    include_postseason: bool = True,
    snapshot_minutes_before: int = DEFAULT_SNAPSHOT_MINUTES_BEFORE,
    kickoff_tz: str = DEFAULT_KICKOFF_TZ,
    markets: Sequence[str] = DEFAULT_PLAYER_PROP_MARKETS,
    bookmakers: Sequence[str] = DEFAULT_BOOKMAKERS,
) -> pd.DataFrame:
    if schedule.empty:
        return pd.DataFrame(columns=MANIFEST_COLUMNS)

    frame = schedule.copy()
    frame["season"] = pd.to_numeric(frame["season"], errors="coerce")
    frame["week"] = pd.to_numeric(frame["week"], errors="coerce")
    frame = frame[frame["season"].notna() & frame["week"].notna()].copy()
    frame["season"] = frame["season"].astype(int)
    frame["week"] = frame["week"].astype(int)

    if end_season is None:
        end_season = int(frame["season"].max())
    frame = frame[frame["season"].between(int(start_season), int(end_season))].copy()
    if not include_postseason and "game_type" in frame.columns:
        frame = frame[frame["game_type"].eq("REG")].copy()

    frame["gameday_dt"] = pd.to_datetime(frame["gameday"], errors="coerce")
    cutoff = _timestamp_naive(current_date)
    frame = frame[frame["gameday_dt"].notna() & frame["gameday_dt"].le(cutoff)].copy()
    if frame.empty:
        return pd.DataFrame(columns=MANIFEST_COLUMNS)

    gametime = frame["gametime"] if "gametime" in frame.columns else pd.Series("12:00", index=frame.index)
    frame["kickoff_ts"] = [
        _kickoff_utc(gameday, time_value, kickoff_tz=kickoff_tz)
        for gameday, time_value in zip(frame["gameday"], gametime)
    ]
    frame["snapshot_ts"] = frame["kickoff_ts"] - pd.to_timedelta(int(snapshot_minutes_before), unit="m")

    week_windows = (
        frame.groupby(["season", "week"], as_index=False)
        .agg(
            events_snapshot_ts=("snapshot_ts", "min"),
            commence_from_ts=("kickoff_ts", "min"),
            commence_to_ts=("kickoff_ts", "max"),
        )
        .reset_index(drop=True)
    )
    week_windows["commence_from_ts"] = week_windows["commence_from_ts"] - pd.to_timedelta(6, unit="h")
    week_windows["commence_to_ts"] = week_windows["commence_to_ts"] + pd.to_timedelta(12, unit="h")
    frame = frame.merge(week_windows, on=["season", "week"], how="left")

    out = pd.DataFrame(
        {
            "season": frame["season"].astype(int),
            "week": frame["week"].astype(int),
            "game_type": frame.get("game_type", pd.Series(pd.NA, index=frame.index)),
            "game_id": frame["game_id"].astype("string"),
            "gameday": frame["gameday"].astype("string"),
            "gametime": frame.get("gametime", pd.Series(pd.NA, index=frame.index)).astype("string"),
            "away_team": frame["away_team"].astype("string"),
            "home_team": frame["home_team"].astype("string"),
            "kickoff_time_utc": frame["kickoff_ts"].apply(_utc_iso),
            "snapshot_time_utc": frame["snapshot_ts"].apply(_utc_iso),
            "events_snapshot_time_utc": frame["events_snapshot_ts"].apply(_utc_iso),
            "commence_time_from_utc": frame["commence_from_ts"].apply(_utc_iso),
            "commence_time_to_utc": frame["commence_to_ts"].apply(_utc_iso),
            "target_markets": json.dumps(list(markets)),
            "target_bookmakers": json.dumps(list(bookmakers)),
            "oddsapi_event_id": pd.NA,
            "events_status": "pending",
            "event_markets_status": "pending",
            "event_odds_status": "pending",
        }
    )
    out = out[MANIFEST_COLUMNS].sort_values(
        ["season", "week", "kickoff_time_utc", "game_id"],
        kind="stable",
    )
    return out.reset_index(drop=True)


def estimate_quota(manifest: pd.DataFrame, *, markets: Sequence[str]) -> dict[str, int]:
    if manifest.empty:
        return {
            "events_requests": 0,
            "event_markets_requests": 0,
            "event_odds_requests_worst_case": 0,
            "discovery_requests": 0,
            "event_odds_credits_worst_case": 0,
        }
    week_groups = manifest[["season", "week"]].drop_duplicates()
    event_count = len(manifest)
    event_odds_credits = event_count * len(markets) * 10
    return {
        "events_requests": int(len(week_groups)),
        "event_markets_requests": int(event_count),
        "event_odds_requests_worst_case": int(event_count),
        "discovery_requests": int(len(week_groups) + event_count),
        "event_odds_credits_worst_case": int(event_odds_credits),
    }


def match_events_to_manifest(manifest: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    if manifest.empty:
        return manifest.copy()
    out = manifest.copy()
    if "oddsapi_event_id" not in out.columns:
        out["oddsapi_event_id"] = pd.NA
    required = {"requested_season", "requested_week", "home_team", "away_team", "event_id"}
    if events.empty or not required.issubset(events.columns):
        return out

    local = out.reset_index(names="_manifest_row").copy()
    local["_home_key"] = local["home_team"].apply(_clean_team)
    local["_away_key"] = local["away_team"].apply(_clean_team)
    local["_kickoff_ts"] = pd.to_datetime(local["kickoff_time_utc"], errors="coerce", utc=True)

    ev = events.copy()
    ev["requested_season"] = pd.to_numeric(ev["requested_season"], errors="coerce")
    ev["requested_week"] = pd.to_numeric(ev["requested_week"], errors="coerce")
    ev = ev[ev["requested_season"].notna() & ev["requested_week"].notna()].copy()
    if ev.empty:
        return out
    ev["requested_season"] = ev["requested_season"].astype(int)
    ev["requested_week"] = ev["requested_week"].astype(int)
    ev["_home_key"] = ev["home_team"].apply(_event_team_to_schedule_abbr)
    ev["_away_key"] = ev["away_team"].apply(_event_team_to_schedule_abbr)
    ev["_event_ts"] = pd.to_datetime(ev["commence_time"], errors="coerce", utc=True)

    merged = local.merge(
        ev[
            [
                "requested_season",
                "requested_week",
                "event_id",
                "_home_key",
                "_away_key",
                "_event_ts",
            ]
        ],
        left_on=["season", "week", "_home_key", "_away_key"],
        right_on=["requested_season", "requested_week", "_home_key", "_away_key"],
        how="left",
    )
    matched = merged[merged["event_id"].notna()].copy()
    if not matched.empty:
        matched["_time_diff_seconds"] = (matched["_kickoff_ts"] - matched["_event_ts"]).abs().dt.total_seconds()
        matched = matched.sort_values(["_manifest_row", "_time_diff_seconds"], kind="stable")
        matched = matched.drop_duplicates("_manifest_row", keep="first")
        row_ids = matched["_manifest_row"].astype(int).to_numpy()
        out.loc[row_ids, "oddsapi_event_id"] = matched["event_id"].to_numpy()

    out["events_status"] = "pending"
    out.loc[out["oddsapi_event_id"].notna(), "events_status"] = "matched"
    weeks_with_events = {
        (int(season), int(week))
        for season, week in ev[["requested_season", "requested_week"]].drop_duplicates().itertuples(index=False)
    }
    if weeks_with_events:
        no_match_mask = out["oddsapi_event_id"].isna() & out.apply(
            lambda row: (int(row["season"]), int(row["week"])) in weeks_with_events,
            axis=1,
        )
        out.loc[no_match_mask, "events_status"] = "no_match"
    return out[MANIFEST_COLUMNS]


def update_market_status(manifest: pd.DataFrame, markets: pd.DataFrame) -> pd.DataFrame:
    if manifest.empty:
        return manifest.copy()
    out = manifest.copy()
    out["event_markets_status"] = "pending"
    missing_event = out["oddsapi_event_id"].isna()
    out.loc[missing_event, "event_markets_status"] = "missing_event"
    if markets.empty or "event_id" not in markets.columns:
        return out

    market_groups = markets.groupby("event_id")["market_key"].agg(lambda values: set(values.dropna().astype(str)))
    for idx, row in out[out["oddsapi_event_id"].notna()].iterrows():
        event_id = str(row["oddsapi_event_id"])
        if event_id not in market_groups:
            continue
        target_markets = set(_json_list(row.get("target_markets")))
        available = market_groups[event_id]
        out.loc[idx, "event_markets_status"] = "available" if target_markets & available else "no_target_markets"
    return out[MANIFEST_COLUMNS]


def update_odds_status(manifest: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    if manifest.empty:
        return manifest.copy()
    out = manifest.copy()
    out["event_odds_status"] = "pending"
    out.loc[out["oddsapi_event_id"].isna(), "event_odds_status"] = "missing_event"
    out.loc[out["event_markets_status"].eq("no_target_markets"), "event_odds_status"] = "no_target_markets"
    if odds.empty or "event_id" not in odds.columns:
        return out[MANIFEST_COLUMNS]
    done_events = set(odds["event_id"].dropna().astype(str))
    out.loc[out["oddsapi_event_id"].astype("string").isin(done_events), "event_odds_status"] = "done"
    return out[MANIFEST_COLUMNS]


def _event_discovery_jobs(manifest: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "season",
        "week",
        "events_snapshot_time_utc",
        "commence_time_from_utc",
        "commence_time_to_utc",
    ]
    return manifest[cols].drop_duplicates().sort_values(["season", "week"], kind="stable").reset_index(drop=True)


def discover_events(
    manifest: pd.DataFrame,
    *,
    api_key: str | None,
    output: Path,
    sport_key: str,
    retries: int,
    sleep: float,
    timeout: float,
    dry_run: bool,
    resume: bool,
    max_requests: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    existing = _read_optional_frame(output)
    jobs = _event_discovery_jobs(manifest)
    if resume and not existing.empty and {"requested_season", "requested_week"}.issubset(existing.columns):
        done = {
            (int(season), int(week))
            for season, week in existing[["requested_season", "requested_week"]].dropna().itertuples(index=False)
        }
        jobs = jobs[~jobs.apply(lambda row: (int(row["season"]), int(row["week"])) in done, axis=1)]
    if max_requests is not None:
        jobs = jobs.head(max(0, int(max_requests)))
    if dry_run:
        return match_events_to_manifest(manifest, existing), existing, int(len(jobs))

    if not api_key:
        raise RuntimeError(f"{ODDS_API_PAID_KEY_SECRET} not configured. Add it to secrets.env.")

    new_frames: list[pd.DataFrame] = []
    for row in jobs.itertuples(index=False):
        frame = fetch_historical_events(
            api_key=api_key,
            date=row.events_snapshot_time_utc,
            sport_key=sport_key,
            commence_time_from=row.commence_time_from_utc,
            commence_time_to=row.commence_time_to_utc,
            retries=retries,
            sleep=sleep,
            timeout=timeout,
        )
        if not frame.empty:
            frame.insert(0, "requested_season", int(row.season))
            frame.insert(1, "requested_week", int(row.week))
            new_frames.append(frame)
        time.sleep(max(0.0, float(sleep)))

    new_rows = pd.concat(new_frames, ignore_index=True, sort=False) if new_frames else pd.DataFrame()
    combined = _append_and_write(
        existing,
        new_rows,
        output,
        subset=["requested_season", "requested_week", "event_id"],
    )
    return match_events_to_manifest(manifest, combined), combined, int(len(jobs))


def discover_event_markets(
    manifest: pd.DataFrame,
    *,
    api_key: str | None,
    output: Path,
    sport_key: str,
    regions: Sequence[str],
    retries: int,
    sleep: float,
    timeout: float,
    dry_run: bool,
    resume: bool,
    max_requests: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    existing = _read_optional_frame(output)
    jobs = manifest[manifest["oddsapi_event_id"].notna()].copy()
    if resume and not existing.empty and {"event_id", "requested_date"}.issubset(existing.columns):
        done = {
            (str(event_id), str(requested_date))
            for event_id, requested_date in existing[["event_id", "requested_date"]].dropna().itertuples(index=False)
        }
        jobs = jobs[
            ~jobs.apply(lambda row: (str(row["oddsapi_event_id"]), str(row["snapshot_time_utc"])) in done, axis=1)
        ]
    if max_requests is not None:
        jobs = jobs.head(max(0, int(max_requests)))
    if dry_run:
        return update_market_status(manifest, existing), existing, int(len(jobs))

    if not api_key:
        raise RuntimeError(f"{ODDS_API_PAID_KEY_SECRET} not configured. Add it to secrets.env.")

    new_frames: list[pd.DataFrame] = []
    for row in jobs.itertuples(index=False):
        bookmakers = _json_list(row.target_bookmakers)
        frame = fetch_historical_event_markets(
            api_key=api_key,
            event_id=str(row.oddsapi_event_id),
            date=str(row.snapshot_time_utc),
            sport_key=sport_key,
            bookmakers=bookmakers,
            regions=regions,
            retries=retries,
            sleep=sleep,
            timeout=timeout,
        )
        if not frame.empty:
            frame.insert(0, "requested_season", int(row.season))
            frame.insert(1, "requested_week", int(row.week))
            frame.insert(2, "requested_game_id", str(row.game_id))
            frame.insert(3, "requested_kickoff_time_utc", str(row.kickoff_time_utc))
            new_frames.append(frame)
        time.sleep(max(0.0, float(sleep)))

    new_rows = pd.concat(new_frames, ignore_index=True, sort=False) if new_frames else pd.DataFrame()
    combined = _append_and_write(
        existing,
        new_rows,
        output,
        subset=["event_id", "requested_date", "bookmaker_key", "market_key"],
    )
    return update_market_status(manifest, combined), combined, int(len(jobs))


def _available_target_markets(manifest_row: pd.Series, markets: pd.DataFrame) -> list[str]:
    target = set(_json_list(manifest_row.get("target_markets")))
    if not target:
        return []
    if markets.empty or "event_id" not in markets.columns or "market_key" not in markets.columns:
        return []
    event_rows = markets[markets["event_id"].astype("string").eq(str(manifest_row["oddsapi_event_id"]))]
    if event_rows.empty:
        return []
    available = set(event_rows["market_key"].dropna().astype(str))
    return sorted(target & available)


def fetch_event_odds(
    manifest: pd.DataFrame,
    markets: pd.DataFrame,
    *,
    api_key: str | None,
    output: Path,
    sport_key: str,
    regions: Sequence[str],
    odds_format: str,
    include_links: bool,
    include_sids: bool,
    retries: int,
    sleep: float,
    timeout: float,
    dry_run: bool,
    resume: bool,
    max_requests: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    existing = _read_optional_frame(output)
    jobs = manifest[manifest["oddsapi_event_id"].notna()].copy()
    if jobs.empty:
        return update_odds_status(manifest, existing), existing, 0, 0
    jobs["_available_markets"] = jobs.apply(lambda row: _available_target_markets(row, markets), axis=1)
    jobs = jobs[jobs["_available_markets"].map(bool)].copy()

    if resume and not existing.empty and {"event_id", "requested_date"}.issubset(existing.columns):
        done = {
            (str(event_id), str(requested_date))
            for event_id, requested_date in existing[["event_id", "requested_date"]].dropna().itertuples(index=False)
        }
        jobs = jobs[
            ~jobs.apply(lambda row: (str(row["oddsapi_event_id"]), str(row["snapshot_time_utc"])) in done, axis=1)
        ]
    if max_requests is not None:
        jobs = jobs.head(max(0, int(max_requests)))
    estimated_credits = int(sum(len(markets_for_event) * 10 for markets_for_event in jobs["_available_markets"]))
    if dry_run:
        return update_odds_status(manifest, existing), existing, int(len(jobs)), estimated_credits

    if not api_key:
        raise RuntimeError(f"{ODDS_API_PAID_KEY_SECRET} not configured. Add it to secrets.env.")

    new_frames: list[pd.DataFrame] = []
    for _, row in jobs.iterrows():
        bookmakers = _json_list(row["target_bookmakers"])
        available_markets = list(row["_available_markets"])
        frame = fetch_historical_event_odds(
            api_key=api_key,
            event_id=str(row["oddsapi_event_id"]),
            date=str(row["snapshot_time_utc"]),
            sport_key=sport_key,
            markets=available_markets,
            bookmakers=bookmakers,
            regions=regions,
            odds_format=odds_format,
            include_links=include_links,
            include_sids=include_sids,
            retries=retries,
            sleep=sleep,
            timeout=timeout,
        )
        if not frame.empty:
            frame.insert(0, "requested_season", int(row["season"]))
            frame.insert(1, "requested_week", int(row["week"]))
            frame.insert(2, "requested_game_id", str(row["game_id"]))
            frame.insert(3, "requested_kickoff_time_utc", str(row["kickoff_time_utc"]))
            new_frames.append(frame)
        time.sleep(max(0.0, float(sleep)))

    new_rows = pd.concat(new_frames, ignore_index=True, sort=False) if new_frames else pd.DataFrame()
    combined = _append_and_write(
        existing,
        new_rows,
        output,
        subset=[
            "event_id",
            "requested_date",
            "bookmaker_key",
            "market_key",
            "player_name",
            "outcome_name",
            "point",
            "price",
        ],
    )
    return update_odds_status(manifest, combined), combined, int(len(jobs)), estimated_credits


def normalize_historical_player_prop_lines(
    odds: pd.DataFrame,
    *,
    manifest: pd.DataFrame | None = None,
    vendors: Sequence[str] = DEFAULT_BOOKMAKERS,
) -> pd.DataFrame:
    if odds.empty:
        return pd.DataFrame(columns=NORMALIZED_LINE_COLUMNS)
    required = {
        "requested_season",
        "requested_week",
        "bookmaker_key",
        "market_key",
        "outcome_name",
        "player_name",
        "price",
        "point",
    }
    if missing := required - set(odds.columns):
        raise ValueError(f"Historical odds rows missing required columns: {sorted(missing)}")

    out = odds.copy()
    if "requested_game_id" not in out.columns and manifest is not None and not manifest.empty:
        lookup = manifest[["oddsapi_event_id", "snapshot_time_utc", "game_id"]].dropna(
            subset=["oddsapi_event_id"]
        )
        lookup = lookup.rename(
            columns={
                "oddsapi_event_id": "event_id",
                "snapshot_time_utc": "requested_date",
                "game_id": "requested_game_id",
            }
        )
        out = out.merge(lookup, on=["event_id", "requested_date"], how="left")

    if "requested_game_id" not in out.columns:
        return pd.DataFrame(columns=NORMALIZED_LINE_COLUMNS)

    out["season"] = pd.to_numeric(out["requested_season"], errors="coerce")
    out["week"] = pd.to_numeric(out["requested_week"], errors="coerce")
    out["game_id"] = out["requested_game_id"].astype("string")
    out["sportsbook"] = out["bookmaker_key"].astype("string").str.lower()
    vendor_set = {str(vendor).strip().lower() for vendor in vendors if str(vendor).strip()}
    if vendor_set:
        out = out[out["sportsbook"].isin(vendor_set)]
    out["market"] = out["market_key"].map(ODDSAPI_MARKET_TO_MODEL_MARKET)
    out = out[out["market"].notna()]
    out["player_name_key"] = out["player_name"].apply(_player_name_key)
    out["player_name_prefix_key"] = out["player_name"].apply(_player_name_prefix_key)
    out["line"] = pd.to_numeric(out["point"], errors="coerce")
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out["outcome_side"] = out["outcome_name"].astype("string").str.lower()
    out = out.dropna(subset=["season", "week", "game_id", "market", "player_name_key", "line", "price"])
    if out.empty:
        return pd.DataFrame(columns=NORMALIZED_LINE_COLUMNS)
    out["season"] = out["season"].astype(int)
    out["week"] = out["week"].astype(int)
    out["line_updated_at"] = _first_present(out, ["market_last_update", "book_last_update", "snapshot_timestamp"])

    key_cols = [
        "season",
        "week",
        "game_id",
        "event_id",
        "requested_date",
        "snapshot_timestamp",
        "commence_time",
        "home_team",
        "away_team",
        "market",
        "market_key",
        "player_name",
        "player_name_key",
        "player_name_prefix_key",
        "line",
        "sportsbook",
    ]
    for col in key_cols:
        if col not in out.columns:
            out[col] = pd.NA

    over = out[out["outcome_side"].str.contains("over", na=False)].copy()
    under = out[out["outcome_side"].str.contains("under", na=False)].copy()
    sort_cols = key_cols + ["line_updated_at"]
    over = over.sort_values(sort_cols, kind="stable").drop_duplicates(key_cols, keep="last")
    under = under.sort_values(sort_cols, kind="stable").drop_duplicates(key_cols, keep="last")
    over = over[key_cols + ["price", "line_updated_at"]].rename(
        columns={"price": "over_odds", "line_updated_at": "over_updated_at"}
    )
    under = under[key_cols + ["price", "line_updated_at"]].rename(
        columns={"price": "under_odds", "line_updated_at": "under_updated_at"}
    )
    merged = over.merge(under, on=key_cols, how="outer")
    if merged.empty:
        return pd.DataFrame(columns=NORMALIZED_LINE_COLUMNS)

    merged["line_updated_at"] = merged["over_updated_at"].where(
        merged["over_updated_at"].notna(),
        merged["under_updated_at"],
    )
    over_prob = moneyline_to_prob(merged["over_odds"])
    under_prob = moneyline_to_prob(merged["under_odds"])
    total_prob = over_prob + under_prob
    merged["implied_probability"] = np.where(total_prob > 0, over_prob / total_prob, over_prob)
    merged["line_balance_distance"] = _line_balance_distance(merged)
    merged["line_is_comparable"] = merged["line_balance_distance"].le(LINE_BALANCE_DISTANCE_LIMIT)
    merged["line_source"] = "odds_api_historical"
    merged = merged.rename(columns={"event_id": "oddsapi_event_id"})
    return merged[NORMALIZED_LINE_COLUMNS].sort_values(
        ["season", "week", "game_id", "market", "player_name_key", "sportsbook", "line"],
        kind="stable",
    )


def _write_manifest(manifest: pd.DataFrame, path: Path) -> Path:
    return write_df(manifest[MANIFEST_COLUMNS], path)


def _print_quota(manifest: pd.DataFrame, markets: Sequence[str]) -> None:
    quota = estimate_quota(manifest, markets=markets)
    print("[oddsapi.backfill] manifest rows:", len(manifest))
    print("[oddsapi.backfill] events requests:", quota["events_requests"])
    print("[oddsapi.backfill] event-markets requests:", quota["event_markets_requests"])
    print("[oddsapi.backfill] discovery requests:", quota["discovery_requests"])
    print("[oddsapi.backfill] worst-case event-odds requests:", quota["event_odds_requests_worst_case"])
    print("[oddsapi.backfill] worst-case event-odds credits:", quota["event_odds_credits_worst_case"])


def _load_or_build_manifest(args: argparse.Namespace) -> pd.DataFrame:
    if args.mode == "manifest" or args.force_manifest or not args.manifest.exists():
        schedule = _load_schedule(args.schedule)
        return build_backfill_manifest(
            schedule,
            start_season=args.start_season,
            end_season=args.end_season,
            current_date=args.current_date,
            include_postseason=not args.regular_only,
            snapshot_minutes_before=args.snapshot_minutes_before,
            kickoff_tz=args.kickoff_tz,
            markets=args.markets,
            bookmakers=args.bookmakers,
        )
    return read_df(args.manifest)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and execute a resumable Odds API historical player-prop backfill.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["manifest", "discover-events", "discover-markets", "fetch-odds", "normalize", "all"],
        default="manifest",
    )
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--events-output", type=Path, default=DEFAULT_EVENTS_OUTPUT_PATH)
    parser.add_argument("--markets-output", type=Path, default=DEFAULT_MARKETS_OUTPUT_PATH)
    parser.add_argument("--odds-output", type=Path, default=DEFAULT_ODDS_OUTPUT_PATH)
    parser.add_argument("--normalized-output", type=Path, default=DEFAULT_NORMALIZED_OUTPUT_PATH)
    parser.add_argument("--start-season", type=int, default=DEFAULT_START_SEASON)
    parser.add_argument("--end-season", type=int, default=None)
    parser.add_argument("--current-date", default=None)
    parser.add_argument("--regular-only", action="store_true")
    parser.add_argument("--snapshot-minutes-before", type=int, default=DEFAULT_SNAPSHOT_MINUTES_BEFORE)
    parser.add_argument("--kickoff-tz", default=DEFAULT_KICKOFF_TZ)
    parser.add_argument("--sport-key", default=DEFAULT_SPORT_KEY)
    parser.add_argument("--markets", nargs="+", default=DEFAULT_PLAYER_PROP_MARKETS)
    parser.add_argument("--bookmakers", nargs="+", default=DEFAULT_BOOKMAKERS)
    parser.add_argument("--regions", nargs="+", default=DEFAULT_REGIONS)
    parser.add_argument("--odds-format", default="american")
    parser.add_argument("--include-links", action="store_true")
    parser.add_argument("--include-sids", action="store_true")
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--timeout", type=float, default=REQUEST_TIMEOUT)
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--force-manifest", action="store_true")
    parser.add_argument("--confirm-paid-backfill", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.set_defaults(resume=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    if args.mode in API_MODES and not args.dry_run and not args.confirm_paid_backfill:
        raise RuntimeError(
            "Historical backfill modes spend the paid Odds API key. "
            "Re-run with --confirm-paid-backfill after reviewing the dry-run counts."
        )

    manifest = _load_or_build_manifest(args)
    _print_quota(manifest, args.markets)

    if args.debug and not manifest.empty:
        print("[oddsapi.backfill] sample:")
        print(manifest.head(20).to_string(index=False))

    if args.mode == "manifest":
        if args.dry_run:
            print(f"[oddsapi.backfill] dry-run: would write manifest -> {args.manifest.resolve()}")
            return 0
        written = _write_manifest(manifest, args.manifest)
        print(f"[oddsapi.backfill] wrote manifest -> {written.resolve()}")
        return 0

    api_key = None if args.dry_run or args.mode == "normalize" else get_odds_api_paid_key()

    events = _read_optional_frame(args.events_output)
    markets = _read_optional_frame(args.markets_output)
    odds = _read_optional_frame(args.odds_output)

    if args.mode in {"discover-events", "all"}:
        manifest, events, planned = discover_events(
            manifest,
            api_key=api_key,
            output=args.events_output,
            sport_key=args.sport_key,
            retries=args.retries,
            sleep=args.sleep,
            timeout=args.timeout,
            dry_run=args.dry_run,
            resume=args.resume,
            max_requests=args.max_requests,
        )
        print(f"[oddsapi.backfill] event discovery requests planned/executed: {planned}")
        if not args.dry_run:
            _write_manifest(manifest, args.manifest)

    if args.mode in {"discover-markets", "all"}:
        if events.empty and args.events_output.exists():
            events = read_df(args.events_output)
        manifest = match_events_to_manifest(manifest, events)
        manifest, markets, planned = discover_event_markets(
            manifest,
            api_key=api_key,
            output=args.markets_output,
            sport_key=args.sport_key,
            regions=args.regions,
            retries=args.retries,
            sleep=args.sleep,
            timeout=args.timeout,
            dry_run=args.dry_run,
            resume=args.resume,
            max_requests=args.max_requests,
        )
        print(f"[oddsapi.backfill] event-market requests planned/executed: {planned}")
        if not args.dry_run:
            _write_manifest(manifest, args.manifest)

    if args.mode in {"fetch-odds", "all"}:
        if markets.empty and args.markets_output.exists():
            markets = read_df(args.markets_output)
        manifest = update_market_status(manifest, markets)
        manifest, odds, planned, estimated_credits = fetch_event_odds(
            manifest,
            markets,
            api_key=api_key,
            output=args.odds_output,
            sport_key=args.sport_key,
            regions=args.regions,
            odds_format=args.odds_format,
            include_links=args.include_links,
            include_sids=args.include_sids,
            retries=args.retries,
            sleep=args.sleep,
            timeout=args.timeout,
            dry_run=args.dry_run,
            resume=args.resume,
            max_requests=args.max_requests,
        )
        print(f"[oddsapi.backfill] event-odds requests planned/executed: {planned}")
        print(f"[oddsapi.backfill] estimated event-odds credits for this batch: {estimated_credits}")
        if not args.dry_run:
            _write_manifest(manifest, args.manifest)

    if args.mode in {"normalize", "all"}:
        if odds.empty and args.odds_output.exists():
            odds = read_df(args.odds_output)
        normalized = normalize_historical_player_prop_lines(
            odds,
            manifest=manifest,
            vendors=args.bookmakers,
        )
        if args.dry_run:
            print(
                "[oddsapi.backfill] dry-run: would write normalized historical lines "
                f"rows={len(normalized)} -> {args.normalized_output.resolve()}"
            )
        else:
            written = write_df(normalized, args.normalized_output)
            print(
                "[oddsapi.backfill] wrote normalized historical lines "
                f"rows={len(normalized)} -> {written.resolve()}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
