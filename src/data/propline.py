from __future__ import annotations

import argparse
from datetime import timezone
import json
import logging
import time
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from src.utils.io import RAW_DIR, read_df
from src.utils.secrets import get_secret
from src.utils.teams import get_team_abbr_from_name, normalize_team_abbr


API_BASE = "https://api.prop-line.com/v1"
DEFAULT_SPORT_KEY = "football_nfl"
DEFAULT_SCHEDULE_PATH = RAW_DIR / "nfl_schedules.parquet"
DEFAULT_KICKOFF_TZ = "America/New_York"
DEFAULT_BOOKMAKERS = ["draftkings", "hardrock", "fanduel"]
DEFAULT_MARKETS = [
    "player_pass_yds",
    "player_pass_tds",
    "player_rush_yds",
    "player_rush_tds",
    "player_reception_yds",
    "player_reception_tds",
    "player_anytime_td",
    "player_1st_td",
    "player_last_td",
    "player_2plus_td",
    "player_3plus_td",
    "player_sacks",
]
REQUEST_TIMEOUT = 30.0
DEFAULT_RETRIES = 2
DEFAULT_SLEEP = 1.0


class PropLineError(RuntimeError):
    """Raised when a PropLine HTTP request fails."""

    def __init__(self, message: str, *, status_code: Optional[int] = None, body: Optional[str] = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


def _build_output_path(season: Optional[int], week: Optional[int]) -> Path:
    base = "propline_player_props"
    if season is not None:
        base = f"{base}_{int(season)}"
    if week is not None:
        base = f"{base}_wk{int(week):02d}"
    return RAW_DIR / f"{base}.parquet"


def _csv_param(values: Iterable[str]) -> str:
    return ",".join(str(value).strip() for value in values if str(value).strip())


def _is_missing(value: Any) -> bool:
    if isinstance(value, (dict, list, tuple, set)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _json_safe(value: Any) -> Any:
    if _is_missing(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return value


def _flatten_nested_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Stringify nested API fields before parquet write/read round-trips."""
    if frame.empty:
        return frame
    out = frame.copy()
    for col in out.columns:
        if out[col].dtype != "object":
            continue
        has_nested = out[col].map(lambda value: isinstance(value, (dict, list, tuple, set))).any()
        if has_nested:
            out[col] = out[col].map(
                lambda value: None
                if _is_missing(value)
                else json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True)
            )
    for col in out.columns:
        if out[col].isna().all():
            out[col] = out[col].astype("string")
    return out


def _write_propline_frame(frame: pd.DataFrame, output: Path) -> Path:
    clean = _flatten_nested_columns(frame)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".csv" or not output.suffix:
        output_path = output.with_suffix(".csv")
        clean.to_csv(output_path, index=False, encoding="utf-8")
        return output_path
    output_path = output if output.suffix == ".parquet" else output.with_suffix(".parquet")
    clean.to_parquet(output_path, engine="pyarrow", index=False)
    return output_path


def _schedule_team_abbr(value: Any) -> Optional[str]:
    if _is_missing(value):
        return None
    text = str(value).strip()
    if not text or text.upper() in {"NONE", "NAN", "NA", "<NA>"}:
        return None
    abbr = get_team_abbr_from_name(text) or normalize_team_abbr(text)
    if abbr == "LAR":
        return "LA"
    return str(abbr).strip().upper() if abbr else None


def _schedule_kickoff_utc(gameday: Any, gametime: Any, *, kickoff_tz: str) -> pd.Timestamp:
    if _is_missing(gameday):
        return pd.NaT
    day = pd.to_datetime(gameday, errors="coerce")
    if pd.isna(day):
        return pd.NaT
    time_text = "12:00" if _is_missing(gametime) else str(gametime).strip()
    if not time_text:
        time_text = "12:00"
    kickoff = pd.Timestamp(f"{day.date()} {time_text}")
    localized = kickoff.to_pydatetime().replace(tzinfo=ZoneInfo(kickoff_tz))
    return pd.Timestamp(localized).tz_convert(timezone.utc)


def _prepare_schedule_week(
    schedule: pd.DataFrame,
    *,
    season: int,
    week: int,
    kickoff_tz: str = DEFAULT_KICKOFF_TZ,
) -> pd.DataFrame:
    required = {"season", "week", "game_id", "home_team", "away_team", "gameday"}
    if schedule.empty or not required.issubset(schedule.columns):
        return pd.DataFrame()

    frame = schedule.copy()
    frame["season"] = pd.to_numeric(frame["season"], errors="coerce")
    frame["week"] = pd.to_numeric(frame["week"], errors="coerce")
    frame = frame[frame["season"].eq(int(season)) & frame["week"].eq(int(week))].copy()
    if frame.empty:
        return pd.DataFrame()

    gametime = frame["gametime"] if "gametime" in frame.columns else pd.Series("12:00", index=frame.index)
    out = pd.DataFrame(
        {
            "_schedule_game_id": frame["game_id"].astype("string"),
            "_home_abbr": frame["home_team"].apply(_schedule_team_abbr),
            "_away_abbr": frame["away_team"].apply(_schedule_team_abbr),
            "_schedule_kickoff_utc": [
                _schedule_kickoff_utc(gameday, time_value, kickoff_tz=kickoff_tz)
                for gameday, time_value in zip(frame["gameday"], gametime)
            ],
        }
    )
    return out.dropna(subset=["_schedule_game_id", "_home_abbr", "_away_abbr"]).drop_duplicates()


def _filter_to_schedule_week(
    frame: pd.DataFrame,
    schedule: pd.DataFrame,
    *,
    season: int,
    week: int,
    kickoff_tz: str = DEFAULT_KICKOFF_TZ,
    kickoff_tolerance_hours: float = 18.0,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    schedule_week = _prepare_schedule_week(
        schedule,
        season=int(season),
        week=int(week),
        kickoff_tz=kickoff_tz,
    )
    if schedule_week.empty:
        logging.warning("PropLine schedule filter skipped: no schedule rows for season=%s week=%s.", season, week)
        return frame

    out = frame.copy()
    out["_home_abbr"] = out["home_team"].apply(_schedule_team_abbr) if "home_team" in out.columns else pd.NA
    out["_away_abbr"] = out["away_team"].apply(_schedule_team_abbr) if "away_team" in out.columns else pd.NA
    out["_event_kickoff_utc"] = (
        pd.to_datetime(out["commence_time"], errors="coerce", utc=True) if "commence_time" in out.columns else pd.NaT
    )
    out["_source_row"] = range(len(out))

    matched = out.merge(schedule_week, on=["_home_abbr", "_away_abbr"], how="inner")
    if matched.empty:
        logging.warning(
            "PropLine schedule filter removed all rows for season=%s week=%s; no team matchup matched schedule.",
            season,
            week,
        )
        return matched.drop(columns=[col for col in matched.columns if col.startswith("_")], errors="ignore")

    matched["_time_diff_seconds"] = (
        matched["_event_kickoff_utc"] - matched["_schedule_kickoff_utc"]
    ).abs().dt.total_seconds()
    tolerance_seconds = float(kickoff_tolerance_hours) * 3600.0
    time_ok = matched["_time_diff_seconds"].isna() | matched["_time_diff_seconds"].le(tolerance_seconds)
    matched = matched[time_ok].copy()
    if matched.empty:
        logging.warning(
            "PropLine schedule filter removed all rows for season=%s week=%s; matched teams but kickoff times were outside %.1fh.",
            season,
            week,
            kickoff_tolerance_hours,
        )
        return matched.drop(columns=[col for col in matched.columns if col.startswith("_")], errors="ignore")

    matched = matched.sort_values(["_source_row", "_time_diff_seconds"], kind="stable")
    matched = matched.drop_duplicates(subset=["_source_row"], keep="first")
    matched["game_id"] = matched["_schedule_game_id"].astype("string")
    matched["schedule_season"] = int(season)
    matched["schedule_week"] = int(week)
    matched["schedule_game_id"] = matched["_schedule_game_id"].astype("string")
    matched["schedule_kickoff_utc"] = matched["_schedule_kickoff_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    matched["event_home_abbr"] = matched["_home_abbr"]
    matched["event_away_abbr"] = matched["_away_abbr"]
    matched["event_schedule_time_diff_seconds"] = matched["_time_diff_seconds"]

    removed = len(frame) - len(matched)
    logging.info(
        "PropLine schedule filter season=%s week=%s kept=%s removed=%s target_games=%s matched_games=%s",
        season,
        week,
        len(matched),
        removed,
        schedule_week["_schedule_game_id"].nunique(),
        matched["_schedule_game_id"].nunique(),
    )
    return matched.drop(columns=[col for col in matched.columns if col.startswith("_")], errors="ignore")


def _request_with_retry(
    session: requests.Session,
    path: str,
    *,
    params: dict[str, Any],
    retries: int,
    sleep: float,
    timeout: float,
) -> Any:
    url = API_BASE + path
    for attempt in range(retries + 1):
        response = session.get(url, params=params, timeout=timeout)
        if response.status_code in {429, 503} and attempt < retries:
            retry_after = response.headers.get("Retry-After")
            try:
                wait = float(retry_after) if retry_after is not None else sleep * (attempt + 1)
            except ValueError:
                wait = sleep * (attempt + 1)
            logging.warning("PropLine throttled request (%s), retrying in %.1fs", response.status_code, wait)
            time.sleep(max(wait, 0.0))
            continue
        if response.status_code >= 500 and attempt < retries:
            wait = sleep * (attempt + 1)
            logging.warning("PropLine server error %s, retrying in %.1fs", response.status_code, wait)
            time.sleep(wait)
            continue
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            body_text: Optional[str] = None
            snippet = ""
            try:
                body_text = response.text
                snippet = f" body={body_text[:200]!r}"
            except Exception:
                snippet = ""
            raise PropLineError(
                f"PropLine request failed for {response.url} ({exc}){snippet}",
                status_code=response.status_code,
                body=body_text,
            ) from exc
        return response.json()
    raise PropLineError(f"PropLine request exhausted retries for {url}")


def _response_events(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        if {"id", "bookmakers"}.issubset(payload.keys()):
            return [payload]
    return []


def _flatten_odds_events(payload: Any, *, requested_markets: Sequence[str], requested_bookmakers: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event in _response_events(payload):
        event_base = {
            "event_id": event.get("id"),
            "sport_key": event.get("sport_key"),
            "home_team": event.get("home_team"),
            "away_team": event.get("away_team"),
            "home_team_key": event.get("home_team_key"),
            "away_team_key": event.get("away_team_key"),
            "home_team_id": event.get("home_team_id"),
            "away_team_id": event.get("away_team_id"),
            "commence_time": event.get("commence_time"),
            "espn_event_id": event.get("espn_event_id"),
            "merged_from_event_ids": event.get("merged_from_event_ids"),
        }
        for bookmaker in event.get("bookmakers") or []:
            if not isinstance(bookmaker, dict):
                continue
            book_base = {
                "bookmaker_key": bookmaker.get("key"),
                "bookmaker_title": bookmaker.get("title"),
                "book_event_id": bookmaker.get("book_event_id"),
                "link": bookmaker.get("link"),
                "app_link": bookmaker.get("app_link"),
            }
            for market in bookmaker.get("markets") or []:
                if not isinstance(market, dict):
                    continue
                market_base = {
                    "market_key": market.get("key"),
                    "market_description": market.get("description"),
                    "market_team": market.get("team"),
                    "market_last_update": market.get("last_update"),
                    "market_suspended_at": market.get("suspended_at"),
                }
                for outcome in market.get("outcomes") or []:
                    if not isinstance(outcome, dict):
                        continue
                    row = {
                        **event_base,
                        **book_base,
                        **market_base,
                        "outcome_name": outcome.get("name"),
                        "player_name": outcome.get("description"),
                        "price": outcome.get("price"),
                        "point": outcome.get("point"),
                        "player_id": outcome.get("player_id"),
                        "outcome_id": outcome.get("outcome_id"),
                        "book_outcome_id": outcome.get("book_outcome_id"),
                        "last_change_at": outcome.get("last_change_at"),
                        "last_seen_at": outcome.get("last_seen_at"),
                        "book_updated_at": outcome.get("book_updated_at"),
                        "liquidity": outcome.get("liquidity"),
                        "payout_multiplier": outcome.get("payout_multiplier"),
                        "dfs_odds_type": outcome.get("dfs_odds_type"),
                        "requested_markets": _csv_param(requested_markets),
                        "requested_bookmakers": _csv_param(requested_bookmakers),
                    }
                    rows.append(row)
    return pd.DataFrame(rows)


def fetch_player_props(
    *,
    api_key: str,
    sport_key: str = DEFAULT_SPORT_KEY,
    markets: Sequence[str] = DEFAULT_MARKETS,
    bookmakers: Sequence[str] = DEFAULT_BOOKMAKERS,
    retries: int = DEFAULT_RETRIES,
    sleep: float = DEFAULT_SLEEP,
    timeout: float = REQUEST_TIMEOUT,
) -> pd.DataFrame:
    session = requests.Session()
    session.headers.update({"X-API-Key": api_key})
    params = {
        "markets": _csv_param(markets),
        "bookmakers": _csv_param(bookmakers),
        "includeBookIds": "true",
    }
    logging.info("Requesting %s/sports/%s/odds params=%s", API_BASE, sport_key, params)
    payload = _request_with_retry(
        session,
        f"/sports/{sport_key}/odds",
        params=params,
        retries=max(0, int(retries)),
        sleep=max(0.0, float(sleep)),
        timeout=float(timeout),
    )
    return _flatten_odds_events(payload, requested_markets=markets, requested_bookmakers=bookmakers)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch NFL player prop lines from PropLine.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--season", type=int, required=True, help="Season label for the local cache file.")
    parser.add_argument("--week", type=int, required=True, help="Week label for the local cache file.")
    parser.add_argument("--sport-key", default=DEFAULT_SPORT_KEY)
    parser.add_argument("--bookmakers", nargs="+", default=DEFAULT_BOOKMAKERS)
    parser.add_argument("--markets", nargs="+", default=DEFAULT_MARKETS)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE_PATH)
    parser.add_argument("--kickoff-tz", default=DEFAULT_KICKOFF_TZ)
    parser.add_argument("--kickoff-tolerance-hours", type=float, default=18.0)
    parser.add_argument("--no-schedule-filter", action="store_true")
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--timeout", type=float, default=REQUEST_TIMEOUT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    output = args.output or _build_output_path(args.season, args.week)
    if args.dry_run:
        logging.info("[dry-run] Would write PropLine player props to %s", output)
        return

    api_key = get_secret("PROPLINE_API_KEY")
    if not api_key:
        raise RuntimeError("PROPLINE_API_KEY not configured. Add it to secrets.env.")

    frame = fetch_player_props(
        api_key=api_key,
        sport_key=args.sport_key,
        markets=args.markets,
        bookmakers=args.bookmakers,
        retries=args.retries,
        sleep=args.sleep,
        timeout=args.timeout,
    )
    if not args.no_schedule_filter:
        if args.schedule.exists():
            schedule = read_df(args.schedule)
            frame = _filter_to_schedule_week(
                frame,
                schedule,
                season=args.season,
                week=args.week,
                kickoff_tz=args.kickoff_tz,
                kickoff_tolerance_hours=args.kickoff_tolerance_hours,
            )
        else:
            logging.warning("PropLine schedule filter skipped: schedule file not found at %s", args.schedule)

    if frame.empty:
        logging.warning("PropLine returned no player prop rows for season=%s week=%s.", args.season, args.week)
    if "propline_feed" not in frame.columns:
        frame.insert(0, "propline_feed", "player_props")
    if "requested_season" not in frame.columns:
        frame.insert(1, "requested_season", int(args.season))
    if "requested_week" not in frame.columns:
        frame.insert(2, "requested_week", int(args.week))
    written = _write_propline_frame(frame, output)
    logging.info("Saved PropLine player prop rows=%s -> %s", len(frame), written)


if __name__ == "__main__":
    main()
