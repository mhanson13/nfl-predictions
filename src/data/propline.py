from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import pandas as pd
import requests

from src.utils.io import RAW_DIR, write_df
from src.utils.secrets import get_secret


API_BASE = "https://api.prop-line.com/v1"
DEFAULT_SPORT_KEY = "football_nfl"
DEFAULT_BOOKMAKERS = ["draftkings", "hardrock", "fanduel"]
DEFAULT_MARKETS = [
    "player_pass_yds",
    "player_rush_yds",
    "player_reception_yds",
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
    if frame.empty:
        logging.warning("PropLine returned no player prop rows for season=%s week=%s.", args.season, args.week)
    else:
        frame.insert(0, "propline_feed", "player_props")
        frame.insert(1, "requested_season", int(args.season))
        frame.insert(2, "requested_week", int(args.week))
    written = write_df(frame, output)
    logging.info("Saved PropLine player prop rows=%s -> %s", len(frame), written)


if __name__ == "__main__":
    main()
