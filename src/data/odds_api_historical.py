from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import pandas as pd
import requests

from src.utils.io import RAW_DIR, write_df
from src.utils.odds import ODDS_API_PAID_KEY_SECRET, get_odds_api_paid_key


API_BASE = "https://api.the-odds-api.com/v4"
DEFAULT_SPORT_KEY = "americanfootball_nfl"
DEFAULT_REGIONS = ["us"]
DEFAULT_BOOKMAKERS = ["draftkings", "fanduel", "hardrockbet"]
DEFAULT_PLAYER_PROP_MARKETS = [
    "player_pass_yds",
    "player_rush_yds",
    "player_reception_yds",
    "player_sacks",
]
REQUEST_TIMEOUT = 30.0
DEFAULT_RETRIES = 2
DEFAULT_SLEEP = 1.0


class OddsApiError(RuntimeError):
    """Raised when The Odds API request fails."""

    def __init__(self, message: str, *, status_code: Optional[int] = None, body: Optional[str] = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


def _redact_url(url: str) -> str:
    return re.sub(r"([?&]apiKey=)[^&]+", r"\1***", url)


def _error_hint(body_text: str | None) -> str:
    if not body_text:
        return ""
    try:
        payload = json.loads(body_text)
    except json.JSONDecodeError:
        return ""
    if not isinstance(payload, dict):
        return ""
    code = str(payload.get("error_code") or "")
    if code == "HISTORICAL_UNAVAILABLE_ON_FREE_USAGE_PLAN":
        return " Historical odds require a paid Odds API usage plan for this key."
    if code in {"INVALID_API_KEY", "API_KEY_INVALID"}:
        return f" Check {ODDS_API_PAID_KEY_SECRET} in secrets.env."
    return ""


def _csv_param(values: Iterable[str]) -> str:
    return ",".join(str(value).strip() for value in values if str(value).strip())


def _date_slug(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "", value.strip())
    return slug or "snapshot"


def _build_events_output_path(date: str) -> Path:
    return RAW_DIR / f"oddsapi_historical_events_{_date_slug(date)}.parquet"


def _build_event_odds_output_path(
    *,
    event_id: str,
    date: str,
    season: int | None = None,
    week: int | None = None,
) -> Path:
    event_slug = re.sub(r"[^0-9A-Za-z]+", "", event_id.strip())[:24] or "event"
    if season is not None and week is not None:
        return RAW_DIR / f"oddsapi_historical_player_props_{int(season)}_wk{int(week):02d}_{event_slug}_{_date_slug(date)}.parquet"
    return RAW_DIR / f"oddsapi_historical_event_odds_{event_slug}_{_date_slug(date)}.parquet"


def _build_event_markets_output_path(
    *,
    event_id: str,
    date: str,
    season: int | None = None,
    week: int | None = None,
) -> Path:
    event_slug = re.sub(r"[^0-9A-Za-z]+", "", event_id.strip())[:24] or "event"
    if season is not None and week is not None:
        return RAW_DIR / f"oddsapi_historical_event_markets_{int(season)}_wk{int(week):02d}_{event_slug}_{_date_slug(date)}.parquet"
    return RAW_DIR / f"oddsapi_historical_event_markets_{event_slug}_{_date_slug(date)}.parquet"


def _request_with_retry(
    session: requests.Session,
    path: str,
    *,
    params: dict[str, Any],
    retries: int,
    sleep: float,
    timeout: float,
) -> tuple[Any, dict[str, str]]:
    url = API_BASE + path
    for attempt in range(retries + 1):
        response = session.get(url, params=params, timeout=timeout)
        if response.status_code in {429, 503} and attempt < retries:
            retry_after = response.headers.get("Retry-After")
            try:
                wait = float(retry_after) if retry_after is not None else sleep * (attempt + 1)
            except ValueError:
                wait = sleep * (attempt + 1)
            logging.warning("Odds API throttled request (%s), retrying in %.1fs", response.status_code, wait)
            time.sleep(max(wait, 0.0))
            continue
        if response.status_code >= 500 and attempt < retries:
            wait = sleep * (attempt + 1)
            logging.warning("Odds API server error %s, retrying in %.1fs", response.status_code, wait)
            time.sleep(wait)
            continue
        if response.status_code >= 400:
            body_text: Optional[str] = None
            snippet = ""
            try:
                body_text = response.text
                snippet = f" body={body_text[:200]!r}"
            except Exception:
                snippet = ""
            safe_url = _redact_url(response.url)
            hint = _error_hint(body_text)
            raise OddsApiError(
                f"Odds API request failed for {safe_url} ({response.status_code} {response.reason}).{hint}{snippet}",
                status_code=response.status_code,
                body=body_text,
            )
        quota_headers = {
            "x_requests_remaining": response.headers.get("x-requests-remaining"),
            "x_requests_used": response.headers.get("x-requests-used"),
            "x_requests_last": response.headers.get("x-requests-last"),
        }
        return response.json(), quota_headers
    raise OddsApiError(f"Odds API request exhausted retries for {url}")


def _historical_wrapper_fields(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"snapshot_timestamp": None, "previous_timestamp": None, "next_timestamp": None}
    return {
        "snapshot_timestamp": payload.get("timestamp"),
        "previous_timestamp": payload.get("previous_timestamp"),
        "next_timestamp": payload.get("next_timestamp"),
    }


def _payload_data(payload: Any) -> Any:
    if isinstance(payload, dict) and "data" in payload:
        return payload.get("data")
    return payload


def _flatten_historical_events(payload: Any, *, requested_date: str) -> pd.DataFrame:
    data = _payload_data(payload)
    if not isinstance(data, list):
        return pd.DataFrame()
    wrapper = _historical_wrapper_fields(payload)
    rows: list[dict[str, Any]] = []
    for event in data:
        if not isinstance(event, dict):
            continue
        rows.append(
            {
                **wrapper,
                "requested_date": requested_date,
                "event_id": event.get("id"),
                "sport_key": event.get("sport_key"),
                "sport_title": event.get("sport_title"),
                "commence_time": event.get("commence_time"),
                "home_team": event.get("home_team"),
                "away_team": event.get("away_team"),
            }
        )
    return pd.DataFrame(rows)


def _flatten_event_odds(
    payload: Any,
    *,
    requested_date: str,
    requested_markets: Sequence[str],
    requested_bookmakers: Sequence[str],
    requested_regions: Sequence[str],
    quota_headers: dict[str, str] | None = None,
) -> pd.DataFrame:
    event = _payload_data(payload)
    if not isinstance(event, dict):
        return pd.DataFrame()
    wrapper = _historical_wrapper_fields(payload)
    quota_headers = quota_headers or {}
    event_base = {
        **wrapper,
        "requested_date": requested_date,
        "requested_markets": _csv_param(requested_markets),
        "requested_bookmakers": _csv_param(requested_bookmakers),
        "requested_regions": _csv_param(requested_regions),
        "x_requests_remaining": quota_headers.get("x_requests_remaining"),
        "x_requests_used": quota_headers.get("x_requests_used"),
        "x_requests_last": quota_headers.get("x_requests_last"),
        "event_id": event.get("id"),
        "sport_key": event.get("sport_key"),
        "sport_title": event.get("sport_title"),
        "commence_time": event.get("commence_time"),
        "home_team": event.get("home_team"),
        "away_team": event.get("away_team"),
    }
    rows: list[dict[str, Any]] = []
    for bookmaker in event.get("bookmakers") or []:
        if not isinstance(bookmaker, dict):
            continue
        book_base = {
            "bookmaker_key": bookmaker.get("key"),
            "bookmaker_title": bookmaker.get("title"),
            "book_last_update": bookmaker.get("last_update"),
        }
        for market in bookmaker.get("markets") or []:
            if not isinstance(market, dict):
                continue
            market_base = {
                "market_key": market.get("key"),
                "market_last_update": market.get("last_update"),
            }
            for outcome in market.get("outcomes") or []:
                if not isinstance(outcome, dict):
                    continue
                rows.append(
                    {
                        **event_base,
                        **book_base,
                        **market_base,
                        "outcome_name": outcome.get("name"),
                        "player_name": outcome.get("description"),
                        "price": outcome.get("price"),
                        "point": outcome.get("point"),
                        "outcome_link": outcome.get("link"),
                        "outcome_sid": outcome.get("sid"),
                    }
                )
    return pd.DataFrame(rows)


def _flatten_event_markets(
    payload: Any,
    *,
    requested_date: str,
    requested_bookmakers: Sequence[str],
    requested_regions: Sequence[str],
    quota_headers: dict[str, str] | None = None,
) -> pd.DataFrame:
    event = _payload_data(payload)
    if not isinstance(event, dict):
        return pd.DataFrame()
    wrapper = _historical_wrapper_fields(payload)
    quota_headers = quota_headers or {}
    event_base = {
        **wrapper,
        "requested_date": requested_date,
        "requested_bookmakers": _csv_param(requested_bookmakers),
        "requested_regions": _csv_param(requested_regions),
        "x_requests_remaining": quota_headers.get("x_requests_remaining"),
        "x_requests_used": quota_headers.get("x_requests_used"),
        "x_requests_last": quota_headers.get("x_requests_last"),
        "event_id": event.get("id"),
        "sport_key": event.get("sport_key"),
        "sport_title": event.get("sport_title"),
        "commence_time": event.get("commence_time"),
        "home_team": event.get("home_team"),
        "away_team": event.get("away_team"),
    }
    rows: list[dict[str, Any]] = []
    for bookmaker in event.get("bookmakers") or []:
        if not isinstance(bookmaker, dict):
            continue
        book_base = {
            "bookmaker_key": bookmaker.get("key"),
            "bookmaker_title": bookmaker.get("title"),
        }
        for market in bookmaker.get("markets") or []:
            if not isinstance(market, dict):
                continue
            rows.append(
                {
                    **event_base,
                    **book_base,
                    "market_key": market.get("key"),
                    "market_last_update": market.get("last_update"),
                }
            )
    return pd.DataFrame(rows)


def fetch_historical_events(
    *,
    api_key: str,
    date: str,
    sport_key: str = DEFAULT_SPORT_KEY,
    event_ids: Sequence[str] | None = None,
    commence_time_from: str | None = None,
    commence_time_to: str | None = None,
    retries: int = DEFAULT_RETRIES,
    sleep: float = DEFAULT_SLEEP,
    timeout: float = REQUEST_TIMEOUT,
) -> pd.DataFrame:
    session = requests.Session()
    params: dict[str, Any] = {
        "apiKey": api_key,
        "date": date,
        "dateFormat": "iso",
    }
    if event_ids:
        params["eventIds"] = _csv_param(event_ids)
    if commence_time_from:
        params["commenceTimeFrom"] = commence_time_from
    if commence_time_to:
        params["commenceTimeTo"] = commence_time_to
    logging.info("Requesting %s/historical/sports/%s/events params=%s", API_BASE, sport_key, {**params, "apiKey": "***"})
    payload, _quota = _request_with_retry(
        session,
        f"/historical/sports/{sport_key}/events",
        params=params,
        retries=max(0, int(retries)),
        sleep=max(0.0, float(sleep)),
        timeout=float(timeout),
    )
    return _flatten_historical_events(payload, requested_date=date)


def fetch_historical_event_odds(
    *,
    api_key: str,
    event_id: str,
    date: str,
    sport_key: str = DEFAULT_SPORT_KEY,
    markets: Sequence[str] = DEFAULT_PLAYER_PROP_MARKETS,
    bookmakers: Sequence[str] = DEFAULT_BOOKMAKERS,
    regions: Sequence[str] = DEFAULT_REGIONS,
    odds_format: str = "american",
    include_links: bool = False,
    include_sids: bool = False,
    retries: int = DEFAULT_RETRIES,
    sleep: float = DEFAULT_SLEEP,
    timeout: float = REQUEST_TIMEOUT,
) -> pd.DataFrame:
    session = requests.Session()
    params: dict[str, Any] = {
        "apiKey": api_key,
        "date": date,
        "dateFormat": "iso",
        "oddsFormat": odds_format,
        "markets": _csv_param(markets),
    }
    if bookmakers:
        params["bookmakers"] = _csv_param(bookmakers)
    else:
        params["regions"] = _csv_param(regions)
    if include_links:
        params["includeLinks"] = "true"
    if include_sids:
        params["includeSids"] = "true"
    logging.info(
        "Requesting %s/historical/sports/%s/events/%s/odds params=%s",
        API_BASE,
        sport_key,
        event_id,
        {**params, "apiKey": "***"},
    )
    payload, quota_headers = _request_with_retry(
        session,
        f"/historical/sports/{sport_key}/events/{event_id}/odds",
        params=params,
        retries=max(0, int(retries)),
        sleep=max(0.0, float(sleep)),
        timeout=float(timeout),
    )
    return _flatten_event_odds(
        payload,
        requested_date=date,
        requested_markets=markets,
        requested_bookmakers=bookmakers,
        requested_regions=regions,
        quota_headers=quota_headers,
    )


def fetch_historical_event_markets(
    *,
    api_key: str,
    event_id: str,
    date: str,
    sport_key: str = DEFAULT_SPORT_KEY,
    bookmakers: Sequence[str] = DEFAULT_BOOKMAKERS,
    regions: Sequence[str] = DEFAULT_REGIONS,
    retries: int = DEFAULT_RETRIES,
    sleep: float = DEFAULT_SLEEP,
    timeout: float = REQUEST_TIMEOUT,
) -> pd.DataFrame:
    session = requests.Session()
    params: dict[str, Any] = {
        "apiKey": api_key,
        "date": date,
        "dateFormat": "iso",
    }
    if bookmakers:
        params["bookmakers"] = _csv_param(bookmakers)
    else:
        params["regions"] = _csv_param(regions)
    logging.info(
        "Requesting %s/historical/sports/%s/events/%s/markets params=%s",
        API_BASE,
        sport_key,
        event_id,
        {**params, "apiKey": "***"},
    )
    payload, quota_headers = _request_with_retry(
        session,
        f"/historical/sports/{sport_key}/events/{event_id}/markets",
        params=params,
        retries=max(0, int(retries)),
        sleep=max(0.0, float(sleep)),
        timeout=float(timeout),
    )
    return _flatten_event_markets(
        payload,
        requested_date=date,
        requested_bookmakers=bookmakers,
        requested_regions=regions,
        quota_headers=quota_headers,
    )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch historical NFL events and player-prop odds from The Odds API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=["events", "event-markets", "event-odds"], required=True)
    parser.add_argument("--date", required=True, help="Historical snapshot timestamp, for example 2024-09-08T16:55:00Z.")
    parser.add_argument("--sport-key", default=DEFAULT_SPORT_KEY)
    parser.add_argument("--event-id", default=None, help="Required for --mode event-markets or event-odds.")
    parser.add_argument("--event-ids", nargs="+", default=None, help="Optional filter for --mode events.")
    parser.add_argument("--commence-time-from", default=None)
    parser.add_argument("--commence-time-to", default=None)
    parser.add_argument("--markets", nargs="+", default=DEFAULT_PLAYER_PROP_MARKETS)
    parser.add_argument("--bookmakers", nargs="+", default=DEFAULT_BOOKMAKERS)
    parser.add_argument("--regions", nargs="+", default=DEFAULT_REGIONS)
    parser.add_argument("--odds-format", default="american")
    parser.add_argument("--include-links", action="store_true")
    parser.add_argument("--include-sids", action="store_true")
    parser.add_argument("--season", type=int, default=None, help="Optional local output label for event odds.")
    parser.add_argument("--week", type=int, default=None, help="Optional local output label for event odds.")
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

    if args.mode in {"event-markets", "event-odds"} and not args.event_id:
        raise ValueError(f"--event-id is required for --mode {args.mode}")

    if args.mode == "events":
        output = args.output or _build_events_output_path(args.date)
        estimated_cost = 1
    elif args.mode == "event-markets":
        output = args.output or _build_event_markets_output_path(
            event_id=str(args.event_id),
            date=args.date,
            season=args.season,
            week=args.week,
        )
        estimated_cost = "1 credit"
    else:
        output = args.output or _build_event_odds_output_path(
            event_id=str(args.event_id),
            date=args.date,
            season=args.season,
            week=args.week,
        )
        estimated_cost = f"up to {10 * len(args.markets)} credits before empty-market adjustment"

    if args.dry_run:
        logging.info("[dry-run] Would request Odds API mode=%s date=%s output=%s", args.mode, args.date, output)
        logging.info("[dry-run] Estimated quota cost: %s", estimated_cost)
        return

    api_key = get_odds_api_paid_key()
    if not api_key:
        raise RuntimeError(f"{ODDS_API_PAID_KEY_SECRET} not configured. Add it to secrets.env.")

    if args.mode == "events":
        frame = fetch_historical_events(
            api_key=api_key,
            date=args.date,
            sport_key=args.sport_key,
            event_ids=args.event_ids,
            commence_time_from=args.commence_time_from,
            commence_time_to=args.commence_time_to,
            retries=args.retries,
            sleep=args.sleep,
            timeout=args.timeout,
        )
        if not frame.empty:
            frame.insert(0, "oddsapi_feed", "historical_events")
    elif args.mode == "event-markets":
        frame = fetch_historical_event_markets(
            api_key=api_key,
            event_id=str(args.event_id),
            date=args.date,
            sport_key=args.sport_key,
            bookmakers=args.bookmakers,
            regions=args.regions,
            retries=args.retries,
            sleep=args.sleep,
            timeout=args.timeout,
        )
        if not frame.empty:
            frame.insert(0, "oddsapi_feed", "historical_event_markets")
            if args.season is not None:
                frame.insert(1, "requested_season", int(args.season))
            if args.week is not None:
                frame.insert(2 if args.season is not None else 1, "requested_week", int(args.week))
    else:
        frame = fetch_historical_event_odds(
            api_key=api_key,
            event_id=str(args.event_id),
            date=args.date,
            sport_key=args.sport_key,
            markets=args.markets,
            bookmakers=args.bookmakers,
            regions=args.regions,
            odds_format=args.odds_format,
            include_links=args.include_links,
            include_sids=args.include_sids,
            retries=args.retries,
            sleep=args.sleep,
            timeout=args.timeout,
        )
        if not frame.empty:
            frame.insert(0, "oddsapi_feed", "historical_event_odds")
            if args.season is not None:
                frame.insert(1, "requested_season", int(args.season))
            if args.week is not None:
                frame.insert(2 if args.season is not None else 1, "requested_week", int(args.week))

    if frame.empty:
        logging.warning("Odds API returned no rows for mode=%s date=%s.", args.mode, args.date)
    written = write_df(frame, output)
    logging.info("Saved Odds API rows=%s -> %s", len(frame), written)


if __name__ == "__main__":
    main()
