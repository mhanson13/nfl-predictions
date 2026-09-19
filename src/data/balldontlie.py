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
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd
import requests

from src.utils.io import RAW_DIR, read_df, write_df
from src.utils.secrets import get_secret


API_BASE = "https://api.balldontlie.io/nfl/v1"
REQUEST_TIMEOUT = 30.0
DEFAULT_SLEEP = 1.1
DEFAULT_RETRIES = 2
DEFAULT_PER_PAGE = 100

HEADERS_KEY = "Authorization"


@dataclass(frozen=True)
class FeedConfig:
    path: str
    description: str
    requires_season: bool = False
    season_param: Optional[str] = None
    week_param: Optional[str] = None
    season_types_param: Optional[str] = None
    paginated: bool = False
    team_ids_param: Optional[str] = None
    requires_team_ids: bool = False


class BallDontLieError(RuntimeError):
    """Raised when a BallDontLie HTTP request fails."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        body: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


FEEDS: dict[str, FeedConfig] = {
    "teams": FeedConfig(
        path="/teams",
        description="Static list of NFL team metadata.",
    ),
    "players": FeedConfig(
        path="/players",
        description="Player directory.",
        paginated=True,
        team_ids_param="team_ids[]",
    ),
    "active_players": FeedConfig(
        path="/players/active",
        description="Currently active player directory.",
        paginated=True,
        team_ids_param="team_ids[]",
    ),
    "games": FeedConfig(
        path="/games",
        description="Season and week game schedule/results.",
        requires_season=True,
        season_param="seasons[]",
        week_param="weeks[]",
        season_types_param="season_types[]",
        paginated=True,
        team_ids_param="team_ids[]",
    ),
    "standings": FeedConfig(
        path="/standings",
        description="Season standings by team.",
        requires_season=True,
        season_param="season",
    ),
    "injuries": FeedConfig(
        path="/player_injuries",
        description="Current player injury records.",
        paginated=True,
        team_ids_param="team_ids[]",
    ),
    "stats": FeedConfig(
        path="/stats",
        description="Game-level player stats.",
        requires_season=True,
        season_param="seasons[]",
        season_types_param="season_types[]",
        paginated=True,
    ),
    "season_stats": FeedConfig(
        path="/season_stats",
        description="Season-level player stat totals.",
        requires_season=True,
        season_param="season",
        season_types_param="season_types[]",
    ),
    "team_stats": FeedConfig(
        path="/team_stats",
        description="Game-level team stats.",
        requires_season=True,
        season_param="seasons",
        season_types_param="season_types[]",
        paginated=True,
        team_ids_param="team_ids",
    ),
    "team_season_stats": FeedConfig(
        path="/team_season_stats",
        description="Season-level team stat totals.",
        requires_season=True,
        season_param="season",
        season_types_param="season_types[]",
        paginated=True,
        team_ids_param="team_ids",
        requires_team_ids=True,
    ),
}

DEFAULT_FEEDS = [
    "teams",
    "players",
    "active_players",
    "games",
    "standings",
    "injuries",
    "stats",
    "season_stats",
    "team_stats",
    "team_season_stats",
]


def _build_output_path(feed: str, season: Optional[int], week: Optional[int]) -> Path:
    base = f"balldontlie_{feed}"
    if season is not None:
        base = f"{base}_{season}"
    if week is not None:
        base = f"{base}_wk{int(week):02d}"
    return RAW_DIR / f"{base}.parquet"


def _cached_feed_has_rows(path: Path) -> bool:
    """Return True when an existing cache file is readable and non-empty."""
    try:
        frame = read_df(path)
    except Exception as exc:
        logging.warning("Cached BallDontLie file %s is unreadable (%s); refetching.", path, exc)
        return False
    if frame.empty:
        logging.info("Cached BallDontLie file %s has no rows; refetching.", path)
        return False
    return True


def _to_dataframe(records: Any) -> pd.DataFrame:
    if isinstance(records, list):
        if not records:
            return pd.DataFrame()
        return pd.json_normalize(records)
    if isinstance(records, dict):
        return pd.json_normalize(records)
    if records is None:
        return pd.DataFrame()
    return pd.DataFrame({"value": [records]})


def _response_records(payload: Any) -> list[Any]:
    if isinstance(payload, dict) and "data" in payload:
        data = payload.get("data")
        if isinstance(data, list):
            return data
        if data is None:
            return []
        return [data]
    if isinstance(payload, list):
        return payload
    if payload is None:
        return []
    return [payload]


def _next_cursor(payload: Any) -> Optional[Any]:
    if not isinstance(payload, dict):
        return None
    meta = payload.get("meta")
    if not isinstance(meta, dict):
        return None
    return meta.get("next_cursor")


def _request_with_retry(
    session: requests.Session,
    path: str,
    *,
    params: list[tuple[str, Any]],
    retries: int,
    sleep: float,
    timeout: float,
) -> Any:
    url = API_BASE + path
    for attempt in range(retries + 1):
        response = session.get(url, params=params, timeout=timeout)
        if response.status_code == 429 and attempt < retries:
            wait = sleep * (attempt + 1)
            logging.warning("BallDontLie rate limit hit (%s), retrying in %.1fs", response.url, wait)
            time.sleep(wait)
            continue
        if response.status_code >= 500 and attempt < retries:
            wait = sleep * (attempt + 1)
            logging.warning(
                "BallDontLie server error %s for %s, retrying in %.1fs",
                response.status_code,
                response.url,
                wait,
            )
            time.sleep(wait)
            continue
        if response.status_code == 404:
            logging.warning("BallDontLie returned 404 for %s; treating as empty dataset", response.url)
            return {"data": []}
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            body_text: Optional[str] = None
            snippet = ""
            try:
                body_text = response.text
                snippet = f" body={body_text[:160]!r}"
            except Exception:
                snippet = ""
            raise BallDontLieError(
                f"BallDontLie request failed for {response.url} ({exc}){snippet}",
                status_code=response.status_code,
                body=body_text,
            ) from exc
        return response.json()
    raise BallDontLieError(f"BallDontLie request exhausted retries for {url}")


def _add_repeated(params: list[tuple[str, Any]], key: Optional[str], values: Iterable[Any]) -> None:
    if not key:
        return
    for value in values:
        if value is not None:
            params.append((key, value))


def _cached_team_ids() -> list[int]:
    path = _build_output_path("teams", None, None)
    if not path.exists():
        return []
    try:
        frame = read_df(path)
    except Exception:
        return []
    if "id" not in frame.columns:
        return []
    return sorted({int(v) for v in frame["id"].dropna().tolist()})


def _fetch_team_ids(
    session: requests.Session,
    *,
    retries: int,
    sleep: float,
    timeout: float,
) -> list[int]:
    cached = _cached_team_ids()
    if cached:
        return cached

    payload = _request_with_retry(
        session,
        FEEDS["teams"].path,
        params=[],
        retries=retries,
        sleep=sleep,
        timeout=timeout,
    )
    frame = _to_dataframe(_response_records(payload))
    if frame.empty or "id" not in frame.columns:
        raise RuntimeError("BallDontLie teams endpoint returned no team IDs.")
    return sorted({int(v) for v in frame["id"].dropna().tolist()})


def _build_params(
    feed: FeedConfig,
    *,
    season: Optional[int],
    week: Optional[int],
    season_types: list[int],
    team_ids: list[int],
) -> list[tuple[str, Any]]:
    params: list[tuple[str, Any]] = []
    if feed.season_param and season is not None:
        params.append((feed.season_param, int(season)))
    if feed.week_param and week is not None:
        params.append((feed.week_param, int(week)))
    _add_repeated(params, feed.season_types_param, season_types)
    _add_repeated(params, feed.team_ids_param, team_ids)
    return params


def _fetch_payload_records(
    session: requests.Session,
    feed: FeedConfig,
    *,
    params: list[tuple[str, Any]],
    per_page: int,
    retries: int,
    sleep: float,
    timeout: float,
) -> list[Any]:
    if not feed.paginated:
        payload = _request_with_retry(
            session,
            feed.path,
            params=params,
            retries=retries,
            sleep=sleep,
            timeout=timeout,
        )
        return _response_records(payload)

    records: list[Any] = []
    cursor: Optional[Any] = None
    while True:
        page_params = list(params)
        page_params.append(("per_page", per_page))
        if cursor is not None:
            page_params.append(("cursor", cursor))
        payload = _request_with_retry(
            session,
            feed.path,
            params=page_params,
            retries=retries,
            sleep=sleep,
            timeout=timeout,
        )
        records.extend(_response_records(payload))
        cursor = _next_cursor(payload)
        if cursor is None:
            break
        if sleep > 0:
            time.sleep(sleep)
    return records


def _fetch_feed(
    session: requests.Session,
    feed_name: str,
    feed: FeedConfig,
    *,
    season: Optional[int],
    week: Optional[int],
    season_types: list[int],
    team_ids: list[int],
    overwrite: bool,
    sleep: float,
    retries: int,
    timeout: float,
    per_page: int,
    dry_run: bool,
) -> Optional[Path]:
    output_path = _build_output_path(feed_name, season if feed.requires_season else None, week)
    if output_path.exists() and not overwrite and _cached_feed_has_rows(output_path):
        logging.info(
            "Skipping %s%s%s (cached at %s)",
            feed_name,
            f" {season}" if season else "",
            f" week {week}" if week else "",
            output_path,
        )
        return None

    resolved_team_ids = team_ids
    if feed.requires_team_ids and not resolved_team_ids:
        if not dry_run:
            resolved_team_ids = _fetch_team_ids(
                session,
                retries=retries,
                sleep=sleep,
                timeout=timeout,
            )

    params = _build_params(
        feed,
        season=season,
        week=week,
        season_types=season_types,
        team_ids=resolved_team_ids,
    )
    logging.info("Requesting %s%s params=%s", API_BASE, feed.path, params)
    if dry_run:
        logging.info("[dry-run] Would write response to %s", output_path)
        return None

    try:
        records = _fetch_payload_records(
            session,
            feed,
            params=params,
            per_page=per_page,
            retries=retries,
            sleep=sleep,
            timeout=timeout,
        )
    except BallDontLieError as exc:
        if exc.status_code in {401, 402, 403}:
            logging.warning(
                "BallDontLie feed %s (season=%s, week=%s) is not authorized for this API key/tier (status=%s); skipping.",
                feed_name,
                season,
                week,
                exc.status_code,
            )
            return None
        raise

    frame = _to_dataframe(records)
    if frame.empty:
        logging.warning("Feed %s returned no rows (season=%s, week=%s)", feed_name, season, week)
    else:
        frame.insert(0, "balldontlie_feed", feed_name)
        insert_at = 1
        if season is not None:
            frame.insert(insert_at, "requested_season", season)
            insert_at += 1
        if week is not None:
            frame.insert(insert_at, "requested_week", week)
            insert_at += 1
        if season_types:
            frame.insert(insert_at, "requested_season_types", ",".join(str(v) for v in season_types))
    actual_path = write_df(frame, output_path)
    logging.info("Saved %s rows -> %s", len(frame), actual_path)
    return actual_path


def _resolve_jobs(
    feeds: Iterable[str],
    seasons: Optional[Iterable[int]],
    weeks: Optional[Iterable[int]],
) -> list[tuple[str, Optional[int], Optional[int]]]:
    chosen: list[tuple[str, Optional[int], Optional[int]]] = []
    season_list = sorted({int(s) for s in seasons}) if seasons else []
    week_list = sorted({int(w) for w in weeks}) if weeks else []

    for feed_name in feeds:
        feed = FEEDS[feed_name]
        if feed.requires_season and not season_list:
            raise ValueError(f"Feed '{feed_name}' requires --seasons.")
        seasons_to_use: list[Optional[int]] = season_list if feed.requires_season else [None]
        if feed.week_param and week_list:
            weeks_to_use: list[Optional[int]] = week_list
        else:
            weeks_to_use = [None]

        for season_val in seasons_to_use:
            for week_val in weeks_to_use:
                chosen.append((feed_name, season_val, week_val))
    return chosen


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch NFL data from BallDontLie.")
    parser.add_argument(
        "--feeds",
        nargs="+",
        choices=sorted(FEEDS.keys()),
        default=DEFAULT_FEEDS,
        help="BallDontLie feeds to fetch.",
    )
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        help="Season years needed for season-scoped feeds.",
    )
    parser.add_argument(
        "--weeks",
        type=int,
        nargs="+",
        help="Optional week filters for feeds that support weeks, such as games.",
    )
    parser.add_argument(
        "--season-types",
        type=int,
        nargs="+",
        choices=[1, 2, 3],
        default=[2],
        help="Season type filters: 1=preseason, 2=regular season, 3=postseason.",
    )
    parser.add_argument(
        "--team-ids",
        type=int,
        nargs="+",
        default=None,
        help="Optional BallDontLie team IDs for feeds that support team filtering.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Refetch even if cached files exist.")
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP, help="Seconds to sleep between API calls.")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="Retry attempts for failed requests.")
    parser.add_argument("--timeout", type=float, default=REQUEST_TIMEOUT, help="Request timeout in seconds.")
    parser.add_argument("--per-page", type=int, default=DEFAULT_PER_PAGE, help="Page size for paginated endpoints.")
    parser.add_argument("--dry-run", action="store_true", help="Log planned downloads without executing.")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging.")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    api_key = get_secret("BALLDONTLIE_API_KEY")
    if not api_key:
        raise RuntimeError("BALLDONTLIE_API_KEY not configured. Add it to secrets.env.")

    session = requests.Session()
    session.headers.update({HEADERS_KEY: api_key})

    pending_jobs = _resolve_jobs(args.feeds, args.seasons, args.weeks)
    if not pending_jobs:
        logging.info("No feeds selected; exiting.")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    per_page = min(max(int(args.per_page), 1), DEFAULT_PER_PAGE)
    sleep = max(0.0, args.sleep)
    retries = max(0, args.retries)
    season_types = sorted({int(v) for v in args.season_types})
    team_ids = sorted({int(v) for v in args.team_ids}) if args.team_ids else []

    for feed_name, season, week in pending_jobs:
        feed = FEEDS[feed_name]
        try:
            _fetch_feed(
                session,
                feed_name,
                feed,
                season=season,
                week=week,
                season_types=season_types,
                team_ids=team_ids,
                overwrite=args.overwrite,
                sleep=sleep,
                retries=retries,
                timeout=args.timeout,
                per_page=per_page,
                dry_run=args.dry_run,
            )
        finally:
            if sleep > 0:
                time.sleep(sleep)


if __name__ == "__main__":
    main()
