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

from src.utils.io import RAW_DIR, write_df
from src.utils.secrets import get_secret


API_BASE = "https://api.sportsdata.io/v3/nfl"
REQUEST_TIMEOUT = 30.0
DEFAULT_SLEEP = 0.3
DEFAULT_RETRIES = 2

HEADERS_KEY = "Ocp-Apim-Subscription-Key"


@dataclass(frozen=True)
class FeedConfig:
    path: str
    requires_season: bool
    description: str
    season_formatter: Optional[str] = None  # Formatting pattern for season strings
    requires_week: bool = False


@dataclass(frozen=True)
class TimeframeSnapshot:
    season: int
    week: int
    api_season: str
    api_week: str
    season_type: int


FEEDS: dict[str, FeedConfig] = {
    "teams": FeedConfig(
        path="/scores/json/Teams",
        requires_season=False,
        description="Static list of franchise/team metadata.",
    ),
    "stadiums": FeedConfig(
        path="/scores/json/Stadiums",
        requires_season=False,
        description="Venue directory containing coordinates and surface information.",
    ),
    "schedules": FeedConfig(
        path="/scores/json/Schedules/{season}",
        requires_season=True,
        description="Full schedule including kickoff times, moneylines, and weather forecasts.",
    ),
    "standings": FeedConfig(
        path="/scores/json/Standings/{season}",
        requires_season=True,
        description="Seasonal standings with win/loss records and playoff clinch data.",
    ),
    "timeframes": FeedConfig(
        path="/scores/json/Timeframes/{season_code}",
        requires_season=True,
        description="Date boundaries for season segments (regular/postseason/preseason).",
        season_formatter="{season}REG",
    ),
    "injuries": FeedConfig(
        path="/scores/json/Injuries/{season_code}",
        requires_season=True,
        description="Weekly injury reports (returns empty outside active season windows).",
        season_formatter="{season}REG",
    ),
    "player_season_projections": FeedConfig(
        path="/projections/json/PlayerSeasonProjectionStats/{season_code}",
        requires_season=True,
        description="Season-level fantasy/stat projections.",
        season_formatter="{season}REG",
    ),
    "player_game_projections": FeedConfig(
        path="/projections/json/PlayerGameProjectionStatsByWeek/{season_code}/{week}",
        requires_season=True,
        description="Weekly player fantasy/stat projections.",
        season_formatter="{season}REG",
        requires_week=True,
    ),
    "dfs_slates": FeedConfig(
        path="/projections/json/DfsSlatesByWeek/{season_code}/{week}",
        requires_season=True,
        description="DFS slates, contests, and player pool for a given week.",
        season_formatter="{season}REG",
        requires_week=True,
    ),
    "betting_futures": FeedConfig(
        path="/odds/json/BettingFuturesBySeason/{season}",
        requires_season=True,
        description="Season-long betting futures (requires Odds access).",
    ),
    "draft_picks": FeedConfig(
        path="/scores/json/PlayerDrafts/{season}",
        requires_season=True,
        description="NFL draft selections by season (may be restricted by plan).",
    ),
    "free_agents": FeedConfig(
        path="/scores/json/FreeAgents",
        requires_season=False,
        description="Current list of free agents.",
    ),
    "players": FeedConfig(
        path="/stats/json/Players",
        requires_season=False,
        description="Complete player directory (large response, ~5k rows).",
    ),
}


def _format_season(season: int, feed: FeedConfig) -> dict[str, str]:
    """
    Build the formatting kwargs required for the feed path.

    Some endpoints expect the plain season (e.g., 2024) while others use a season code
    such as '2024REG'. We expose both `season` and `season_code` placeholders so the
    path can use whichever is appropriate.
    """
    season_str = str(season)
    if feed.season_formatter:
        season_code = feed.season_formatter.format(season=season_str)
    else:
        season_code = season_str
    return {"season": season_str, "season_code": season_code}


def _build_output_path(feed: str, season: Optional[int], week: Optional[int]) -> Path:
    base = f"sportsdataio_{feed}"
    if season is not None:
        base = f"{base}_{season}"
    if week is not None:
        base = f"{base}_wk{int(week):02d}"
    return RAW_DIR / f"{base}.parquet"


def _to_dataframe(payload: Any) -> pd.DataFrame:
    if isinstance(payload, list):
        if not payload:
            return pd.DataFrame()
        return pd.json_normalize(payload)
    if isinstance(payload, dict):
        return pd.json_normalize(payload)
    return pd.DataFrame({"value": [payload]})


def _request_with_retry(
    session: requests.Session,
    url: str,
    *,
    retries: int,
    sleep: float,
    timeout: float,
) -> Any:
    for attempt in range(retries + 1):
        response = session.get(url, timeout=timeout)
        if response.status_code == 429 and attempt < retries:
            wait = sleep * (attempt + 1)
            logging.warning("sportsdataio rate limit hit (%s), retrying in %.1fs", url, wait)
            time.sleep(wait)
            continue
        if response.status_code >= 500 and attempt < retries:
            wait = sleep * (attempt + 1)
            logging.warning(
                "sportsdataio server error %s for %s, retrying in %.1fs",
                response.status_code,
                url,
                wait,
            )
            time.sleep(wait)
            continue
        if response.status_code == 404:
            logging.warning("sportsdataio returned 404 for %s; treating as empty dataset", url)
            return []
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            snippet = ""
            try:
                snippet = f" body={response.text[:160]!r}"
            except Exception:
                snippet = ""
            raise RuntimeError(f"SportsDataIO request failed for {url} ({exc}){snippet}") from exc
        return response.json()
    raise RuntimeError(f"SportsDataIO request exhausted retries for {url}")


def _get_current_timeframe(session: requests.Session, timeout: float) -> TimeframeSnapshot:
    url = f"{API_BASE}/scores/json/Timeframes/current"
    payload = _request_with_retry(
        session,
        url,
        retries=0,
        sleep=DEFAULT_SLEEP,
        timeout=timeout,
    )
    if not payload:
        raise RuntimeError("SportsDataIO timeframes/current returned no data")
    data = payload[0]
    api_season = str(data.get("ApiSeason") or data.get("Season"))
    api_week = str(data.get("ApiWeek") or data.get("Week"))
    try:
        season_val = int(api_season[:4])
    except (TypeError, ValueError):
        season_val = int(data["Season"])
    try:
        week_val = int(api_week)
    except (TypeError, ValueError):
        week_val = int(data["Week"])
    season_type = int(data.get("SeasonType", 1) or 1)
    logging.info(
        "SportsDataIO current timeframe: season=%s week=%s (season_type=%s)",
        season_val,
        week_val,
        season_type,
    )
    return TimeframeSnapshot(
        season=season_val,
        week=week_val,
        api_season=api_season,
        api_week=api_week,
        season_type=season_type,
    )


def _fetch_feed(
    session: requests.Session,
    feed_name: str,
    feed: FeedConfig,
    *,
    season: Optional[int],
    week: Optional[int],
    overwrite: bool,
    sleep: float,
    retries: int,
    timeout: float,
    dry_run: bool,
) -> Optional[Path]:
    output_path = _build_output_path(
        feed_name,
        season if feed.requires_season else None,
        week if feed.requires_week else None,
    )
    if output_path.exists() and not overwrite:
        logging.info("Skipping %s%s (cached at %s)", feed_name, f" {season}" if season else "", output_path)
        return None

    url_kwargs = {}
    if feed.requires_season:
        if season is None:
            raise ValueError(f"Feed '{feed_name}' requires --seasons.")
        url_kwargs = _format_season(season, feed)
    if feed.requires_week:
        if week is None:
            raise ValueError(f"Feed '{feed_name}' requires --weeks.")
        url_kwargs["week"] = str(int(week))

    url = API_BASE + feed.path.format(**url_kwargs)
    logging.info("Requesting %s", url)
    if dry_run:
        logging.info("[dry-run] Would write response to %s", output_path)
        return None

    payload = _request_with_retry(
        session,
        url,
        retries=retries,
        sleep=sleep,
        timeout=timeout,
    )
    frame = _to_dataframe(payload)
    if frame.empty:
        logging.warning("Feed %s returned no rows (season=%s)", feed_name, season)
    else:
        frame.insert(0, "sportsdataio_feed", feed_name)
        insert_at = 1
        if feed.requires_season:
            frame.insert(insert_at, "season", season)
            insert_at += 1
        if feed.requires_week:
            frame.insert(insert_at, "week", week)
    write_df(frame, output_path)
    logging.info(
        "Saved %s rows -> %s",
        len(frame),
        output_path,
    )
    return output_path


def _resolve_jobs(
    feeds: Iterable[str],
    seasons: Optional[Iterable[int]],
    weeks: Optional[Iterable[int]],
    session: requests.Session,
    timeout: float,
) -> list[tuple[str, Optional[int], Optional[int]]]:
    chosen: list[tuple[str, Optional[int], Optional[int]]] = []
    season_list = sorted({int(s) for s in seasons}) if seasons else []
    week_list = sorted({int(w) for w in weeks}) if weeks else []
    timeframe_cache: Optional[TimeframeSnapshot] = None

    def ensure_timeframe() -> TimeframeSnapshot:
        nonlocal timeframe_cache
        if timeframe_cache is None:
            timeframe_cache = _get_current_timeframe(session, timeout)
        return timeframe_cache

    for feed_name in feeds:
        feed = FEEDS[feed_name]

        if feed.requires_season:
            seasons_to_use = season_list if season_list else [ensure_timeframe().season]
        else:
            seasons_to_use = [None]

        if feed.requires_week:
            weeks_to_use = week_list if week_list else [ensure_timeframe().week]
        else:
            weeks_to_use = [None]

        for season_val in seasons_to_use:
            for week_val in weeks_to_use:
                chosen.append(
                    (
                        feed_name,
                        season_val if feed.requires_season else None,
                        week_val if feed.requires_week else None,
                    )
                )
    return chosen


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch NFL data from SportsDataIO.")
    parser.add_argument(
        "--feeds",
        nargs="+",
        choices=sorted(FEEDS.keys()),
        default=[
            "teams",
            "stadiums",
            "schedules",
            "standings",
            "player_season_projections",
            "player_game_projections",
            "dfs_slates",
            "betting_futures",
            "draft_picks",
            "free_agents",
        ],
        help="SportsDataIO feeds to fetch.",
    )
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        help="Season years needed for season-scoped feeds (e.g., 2023 2024).",
    )
    parser.add_argument(
        "--weeks",
        type=int,
        nargs="+",
        help="Week numbers needed for weekly feeds (default: current regular-season week).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Refetch even if cached files exist.")
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP, help="Seconds to sleep between calls.")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="Retry attempts for failed requests.")
    parser.add_argument("--timeout", type=float, default=REQUEST_TIMEOUT, help="Request timeout in seconds.")
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
    logging.getLogger("urllib3").setLevel(logging.DEBUG if args.debug else logging.WARNING)

    api_key = get_secret("SPORTSDATAIO_API_KEY")
    if not api_key:
        raise RuntimeError("SPORTSDATAIO_API_KEY not configured. Add it to secrets.env.")

    session = requests.Session()
    session.headers.update({HEADERS_KEY: api_key})

    pending_jobs = _resolve_jobs(args.feeds, args.seasons, args.weeks, session, args.timeout)
    if not pending_jobs:
        logging.info("No feeds selected; exiting.")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for feed_name, season, week in pending_jobs:
        feed = FEEDS[feed_name]
        try:
            _fetch_feed(
                session,
                feed_name,
                feed,
                season=season,
                week=week,
                overwrite=args.overwrite,
                sleep=max(0.0, args.sleep),
                retries=max(0, args.retries),
                timeout=args.timeout,
                dry_run=args.dry_run,
            )
        finally:
            if args.sleep > 0:
                time.sleep(args.sleep)


if __name__ == "__main__":
    main()
