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
import time
import concurrent.futures
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import requests

from src.utils.io import RAW_DIR, read_df, write_df
from src.utils.logging_config import setup_logging

NEWS_ENDPOINT = "https://site.api.espn.com/apis/fantasy/v2/games/ffl/news/players"
META_PATH = RAW_DIR / "espn_player_news_meta.json"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


def _coerce_int(val: Any) -> Optional[int]:
    try:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        return int(val)
    except Exception:
        return None


def _fetch_player_news(player_id: int, limit: int) -> List[Dict[str, Any]]:
    params = {"limit": limit, "playerId": str(player_id)}
    try:
        resp = requests.get(NEWS_ENDPOINT, params=params, timeout=10, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    feed = data.get("feed") or []
    rows: List[Dict[str, Any]] = []
    for item in feed:
        rows.append(
            {
                "player_id": player_id,
                "news_id": item.get("id"),
                "type": item.get("type"),
                "headline": item.get("headline"),
                "description": item.get("description"),
                "story": item.get("story"),
                "source": item.get("source"),
                "byline": item.get("byline"),
                "published": item.get("published") or item.get("categorized"),
                "last_modified": item.get("lastModified"),
                "categorized": item.get("categorized"),
                "link_web": ((item.get("links") or {}).get("web") or {}).get("href"),
                "link_mobile": ((item.get("links") or {}).get("mobile") or {}).get("href"),
                "link_api": ((item.get("links") or {}).get("api") or {}).get("self", {}).get("href"),
                "premium": item.get("premium"),
                "is_live_blog": item.get("isLiveBlog"),
                "allow_comments": item.get("allowComments"),
                "allow_search": item.get("allowSearch"),
                "allow_reactions": item.get("allowContentReactions"),
            }
        )
    return rows

def _fetch_player_news_with_delay(player_id: int, limit: int, delay: float) -> List[Dict[str, Any]]:
    rows = _fetch_player_news(player_id, limit)
    if delay > 0:
        time.sleep(delay)
    return rows


def _load_roster_player_ids(seasons: Iterable[int]) -> pd.DataFrame:
    roster_path = RAW_DIR / "nfl_rosters.parquet"
    if not roster_path.exists():
        raise FileNotFoundError(
            f"Expected roster data at {roster_path}. Run: python -m src.data.nflverse --season <year> --rosters"
        )

    roster = read_df(roster_path)
    roster = roster[roster["season"].isin(list(seasons))].copy()
    roster["espn_id"] = roster["espn_id"].apply(_coerce_int)
    out = (
        roster.dropna(subset=["espn_id"])
        .groupby("espn_id", as_index=False)
        .agg(
            first_season=("season", "min"),
            last_season=("season", "max"),
        )
    )
    return out


def _filter_time_window(df: pd.DataFrame, season: int) -> pd.DataFrame:
    """Keep news within a reasonable window around the target season."""
    if df.empty:
        return df
    df = df.copy()
    df["published_dt"] = pd.to_datetime(df["published"], errors="coerce", utc=True)
    start_bound = pd.Timestamp(year=season - 1, month=6, day=1, tz="UTC")
    end_bound = pd.Timestamp(year=season + 1, month=3, day=1, tz="UTC")
    df = df[(df["published_dt"].notna()) & (df["published_dt"] >= start_bound) & (df["published_dt"] <= end_bound)]
    df["season"] = season
    return df.drop(columns=["published_dt"])


def _load_meta() -> Optional[Dict[str, Any]]:
    if not META_PATH.exists():
        return None
    try:
        with open(META_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _write_meta(timestamp: datetime) -> None:
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(META_PATH, "w", encoding="utf-8") as fh:
        json.dump({"last_run_utc": timestamp.replace(microsecond=0).isoformat() + "Z"}, fh)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, nargs="+", required=True, help="Season year(s) to collect news for.")
    ap.add_argument("--limit", type=int, default=10, help="Max news items to request per player (default: 10).")
    ap.add_argument("--sleep", type=float, default=0.15, help="Seconds to sleep between requests (default: 0.15).")
    ap.add_argument("--max-workers", type=int, default=1, help="Concurrent request workers (default: 1).")
    ap.add_argument("--min-interval-minutes", type=float, default=240.0,
                    help="Minimum interval between fetches (default: 240 minutes / 4 hours).")
    ap.add_argument("--grace-minutes", type=float, default=15.0,
                    help="Always skip if the last run occurred less than this many minutes ago (default: 15).")
    ap.add_argument("--force-refresh", action="store_true", help="Ignore caching interval and fetch news now.")
    ap.add_argument("--debug", action="store_true", help="Enable verbose logging.")
    args = ap.parse_args()

    setup_logging("nfl_predictions", level="DEBUG" if args.debug else "INFO")

    now_utc = datetime.utcnow()
    if not args.force_refresh:
        meta = _load_meta()
        if meta and isinstance(meta, dict) and meta.get("last_run_utc"):
            try:
                last_run = datetime.fromisoformat(str(meta["last_run_utc"]).replace("Z", "+00:00"))
            except ValueError:
                last_run = None
            if last_run is not None:
                delta = now_utc - last_run.replace(tzinfo=None)
                if delta < timedelta(minutes=args.grace_minutes):
                    print(
                        "[espn_player_news] Last run was "
                        f"{delta.total_seconds()/60:.1f} minutes ago (<{args.grace_minutes}m); skipping."
                    )
                    return
                if delta < timedelta(minutes=args.min_interval_minutes):
                    print(
                        "[espn_player_news] Last run was "
                        f"{delta.total_seconds()/60:.1f} minutes ago (<{args.min_interval_minutes}m); skipping."
                    )
                    return

    seasons = sorted(set(args.season))
    roster_ids = _load_roster_player_ids(seasons)
    if roster_ids.empty:
        print("[espn_player_news] No rostered players found for requested seasons.")
        return

    news_frames: List[pd.DataFrame] = []
    total_players = 0
    processed = 0

    for season in seasons:
        player_mask = (roster_ids["first_season"] <= season) & (roster_ids["last_season"] >= season)
        player_ids = roster_ids["espn_id"].where(player_mask).dropna().astype(int)
        if player_ids.empty:
            print(f"[espn_player_news] No players with ESPN ids found for season {season}.")
            continue

        total_players += len(player_ids)
        season_rows: List[Dict[str, Any]] = []
        throttle_delay = max(args.sleep, 0.0)
        max_workers = max(1, args.max_workers)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_fetch_player_news_with_delay, pid, args.limit, throttle_delay): pid
                for pid in player_ids
            }
            for future in concurrent.futures.as_completed(futures):
                pid = futures[future]
                processed += 1
                try:
                    rows = future.result()
                except Exception as exc:
                    if args.debug:
                        print(f"[espn_player_news] player {pid} request failed: {exc}")
                    rows = []
                if rows:
                    season_rows.extend(rows)
                if processed % 100 == 0:
                    print(f"[espn_player_news] processed {processed}/{total_players} player ids...")

        df_season = pd.DataFrame(season_rows)
        if not df_season.empty:
            df_season = df_season.drop_duplicates(subset=["player_id", "news_id"])
            df_season = _filter_time_window(df_season, season)
            news_frames.append(df_season)
            print(f"[espn_player_news] season {season}: collected {len(df_season)} news rows.")
        else:
            print(f"[espn_player_news] season {season}: no news rows collected.")

    if not news_frames:
        print("[espn_player_news] No news collected; nothing to write.")
        _write_meta(now_utc)
        return

    combined = pd.concat(news_frames, ignore_index=True)
    combined["player_id"] = combined["player_id"].apply(_coerce_int)
    combined = combined.dropna(subset=["player_id", "published"])

    out_path = RAW_DIR / "espn_player_news.parquet"
    if out_path.exists():
        existing = read_df(out_path)
        combined = pd.concat([existing, combined], ignore_index=True)
        combined = combined.drop_duplicates(subset=["player_id", "news_id", "published", "season"])

    write_df(combined, out_path)
    print(f"[espn_player_news] Saved {len(combined)} rows -> {out_path}")
    _write_meta(now_utc)


if __name__ == "__main__":
    main()
