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
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests
from pathlib import Path

from src.utils.io import RAW_DIR
from src.utils.logging import configure as configure_logging

ESPN_NEWS_URL = "http://site.api.espn.com/apis/site/v2/sports/football/nfl/news"

TEAM_NAME_TO_ABBR: Dict[str, str] = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LAR",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
    # Historical / alternate labels that surface in ESPN categories.
    "St. Louis Rams": "LAR",
    "San Diego Chargers": "LAC",
    "Oakland Raiders": "LV",
    "Phoenix Cardinals": "ARI",
    "Washington Redskins": "WAS",
    "Washington Football Team": "WAS",
    "Los Angeles Raiders": "LV",
}


@dataclass
class ArticleRecord:
    season: int
    news_id: str
    headline: str
    description: str
    published: str
    last_modified: Optional[str]
    article_type: Optional[str]
    byline: Optional[str]
    link_web: Optional[str]
    link_mobile: Optional[str]
    source: Optional[str]
    team_abbr: Optional[str]


def _map_team(description: Optional[str]) -> Optional[str]:
    if not description:
        return None
    desc = description.strip()
    if not desc:
        return None
    # Some category descriptions look like "San Francisco 49ers News".
    if desc.endswith(" News"):
        base = desc[:-5]
        if base in TEAM_NAME_TO_ABBR:
            return TEAM_NAME_TO_ABBR[base]
    return TEAM_NAME_TO_ABBR.get(desc)


def _extract_links(article: dict) -> tuple[Optional[str], Optional[str]]:
    links = article.get("links") or {}
    web_link = None
    mobile_link = None
    if isinstance(links, dict):
        web_info = links.get("web")
        mobile_info = links.get("mobile")
        if isinstance(web_info, dict):
            web_link = web_info.get("href")
        if isinstance(mobile_info, dict):
            mobile_link = mobile_info.get("href")
    return web_link, mobile_link


def _extract_source(article: dict) -> Optional[str]:
    source = article.get("dataSourceIdentifier") or article.get("source")
    if isinstance(source, str):
        return source
    return None


def fetch_articles(limit: int = 50) -> List[dict]:
    params = {"limit": limit}
    resp = requests.get(ESPN_NEWS_URL, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    return data.get("articles", [])


def normalize_articles(articles: Iterable[dict], season: int) -> pd.DataFrame:
    records: List[ArticleRecord] = []
    for article in articles:
        published = article.get("published")
        if not published or str(season) not in str(published):
            # Skip non-target seasons.
            continue
        headline = article.get("headline", "")
        description = article.get("description", "")
        last_modified = article.get("lastModified")
        article_type = article.get("type")
        byline = article.get("byline")
        link_web, link_mobile = _extract_links(article)
        source = _extract_source(article)
        news_id = str(article.get("id", ""))

        teams = []
        for category in article.get("categories", []):
            if category.get("type") != "team":
                continue
            abbr = _map_team(category.get("description"))
            if abbr:
                teams.append(abbr)
        if not teams:
            # League-wide story.
            teams = [None]

        for team in teams:
            records.append(
                ArticleRecord(
                    season=season,
                    news_id=news_id,
                    headline=headline,
                    description=description,
                    published=published,
                    last_modified=last_modified,
                    article_type=article_type,
                    byline=byline,
                    link_web=link_web,
                    link_mobile=link_mobile,
                    source=source,
                    team_abbr=team,
                )
            )

    if not records:
        return pd.DataFrame()
    df = pd.DataFrame([r.__dict__ for r in records])
    df["team_abbr"] = df["team_abbr"].astype("object")
    return df


def save_articles(df: pd.DataFrame) -> Path:
    output_path = RAW_DIR / "espn_news.parquet"
    if df.empty:
        output_path.unlink(missing_ok=True)
        return output_path
    df = df.sort_values("published", ascending=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch league-wide ESPN NFL news articles.")
    parser.add_argument("--season", type=int, required=True, help="Season year (used to filter articles by published year).")
    parser.add_argument("--limit", type=int, default=50, help="Number of articles to request from the API (default: 50).")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug output.")
    args = parser.parse_args()

    configure_logging(args.debug)

    if args.debug:
        print(f"[espn_team_news] fetching up to {args.limit} articles for season {args.season}")

    articles = fetch_articles(limit=args.limit)
    if args.debug:
        print(f"[espn_team_news] raw articles received: {len(articles)}")
    df = normalize_articles(articles, season=args.season)
    path = save_articles(df)
    print(f"[espn_team_news] saved {len(df)} rows -> {path}")


if __name__ == "__main__":
    main()
