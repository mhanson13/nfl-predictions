from __future__ import annotations
import argparse
from pathlib import Path
from io import StringIO
from typing import List, Optional

import pandas as pd
import requests

from src.utils.io import RAW_DIR, write_df
from src.utils.logging import configure as configure_logging

BASE_URLS = {
    "passing": "https://www.espn.com/nfl/stats/player/_/season/{year}/seasontype/{stype}",
    "rushing": "https://www.espn.com/nfl/stats/player/_/stat/rushing/season/{year}/seasontype/{stype}",
    "receiving": "https://www.espn.com/nfl/stats/player/_/stat/receiving/season/{year}/seasontype/{stype}",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


def _fetch_page(url: str) -> Optional[pd.DataFrame]:
    """Fetch a single ESPN stats page and return the main table if present."""
    try:
        # Some ESPN pages require a real HTTP request for cookies before read_html
        sess = requests.Session()
        sess.get(url, headers=HEADERS, timeout=20)
        html = sess.get(url, headers=HEADERS, timeout=20).text
        tables = pd.read_html(StringIO(html))
        # Heuristic: last (or only) table is the player stats grid
        if not tables:
            return None
        # Filter out small non-stats tables
        tables = [t for t in tables if t.shape[1] >= 5 and t.shape[0] >= 5]
        if not tables:
            return None
        df = tables[-1]
        # Drop completely empty columns and header repeat rows
        df = df.dropna(axis=1, how="all")
        # Remove header rows that ESPN repeats in the middle of the table
        if df.columns.tolist() == list(df.iloc[0].values):
            df = df[1:].reset_index(drop=True)
        return df
    except Exception:
        return None


def _fetch_category(year: int, stype: int, category: str) -> pd.DataFrame:
    """Download and combine all pages for a category/year/season type.
    stype: 2=regular season, 3=postseason
    """
    if category not in BASE_URLS:
        raise ValueError(f"Unsupported category: {category}")

    frames: List[pd.DataFrame] = []
    seen = set()
    page = 1
    while True:
        url = BASE_URLS[category].format(year=year, stype=stype)
        if page > 1:
            url = f"{url}?page={page}"
        df = _fetch_page(url)
        if df is None or df.empty:
            break
        # Deduplicate by first two columns and page index
        keycols = df.columns[:2].tolist()
        df["_key"] = (
            df[keycols[0]].astype(str).str.strip().fillna("") + "|" + df[keycols[1]].astype(str).str.strip().fillna("")
        )
        new = df[~df["_key"].isin(seen)].copy()
        if new.empty:
            break
        seen.update(new["_key"].tolist())
        new.drop(columns=["_key"], inplace=True, errors="ignore")
        new["season"] = year
        new["season_type"] = stype
        new["category"] = category
        new["page"] = page
        frames.append(new)
        page += 1

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    # Normalize column names to snake_case and deduplicate
    cols = (
        out.columns.astype(str).str.strip().str.lower().str.replace(r"[^0-9a-z]+", "_", regex=True).str.strip("_")
    )
    # Deduplicate columns by suffixing _1, _2, ...
    seen = {}
    new_cols = []
    for c in cols:
        n = seen.get(c, 0)
        if n == 0:
            new_cols.append(c)
        else:
            new_cols.append(f"{c}_{n}")
        seen[c] = n + 1
    out.columns = new_cols
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, nargs="+", required=True, help="Year(s) to fetch, e.g., 2004 2025")
    ap.add_argument("--season_type", type=int, choices=[2, 3], default=2, help="2=regular, 3=postseason")
    ap.add_argument("--category", nargs="+", default=["passing", "rushing", "receiving"],
                    choices=["passing", "rushing", "receiving"], help="Categories to fetch")
    ap.add_argument("--debug", action="store_true", help="Enable verbose debug output")
    args = ap.parse_args()

    configure_logging(args.debug)

    for year in args.season:
        for cat in args.category:
            df = _fetch_category(year, args.season_type, cat)
            if df is None or df.empty:
                print(f"[espn_players] No data for {cat} {year} stype={args.season_type}")
                continue
            out_path = RAW_DIR / f"espn_playerstats_{cat}_{year}_stype{args.season_type}.parquet"
            write_df(df, out_path)
            print(f"[espn_players] Saved {len(df)} rows -> {out_path}")


if __name__ == "__main__":
    main()
