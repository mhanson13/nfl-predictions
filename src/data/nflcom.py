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
import re
import time
import argparse
from dataclasses import dataclass
from typing import List, Dict
import pandas as pd
from urllib.parse import urlencode
from tqdm import tqdm
from src.utils.io import RAW_DIR, write_df, read_df
from src.utils.logging import configure as configure_logging

CATEGORIES = {
    "passing": "offense/passing",
    "rushing": "offense/rushing",
    "receiving": "offense/receiving",
    "scoring": "offense/scoring",
    "downs": "offense/downs",
}
BASE = "https://www.nfl.com/stats/team-stats/{path}/{year}/reg/all"

def _page_url(path: str, year: int, page: int | None = None) -> str:
    base = BASE.format(path=path, year=year)
    if page is None or page <= 1:
        return base
    # NFL uses query parameter 'page' for pagination when present
    return base + f"?{urlencode({'page': page})}"

def fetch_team_stats(year: int, category: str, max_pages: int = 10, pause: float = 0.8) -> pd.DataFrame:
    assert category in CATEGORIES, f"Unknown category: {category}"
    path = CATEGORIES[category]
    frames: List[pd.DataFrame] = []
    for page in range(1, max_pages + 1):
        url = _page_url(path, year, page=None if page == 1 else page)
        try:
            tables = pd.read_html(url, flavor=None, displayed_only=False)
        except Exception as e:
            if page == 1 and not frames:
                raise
            else:
                break
        # Heuristic: pick the largest table on the page
        if not tables:
            break
        df = max(tables, key=lambda x: x.shape[0]*x.shape[1])
        df["page"] = page
        df["category"] = category
        df["year"] = year
        frames.append(df)
        time.sleep(pause)
        # If table rows < teams, likely no pagination
        if df.shape[0] < 20:
            break
    if not frames:
        raise RuntimeError(f"No tables parsed for {category} {year}.")
    out = pd.concat(frames, ignore_index=True)
    # Clean: Flatten multiindex columns, strip
    out.columns = [" ".join(map(str, c)).strip() for c in out.columns.values]
    for c in out.columns:
        if isinstance(c, str):
            out.rename(columns={c: c.strip()}, inplace=True)
    # Standardize team column name
    team_col = next((c for c in out.columns if re.search(r"team", str(c), re.I)), None)
    if team_col and team_col != "Team":
        out.rename(columns={team_col: "Team"}, inplace=True)
    return out

def fetch_multi(years: List[int], categories: List[str], force: bool = False) -> Dict[str, pd.DataFrame]:
    results: Dict[str, pd.DataFrame] = {}
    for cat in categories:
        frames = []
        for y in tqdm(years, desc=f"Category={cat}"):
            per_year_path = RAW_DIR / f"nflcom_teamstats_{cat}_{y}.parquet"
            df = None
            if per_year_path.exists() and not force:
                try:
                    df = read_df(per_year_path)
                except Exception:
                    df = None
            if df is None or df.empty:
                df = fetch_team_stats(y, cat)
                # Save per-year cache
                write_df(df, per_year_path)
            frames.append(df)
        results[cat] = pd.concat(frames, ignore_index=True)
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, nargs="+", required=True, help="e.g., 2025 2024 or range")
    ap.add_argument("--category", type=str, nargs="+", default=list(CATEGORIES.keys()))
    ap.add_argument("--out_prefix", type=str, default="nflcom_teamstats")
    ap.add_argument("--force", action="store_true", help="Force re-download even if per-year cached files exist")
    ap.add_argument("--debug", action="store_true", help="Enable verbose debug output")
    args = ap.parse_args()

    configure_logging(args.debug)

    # Expand year ranges if given like 2010-2020
    years = []
    for y in args.year:
        years.append(int(y))
    out = fetch_multi(years, args.category, force=args.force)
    for cat, df in out.items():
        write_df(df, RAW_DIR / f"{args.out_prefix}_{cat}.parquet")

if __name__ == "__main__":
    main()
