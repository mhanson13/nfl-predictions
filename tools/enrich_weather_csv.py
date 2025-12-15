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
from pathlib import Path
import pandas as pd

WEATHER_KEYS = (
    "kickoff",
    "roof",
    "venue_lat",
    "venue_lon",
    "roof_is_dome",
)

def _weather_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if isinstance(c, str) and c.startswith("weather_")]
    # plus key context
    for k in WEATHER_KEYS:
        if k in df.columns:
            cols.append(k)
    return sorted(set(cols))


def _fill_left(base: pd.DataFrame, add: pd.DataFrame, on: list[str]) -> pd.DataFrame:
    """Left-merge 'add' into 'base' on keys and fill only missing values for weather/context columns.
    Returns updated base.
    """
    if add is None or add.empty:
        return base
    # Ensure key types compatible
    for k in on:
        if k in base.columns:
            base[k] = base[k].astype(str)
        if k in add.columns:
            add[k] = add[k].astype(str)
    wcols = _weather_cols(add)
    if not all(k in add.columns for k in on):
        return base
    r = base.merge(add[on + wcols].drop_duplicates(subset=on), on=on, how="left", suffixes=("", "_wx"))
    # Helper to get a 1-D Series even when duplicate column names exist
    def _as_series(df: pd.DataFrame, col: str) -> pd.Series:
        obj = df[col]
        if isinstance(obj, pd.DataFrame):
            # Select the first actual column by name to avoid iloc typing issues
            first = obj.columns[0]
            return obj[first]
        return obj  # type: ignore
    # Helper to assign into the first occurrence of a (possibly duplicated) column name
    def _assign_first(df: pd.DataFrame, col: str, values: pd.Series) -> None:
        loc = df.columns.get_loc(col)
        if isinstance(loc, slice):
            idx = loc.start
        elif isinstance(loc, (list, tuple)):
            idx = loc[0]
        else:
            idx = int(loc)
        df.iloc[:, idx] = values
    # Fill base columns from *_wx counterparts if base is null
    for c in wcols:
        wxc = f"{c}_wx"
        if c in r.columns and wxc in r.columns:
            base_s = _as_series(r, c)
            wx_s = _as_series(r, wxc)
            filled = base_s.where(base_s.notna(), wx_s)
            _assign_first(r, c, filled)
            # drop the temp column (safe if duplicated names are represented once)
            if wxc in r.columns:
                r.drop(columns=[wxc], inplace=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path, help="Path to predictions_full.csv to enrich")
    ap.add_argument("--weather", type=Path, default=Path("data/processed/weather_games.parquet"), help="Weather parquet produced by src.data.weather")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if not args.weather.exists():
        print(f"[enrich] Weather file not found: {args.weather}")
        return
    w = pd.read_parquet(args.weather)

    # Normalize key dtypes
    for k in ("game_id", "game_uid", "game_uid_relaxed"):
        if k in df.columns:
            df[k] = df[k].astype(str)
        if k in w.columns:
            w[k] = w[k].astype(str)

    # Merge priority: game_uid -> game_uid_relaxed -> (season, week, home_team, away_team)
    before_hit = df.filter(regex=r"^weather_|^(kickoff|roof|venue_lat|venue_lon|roof_is_dome)$").notna().any(axis=1)

    if "game_uid" in df.columns and "game_uid" in w.columns:
        df = _fill_left(df, w, ["game_uid"])
    if "game_uid_relaxed" in df.columns and "game_uid_relaxed" in w.columns:
        df = _fill_left(df, w.rename(columns={"game_uid_relaxed": "guid_rel"}).rename(columns={"guid_rel": "game_uid_relaxed"}), ["game_uid_relaxed"])

    # Fallback on season/week/home/away if present
    keys = [c for c in ("season", "week", "home_team", "away_team") if c in df.columns and c in w.columns]
    if len(keys) == 4:
        df = _fill_left(df, w, keys)

    after_hit = df.filter(regex=r"^weather_|^(kickoff|roof|venue_lat|venue_lon|roof_is_dome)$").notna().any(axis=1)
    added = int((after_hit & ~before_hit).sum())
    print(f"[enrich] Filled weather/context for {added} additional rows")

    # Write back (in-place overwrite)
    df.to_csv(args.csv, index=False)
    print(f"[enrich] Updated -> {args.csv}")


if __name__ == "__main__":
    main()
