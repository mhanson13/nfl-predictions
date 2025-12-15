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
from typing import List, Any
import pandas as pd
from tqdm import tqdm
from sportsdataverse import nfl as sdv_nfl  # type: ignore
from src.utils.io import RAW_DIR, write_df, read_df


def _to_pandas(df: Any) -> pd.DataFrame:
    """
    Convert a variety of possible return types into a pandas.DataFrame.
    Handles:
      - pandas.DataFrame (returned as-is)
      - polars.DataFrame (to_pandas)
      - dict / list[dict] via json_normalize / DataFrame
      - list[pd.DataFrame] concatenated
    Returns an empty DataFrame if conversion is not possible.
    """
    # pandas already?
    if isinstance(df, pd.DataFrame):
        return df

    # polars?
    try:
        import polars as pl  # type: ignore
        if isinstance(df, pl.DataFrame):
            return df.to_pandas()
    except Exception:
        pass

    # dict -> normalize
    if isinstance(df, dict):
        try:
            return pd.json_normalize(df)
        except Exception:
            try:
                return pd.DataFrame([df])
            except Exception:
                return pd.DataFrame()

    # list -> try list of dicts or list of DataFrames
    if isinstance(df, list):
        if not df:
            return pd.DataFrame()
        if all(isinstance(x, dict) for x in df):
            try:
                return pd.DataFrame(df)
            except Exception:
                return pd.json_normalize(df)
        if all(isinstance(x, pd.DataFrame) for x in df):
            try:
                return pd.concat(df, ignore_index=True)
            except Exception:
                # fall back to concatenating as-is (shouldn't happen often)
                out = pd.DataFrame()
                for x in df:
                    try:
                        out = pd.concat([out, x], ignore_index=True)
                    except Exception:
                        continue
                return out
        # mixed list: try to coerce dicts → DataFrame then concat
        frames = []
        for x in df:
            frames.append(_to_pandas(x))
        frames = [f for f in frames if isinstance(f, pd.DataFrame) and not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # unknown type
    return pd.DataFrame()


def get_schedules(seasons: List[int], force: bool = False) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for season in seasons:
        cache_path = RAW_DIR / f"espn_schedule_{season}.parquet"
        if cache_path.exists() and not force:
            try:
                frames.append(read_df(cache_path))
                continue
            except Exception:
                pass
        raw = sdv_nfl.espn_nfl_schedule(dates=season, return_as_pandas=True, limit=900)
        df = _to_pandas(raw)
        if df is None or df.empty:
            continue
        df["season"] = season
        # write per-season cache file
        write_df(df, cache_path)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def get_pbp(seasons: List[int], force: bool = False) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for season in seasons:
        cache_path = RAW_DIR / f"espn_pbp_{season}.parquet"
        if cache_path.exists() and not force:
            try:
                frames.append(read_df(cache_path))
                continue
            except Exception:
                pass

        raw_sched = sdv_nfl.espn_nfl_schedule(dates=season, return_as_pandas=True, limit=900)
        sched = _to_pandas(raw_sched)
        if sched is None or sched.empty:
            continue

        ids_col = (
            "id"
            if "id" in sched.columns
            else ("game_id" if "game_id" in sched.columns else None)
        )
        if not ids_col:
            for cand in ["gameId", "event_id", "eventId"]:
                if cand in sched.columns:
                    ids_col = cand
                    break
        if not ids_col:
            continue

        game_ids = sched[ids_col].astype(str).tolist()
        season_frames: List[pd.DataFrame] = []
        for gid in tqdm(game_ids, desc=f"PBP season {season}"):
            try:
                raw_pbp = sdv_nfl.NFLPlayProcess(gameId=int(gid)).espn_nfl_pbp()
                pbp = _to_pandas(raw_pbp)
                if pbp is None or pbp.empty:
                    continue
                pbp["season"] = season
                pbp["game_id"] = gid
                season_frames.append(pbp)
            except Exception:
                continue
        if season_frames:
            season_df = pd.concat(season_frames, ignore_index=True)
            write_df(season_df, cache_path)
            frames.append(season_df)

    # Only concat valid DataFrames
    frames = [f for f in frames if isinstance(f, pd.DataFrame) and not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, nargs="+", required=True, help="e.g., 2024 2025")
    ap.add_argument("--pbp", action="store_true", help="Fetch ESPN PBP via sportsdataverse")
    ap.add_argument("--schedules", action="store_true", help="Fetch schedules")
    ap.add_argument("--force", action="store_true", help="Force re-download even if cached season files exist")
    args = ap.parse_args()

    seasons = [int(s) for s in args.season]

    if args.schedules:
        sched = get_schedules(seasons, force=args.force)
        if not sched.empty:
            write_df(sched, RAW_DIR / "espn_schedule.parquet")

    if args.pbp:
        pbp = get_pbp(seasons, force=args.force)
        if not pbp.empty:
            write_df(pbp, RAW_DIR / "espn_pbp.parquet")


if __name__ == "__main__":
    main()
