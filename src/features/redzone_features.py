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

"""Red-zone trip and touchdown features for offense/defense rolling windows."""

from __future__ import annotations

from typing import Callable, List

import numpy as np
import pandas as pd


def _normalize_team(series: pd.Series, norm: Callable[[str, str | None], str]) -> pd.Series:
    return series.astype(str).str.upper().str.strip().apply(lambda v: norm(v, None))


def build_redzone_features(
    pbp: pd.DataFrame,
    seasons: List[int],
    norm_team: Callable[[str, str | None], str],
) -> pd.DataFrame:
    """
    Derive red-zone efficiency metrics (for/against) at the team-week level.

    Uses drive-level entry points to count trips and TDs, then computes 3-game
    rolling averages to keep modeling inputs leakage-safe.
    """
    if pbp is None or pbp.empty:
        return pd.DataFrame()

    if "season" not in pbp.columns or "week" not in pbp.columns:
        return pd.DataFrame()
    df = pbp.copy()
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df["week"] = pd.to_numeric(df["week"], errors="coerce")
    df = df.dropna(subset=["season", "week"])
    df = df[df["season"].isin(seasons)]
    if df.empty:
        return pd.DataFrame()

    if "yardline_100" not in df.columns:
        return pd.DataFrame()
    df["yardline_100"] = pd.to_numeric(df["yardline_100"], errors="coerce")
    df = df[df["yardline_100"] <= 20]
    df = df.dropna(subset=["posteam", "defteam"])
    if df.empty:
        return pd.DataFrame()

    df["team"] = _normalize_team(df["posteam"], norm_team)
    df["def_team"] = _normalize_team(df["defteam"], norm_team)
    df["drive_id"] = df["game_id"].astype(str) + "_" + df["drive"].astype(str)
    df["is_td"] = df.get("touchdown", 0).fillna(0).astype(int)

    first_entries = df.sort_values(["game_id", "drive", "play_id"]).drop_duplicates(
        subset=["game_id", "drive", "team"]
    )

    trips_for = (
        first_entries.groupby(["season", "week", "team"], as_index=False)
        .size()
        .rename(columns={"size": "rz_trips_for"})
    )
    tds_for = (
        df[df["is_td"] == 1]
        .groupby(["season", "week", "team"], as_index=False)
        .size()
        .rename(columns={"size": "rz_tds_for"})
    )

    trips_against = (
        first_entries.groupby(["season", "week", "def_team"], as_index=False)
        .size()
        .rename(columns={"def_team": "team", "size": "rz_trips_against"})
    )
    tds_against = (
        df[df["is_td"] == 1]
        .groupby(["season", "week", "def_team"], as_index=False)
        .size()
        .rename(columns={"def_team": "team", "size": "rz_tds_against"})
    )

    merged = trips_for.merge(tds_for, on=["season", "week", "team"], how="left")
    merged = merged.merge(trips_against, on=["season", "week", "team"], how="left")
    merged = merged.merge(tds_against, on=["season", "week", "team"], how="left")
    merged = merged.fillna(0)

    merged["rz_trips_for"] = merged["rz_trips_for"].astype(int)
    merged["rz_tds_for"] = merged["rz_tds_for"].astype(int)
    merged["rz_trips_against"] = merged["rz_trips_against"].astype(int)
    merged["rz_tds_against"] = merged["rz_tds_against"].astype(int)

    merged["rz_td_rate_for"] = np.where(
        merged["rz_trips_for"] > 0,
        merged["rz_tds_for"] / merged["rz_trips_for"],
        np.nan,
    )
    merged["rz_td_rate_against"] = np.where(
        merged["rz_trips_against"] > 0,
        merged["rz_tds_against"] / merged["rz_trips_against"],
        np.nan,
    )

    merged = merged.sort_values(["team", "season", "week"]).reset_index(drop=True)
    grouped = merged.groupby("team", group_keys=False)

    merged["rz_td_rate_for_rolling3"] = grouped["rz_td_rate_for"].transform(
        lambda s: s.shift(1).rolling(window=3, min_periods=1).mean()
    )
    merged["rz_td_rate_against_rolling3"] = grouped["rz_td_rate_against"].transform(
        lambda s: s.shift(1).rolling(window=3, min_periods=1).mean()
    )
    merged["rz_trips_for_rolling3"] = grouped["rz_trips_for"].transform(
        lambda s: s.shift(1).rolling(window=3, min_periods=1).mean()
    )
    merged["rz_trips_against_rolling3"] = grouped["rz_trips_against"].transform(
        lambda s: s.shift(1).rolling(window=3, min_periods=1).mean()
    )

    cols = [
        "season",
        "week",
        "team",
        "rz_trips_for",
        "rz_tds_for",
        "rz_trips_against",
        "rz_tds_against",
        "rz_td_rate_for",
        "rz_td_rate_against",
        "rz_td_rate_for_rolling3",
        "rz_td_rate_against_rolling3",
        "rz_trips_for_rolling3",
        "rz_trips_against_rolling3",
    ]
    features = merged[cols].copy()
    features["season"] = features["season"].astype("Int64")
    features["week"] = features["week"].astype("Int64")
    return features
