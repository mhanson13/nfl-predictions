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

"""Passing EPA rolling-rate features derived from play-by-play dropbacks."""

from __future__ import annotations

from typing import Callable, List

import numpy as np
import pandas as pd


def build_passing_epa_features(
    pbp: pd.DataFrame,
    seasons: List[int],
    norm_team: Callable[[str, str | None], str],
) -> pd.DataFrame:
    """
    Compute per-team EPA per dropback and rolling differentials.

    Only prior games are considered for each week to prevent leakage. Returns
    season/week keyed features ready to merge into matchup_features.
    """
    if pbp is None or pbp.empty:
        return pd.DataFrame()

    df = pbp.copy()
    if "season" not in df.columns or "week" not in df.columns:
        return pd.DataFrame()
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df["week"] = pd.to_numeric(df["week"], errors="coerce")
    df = df.dropna(subset=["season", "week"])
    df = df[df["season"].isin(seasons)]
    if df.empty:
        return pd.DataFrame()

    dropback_flag = pd.Series(0, index=df.index, dtype="int64")
    for col in ("pass_attempt", "qb_scramble", "sack"):
        if col in df.columns:
            dropback_flag = dropback_flag | df[col].fillna(0).astype(int)
    df["dropback"] = dropback_flag
    df = df[df["dropback"] > 0]
    if df.empty:
        return pd.DataFrame()

    df["team"] = df["posteam"].astype(str).str.upper().str.strip().apply(lambda v: norm_team(v, None))
    df = df.dropna(subset=["team"])
    df["epa"] = pd.to_numeric(df.get("epa"), errors="coerce").fillna(0.0)

    grouped = (
        df.groupby(["season", "week", "team"], as_index=False)
        .agg(
            epa_dropbacks=("epa", "sum"),
            dropbacks=("dropback", "sum"),
        )
    )
    grouped = grouped[grouped["dropbacks"] > 0]
    if grouped.empty:
        return pd.DataFrame()
    grouped["pass_epa_per_db"] = grouped["epa_dropbacks"] / grouped["dropbacks"]

    grouped = grouped.sort_values(["team", "season", "week"]).reset_index(drop=True)
    team_groups = grouped.groupby("team", group_keys=False)
    grouped["pass_epa_per_db_rolling3"] = team_groups["pass_epa_per_db"].transform(
        lambda s: s.shift(1).rolling(window=3, min_periods=1).mean()
    )
    grouped["pass_epa_per_db_rolling5"] = team_groups["pass_epa_per_db"].transform(
        lambda s: s.shift(1).rolling(window=5, min_periods=1).mean()
    )

    grouped["league_roll3_mean"] = grouped.groupby(["season", "week"])["pass_epa_per_db_rolling3"].transform(
        "mean"
    )
    grouped["pass_epa_per_db_rolling3_diff_lg"] = (
        grouped["pass_epa_per_db_rolling3"] - grouped["league_roll3_mean"]
    )

    cols = [
        "season",
        "week",
        "team",
        "pass_epa_per_db",
        "pass_epa_per_db_rolling3",
        "pass_epa_per_db_rolling5",
        "pass_epa_per_db_rolling3_diff_lg",
    ]
    features = grouped[cols].copy()
    for col in cols[3:]:
        features[col] = features[col].astype(float)
    features["season"] = features["season"].astype("Int64")
    features["week"] = features["week"].astype("Int64")
    return features
