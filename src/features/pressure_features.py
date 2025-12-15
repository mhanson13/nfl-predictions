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

"""Pressure and sack-rate metrics computed from dropback-level play data."""

from __future__ import annotations

from typing import Callable, List

import numpy as np
import pandas as pd


def build_pressure_features(
    pbp: pd.DataFrame,
    seasons: List[int],
    norm_team: Callable[[str, str | None], str],
) -> pd.DataFrame:
    """
    Produce team-week pressure allowed/generated rates plus rolling averages.

    Requires dropback flags (`pass_attempt`, `qb_scramble`, `sack`) and maps
    both offense and defense to a common team key for later merges.
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

    for col in ("posteam", "defteam"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.upper().str.strip().apply(lambda v: norm_team(v, None))
    df = df.dropna(subset=["posteam", "defteam"])
    if df.empty:
        return pd.DataFrame()

    dropback = pd.Series(0, index=df.index, dtype="int64")
    for col in ("pass_attempt", "qb_scramble", "sack"):
        if col in df.columns:
            dropback = dropback | df[col].fillna(0).astype(int)

    qb_hit_cols = [c for c in df.columns if "qb_hit" in c.lower()]
    qb_hits_flag = pd.Series(0, index=df.index, dtype="int64")
    for col in qb_hit_cols:
        qb_hits_flag = qb_hits_flag | df[col].notna().astype(int)
    if "qb_hit" in df.columns:
        qb_hits_flag = qb_hits_flag | df["qb_hit"].fillna(0).astype(int)

    df["dropback"] = dropback
    df["pressure_event"] = qb_hits_flag | df.get("sack", 0).fillna(0).astype(int)
    df["sack_event"] = df.get("sack", 0).fillna(0).astype(int)
    df = df[df["dropback"] > 0]
    if df.empty:
        return pd.DataFrame()

    off = (
        df.groupby(["season", "week", "posteam"], as_index=False)
        .agg(
            dropbacks=("dropback", "sum"),
            pressures=("pressure_event", "sum"),
            sacks=("sack_event", "sum"),
        )
        .rename(columns={"posteam": "team"})
    )
    off["pressures_allowed_per_db"] = off["pressures"] / off["dropbacks"].replace(0, np.nan)
    off["sack_rate_allowed"] = off["sacks"] / off["dropbacks"].replace(0, np.nan)

    defense = (
        df.groupby(["season", "week", "defteam"], as_index=False)
        .agg(
            dropbacks=("dropback", "sum"),
            pressures_generated=("pressure_event", "sum"),
            sacks_generated=("sack_event", "sum"),
        )
        .rename(columns={"defteam": "team"})
    )
    defense["pressures_generated_per_db"] = defense["pressures_generated"] / defense["dropbacks"].replace(0, np.nan)
    defense["sack_rate_generated"] = defense["sacks_generated"] / defense["dropbacks"].replace(0, np.nan)

    merged = off.merge(defense, on=["season", "week", "team"], how="outer")
    merged = merged.sort_values(["team", "season", "week"]).reset_index(drop=True)
    group = merged.groupby("team", group_keys=False)
    for col in [
        "pressures_allowed_per_db",
        "sack_rate_allowed",
        "pressures_generated_per_db",
        "sack_rate_generated",
    ]:
        merged[f"{col}_rolling3"] = group[col].transform(
            lambda s: s.shift(1).rolling(window=3, min_periods=1).mean()
        )

    cols = [
        "season",
        "week",
        "team",
        "pressures_allowed_per_db",
        "sack_rate_allowed",
        "pressures_allowed_per_db_rolling3",
        "sack_rate_allowed_rolling3",
        "pressures_generated_per_db",
        "sack_rate_generated",
        "pressures_generated_per_db_rolling3",
        "sack_rate_generated_rolling3",
    ]
    features = merged[cols].copy()
    features["season"] = features["season"].astype("Int64")
    features["week"] = features["week"].astype("Int64")
    for col in cols[3:]:
        features[col] = features[col].astype(float)
    return features
