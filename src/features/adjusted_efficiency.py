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

"""Opponent-adjusted offensive/defensive EPA features derived from PBP + schedule."""

from __future__ import annotations

from typing import Callable, List

import numpy as np
import pandas as pd


def _build_team_game_index(sched: pd.DataFrame, norm_team: Callable[[str, str | None], str]) -> pd.DataFrame:
    """Expand schedule rows into a team/opponent index for later joins."""
    rows: list[dict[str, object]] = []
    for _, row in sched.iterrows():
        season = row["season"]
        week = row["week"]
        for prefix, opp_prefix in (("home", "away"), ("away", "home")):
            team = row.get(f"{prefix}_team")
            opponent = row.get(f"{opp_prefix}_team")
            if not isinstance(team, str) or not isinstance(opponent, str):
                continue
            rows.append(
                {
                    "season": season,
                    "week": week,
                    "team": norm_team(team, None),
                    "opponent": norm_team(opponent, None),
                    "game_id": row.get("game_id"),
                }
            )
    return pd.DataFrame(rows)


def build_adjusted_efficiency_features(
    pbp: pd.DataFrame,
    sched_df: pd.DataFrame,
    seasons: List[int],
    norm_team: Callable[[str, str | None], str],
) -> pd.DataFrame:
    """
    Compute adjusted offensive/defensive efficiency deltas versus opponent averages.

    Joins play-level EPA outputs with schedule context so each team-week is
    normalized by how their opponent typically performs.
    """
    if pbp is None or pbp.empty or sched_df is None or sched_df.empty:
        return pd.DataFrame()

    sched = sched_df.copy()
    if not {"season", "week", "home_team", "away_team"}.issubset(sched.columns):
        return pd.DataFrame()
    sched["season"] = pd.to_numeric(sched["season"], errors="coerce")
    sched["week"] = pd.to_numeric(sched["week"], errors="coerce")
    sched = sched.dropna(subset=["season", "week"])
    sched = sched[sched["season"].isin(seasons)]
    if sched.empty:
        return pd.DataFrame()

    games = _build_team_game_index(sched, norm_team)
    if games.empty:
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

    for col in ("posteam", "defteam"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.upper().str.strip().apply(lambda v: norm_team(v, None))
    df["epa"] = pd.to_numeric(df.get("epa"), errors="coerce").fillna(0.0)
    df["play_mask"] = (
        (df.get("play_type").notna())
        | (df.get("pass_attempt", 0).fillna(0).astype(int) == 1)
        | (df.get("rush_attempt", 0).fillna(0).astype(int) == 1)
    ).astype(int)
    df = df[df["play_mask"] > 0]
    if df.empty:
        return pd.DataFrame()

    off_stats = (
        df.groupby(["season", "week", "posteam"], as_index=False)
        .agg(
            off_plays=("play_mask", "sum"),
            off_epa=("epa", "sum"),
        )
        .rename(columns={"posteam": "team"})
    )
    off_stats["off_eff_raw"] = off_stats["off_epa"] / off_stats["off_plays"].replace(0, np.nan)

    def_stats = (
        df.groupby(["season", "week", "defteam"], as_index=False)
        .agg(
            def_plays=("play_mask", "sum"),
            def_epa=("epa", "sum"),
        )
        .rename(columns={"defteam": "team"})
    )
    def_stats["def_eff_raw"] = def_stats["def_epa"] / def_stats["def_plays"].replace(0, np.nan)

    team_stats = games.merge(off_stats[["season", "week", "team", "off_eff_raw"]], on=["season", "week", "team"], how="left")
    team_stats = team_stats.merge(def_stats[["season", "week", "team", "def_eff_raw"]], on=["season", "week", "team"], how="left")

    def rolling_mean(series: pd.Series, window: int) -> pd.Series:
        return series.transform(lambda s: s.shift(1).rolling(window=window, min_periods=1).mean())

    team_stats = team_stats.sort_values(["team", "season", "week"]).reset_index(drop=True)
    grouped = team_stats.groupby("team", group_keys=False)
    team_stats["off_eff_prev"] = grouped["off_eff_raw"].transform(lambda s: s.shift(1))
    team_stats["def_eff_prev"] = grouped["def_eff_raw"].transform(lambda s: s.shift(1))

    team_stats["off_eff_rolling3"] = grouped["off_eff_raw"].transform(
        lambda s: s.shift(1).rolling(window=3, min_periods=1).mean()
    )
    team_stats["def_eff_rolling3"] = grouped["def_eff_raw"].transform(
        lambda s: s.shift(1).rolling(window=3, min_periods=1).mean()
    )

    # Opponent prior form
    opp_def = team_stats[["season", "week", "team", "def_eff_prev", "def_eff_rolling3"]].rename(
        columns={
            "team": "opponent",
            "def_eff_prev": "opp_avg_allowed_eff",
            "def_eff_rolling3": "opp_def_rolling3",
        }
    )
    opp_off = team_stats[["season", "week", "team", "off_eff_prev", "off_eff_rolling3"]].rename(
        columns={
            "team": "opponent",
            "off_eff_prev": "opp_avg_produced_eff",
            "off_eff_rolling3": "opp_off_rolling3",
        }
    )

    team_stats = team_stats.merge(
        opp_def[["season", "week", "opponent", "opp_avg_allowed_eff", "opp_def_rolling3"]],
        on=["season", "week", "opponent"],
        how="left",
    )
    team_stats = team_stats.merge(
        opp_off[["season", "week", "opponent", "opp_avg_produced_eff", "opp_off_rolling3"]],
        on=["season", "week", "opponent"],
        how="left",
    )

    team_stats["off_adj_eff_raw"] = team_stats["off_eff_raw"] - team_stats["opp_avg_allowed_eff"]
    team_stats["def_adj_eff_raw"] = team_stats["def_eff_raw"] - team_stats["opp_avg_produced_eff"]

    grouped_final = team_stats.groupby("team", group_keys=False)
    team_stats["off_adj_eff_rolling3"] = grouped_final["off_adj_eff_raw"].transform(
        lambda s: s.shift(1).rolling(window=3, min_periods=1).mean()
    )
    team_stats["def_adj_eff_rolling3"] = grouped_final["def_adj_eff_raw"].transform(
        lambda s: s.shift(1).rolling(window=3, min_periods=1).mean()
    )

    cols = [
        "season",
        "week",
        "team",
        "off_adj_eff_raw",
        "off_adj_eff_rolling3",
        "def_adj_eff_raw",
        "def_adj_eff_rolling3",
    ]
    features = team_stats[cols].copy()
    features["season"] = features["season"].astype("Int64")
    features["week"] = features["week"].astype("Int64")
    for col in cols[3:]:
        features[col] = features[col].astype(float)
    return features
