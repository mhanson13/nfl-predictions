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

"""Drive-level EPA momentum features derived from play-by-play data."""

from __future__ import annotations

from typing import Callable, List

import numpy as np
import pandas as pd


def build_drive_epa_features(
    pbp: pd.DataFrame,
    seasons: List[int],
    norm_team: Callable[[str, str | None], str],
) -> pd.DataFrame:
    """Compute per-team drive-level EPA rolling features.

    Aggregates EPA at the drive level (one row per team-game-drive), then
    rolls with ``.shift(1)`` before the rolling window to prevent leakage.
    Returns a season/week-keyed DataFrame ready to merge into matchup_features.

    Features produced (all 3-game rolling averages, leakage-free):
    - ``drive_epa_mean_rolling3``        — average EPA per drive
    - ``drive_epa_first_drive_rolling3`` — EPA on the team's 1st drive of the game
    - ``drive_epa_q4_rolling3``          — EPA on drives that started in Q4
    - ``drive_epa_red_zone_rolling3``    — EPA on drives that entered the red zone
    - ``drive_completion_rate_rolling3`` — fraction of drives that ended in a score
    - ``drives_per_game_rolling3``       — total drives per game (pace proxy)

    A league-average normalised differential is also emitted for
    ``drive_epa_mean`` (mirrors the pattern in ``passing_epa_features.py``).
    """
    if pbp is None or pbp.empty:
        return pd.DataFrame()

    required = {"game_id", "posteam", "season", "week", "fixed_drive", "epa"}
    if not required.issubset(pbp.columns):
        return pd.DataFrame()

    df = pbp.copy()
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df["week"] = pd.to_numeric(df["week"], errors="coerce")
    df = df.dropna(subset=["season", "week", "posteam", "fixed_drive"])
    df = df[df["season"].isin(seasons)]
    if df.empty:
        return pd.DataFrame()

    df["team"] = df["posteam"].astype(str).str.upper().str.strip().apply(lambda v: norm_team(v, None))
    df = df.dropna(subset=["team"])
    df["epa"] = pd.to_numeric(df.get("epa"), errors="coerce").fillna(0.0)
    df["fixed_drive"] = pd.to_numeric(df["fixed_drive"], errors="coerce")
    df = df.dropna(subset=["fixed_drive"])

    # Optional columns — coerce once; missing columns become NaN
    if "qtr" in df.columns:
        df["qtr"] = pd.to_numeric(df["qtr"], errors="coerce")
    else:
        df["qtr"] = np.nan

    if "drive_inside20" in df.columns:
        df["drive_inside20"] = pd.to_numeric(df["drive_inside20"], errors="coerce").fillna(0.0)
    else:
        df["drive_inside20"] = 0.0

    if "drive_ended_with_score" in df.columns:
        df["drive_ended_with_score"] = pd.to_numeric(df["drive_ended_with_score"], errors="coerce").fillna(0.0)
    else:
        df["drive_ended_with_score"] = 0.0

    # ── Drive-level aggregation ───────────────────────────────────────────────
    # One row per (season, week, team, fixed_drive)
    drive_agg = (
        df.groupby(["season", "week", "game_id", "team", "fixed_drive"], as_index=False)
        .agg(
            drive_epa=("epa", "sum"),
            drive_min_qtr=("qtr", "min"),       # quarter the drive started in
            drive_in_rz=("drive_inside20", "max"),
            drive_scored=("drive_ended_with_score", "max"),
        )
    )

    if drive_agg.empty:
        return pd.DataFrame()

    # Mark whether this is the team's 1st drive of the game
    drive_agg = drive_agg.sort_values(["game_id", "team", "fixed_drive"]).copy()
    drive_agg["drive_rank"] = drive_agg.groupby(["game_id", "team"]).cumcount() + 1

    # ── Game-level aggregation ────────────────────────────────────────────────
    # For each metric, compute the game-level summary before rolling.

    def _safe_mean(s: pd.Series) -> float:
        """Mean of s, NaN if empty."""
        return s.mean() if len(s) > 0 else np.nan

    game_groups = drive_agg.groupby(["season", "week", "game_id", "team"])

    game_df = game_groups.apply(
        lambda g: pd.Series({
            "drive_epa_mean": _safe_mean(g["drive_epa"]),
            "drive_epa_first_drive": _safe_mean(g.loc[g["drive_rank"] == 1, "drive_epa"]),
            "drive_epa_q4": _safe_mean(g.loc[g["drive_min_qtr"] == 4, "drive_epa"]),
            "drive_epa_red_zone": _safe_mean(g.loc[g["drive_in_rz"] == 1, "drive_epa"]),
            "drive_completion_rate": g["drive_scored"].mean(),
            "drives_per_game": float(len(g)),
        }),
        include_groups=False,
    ).reset_index()

    if game_df.empty:
        return pd.DataFrame()

    # ── Rolling features ──────────────────────────────────────────────────────
    # Sort chronologically and apply .shift(1) before .rolling(3) (leakage prevention).
    raw_cols = [
        "drive_epa_mean",
        "drive_epa_first_drive",
        "drive_epa_q4",
        "drive_epa_red_zone",
        "drive_completion_rate",
        "drives_per_game",
    ]

    game_df = game_df.sort_values(["team", "season", "week"]).reset_index(drop=True)
    team_groups = game_df.groupby("team", group_keys=False)

    for col in raw_cols:
        out_col = f"{col}_rolling3"
        game_df[out_col] = team_groups[col].transform(
            lambda s: s.shift(1).rolling(window=3, min_periods=1).mean()
        )

    # League-average normalised differential for drive_epa_mean (mirrors passing_epa pattern)
    game_df["league_drive_epa_mean_rolling3"] = game_df.groupby(
        ["season", "week"]
    )["drive_epa_mean_rolling3"].transform("mean")
    game_df["drive_epa_mean_rolling3_diff_lg"] = (
        game_df["drive_epa_mean_rolling3"] - game_df["league_drive_epa_mean_rolling3"]
    )

    # ── Deduplicate to one row per team-season-week (take last game in week) ──
    game_df = game_df.sort_values(["team", "season", "week", "game_id"])
    game_df = game_df.drop_duplicates(subset=["season", "week", "team"], keep="last")

    # ── Assemble output ───────────────────────────────────────────────────────
    rolling_cols = [f"{c}_rolling3" for c in raw_cols] + ["drive_epa_mean_rolling3_diff_lg"]
    keep_cols = ["season", "week", "team"] + raw_cols + rolling_cols

    features = game_df[keep_cols].copy()
    for col in raw_cols + rolling_cols:
        features[col] = features[col].astype(float)
    features["season"] = features["season"].astype("Int64")
    features["week"] = features["week"].astype("Int64")

    return features
