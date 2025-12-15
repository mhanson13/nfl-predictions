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

"""Quarterback availability / injury deltas derived from schedule, roster, and injury feeds."""

from __future__ import annotations

from typing import Callable, List, Optional

import pandas as pd


SEVERITY_WEIGHTS = {
    "out": 2.0,
    "injured reserve": 2.0,
    "doubt": 1.5,
    "question": 1.0,
    "limited": 0.5,
    "did not practice": 0.7,
    "dnp": 0.7,
}


def _normalize_name(value: Optional[str]) -> Optional[str]:
    """Upper-case and trim player names so roster / injury sheets can be aligned."""
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned.upper() if cleaned else None


def _score_status(status_text: str) -> float:
    """Map arbitrary status/practice blurbs to a numeric severity weight."""
    if not isinstance(status_text, str):
        return 0.0
    text = status_text.lower()
    for token, weight in SEVERITY_WEIGHTS.items():
        if token in text:
            return weight
    return 0.0


def build_qb_health_features(
    sched_df: pd.DataFrame,
    injuries_df: pd.DataFrame,
    roster_df: pd.DataFrame,
    seasons: List[int],
    norm_team: Callable[[str, Optional[str]], str],
) -> pd.DataFrame:
    """
    Build QB injury availability features for every team-week in the provided schedule window.

    Parameters
    ----------
    sched_df
        Season schedule with per-game QB identifiers (home_* / away_* columns are expected).
    injuries_df
        Weekly injury report rows; only QB rows are consumed if position is provided.
    roster_df
        Roster reference used to backfill QB identifiers when the schedule does not contain ids.
    seasons
        Inclusive list of seasons to keep; data outside of this list is ignored.
    norm_team
        Callable used to normalize team abbreviations (shared with the broader pipeline).

    Returns
    -------
    pd.DataFrame
        One row per team-week containing qb_status_flag, qb_status_delta_rolling3, qb_missed_last_game,
        and qb_games_started_rolling5. The returned frame is detached from the inputs.
    """
    if sched_df is None or sched_df.empty:
        return pd.DataFrame()

    sched = sched_df.copy()
    required = {"season", "week", "home_team", "away_team"}
    if not required.issubset(sched.columns):
        return pd.DataFrame()

    sched["season"] = pd.to_numeric(sched["season"], errors="coerce").astype("Int64")
    sched["week"] = pd.to_numeric(sched["week"], errors="coerce").astype("Int64")
    sched = sched.dropna(subset=["season", "week"])
    sched = sched[sched["season"].isin(seasons)]
    if sched.empty:
        return pd.DataFrame()

    team_rows: list[dict[str, object]] = []
    for _, row in sched.iterrows():
        season = int(row["season"])
        week = int(row["week"])
        for prefix, opponent_prefix in (("home", "away"), ("away", "home")):
            team = row.get(f"{prefix}_team")
            opponent = row.get(f"{opponent_prefix}_team")
            if not isinstance(team, str) or not isinstance(opponent, str):
                continue
            team_norm = norm_team(team, None)
            opp_norm = norm_team(opponent, None)
            team_rows.append(
                {
                    "season": season,
                    "week": week,
                    "team": team_norm,
                    "opponent": opp_norm,
                    "game_id": row.get("game_id"),
                    "qb_name": _normalize_name(row.get(f"{prefix}_qb_name")),
                    "qb_id": str(row.get(f"{prefix}_qb_id")).strip() if row.get(f"{prefix}_qb_id") else None,
                }
            )
    if not team_rows:
        return pd.DataFrame()

    team_games = pd.DataFrame(team_rows)
    team_games = team_games.dropna(subset=["team"]).reset_index(drop=True)

    # Prepare injury sheet filtered to QBs.
    qb_injuries = injuries_df.copy() if injuries_df is not None else pd.DataFrame()
    if qb_injuries.empty:
        qb_injuries = pd.DataFrame(columns=["season", "week", "team", "player_name_clean", "severity"])
    else:
        qb_injuries["season"] = pd.to_numeric(qb_injuries.get("season"), errors="coerce")
        qb_injuries["week"] = pd.to_numeric(qb_injuries.get("week"), errors="coerce")
        qb_injuries = qb_injuries.dropna(subset=["season", "week"])
        qb_injuries = qb_injuries[qb_injuries["season"].isin(seasons)]
        pos_col = next((c for c in qb_injuries.columns if c.lower() in {"position", "pos"}), None)
        if pos_col:
            qb_injuries = qb_injuries[qb_injuries[pos_col].astype(str).str.upper().str.contains("QB", na=False)]
        team_col = next((c for c in qb_injuries.columns if "team" in c.lower()), None)
        if team_col:
            qb_injuries["team"] = qb_injuries[team_col].astype(str).str.upper().str.strip().apply(lambda v: norm_team(v, None))
        else:
            qb_injuries["team"] = pd.NA
        qb_injuries["player_name_clean"] = qb_injuries.get("player_name") or qb_injuries.get("full_name")
        qb_injuries["player_name_clean"] = qb_injuries["player_name_clean"].apply(_normalize_name)
        status_col = next((c for c in qb_injuries.columns if "status" in c.lower()), None)
        practice_col = next((c for c in qb_injuries.columns if "practice" in c.lower()), None)
        if status_col is None and practice_col is None:
            qb_injuries["severity"] = 0.0
        else:
            status_text = qb_injuries[status_col].astype(str) if status_col else ""
            practice_text = qb_injuries[practice_col].astype(str) if practice_col else ""
            qb_injuries["severity"] = [
                max(_score_status(a), _score_status(b))
                for a, b in zip(status_text, practice_text)
            ]
        qb_injuries = qb_injuries[
            ["season", "week", "team", "player_name_clean", "severity", "player_id"]
        ].rename(columns={"player_id": "inj_player_id"})

    roster_lookup = pd.DataFrame()
    if roster_df is not None and not roster_df.empty:
        roster_lookup = roster_df.copy()
        roster_lookup["player_name_clean"] = roster_lookup.get("full_name", roster_lookup.get("player_name")).apply(
            _normalize_name
        )
        roster_lookup["gsis_id"] = roster_lookup.get("gsis_id", roster_lookup.get("player_id"))
        roster_lookup = roster_lookup[
            ["player_name_clean", "gsis_id", "team"]
        ].rename(columns={"gsis_id": "roster_player_id"})
        roster_lookup["team"] = roster_lookup["team"].astype(str).str.upper().str.strip().apply(lambda v: norm_team(v, None))

    # Combine schedule rows with roster lookups so we always have a QB identifier per team-week.
    merged = team_games.copy()
    merged["qb_name_clean"] = merged["qb_name"]

    if not roster_lookup.empty:
        merged = merged.merge(
            roster_lookup,
            left_on=["qb_name_clean", "team"],
            right_on=["player_name_clean", "team"],
            how="left",
        )
        merged["qb_player_id"] = merged.apply(
            lambda r: r["qb_id"] or r["roster_player_id"],
            axis=1,
        )
    else:
        merged["qb_player_id"] = merged["qb_id"]

    if not qb_injuries.empty:
        inj_join_cols = ["season", "week"]
        merged = merged.merge(
            qb_injuries,
            left_on=inj_join_cols + ["qb_player_id"],
            right_on=inj_join_cols + ["inj_player_id"],
            how="left",
        )
        missing_mask = merged["severity"].isna()
        merged.loc[missing_mask, :] = merged.loc[missing_mask, :].merge(
            qb_injuries.drop(columns=["inj_player_id"]),
            left_on=inj_join_cols + ["team", "qb_name_clean"],
            right_on=inj_join_cols + ["team", "player_name_clean"],
            how="left",
            suffixes=("", "_namefallback"),
        )
        if "severity_namefallback" in merged.columns:
            merged["severity"] = merged["severity"].fillna(merged["severity_namefallback"])
        merged["severity"] = merged["severity"].fillna(0.0)
    else:
        merged["severity"] = 0.0

    merged = merged.sort_values(["team", "season", "week"]).reset_index(drop=True)
    group = merged.groupby("team", group_keys=False)
    merged["qb_status_flag"] = merged["severity"]
    merged["qb_status_prev"] = group["qb_status_flag"].shift(1)
    merged["qb_status_delta_rolling3"] = (
        group["qb_status_flag"]
        .transform(lambda s: s.shift(1).rolling(window=3, min_periods=1).mean())
    )
    merged["qb_status_delta_rolling3"] = merged["qb_status_flag"] - merged["qb_status_delta_rolling3"]
    merged["qb_status_delta_rolling3"] = merged["qb_status_delta_rolling3"].fillna(0.0)
    merged["qb_missed_last_game"] = (merged["qb_status_prev"].fillna(0) >= 1.0).astype("Int64")
    merged["qb_games_started_rolling5"] = (
        group["qb_status_flag"]
        .transform(lambda s: s.apply(lambda v: 1 if pd.notna(v) else 0).rolling(window=5, min_periods=1).sum())
    )

    feature_cols = [
        "season",
        "week",
        "team",
        "qb_status_flag",
        "qb_status_delta_rolling3",
        "qb_missed_last_game",
        "qb_games_started_rolling5",
    ]
    features = merged[feature_cols].copy()
    features["season"] = features["season"].astype("Int64")
    features["week"] = features["week"].astype("Int64")
    features["qb_status_flag"] = features["qb_status_flag"].astype(float)
    features["qb_status_delta_rolling3"] = features["qb_status_delta_rolling3"].astype(float)
    features["qb_games_started_rolling5"] = features["qb_games_started_rolling5"].astype(float)
    features["qb_missed_last_game"] = features["qb_missed_last_game"].fillna(0).astype("Int64")
    return features
