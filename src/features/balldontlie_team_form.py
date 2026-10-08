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

"""Leakage-safe weekly team form features from BallDontLie game stats."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd


RAW_FEATURE_COLUMNS = [
    "bdl_points_for",
    "bdl_points_allowed",
    "bdl_point_diff",
    "bdl_total_yards",
    "bdl_total_yards_allowed",
    "bdl_net_passing_yards",
    "bdl_net_passing_yards_allowed",
    "bdl_rushing_yards",
    "bdl_rushing_yards_allowed",
    "bdl_first_downs",
    "bdl_first_downs_allowed",
    "bdl_turnovers",
    "bdl_takeaways",
    "bdl_yards_per_play",
    "bdl_third_down_rate",
    "bdl_red_zone_score_rate",
    "bdl_sack_rate_allowed",
    "bdl_sack_rate_generated",
]


def _call_norm(norm_team: Callable[..., str], value: object, fallback: object = None) -> str | None:
    if value is None or pd.isna(value):
        return None
    try:
        normalized = norm_team(value, fallback)
    except TypeError:
        normalized = norm_team(value)
    except Exception:
        normalized = value
    if normalized is None or pd.isna(normalized):
        return None
    text = str(normalized).strip().upper()
    return text or None


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def _coalesce_column(frame: pd.DataFrame, candidates: Sequence[str]) -> pd.Series:
    for column in candidates:
        if column in frame.columns:
            return frame[column]
    return pd.Series(pd.NA, index=frame.index)


def _schedule_team_index(
    schedule: pd.DataFrame | None,
    seasons: Sequence[int],
    norm_team: Callable[..., str],
) -> pd.DataFrame:
    if schedule is None or schedule.empty:
        return pd.DataFrame(columns=["season", "week", "team"])

    home_col = "home_team" if "home_team" in schedule.columns else "home_abbreviation"
    away_col = "away_team" if "away_team" in schedule.columns else "away_abbreviation"
    required = {"season", "week", home_col, away_col}
    if not required.issubset(schedule.columns):
        return pd.DataFrame(columns=["season", "week", "team"])

    sched = schedule[list(required)].copy()
    sched["season"] = pd.to_numeric(sched["season"], errors="coerce")
    sched["week"] = pd.to_numeric(sched["week"], errors="coerce")
    sched = sched.dropna(subset=["season", "week"])
    sched["season"] = sched["season"].astype(int)
    sched["week"] = sched["week"].astype(int)
    if seasons:
        season_set = {int(season) for season in seasons}
        sched = sched[sched["season"].isin(season_set)]
    if sched.empty:
        return pd.DataFrame(columns=["season", "week", "team"])

    def _norm_schedule_team(value: object) -> str | None:
        team = _call_norm(norm_team, value)
        # nflverse uses LA for the Rams in modern schedules; BallDontLie uses LAR.
        return "LAR" if team == "LA" else team

    home = sched[["season", "week", home_col]].rename(columns={home_col: "team"})
    away = sched[["season", "week", away_col]].rename(columns={away_col: "team"})
    teams = pd.concat([home, away], ignore_index=True)
    teams["team"] = teams["team"].apply(_norm_schedule_team)
    teams = teams.dropna(subset=["team"])
    return teams.drop_duplicates(subset=["season", "week", "team"]).sort_values(["season", "team", "week"])


def _completed_game_stats(
    team_stats: pd.DataFrame,
    seasons: Sequence[int],
    norm_team: Callable[..., str],
) -> pd.DataFrame:
    if team_stats is None or team_stats.empty:
        return pd.DataFrame(columns=["season", "week", "team", *RAW_FEATURE_COLUMNS])

    required = {"game.id", "game.week", "team.abbreviation"}
    if not required.issubset(team_stats.columns):
        return pd.DataFrame(columns=["season", "week", "team", *RAW_FEATURE_COLUMNS])

    frame = team_stats.copy()
    frame["season"] = pd.to_numeric(
        _coalesce_column(frame, ["game.season", "season", "requested_season"]),
        errors="coerce",
    )
    frame["week"] = pd.to_numeric(frame["game.week"], errors="coerce")
    frame = frame.dropna(subset=["season", "week", "game.id", "team.abbreviation"])
    if frame.empty:
        return pd.DataFrame(columns=["season", "week", "team", *RAW_FEATURE_COLUMNS])
    frame["season"] = frame["season"].astype(int)
    frame["week"] = frame["week"].astype(int)
    if seasons:
        season_set = {int(season) for season in seasons}
        frame = frame[frame["season"].isin(season_set)]
    if frame.empty:
        return pd.DataFrame(columns=["season", "week", "team", *RAW_FEATURE_COLUMNS])

    status_state = frame.get("game.status_state")
    status = frame.get("game.status")
    if status_state is not None:
        final_mask = status_state.astype("string").str.lower().eq("final")
    elif status is not None:
        final_mask = status.astype("string").str.lower().eq("final")
    else:
        final_mask = pd.Series(True, index=frame.index)
    frame = frame[final_mask].copy()
    if frame.empty:
        return pd.DataFrame(columns=["season", "week", "team", *RAW_FEATURE_COLUMNS])

    frame["team"] = frame.apply(
        lambda row: _call_norm(norm_team, row.get("team.abbreviation"), row.get("team.full_name")),
        axis=1,
    )
    frame = frame.dropna(subset=["team"])
    frame["game_id"] = frame["game.id"].astype("string")

    home_away = frame.get("home_away", pd.Series(pd.NA, index=frame.index)).astype("string").str.lower()
    home_score = _num(frame, "game.home_team_score")
    visitor_score = _num(frame, "game.visitor_team_score")
    frame["bdl_points_for"] = np.where(home_away.eq("home"), home_score, visitor_score)
    frame["bdl_points_allowed"] = np.where(home_away.eq("home"), visitor_score, home_score)
    frame["bdl_point_diff"] = frame["bdl_points_for"] - frame["bdl_points_allowed"]

    frame["bdl_total_yards"] = _num(frame, "total_yards")
    frame["bdl_net_passing_yards"] = _num(frame, "net_passing_yards")
    frame["bdl_rushing_yards"] = _num(frame, "rushing_yards")
    frame["bdl_first_downs"] = _num(frame, "first_downs")
    frame["bdl_turnovers"] = _num(frame, "turnovers")
    frame["bdl_yards_per_play"] = _num(frame, "yards_per_play")
    if frame["bdl_yards_per_play"].isna().all():
        frame["bdl_yards_per_play"] = _rate(_num(frame, "total_yards"), _num(frame, "total_offensive_plays"))
    frame["bdl_third_down_rate"] = _rate(_num(frame, "third_down_conversions"), _num(frame, "third_down_attempts"))
    frame["bdl_red_zone_score_rate"] = _rate(_num(frame, "red_zone_scores"), _num(frame, "red_zone_attempts"))
    dropbacks = _num(frame, "passing_attempts").add(_num(frame, "sacks"), fill_value=0)
    frame["bdl_sack_rate_allowed"] = _rate(_num(frame, "sacks"), dropbacks)

    base_cols = [
        "season",
        "week",
        "game_id",
        "team",
        "bdl_points_for",
        "bdl_points_allowed",
        "bdl_point_diff",
        "bdl_total_yards",
        "bdl_net_passing_yards",
        "bdl_rushing_yards",
        "bdl_first_downs",
        "bdl_turnovers",
        "bdl_yards_per_play",
        "bdl_third_down_rate",
        "bdl_red_zone_score_rate",
        "bdl_sack_rate_allowed",
    ]
    base = frame[base_cols].copy()
    base = base.drop_duplicates(subset=["season", "week", "game_id", "team"], keep="last")

    opponent_values = base[
        [
            "game_id",
            "team",
            "bdl_total_yards",
            "bdl_net_passing_yards",
            "bdl_rushing_yards",
            "bdl_first_downs",
            "bdl_turnovers",
            "bdl_sack_rate_allowed",
        ]
    ].rename(
        columns={
            "team": "opponent",
            "bdl_total_yards": "bdl_total_yards_allowed",
            "bdl_net_passing_yards": "bdl_net_passing_yards_allowed",
            "bdl_rushing_yards": "bdl_rushing_yards_allowed",
            "bdl_first_downs": "bdl_first_downs_allowed",
            "bdl_turnovers": "bdl_takeaways",
            "bdl_sack_rate_allowed": "bdl_sack_rate_generated",
        }
    )
    opponent_join = base[["game_id", "team"]].merge(opponent_values, on="game_id", how="left")
    opponent_join = opponent_join[opponent_join["team"] != opponent_join["opponent"]]
    opponent_join = opponent_join.drop_duplicates(subset=["game_id", "team"], keep="last")
    base = base.merge(opponent_join.drop(columns=["opponent"]), on=["game_id", "team"], how="left")

    out = base[["season", "week", "team", *RAW_FEATURE_COLUMNS]].copy()
    for column in RAW_FEATURE_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.sort_values(["season", "team", "week"])


def build_balldontlie_team_form_features(
    team_stats: pd.DataFrame,
    seasons: Sequence[int],
    norm_team: Callable[..., str],
    schedule: pd.DataFrame | None = None,
    rolling_windows: Sequence[int] = (3,),
) -> pd.DataFrame:
    """Return team-week BallDontLie form features with prior-only windows.

    Non-final BallDontLie game rows are ignored before feature calculation.
    When a schedule is supplied, the output includes rows for future/current
    weeks so their ``*_prev`` and ``*_rollingN`` values are available even
    though same-week raw stats remain null.
    """

    completed = _completed_game_stats(team_stats, seasons, norm_team)
    team_index = _schedule_team_index(schedule, seasons, norm_team)
    if team_index.empty:
        out = completed.copy()
    else:
        out = team_index.merge(completed, on=["season", "week", "team"], how="left")
    if out.empty:
        return pd.DataFrame(columns=["season", "week", "team", *RAW_FEATURE_COLUMNS])

    out["season"] = pd.to_numeric(out["season"], errors="coerce")
    out["week"] = pd.to_numeric(out["week"], errors="coerce")
    out = out.dropna(subset=["season", "week", "team"])
    out["season"] = out["season"].astype(int)
    out["week"] = out["week"].astype(int)
    out = out.sort_values(["season", "team", "week"]).reset_index(drop=True)

    group = out.groupby(["season", "team"], group_keys=False)
    derived: dict[str, pd.Series] = {}
    for column in RAW_FEATURE_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan
        out[column] = pd.to_numeric(out[column], errors="coerce")
        derived[f"{column}_prev"] = group[column].shift(1)
        for window in rolling_windows:
            derived[f"{column}_rolling{int(window)}"] = group[column].transform(
                lambda series, w=int(window): series.shift(1).rolling(window=w, min_periods=1).mean()
            )

    if derived:
        out = pd.concat([out, pd.DataFrame(derived, index=out.index)], axis=1)
    out["season"] = out["season"].astype("Int64")
    out["week"] = out["week"].astype("Int64")
    return out
