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

"""Long-memory, pre-game team-strength ratings.

The rolling 3/5-game windows elsewhere forget everything older than a month
and restart cold each September. These ratings carry information across the
whole history with explicit off-season regression to the mean:

* ``strength_elo``          Elo rating (margin-of-victory adjusted)
* ``strength_pdiff_ewm``    exponentially weighted point differential per game
* ``strength_pass_epa_ewm``     EWM offensive dropback EPA
* ``strength_pass_epa_def_ewm`` EWM dropback EPA allowed (the opponent's per-game value)
* ``strength_season_games`` games already played this season (rating maturity)

Every value for a game is computed only from games completed in *earlier
weeks*, so a row never sees its own result. Weeks are processed in order and
all games in a week read the state before any of them update it.
Columns are emitted as ``*_home`` / ``*_away``; ``make_feature_diffs`` adds
the ``*_diff`` model inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ELO_MEAN = 1505.0
ELO_K = 20.0
ELO_HOME_ADV = 48.0
ELO_SEASON_CARRY = 2.0 / 3.0  # keep 2/3 of the deviation from the mean across the off-season
EWM_HALFLIFE_GAMES = 8.0
EWM_SEASON_CARRY = 0.5  # deviation from the league mean kept across the off-season

# (output name, column read for the home team, column read for the away team).
# ``pass_epa_per_db_*`` holds that game's own result, so it is only read after
# the game has been played. Do NOT add ``epa_per_play_*`` /
# ``def_epa_per_play_allowed_*``: those are full-season team aggregates
# (constant within a team-season), i.e. they contain future games.
_EPA_STATS: tuple[tuple[str, str, str], ...] = (
    ("strength_pass_epa_ewm", "pass_epa_per_db_home", "pass_epa_per_db_away"),
    ("strength_pass_epa_def_ewm", "pass_epa_per_db_away", "pass_epa_per_db_home"),
)
STRENGTH_COLUMNS: tuple[str, ...] = (
    "strength_elo",
    "strength_pdiff_ewm",
    *(name for name, _, _ in _EPA_STATS),
    "strength_season_games",
)


@dataclass
class _TeamState:
    elo: float = ELO_MEAN
    season: int | None = None
    season_games: int = 0
    ewm: dict[str, float] = field(default_factory=dict)


def _elo_expected(diff: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))


def _mov_multiplier(point_diff: float, winner_elo_edge: float) -> float:
    """FiveThirtyEight-style margin-of-victory multiplier (damps blowouts by favourites)."""
    return float(np.log(abs(point_diff) + 1.0) * (2.2 / (winner_elo_edge * 0.001 + 2.2)))


def _per_game_stats(row: pd.Series, side: str) -> dict[str, float]:
    other = "away" if side == "home" else "home"
    pts_for = float(row[f"{side}_score"])
    pts_against = float(row[f"{other}_score"])
    stats = {"strength_pdiff_ewm": pts_for - pts_against}
    for name, home_col, away_col in _EPA_STATS:
        col = home_col if side == "home" else away_col
        value = pd.to_numeric(row.get(col), errors="coerce")
        if pd.notna(value):
            stats[name] = float(value)
    return stats


def add_team_strength_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with pre-game ``strength_*_home`` / ``strength_*_away`` columns added."""
    required = {"season", "week", "home_team", "away_team", "home_score", "away_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"add_team_strength_features needs columns {sorted(missing)}")

    out = df.copy()
    for col in STRENGTH_COLUMNS:
        out[f"{col}_home"] = np.nan
        out[f"{col}_away"] = np.nan

    season = pd.to_numeric(out["season"], errors="coerce")
    week = pd.to_numeric(out["week"], errors="coerce")
    valid = season.notna() & week.notna() & out["home_team"].notna() & out["away_team"].notna()
    order = (
        pd.DataFrame({"season": season, "week": week}, index=out.index)[valid]
        .sort_values(["season", "week"], kind="mergesort")
    )

    alpha = 1.0 - 0.5 ** (1.0 / EWM_HALFLIFE_GAMES)
    teams: dict[str, _TeamState] = {}
    league_mean: dict[str, float] = {}
    league_n: dict[str, int] = {}

    def team_state(team: str, season_value: int) -> _TeamState:
        state = teams.setdefault(team, _TeamState())
        if state.season is not None and state.season != season_value:
            state.elo = ELO_MEAN + (state.elo - ELO_MEAN) * ELO_SEASON_CARRY
            for key, value in state.ewm.items():
                mean = league_mean.get(key, value)
                state.ewm[key] = mean + (value - mean) * EWM_SEASON_CARRY
            state.season_games = 0
        state.season = season_value
        return state

    for (season_value, _week_value), week_rows in order.groupby(["season", "week"], sort=True):
        season_int = int(season_value)
        idx = week_rows.index
        # 1) pre-game snapshot for every game in the week
        for i in idx:
            row = out.loc[i]
            for side in ("home", "away"):
                state = team_state(str(row[f"{side}_team"]), season_int)
                out.at[i, f"strength_elo_{side}"] = state.elo
                out.at[i, f"strength_season_games_{side}"] = state.season_games
                for key in STRENGTH_COLUMNS[1:-1]:
                    out.at[i, f"{key}_{side}"] = state.ewm.get(key, league_mean.get(key, np.nan))
        # 2) update with this week's completed games
        for i in idx:
            row = out.loc[i]
            if pd.isna(row["home_score"]) or pd.isna(row["away_score"]):
                continue
            home = teams[str(row["home_team"])]
            away = teams[str(row["away_team"])]
            point_diff = float(row["home_score"]) - float(row["away_score"])
            elo_diff = home.elo + ELO_HOME_ADV - away.elo
            expected = _elo_expected(elo_diff)
            actual = 1.0 if point_diff > 0 else (0.0 if point_diff < 0 else 0.5)
            winner_edge = elo_diff if point_diff >= 0 else -elo_diff
            shift = ELO_K * _mov_multiplier(point_diff, winner_edge) * (actual - expected) if point_diff else 0.0
            home.elo += shift
            away.elo -= shift
            for side, state in (("home", home), ("away", away)):
                state.season_games += 1
                for key, value in _per_game_stats(row, side).items():
                    n = league_n.get(key, 0)
                    league_mean[key] = (league_mean.get(key, 0.0) * n + value) / (n + 1)
                    league_n[key] = n + 1
                    prev = state.ewm.get(key)
                    state.ewm[key] = value if prev is None else (1.0 - alpha) * prev + alpha * value
    return out
