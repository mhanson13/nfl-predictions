"""Complete the team-week injury aggregate before it is merged onto games.

Two gaps otherwise become all-NaN injury features:

* a team with nobody on the report in a week where reports exist (the raw
  file simply has no rows for it) -> genuinely zero listed players;
* a whole week missing upstream (e.g. nflverse has no 2023 postseason reports
  after the Wild Card round, and upcoming weeks before Wednesday's first
  report) -> carry forward the team's latest *earlier* report from the same
  season. Only past reports are used, so this is leak-free; nothing crosses
  the off-season.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd


def complete_injury_grid(
    inj_agg: pd.DataFrame,
    games: pd.DataFrame,
    value_cols: Sequence[str],
) -> pd.DataFrame:
    """Return ``inj_agg`` (season, week, team, *value_cols) covering every scheduled team-week."""
    if inj_agg is None or inj_agg.empty or games is None or games.empty:
        return inj_agg
    keys = ["season", "week", "team"]
    teams = pd.concat(
        [
            games[["season", "week", "home_team"]].rename(columns={"home_team": "team"}),
            games[["season", "week", "away_team"]].rename(columns={"away_team": "team"}),
        ],
        ignore_index=True,
    ).dropna()
    teams["season"] = pd.to_numeric(teams["season"], errors="coerce")
    teams["week"] = pd.to_numeric(teams["week"], errors="coerce")
    teams = teams.dropna().drop_duplicates()
    teams = teams.astype({"season": inj_agg["season"].dtype, "week": inj_agg["week"].dtype})

    reported_weeks = inj_agg[["season", "week"]].drop_duplicates()
    reported_weeks["_week_reported"] = True
    grid = teams.merge(inj_agg, on=keys, how="outer").merge(reported_weeks, on=["season", "week"], how="left")
    grid["_week_reported"] = grid["_week_reported"].eq(True)
    cols = [c for c in value_cols if c in grid.columns]

    # 1) week has reports but this team has no rows -> nobody listed
    absent = grid["_week_reported"] & grid[cols].isna().all(axis=1)
    grid.loc[absent, cols] = 0.0

    # 2) whole week missing upstream -> latest earlier report from the same season
    grid = grid.sort_values(["team", "season", "week"], kind="mergesort")
    carried = grid.groupby(["team", "season"], sort=False)[cols].ffill()
    missing_week = ~grid["_week_reported"]
    grid.loc[missing_week, cols] = carried.loc[missing_week, cols]

    return grid.drop(columns=["_week_reported"]).sort_values(keys, kind="mergesort").reset_index(drop=True)
