from __future__ import annotations

import pandas as pd


def filter_before_week(
    df: pd.DataFrame,
    season: int | None,
    week: int | None,
    *,
    season_col: str = "season",
    week_col: str = "week",
) -> pd.DataFrame:
    """Return rows strictly before a season/week cutoff."""
    if season is None or week is None:
        return df
    if season_col not in df.columns or week_col not in df.columns:
        return df

    out = df.copy()
    season_values = pd.to_numeric(out[season_col], errors="coerce")
    week_values = pd.to_numeric(out[week_col], errors="coerce")
    keep = (season_values < int(season)) | (
        (season_values == int(season)) & (week_values < int(week))
    )
    return out[keep.fillna(False)].copy()
