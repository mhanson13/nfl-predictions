"""Full player display names for published projection and prop CSVs.

nflverse ``player_name`` is abbreviated ("J.Herbert"). These helpers swap in the full
name by player_id for output only; internal matching keys are built before this runs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.io import RAW_DIR

DEFAULT_PLAYER_STATS_PATH = RAW_DIR / "nfl_player_stats.parquet"
DEFAULT_ROSTERS_PATH = RAW_DIR / "nfl_rosters.parquet"


def _latest_name_by_id(frame: pd.DataFrame, id_col: str, name_col: str) -> pd.Series:
    if frame.empty or id_col not in frame.columns or name_col not in frame.columns:
        return pd.Series(dtype="object")
    cols = [id_col, name_col] + [c for c in ("season", "week") if c in frame.columns]
    out = frame[cols].copy()
    out[name_col] = out[name_col].astype("string").str.strip()
    out = out.dropna(subset=[id_col, name_col])
    # A single-word name ("Emmett") is not a usable display name.
    out = out[out[name_col].str.contains(" ", regex=False)]
    sort_cols = [c for c in ("season", "week") if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, kind="stable")
    return out.groupby(id_col)[name_col].last()


def load_display_names(
    player_stats_path: Path = DEFAULT_PLAYER_STATS_PATH,
    rosters_path: Path = DEFAULT_ROSTERS_PATH,
) -> pd.Series:
    """Map player_id to a full display name ("Justin Herbert", not "J.Herbert").

    nflverse player stats ``player_display_name`` first, roster ``full_name`` as fallback.
    """
    names = pd.Series(dtype="object")
    if rosters_path is not None and Path(rosters_path).exists():
        try:
            rosters = pd.read_parquet(rosters_path, columns=["season", "week", "gsis_id", "full_name"])
        except Exception:
            rosters = pd.DataFrame()
        names = _latest_name_by_id(rosters, "gsis_id", "full_name")
    if player_stats_path is not None and Path(player_stats_path).exists():
        try:
            stats = pd.read_parquet(player_stats_path, columns=["season", "week", "player_id", "player_display_name"])
        except Exception:
            stats = pd.DataFrame()
        stat_names = _latest_name_by_id(stats, "player_id", "player_display_name")
        names = stat_names.combine_first(names)
    return names


def apply_display_names(frame: pd.DataFrame, names: pd.Series) -> pd.DataFrame:
    """Replace ``player_name`` with the full name for output only.

    Runs after odds matching, so internal name keys are unaffected. Rows without a
    known player_id keep their existing name.
    """
    if frame.empty or names.empty or "player_id" not in frame.columns or "player_name" not in frame.columns:
        return frame
    out = frame.copy()
    full = out["player_id"].map(names)
    out["player_name"] = full.where(full.notna(), out["player_name"])
    return out
