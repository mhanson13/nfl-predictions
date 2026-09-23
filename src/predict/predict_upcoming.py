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

"""Predict upcoming NFL games using trained models and parquet artifacts with engine fallbacks."""

from __future__ import annotations
import argparse
import re
import shutil
import warnings
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
from dateutil import tz
from src.utils.io import RAW_DIR, PROC_DIR, read_df
from src.utils.logging_config import setup_logging
from src.utils.odds import ODDS_API_FREE_KEY_SECRET, fetch_odds, get_odds_api_free_key
from src.utils.teams import get_team_abbr_from_name
from src.predict.utils import apply_probability_caps, moneyline_to_prob
from src.predict.volatility import (
    DEFAULT_VOLATILITY_ARTIFACT,
    apply_shrinkage as apply_volatility_shrinkage,
    load_volatility_artifact,
    score_volatility,
)





def _column_or_default(

    df: pd.DataFrame,

    column: str,

    default_value: Any = pd.NA,

    dtype: Optional[str] = None,

) -> pd.Series:

    """Return an existing column or a Series aligned to df.index filled with default_value."""

    if column in df.columns:

        return df[column]

    if dtype is not None:

        return pd.Series(default_value, index=df.index, dtype=dtype)

    if default_value is pd.NA:

        return pd.Series(pd.NA, index=df.index, dtype="object")

    return pd.Series(default_value, index=df.index)


def _coerce_probability_series(values: Any, index: pd.Index) -> pd.Series:
    """Return numeric probabilities aligned to *index*."""
    if isinstance(values, pd.Series):
        series = values.reindex(index) if not values.index.equals(index) else values.copy()
    else:
        series = pd.Series(np.asarray(values, dtype=float), index=index)
    return pd.to_numeric(series, errors="coerce").astype(float)


def _has_probability_signal(values: Any) -> bool:
    """True when a probability vector contains more than one finite value."""
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return len(series) > 1 and series.nunique(dropna=True) > 1


def _is_collapsed_probability(values: Any) -> bool:
    """True when multiple finite probabilities have collapsed to one value."""
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return len(series) > 1 and series.nunique(dropna=True) <= 1


def _probability_unique_count(values: Any) -> int:
    """Count distinct finite probabilities."""
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return int(series.nunique(dropna=True))


def _probability_range(values: Any) -> float:
    """Return max-min spread for finite probabilities."""
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if series.empty:
        return 0.0
    return float(series.max() - series.min())


def _is_saturated_probability_signal(values: Any) -> bool:
    """True when probabilities are mostly pinned to extreme tails."""
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if len(series) < 2:
        return False
    tail_rate = float(((series <= 0.02) | (series >= 0.98)).mean())
    one_sided_tail_rate = float((series >= 0.95).mean() + (series <= 0.05).mean())
    return tail_rate >= 0.5 or one_sided_tail_rate >= 0.75


def _fallback_if_probability_collapsed(
    candidate: pd.Series,
    fallback: pd.Series,
    *,
    stage: str,
) -> tuple[pd.Series, str | None]:
    """Use fallback probabilities when a stage destroys or overstates slate-level signal."""
    candidate_unique = _probability_unique_count(candidate)
    fallback_unique = _probability_unique_count(fallback)
    candidate_range = _probability_range(candidate)
    fallback_range = _probability_range(fallback)
    severe_compression = fallback_unique >= 8 and candidate_unique <= max(2, fallback_unique // 4)
    severe_range_compression = fallback_range >= 0.03 and candidate_range <= max(0.005, fallback_range * 0.25)
    saturation = (
        fallback_unique > 1
        and _is_saturated_probability_signal(candidate)
        and not _is_saturated_probability_signal(fallback)
    )
    reason = None
    if candidate_unique <= 1 or severe_compression or severe_range_compression:
        reason = f"{stage}_collapsed"
    elif saturation:
        reason = f"{stage}_saturated"
    if fallback_unique > 1 and reason is not None:
        clipped = fallback.clip(0.02, 0.98)
        if _has_probability_signal(clipped):
            return clipped, reason
        lightly_clipped = fallback.clip(0.001, 0.999)
        return lightly_clipped, reason
    return candidate, None


def _best_probability_signal(preds: pd.DataFrame, index: pd.Index) -> tuple[pd.Series, str]:
    """Return the best available probability column that still has ranking signal."""
    candidates: list[tuple[bool, float, pd.Series, str]] = []
    for col in ("home_win_prob_capped", "home_win_prob_calibrated", "home_win_prob_model_raw"):
        if col not in preds.columns:
            continue
        series = _coerce_probability_series(preds[col], index)
        prob_range = _probability_range(series)
        if _has_probability_signal(series):
            candidates.append((_is_saturated_probability_signal(series), prob_range, series, col))
    if candidates:
        non_saturated = [item for item in candidates if not item[0]]
        pool = non_saturated or candidates
        _, _, best_series, best_col = max(pool, key=lambda item: item[1])
        return best_series, best_col
    if "home_win_prob" in preds.columns:
        return _coerce_probability_series(preds["home_win_prob"], index), "home_win_prob"
    return pd.Series(np.nan, index=index, dtype=float), "none"


def _append_quality_fallback(preds: pd.DataFrame, reason: str) -> None:
    """Append a fallback reason to the exported quality diagnostic column."""
    existing = (
        str(preds["home_win_prob_quality_fallback"].iloc[0])
        if "home_win_prob_quality_fallback" in preds.columns and len(preds)
        else ""
    )
    reasons = [part for part in existing.split(";") if part and part.lower() != "nan"]
    if reason not in reasons:
        reasons.append(reason)
    preds["home_win_prob_quality_fallback"] = ";".join(reasons)


def _should_skip_volatility_shrinkage(
    labels: Any,
    *,
    min_coverage: float = 0.0,
    max_coverage: float,
) -> tuple[bool, float]:
    """Return whether volatility labels cover too little or too much of the slate to be useful."""
    series = pd.to_numeric(pd.Series(labels), errors="coerce").dropna()
    if series.empty:
        return False, 0.0
    coverage = float((series == 1).mean())
    min_cov = float(np.clip(min_coverage, 0.0, 1.0))
    max_cov = float(np.clip(max_coverage, 0.0, 1.0))
    return coverage < min_cov or coverage > max_cov, coverage


def _volatility_artifact_skip_reason(
    artifact: Any,
    *,
    min_auc: float = 0.55,
    min_pred_positive_rate: float = 0.02,
    max_pred_positive_rate: float = 0.85,
) -> str | None:
    """Return a diagnostic reason when a volatility artifact is too weak to adjust predictions."""
    metrics = getattr(artifact, "metrics", None) or {}
    auc = metrics.get("auc")
    if auc is not None and np.isfinite(float(auc)) and float(auc) < min_auc:
        return f"auc={float(auc):.3f}<min_auc={min_auc:.3f}"
    pred_rate = metrics.get("pred_positive_rate")
    if pred_rate is not None and np.isfinite(float(pred_rate)):
        pred_rate = float(pred_rate)
        if pred_rate < min_pred_positive_rate:
            return f"pred_positive_rate={pred_rate:.3f}<min={min_pred_positive_rate:.3f}"
        if pred_rate > max_pred_positive_rate:
            return f"pred_positive_rate={pred_rate:.3f}>max={max_pred_positive_rate:.3f}"
    return None


def _apply_probability_calibrator(calibrator: Any, probabilities: pd.Series) -> pd.Series:
    """Apply a probability calibrator while preserving probability, not class, output."""
    values = pd.to_numeric(probabilities, errors="coerce").astype(float)
    arr = values.to_numpy(dtype=float).reshape(-1, 1)
    if hasattr(calibrator, "predict_proba"):
        calibrated = np.asarray(calibrator.predict_proba(arr))[:, 1]
    elif hasattr(calibrator, "predict"):
        calibrated = calibrator.predict(values)
    elif hasattr(calibrator, "transform"):
        calibrated = calibrator.transform(values)
    else:
        raise TypeError(f"Unsupported calibrator type: {type(calibrator).__name__}")
    return _coerce_probability_series(calibrated, probabilities.index)


QB_PLAYER_PROJECTION_COLUMNS = [
    "season",
    "week",
    "game_id",
    "kickoff",
    "kickoff_mt",
    "team",
    "team_side",
    "player_rank",
    "player_id",
    "player_name",
    "jersey_number",
    "games_sampled",
    "projected_passing_yards",
    "projected_passing_tds",
    "projected_rushing_tds",
    "projected_passing_attempts",
    "projected_interceptions",
    "projected_completions",
    "projected_rushing_yards",
]

OFFENSE_PLAYER_PROJECTION_COLUMNS = [
    "season",
    "week",
    "game_id",
    "kickoff",
    "kickoff_mt",
    "team",
    "team_side",
    "player_rank",
    "player_id",
    "player_name",
    "jersey_number",
    "games_sampled",
    "projected_rushing_yards",
    "projected_rushing_tds",
    "projected_carries",
    "projected_receiving_yards",
    "projected_receiving_tds",
    "projected_receptions",
    "projected_targets",
    "projected_total_yards",
    "projected_total_tds",
    "penalties_per_game",
    "penalty_count",
    "penalty_yards",
]


def _write_player_projection_csv(
    frame: pd.DataFrame,
    save_path: str,
    columns: list[str],
    label: str,
    *,
    debug: bool = False,
) -> pd.DataFrame:
    """Write player projection output, including explicit empty current-week files."""
    output = frame.copy()
    for col in columns:
        if col not in output.columns:
            output[col] = pd.NA
    output = output[columns]
    _backup_existing_file(Path(save_path))
    output.to_csv(save_path, index=False)
    if debug:
        print(f"[predict][players] saved {label} projections -> {save_path} (rows={len(output)})")
    return output


def _recent_player_stat_threshold(
    subset: pd.DataFrame,
    target_season: int,
    label: str,
    *,
    debug: bool = False,
) -> int:
    """Return recency floor, relaxing it when the player-stat feed is stale."""
    target_floor = int(target_season) - 1
    if subset is None or subset.empty or "season" not in subset.columns:
        return target_floor
    seasons = pd.to_numeric(subset["season"], errors="coerce").dropna()
    if seasons.empty:
        return target_floor
    latest = int(seasons.max())
    if latest < target_floor and debug:
        print(
            f"[predict][players][{label}] player stats only available through {latest}; "
            f"using {latest} as the recency floor for {target_season} projections."
        )
    return min(target_floor, latest)





def _round_or_none(value: Any, digits: int) -> Optional[float]:

    """Round numeric-like scalars, returning None for missing values."""

    if pd.isna(value):

        return None

    return round(float(value), digits)





MOUNTAIN_TZ = tz.gettz("America/Denver")


def _format_mountain_time(series: pd.Series) -> pd.Series:
    """Return a series of Mountain Time-formatted strings for kickoff timestamps."""
    if series is None or len(series) == 0:
        return pd.Series([], dtype="object")
    kickoff_dt = pd.to_datetime(series, utc=True, errors="coerce")
    if kickoff_dt.isna().all():
        return pd.Series([""] * len(series), index=series.index, dtype="object")
    if MOUNTAIN_TZ is None:
        formatted = kickoff_dt.dt.strftime("%Y-%m-%d %H:%M UTC")
    else:
        kickoff_mt = kickoff_dt.dt.tz_convert(MOUNTAIN_TZ)
        formatted = kickoff_mt.dt.strftime("%Y-%m-%d %H:%M %Z")
    return formatted.fillna("")




PREDICTIONS_ARCHIVE_DIR = Path("predictions")





def _backup_existing_file(path: Path) -> None:

    """Rename an existing file to the next available .backup_N suffix."""

    if not path.exists():

        return

    counter = 1

    while True:

        backup_path = path.with_name(f"{path.name}.backup_{counter}")

        if not backup_path.exists():

            path.rename(backup_path)

            return

        counter += 1





def _archive_prediction_file(source_path: Optional[str], week: Optional[Any]) -> Optional[Path]:

    """Copy the freshly generated prediction file into the archive directory with a week prefix."""

    if not source_path:

        return None

    src = Path(source_path)

    if not src.exists():

        return None

    try:

        week_int = int(week) if week is not None else None

    except Exception:

        week_int = None

    prefix = f"w{week_int}" if week_int is not None else "wNA"

    archive_dir = PREDICTIONS_ARCHIVE_DIR

    archive_dir.mkdir(parents=True, exist_ok=True)

    target_name = f"{prefix}_{src.name}"

    target_path = archive_dir / target_name

    _backup_existing_file(target_path)

    shutil.copy2(src, target_path)

    return target_path





def _load_penalty_frame() -> pd.DataFrame:
    path = RAW_DIR / "nfl_pbp.parquet"
    if not path.exists():
        return pd.DataFrame()
    columns = [

        "season",

        "week",

        "game_id",

        "season_type",

        "penalty",

        "penalty_team",

        "penalty_player_id",

        "penalty_player_name",

        "penalty_yards",

    ]

    df = read_df(path, columns=columns)
    df = df.copy()

    df["season"] = pd.to_numeric(_column_or_default(df, "season"), errors="coerce")

    df["week"] = pd.to_numeric(_column_or_default(df, "week"), errors="coerce")

    df = df[pd.notna(df["season"]) & pd.notna(df["week"])]

    if df.empty:

        return df

    df["season"] = df["season"].astype("int64")

    df["week"] = df["week"].astype("int64")

    season_type_series = _column_or_default(df, "season_type", default_value="REG", dtype="object")

    df["season_type"] = season_type_series.fillna("").astype(str)

    mask_reg = df["season_type"].str.upper().isin({"REG", "R", "2"})

    if mask_reg.any():

        df = df[mask_reg]

    df.drop(columns=["season_type"], inplace=True, errors="ignore")

    df["penalty"] = pd.to_numeric(_column_or_default(df, "penalty"), errors="coerce").fillna(0).astype("int64")

    df = df[df["penalty"] > 0]

    if df.empty:

        return df

    game_ids = _column_or_default(df, "game_id", default_value="", dtype="object")

    df["game_id"] = game_ids.fillna("").astype(str)

    penalty_team = _column_or_default(df, "penalty_team", default_value="", dtype="object")

    df["penalty_team"] = penalty_team.fillna("").astype(str).str.upper()

    player_ids = _column_or_default(df, "penalty_player_id", default_value="", dtype="object")

    df["penalty_player_id"] = player_ids.fillna("").astype(str)

    player_names = _column_or_default(df, "penalty_player_name", default_value="", dtype="object")

    df["penalty_player_name"] = player_names.fillna("").astype(str)

    df["penalty_yards"] = pd.to_numeric(_column_or_default(df, "penalty_yards"), errors="coerce").fillna(0.0)

    df = df[df["penalty_team"] != ""]

    return df





def _select_penalty_window(penalties: pd.DataFrame, season: int, week: Optional[int]) -> pd.DataFrame:

    if penalties is None or penalties.empty:

        return pd.DataFrame()

    current = pd.DataFrame()

    if week is not None:

        try:

            week_int = int(week)

        except Exception:

            week_int = None

        if week_int is not None:

            current = penalties[(penalties["season"] == season) & (penalties["week"] < week_int)].copy()

    else:

        current = penalties[penalties["season"] == season].copy()

    history = penalties[penalties["season"] < season].copy()

    if history.empty and current.empty:

        return pd.DataFrame()

    subset = pd.concat([history, current], ignore_index=True)

    return subset





def _aggregate_player_penalties(penalties: pd.DataFrame) -> pd.DataFrame:

    if penalties is None or penalties.empty:

        return pd.DataFrame()

    players = penalties[penalties["penalty_player_id"] != ""].copy()

    if players.empty:

        return pd.DataFrame()

    players["player_id"] = players["penalty_player_id"].astype(str)

    players["team"] = players["penalty_team"].astype(str).str.upper()

    players["game_key"] = players["game_id"].astype(str)

    agg = players.groupby(["player_id", "team"], as_index=False).agg(

        penalty_count=("penalty", "sum"),

        penalty_yards=("penalty_yards", "sum"),

        penalty_games=("game_key", "nunique"),

        penalty_player_name=("penalty_player_name", "last"),

    )

    if agg.empty:

        return agg

    games = agg["penalty_games"].replace({0: pd.NA})

    agg["penalty_rate_per_game"] = (agg["penalty_count"] / games).astype(float)

    return agg





def _aggregate_team_penalties(penalties: pd.DataFrame) -> pd.DataFrame:

    if penalties is None or penalties.empty:

        return pd.DataFrame()

    penalties = penalties.copy()

    penalties["game_key"] = penalties["game_id"].astype(str)

    agg = penalties.groupby("penalty_team", as_index=False).agg(

        penalty_count=("penalty", "sum"),

        penalty_yards=("penalty_yards", "sum"),

        penalty_games=("game_key", "nunique"),

    )

    if agg.empty:

        return agg

    games = agg["penalty_games"].replace({0: pd.NA})

    agg["penalties_per_game"] = (agg["penalty_count"] / games).astype(float)

    agg["penalty_yards_per_game"] = (agg["penalty_yards"] / games).astype(float)

    agg.rename(columns={"penalty_team": "team"}, inplace=True)

    agg["team"] = agg["team"].astype(str).str.upper()

    return agg





def _attach_team_penalty_metrics(
    preds: pd.DataFrame,
    penalties: pd.DataFrame,
    season: int,
    week: Optional[int],
    debug: bool = False,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Attach rolling penalty rates per team to the predictions matrix.

    Returns the enriched DataFrame and list of new column names so callers can
    drop or log the additions if needed.
    """
    if penalties is None or penalties.empty or preds.empty:

        return preds, []

    subset = _select_penalty_window(penalties, season, week)

    team_stats = _aggregate_team_penalties(subset)

    if team_stats.empty:

        if debug:

            print("[predict][penalties] no team penalty history available for attachment")

        return preds, []

    team_stats.set_index("team", inplace=True)

    cols_added: List[str] = []

    for prefix, team_col in (("home", "home_team"), ("away", "away_team")):

        rate_col = f"{prefix}_penalties_per_game"

        yards_col = f"{prefix}_penalty_yards_per_game"

        preds[rate_col] = preds[team_col].astype(str).str.upper().map(team_stats["penalties_per_game"])

        preds[yards_col] = preds[team_col].astype(str).str.upper().map(team_stats["penalty_yards_per_game"])

        cols_added.extend([rate_col, yards_col])

    return preds, cols_added





def _make_feature_diffs(df: pd.DataFrame) -> pd.DataFrame:

    """Create *_diff = *_home - *_away features to mirror training preprocessing."""

    feat_df = df.copy()

    cols = set(feat_df.columns)

    diffs: Dict[str, pd.Series] = {}

    for c in list(cols):

        if not isinstance(c, str) or not c.endswith("_home"):

            continue

        base = c[:-5]

        away_col = base + "_away"

        if away_col not in cols:

            continue

        s_home = pd.to_numeric(feat_df[c], errors="coerce")

        s_away = pd.to_numeric(feat_df[away_col], errors="coerce")

        diffs[f"{base}_diff"] = s_home.astype(float) - s_away.astype(float)

    if diffs:

        feat_df = pd.concat([feat_df, pd.DataFrame(diffs, index=feat_df.index)], axis=1)

    return feat_df.copy()





def _cleanup_prediction_columns(df: pd.DataFrame) -> pd.DataFrame:

    """Drop duplicate, all-null, and all-zero numeric columns."""

    cleaned = df.copy()
    preserve_zero_cols = {
        "home_win_prob_model_raw",
        "home_win_prob_calibrated",
        "home_win_prob_capped",
        "home_win_prob_raw",
        "volatility_prob",
        "volatility_label",
    }

    idx = pd.Index(cleaned.columns)

    if idx.duplicated().any():

        dup_mask = idx.duplicated(keep="last")

        if dup_mask.any():

            cleaned = cleaned.loc[:, ~dup_mask]

    all_na = [c for c in cleaned.columns if cleaned[c].isna().all()]

    if all_na:

        cleaned = cleaned.drop(columns=all_na)

    numeric_cols = cleaned.select_dtypes(include="number").columns

    all_zero = []

    for col in numeric_cols:
        if col in preserve_zero_cols:
            continue

        series = pd.to_numeric(cleaned[col], errors="coerce")

        if not series.notna().any():

            continue

        if series.fillna(0).ne(0).any():

            continue

        all_zero.append(col)

    if all_zero:

        cleaned = cleaned.drop(columns=all_zero)

    return cleaned





def _normalize_team_name(name: Optional[str]) -> Optional[str]:
    """
    Normalize team name to standard abbreviation.
    
    Uses centralized team mapping from src.utils.teams module.
    
    Args:
        name: Team name to normalize
        
    Returns:
        Team abbreviation or None if not found
    """
    return get_team_abbr_from_name(name)



def _load_odds_api_key() -> Optional[str]:
    return get_odds_api_free_key()



def _slugify_bookmaker(name: str) -> str:

    return re.sub(r"[^a-z0-9]+", "_", (name or "").lower()).strip("_") or "bookmaker"



def _attach_odds_spreads(preds: pd.DataFrame, odds: Optional[list[dict]], debug: bool = False) -> List[str]:

    if odds is None or preds.empty:

        return []

    cols_needed: Set[str] = set()

    updates: List[Tuple[pd.Series, Dict[str, Any]]] = []

    for event in odds:

        try:

            home_abbr = _normalize_team_name(event.get("home_team"))

            away_abbr = _normalize_team_name(event.get("away_team"))

            if not home_abbr or not away_abbr:

                continue

            mask = (preds["home_team"].astype(str).str.upper() == home_abbr) & (

                preds["away_team"].astype(str).str.upper() == away_abbr

            )

            if not mask.any():

                continue

            commence = event.get("commence_time")

            for book in event.get("bookmakers", []):

                book_key = book.get("key") or book.get("title") or "bookmaker"

                slug = _slugify_bookmaker(book_key)

                market = None

                for mk in book.get("markets", []):

                    if mk.get("key") == "spreads":

                        market = mk

                        break

                if market is None:

                    continue

                outcomes = market.get("outcomes", [])

                home_line = home_price = away_line = away_price = None

                for outcome in outcomes:

                    name = outcome.get("name")

                    abbr = _normalize_team_name(name)

                    if abbr == home_abbr:

                        home_line = outcome.get("point")

                        home_price = outcome.get("price")

                    elif abbr == away_abbr:

                        away_line = outcome.get("point")

                        away_price = outcome.get("price")

                columns = {

                    f"odds_{slug}_home_spread": home_line,

                    f"odds_{slug}_home_price": home_price,

                    f"odds_{slug}_away_spread": away_line,

                    f"odds_{slug}_away_price": away_price,

                }

                if commence:

                    columns[f"odds_{slug}_last_update"] = book.get("last_update") or commence

                cols_needed.update(columns.keys())

                updates.append((mask, columns))

        except Exception:

            if debug:

                print("[predict][odds] failed to merge an odds event")

            continue

    if not updates:

        return []

    missing = [col for col in cols_needed if col not in preds.columns]

    for col in missing:

        # Avoid DataFrame reindex (which fails when existing columns contain duplicates)

        preds.loc[:, col] = pd.NA

    for mask, columns in updates:

        for col, value in columns.items():

            preds.loc[mask, col] = value

    return sorted(cols_needed)











def _load_player_stats_frame() -> pd.DataFrame:

    path = RAW_DIR / "nfl_player_stats.parquet"

    if not path.exists():

        return pd.DataFrame()

    df = read_df(path)
    df = df.copy()

    df["season"] = pd.to_numeric(_column_or_default(df, "season"), errors="coerce")

    df["week"] = pd.to_numeric(_column_or_default(df, "week"), errors="coerce")

    df = df[pd.notna(df["season"]) & pd.notna(df["week"])]

    df["season"] = df["season"].astype("int64")

    df["week"] = df["week"].astype("int64")

    season_type_raw = _column_or_default(df, "season_type")
    season_type_num = pd.to_numeric(season_type_raw, errors="coerce")
    season_type_text = season_type_raw.fillna("").astype(str).str.strip().str.upper()
    regular_season = (
        season_type_num.eq(2)
        | season_type_text.isin(["", "REG", "REGULAR", "REGULAR_SEASON", "REGULAR SEASON"])
    )
    df = df[regular_season].copy()

    if "recent_team" in df.columns:
        recent_team = _column_or_default(df, "recent_team", default_value="", dtype="object")
    else:
        recent_team = _column_or_default(df, "team", default_value="", dtype="object")

    df["recent_team"] = recent_team.fillna("").astype(str).str.upper()

    player_ids = _column_or_default(df, "player_id", default_value="", dtype="object")

    df["player_id"] = player_ids.fillna("").astype(str)

    if "player_name" not in df.columns:

        if "player_display_name" in df.columns:

            df["player_name"] = df["player_display_name"].astype(str)

        else:

            df["player_name"] = ""

    player_names = _column_or_default(df, "player_name", default_value="", dtype="object")

    df["player_name"] = player_names.fillna("").astype(str)

    return df





def _load_roster_frame() -> pd.DataFrame:

    path = RAW_DIR / "nfl_rosters.parquet"

    if not path.exists():

        return pd.DataFrame()

    df = read_df(path)
    df = df.copy()

    df["season"] = pd.to_numeric(_column_or_default(df, "season"), errors="coerce")

    df = df[pd.notna(df["season"])]

    df["season"] = df["season"].astype("int64")

    team_series = _column_or_default(df, "team", default_value="", dtype="object")

    df["team"] = team_series.fillna("").astype(str).str.upper()

    gsis_ids = _column_or_default(df, "gsis_id", default_value="", dtype="object")

    df["gsis_id"] = gsis_ids.fillna("").astype(str)

    if "jersey_number" not in df.columns:

        df["jersey_number"] = pd.NA

    if "full_name" not in df.columns:

        df["full_name"] = pd.NA

    status_series = _column_or_default(df, "status", default_value="", dtype="object")

    df["status"] = status_series.fillna("").astype(str).str.upper()

    position_series = _column_or_default(df, "position", default_value="", dtype="object")

    df["position"] = position_series.fillna("").astype(str).str.upper()

    depth_chart_series = _column_or_default(df, "depth_chart_position", default_value="", dtype="object")

    df["depth_chart_position"] = depth_chart_series.fillna("").astype(str).str.upper()

    return df[

        ["season", "team", "gsis_id", "jersey_number", "full_name", "status", "position", "depth_chart_position"]

    ]



def _active_player_ids(roster: pd.DataFrame, season: int, team: str, active_status: Optional[Set[str]]) -> Optional[Set[str]]:

    if roster is None or roster.empty:

        return None

    df = roster[(roster["season"] == season) & (roster["team"] == team)].copy()

    if df.empty:

        return None

    if active_status and "status" in df.columns:

        df = df[df["status"].isin(active_status)]

        if df.empty:

            return None

    ids = df["gsis_id"].dropna().astype(str)

    if ids.empty:

        return None

    return set(ids)





def _active_roster_records(

    roster: pd.DataFrame,

    season: int,

    team: str,

    active_status: Optional[Set[str]],

    allowed_positions: Optional[Set[str]] = None,

) -> pd.DataFrame:

    if roster is None or roster.empty:

        return pd.DataFrame()

    df = roster[(roster["season"] == season) & (roster["team"] == team)].copy()

    if df.empty:

        return pd.DataFrame()

    if active_status and "status" in df.columns:

        df = df[df["status"].isin(active_status)]

    if df.empty:

        return pd.DataFrame()

    position_series = _column_or_default(df, "position", default_value="", dtype="object")

    df["position"] = position_series.fillna("").astype(str).str.upper()

    if allowed_positions is not None:

        df = df[df["position"].isin(allowed_positions)]

        if df.empty:

            return pd.DataFrame()

    depth_series = _column_or_default(df, "depth_chart_position", default_value="", dtype="object")

    df["depth_chart_position"] = depth_series.fillna("").astype(str).str.upper()

    player_names = _column_or_default(df, "full_name", default_value="", dtype="object")

    df["player_name"] = player_names.fillna("").astype(str)

    if "jersey_number" not in df.columns:

        df["jersey_number"] = pd.NA

    jersey_series = _column_or_default(df, "jersey_number")

    df["jersey_number"] = jersey_series

    gsis = df.get("gsis_id")

    if gsis is None:

        return pd.DataFrame()

    df = df.dropna(subset=["gsis_id"]).copy()

    df["player_id"] = df["gsis_id"].astype(str)

    df = df.drop_duplicates(subset=["player_id"], keep="last")

    return df[["player_id", "player_name", "jersey_number", "position", "depth_chart_position", "status"]]









def _attach_roster_info(

    agg: pd.DataFrame,

    roster: pd.DataFrame,

    season: int,

    team_col: str,

    fill_name: bool = False,

) -> pd.DataFrame:

    agg = agg.copy()

    if "jersey_number" not in agg.columns:

        agg["jersey_number"] = pd.NA

    if roster is None or roster.empty:

        team_series = _column_or_default(agg, team_col, default_value="", dtype="object")

        agg[team_col] = team_series.fillna("").astype(str).str.upper()

        return agg



    roster_all = roster

    roster_season = roster_all[roster_all["season"] == season]



    def _apply(subset: pd.DataFrame, mask: pd.Series, update_team: bool = True) -> None:

        subset = subset.dropna(subset=["gsis_id"]).drop_duplicates(subset=["gsis_id"], keep="last")

        if subset.empty:

            return

        jersey_map = subset.set_index("gsis_id")["jersey_number"]

        team_map = subset.set_index("gsis_id")["team"]

        name_map = subset.set_index("gsis_id")["full_name"]



        idx = agg.index[mask]

        if not len(idx):

            return



        jerseys = agg.loc[idx, "player_id"].map(jersey_map)

        jersey_fill_mask = agg.loc[idx, "jersey_number"].isna() & jerseys.notna()

        if jersey_fill_mask.any():

            agg.loc[idx[jersey_fill_mask], "jersey_number"] = jerseys.loc[idx[jersey_fill_mask]]



        if update_team:

            teams = agg.loc[idx, "player_id"].map(team_map)

            team_current = agg.loc[idx, team_col].fillna("")

            team_fill_mask = teams.notna() & (

                team_current.eq("") | team_current.eq("TOT") | team_current.eq("ALL")

            )

            if team_fill_mask.any():

                agg.loc[idx[team_fill_mask], team_col] = teams.loc[idx[team_fill_mask]].astype(str).str.upper()



        if fill_name and "player_name" in agg.columns:

            names = agg.loc[idx, "player_id"].map(name_map)

            name_current = agg.loc[idx, "player_name"].fillna("")

            name_fill_mask = names.notna() & name_current.eq("")

            if name_fill_mask.any():

                agg.loc[idx[name_fill_mask], "player_name"] = names.loc[idx[name_fill_mask]].astype(str)



    all_mask = pd.Series(True, index=agg.index)

    if not roster_season.empty:

        _apply(roster_season, all_mask, update_team=True)

    missing_mask = agg["jersey_number"].isna()

    if missing_mask.any():

        _apply(roster_all, missing_mask, update_team=False)



    team_series = _column_or_default(agg, team_col, default_value="", dtype="object")

    agg[team_col] = team_series.fillna("").astype(str).str.upper()

    return agg





def _select_stats_window(stats: pd.DataFrame, season: int, week: Optional[int]) -> pd.DataFrame:

    current = pd.DataFrame()

    if week is not None:

        try:

            week_int = int(week)

        except Exception:

            week_int = None

        if week_int is not None:

            current = stats[(stats["season"] == season) & (stats["week"] < week_int)].copy()

    else:

        current = stats[stats["season"] == season].copy()



    history = stats[stats["season"] < season].copy()

    subset = pd.concat([history, current], ignore_index=True)

    return subset





def _player_qb_predictions(
    pred_games: pd.DataFrame,
    save_path: Optional[str],
    stats: pd.DataFrame,
    roster: pd.DataFrame,
    debug: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Summarize recent QB production per team and save a projection table.

    Filters stats to quarterbacks, aggregates rolling windows, and ensures only
    active roster members (when roster info is available) are emitted.
    """
    if not save_path:

        return None

    if pred_games is None or pred_games.empty:

        return _write_player_projection_csv(pd.DataFrame(), save_path, QB_PLAYER_PROJECTION_COLUMNS, "QB", debug=debug)

    if stats is None or stats.empty:

        if debug:

            print("[predict][players] no player stats available for QB projections")

        return _write_player_projection_csv(pd.DataFrame(), save_path, QB_PLAYER_PROJECTION_COLUMNS, "QB", debug=debug)



    position_group = _column_or_default(stats, "position_group", default_value="", dtype="object")

    qb_stats = stats[position_group.fillna("").isin(["QB"])].copy()

    if qb_stats.empty:

        if debug:

            print("[predict][players] no QB stats available")

        return _write_player_projection_csv(pd.DataFrame(), save_path, QB_PLAYER_PROJECTION_COLUMNS, "QB", debug=debug)



    for col in [

        "passing_yards",

        "passing_tds",

        "rushing_tds",

        "attempts",

        "interceptions",

        "completions",

        "rushing_yards",

    ]:

        if col not in qb_stats.columns:

            qb_stats[col] = 0.0



    active_statuses: Optional[Set[str]] = {"ACT"} if (roster is not None and not roster.empty and "status" in roster.columns) else None

    active_cache: Dict[Tuple[int, str], Optional[Set[str]]] = {}

    

    results: List[Dict[str, Any]] = []

    grouped = pred_games.groupby(["season", "week"])

    for (season, week), games in grouped:

        subset = _select_stats_window(qb_stats, int(season), int(week))

        if subset.empty:

            continue

        subset = subset[subset["recent_team"].notna() & (subset["recent_team"] != "")]

        if subset.empty:

            continue

        subset["season_week"] = subset["season"].astype(str) + "_" + subset["week"].astype(str)

        agg = subset.groupby(["player_id", "recent_team"], as_index=False).agg(

            player_name=("player_name", "last"),

            games_played=("season_week", "nunique"),

            passing_yards=("passing_yards", "sum"),

            passing_tds=("passing_tds", "sum"),

            rushing_tds=("rushing_tds", "sum"),

            passing_attempts=("attempts", "sum"),

            interceptions=("interceptions", "sum"),

            completions=("completions", "sum"),

            rushing_yards=("rushing_yards", "sum"),

        )

        if agg.empty:

            continue

        last_season = subset.groupby("player_id")["season"].max()

        agg["last_season"] = pd.to_numeric(agg["player_id"].map(last_season), errors="coerce")

        last_team = (

            subset.sort_values(["season", "week"])

            .groupby("player_id")

            .tail(1)

            .set_index("player_id")["recent_team"]

        )

        agg["last_team"] = agg["player_id"].map(last_team).astype(str).str.upper()

        recent_threshold = _recent_player_stat_threshold(subset, int(season), "qb", debug=debug)

        agg = agg[agg["last_season"].notna() & (agg["last_season"] >= recent_threshold)]

        agg = agg[agg["recent_team"] == agg["last_team"]]

        if agg.empty:

            continue

        agg["season"] = season

        agg = _attach_roster_info(agg, roster, int(season), team_col="recent_team")

        agg.drop(columns=["last_season", "last_team"], inplace=True, errors="ignore")

        gp = agg["games_played"].replace({0: pd.NA})

        agg["projected_passing_yards"] = (agg["passing_yards"] / gp).astype(float)

        agg["projected_passing_tds"] = (agg["passing_tds"] / gp).astype(float)

        agg["projected_rushing_tds"] = (agg["rushing_tds"] / gp).astype(float)

        agg["projected_passing_attempts"] = (agg["passing_attempts"] / gp).astype(float)

        agg["projected_interceptions"] = (agg["interceptions"] / gp).astype(float)

        agg["projected_completions"] = (agg["completions"] / gp).astype(float)

        agg["projected_rushing_yards"] = (agg["rushing_yards"] / gp).astype(float)



        for _, game in games.iterrows():

            game_id = game.get("game_id")

            for side in ("home", "away"):

                team = str(game.get(f"{side}_team", "")).upper()

                if not team:

                    continue

                players = agg[agg["recent_team"] == team].copy()

                if roster is not None and not roster.empty:

                    key = (int(season), team)

                    if key not in active_cache:

                        active_cache[key] = _active_player_ids(roster, int(season), team, active_statuses)

                    valid_ids = active_cache.get(key)

                    if valid_ids is None:

                        if debug:

                            print(f"[predict][players][qb] missing roster data for {season} {team}, skipping")

                        continue

                    players = players[players["player_id"].isin(valid_ids)]

                if players.empty:

                    continue

                players = players.sort_values(

                    ["projected_passing_yards", "projected_passing_attempts"],

                    ascending=[False, False]

                )

                for rank, row in enumerate(players.itertuples(index=False), start=1):
                    results.append(
                        {
                            "season": season,
                            "week": week,
                            "game_id": game_id,
                            "kickoff": game.get("kickoff"),
                            "kickoff_mt": game.get("kickoff_mt"),
                            "team": team,
                            "team_side": side,
                            "player_rank": rank,
                            "player_id": row.player_id,
                            "player_name": row.player_name,
                            "jersey_number": row.jersey_number,

                            "games_sampled": row.games_played,

                            "projected_passing_yards": _round_or_none(row.projected_passing_yards, 1),

                            "projected_passing_tds": _round_or_none(row.projected_passing_tds, 2),

                            "projected_rushing_tds": _round_or_none(row.projected_rushing_tds, 2),

                            "projected_passing_attempts": _round_or_none(row.projected_passing_attempts, 2),

                            "projected_interceptions": _round_or_none(row.projected_interceptions, 2),

                            "projected_completions": _round_or_none(row.projected_completions, 2),

                            "projected_rushing_yards": _round_or_none(row.projected_rushing_yards, 1),

                        }

                    )

    if not results:
        if debug:
            print("[predict][players][qb] no projection rows survived filters; writing empty current-week file")
        return _write_player_projection_csv(pd.DataFrame(), save_path, QB_PLAYER_PROJECTION_COLUMNS, "QB", debug=debug)

    df_qb = pd.DataFrame(results)
    return _write_player_projection_csv(df_qb, save_path, QB_PLAYER_PROJECTION_COLUMNS, "QB", debug=debug)







def _player_offense_predictions(
    pred_games: pd.DataFrame,
    save_path: Optional[str],
    stats: pd.DataFrame,
    roster: pd.DataFrame,
    penalties: pd.DataFrame,
    debug: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Export per-team offensive player projections (RB/WR/TE) with penalty context.

    Combines rushing/receiving workload with per-opponent pace so downstream
    tabs can rank players and filter by sample size.
    """
    if not save_path:

        return None

    if pred_games is None or pred_games.empty or stats is None or stats.empty:

        if debug and (stats is None or stats.empty):
            print("[predict][players] no player stats available for offensive projections")
        return _write_player_projection_csv(pd.DataFrame(), save_path, OFFENSE_PLAYER_PROJECTION_COLUMNS, "offensive", debug=debug)



    penalties_available = penalties is not None and not penalties.empty



    position_group = _column_or_default(stats, "position_group", default_value="", dtype="object")

    offense_stats = stats[position_group.fillna("").isin(["RB", "WR", "TE"])].copy()

    if offense_stats.empty:

        if debug:

            print("[predict][players] no offensive skill-position stats available")

        return _write_player_projection_csv(pd.DataFrame(), save_path, OFFENSE_PLAYER_PROJECTION_COLUMNS, "offensive", debug=debug)



    needed_cols = [

        "rushing_yards",

        "rushing_tds",

        "carries",

        "receiving_yards",

        "receiving_tds",

        "receptions",

        "targets",

    ]

    for col in needed_cols:

        if col not in offense_stats.columns:

            offense_stats[col] = 0.0



    active_statuses: Optional[Set[str]] = {"ACT"} if (roster is not None and not roster.empty and "status" in roster.columns) else None

    active_cache: Dict[Tuple[int, str], Optional[Set[str]]] = {}

    roster_cache: Dict[Tuple[int, str], pd.DataFrame] = {}

    allowed_positions: Set[str] = {"RB", "WR", "TE", "FB"}



    results: List[Dict[str, Any]] = []

    grouped = pred_games.groupby(["season", "week"])

    for (season, week), games in grouped:

        subset = _select_stats_window(offense_stats, int(season), int(week))

        penalty_stats = pd.DataFrame()

        if penalties_available:

            penalty_subset = _select_penalty_window(penalties, int(season), int(week))

            penalty_stats = _aggregate_player_penalties(penalty_subset)

            if not penalty_stats.empty:

                penalty_stats["team"] = penalty_stats["team"].astype(str).str.upper()

        if subset.empty:

            continue

        subset = subset[subset["recent_team"].notna() & (subset["recent_team"] != "")]

        if subset.empty:

            continue

        subset["season_week"] = subset["season"].astype(str) + "_" + subset["week"].astype(str)

        agg = subset.groupby(["player_id", "recent_team"], as_index=False).agg(

            player_name=("player_name", "last"),

            games_played=("season_week", "nunique"),

            rushing_yards=("rushing_yards", "sum"),

            rushing_tds=("rushing_tds", "sum"),

            carries=("carries", "sum"),

            receiving_yards=("receiving_yards", "sum"),

            receiving_tds=("receiving_tds", "sum"),

            receptions=("receptions", "sum"),

            targets=("targets", "sum"),

        )

        if agg.empty:

            continue

        last_season = subset.groupby("player_id")["season"].max()

        agg["last_season"] = pd.to_numeric(agg["player_id"].map(last_season), errors="coerce")

        last_team = (

            subset.sort_values(["season", "week"])

            .groupby("player_id")

            .tail(1)

            .set_index("player_id")["recent_team"]

        )

        agg["last_team"] = agg["player_id"].map(last_team).astype(str).str.upper()

        recent_threshold = _recent_player_stat_threshold(subset, int(season), "offense", debug=debug)

        agg = agg[agg["last_season"].notna() & (agg["last_season"] >= recent_threshold)]

        agg = agg[agg["recent_team"] == agg["last_team"]]

        if agg.empty:

            continue

        agg["season"] = season

        agg = _attach_roster_info(agg, roster, int(season), team_col="recent_team")

        agg.drop(columns=["last_season", "last_team"], inplace=True, errors="ignore")

        gp = agg["games_played"].replace({0: pd.NA})

        agg["projected_rushing_yards"] = (agg["rushing_yards"] / gp).astype(float)

        agg["projected_rushing_tds"] = (agg["rushing_tds"] / gp).astype(float)

        agg["projected_carries"] = (agg["carries"] / gp).astype(float)

        agg["projected_receiving_yards"] = (agg["receiving_yards"] / gp).astype(float)

        agg["projected_receiving_tds"] = (agg["receiving_tds"] / gp).astype(float)

        agg["projected_receptions"] = (agg["receptions"] / gp).astype(float)

        agg["projected_targets"] = (agg["targets"] / gp).astype(float)

        agg["projected_total_yards"] = agg["projected_rushing_yards"].fillna(0) + agg["projected_receiving_yards"].fillna(0)

        agg["projected_total_tds"] = agg["projected_rushing_tds"].fillna(0) + agg["projected_receiving_tds"].fillna(0)



        for _, game in games.iterrows():

            game_id = game.get("game_id")

            for side in ("home", "away"):

                team = str(game.get(f"{side}_team", "")).upper()

                if not team:

                    continue

                players = agg[agg["recent_team"] == team].copy()

                roster_details = pd.DataFrame()

                if roster is not None and not roster.empty:

                    key = (int(season), team)

                    if key not in active_cache:

                        active_cache[key] = _active_player_ids(roster, int(season), team, active_statuses)

                    if key not in roster_cache:

                        roster_cache[key] = _active_roster_records(roster, int(season), team, active_statuses, allowed_positions)

                    valid_ids = active_cache.get(key)

                    roster_details = roster_cache.get(key, pd.DataFrame())

                    skill_ids = set(roster_details["player_id"]) if roster_details is not None and not roster_details.empty else set()

                    if not valid_ids:

                        if debug:

                            print(f"[predict][players][offense] missing roster data for {season} {team}, skipping")

                        continue

                    valid_ids = set(valid_ids)

                    if skill_ids:

                        valid_ids &= skill_ids

                    if not valid_ids:

                        if debug:

                            print(f"[predict][players][offense] no ACT skill players for {season} {team}, skipping")

                        continue

                    players = players[players["player_id"].isin(valid_ids)]

                    if not roster_details.empty:

                        players = players.merge(

                            roster_details[["player_id", "player_name", "jersey_number"]],

                            on="player_id",

                            how="left",

                            suffixes=("", "_roster"),

                        )

                        if "player_name_roster" in players.columns:

                            name_fill_mask = players["player_name"].isna() | (players["player_name"] == "")

                            if name_fill_mask.any():

                                players.loc[name_fill_mask, "player_name"] = players.loc[name_fill_mask, "player_name_roster"]

                            players.drop(columns=["player_name_roster"], inplace=True)

                        if "jersey_number_roster" in players.columns:

                            jersey_fill_mask = players["jersey_number"].isna()

                            if jersey_fill_mask.any():

                                players.loc[jersey_fill_mask, "jersey_number"] = players.loc[jersey_fill_mask, "jersey_number_roster"]

                            players.drop(columns=["jersey_number_roster"], inplace=True)

                if not penalty_stats.empty:

                    team_penalties = penalty_stats[penalty_stats["team"] == team]

                    if not team_penalties.empty:

                        players = players.merge(

                            team_penalties[

                                [

                                    "player_id",

                                    "penalty_count",

                                    "penalty_yards",

                                    "penalty_games",

                                    "penalty_rate_per_game",

                                    "penalty_player_name",

                                ]

                            ],

                            on="player_id",

                            how="left",

                        )

                        if "penalty_player_name" in players.columns:

                            name_fill_mask = players["player_name"].isna() | (players["player_name"] == "")

                            if name_fill_mask.any():

                                players.loc[name_fill_mask, "player_name"] = players.loc[name_fill_mask, "penalty_player_name"]

                            players.drop(columns=["penalty_player_name"], inplace=True)

                if players.empty:

                    if debug:

                        print(f"[predict][players][offense] no projection data for {season} {team}, skipping")

                    continue

                if not roster_details.empty and len(players) < 10:

                    missing = roster_details[~roster_details["player_id"].isin(players["player_id"])]

                    if not missing.empty:

                        filler_needed = 10 - len(players)

                        filler_rows: List[Dict[str, Any]] = []

                        for filler in missing.itertuples(index=False):

                            filler_rows.append(

                                {

                                    "player_id": filler.player_id,

                                    "recent_team": team,

                                    "player_name": filler.player_name,

                                    "jersey_number": filler.jersey_number,

                                    "games_played": 0,

                                    "projected_rushing_yards": 0.0,

                                    "projected_rushing_tds": 0.0,

                                    "projected_carries": 0.0,

                                    "projected_receiving_yards": 0.0,

                                    "projected_receiving_tds": 0.0,

                                    "projected_receptions": 0.0,

                                    "projected_targets": 0.0,

                                    "projected_total_yards": 0.0,

                                    "projected_total_tds": 0.0,

                                }

                            )

                            if len(filler_rows) >= filler_needed:

                                break

                        if filler_rows:

                            filler_df = pd.DataFrame(filler_rows)

                            players = pd.concat([players, filler_df], ignore_index=True, sort=False)

                for col, fill_value in (

                    ("penalty_count", 0),

                    ("penalty_games", 0),

                    ("penalty_yards", 0.0),

                    ("penalty_rate_per_game", 0.0),

                ):

                    if col not in players.columns:

                        players[col] = fill_value

                    else:

                        players[col] = players[col].fillna(fill_value)

                players = players.sort_values(

                    ["projected_total_yards", "projected_total_tds", "projected_receptions"],

                    ascending=[False, False, False]

                ).head(10)

                if len(players) < 10 and debug:

                    print(f"[predict][players][offense] only {len(players)} ACT skill players found for {season} {team}")

                for rank, row in enumerate(players.itertuples(index=False), start=1):
                    results.append(
                        {
                            "season": season,
                            "week": week,
                            "game_id": game_id,
                            "kickoff": game.get("kickoff"),
                            "kickoff_mt": game.get("kickoff_mt"),
                            "team": team,
                            "team_side": side,
                            "player_rank": rank,
                            "player_id": row.player_id,
                            "player_name": row.player_name,
                            "jersey_number": row.jersey_number,

                            "games_sampled": row.games_played,

                            "projected_rushing_yards": _round_or_none(row.projected_rushing_yards, 1),

                            "projected_rushing_tds": _round_or_none(row.projected_rushing_tds, 2),

                            "projected_carries": _round_or_none(row.projected_carries, 2),

                            "projected_receiving_yards": _round_or_none(row.projected_receiving_yards, 1),

                            "projected_receiving_tds": _round_or_none(row.projected_receiving_tds, 2),

                            "projected_receptions": _round_or_none(row.projected_receptions, 2),

                            "projected_targets": _round_or_none(row.projected_targets, 2),

                            "projected_total_yards": _round_or_none(row.projected_total_yards, 1),

                            "projected_total_tds": _round_or_none(row.projected_total_tds, 2),

                            "penalties_per_game": _round_or_none(getattr(row, "penalty_rate_per_game", pd.NA), 3),

                            "penalty_count": None if pd.isna(getattr(row, "penalty_count", pd.NA)) else int(getattr(row, "penalty_count", 0)),

                            "penalty_yards": _round_or_none(getattr(row, "penalty_yards", pd.NA), 1),

                        }

                    )

    if not results:
        if debug:
            print("[predict][players][offense] no projection rows survived filters; writing empty current-week file")
        return _write_player_projection_csv(pd.DataFrame(), save_path, OFFENSE_PLAYER_PROJECTION_COLUMNS, "offensive", debug=debug)

    df_off = pd.DataFrame(results)
    return _write_player_projection_csv(df_off, save_path, OFFENSE_PLAYER_PROJECTION_COLUMNS, "offensive", debug=debug)





def _select_pbp_window(pbp: pd.DataFrame, season: int, week: Optional[int]) -> pd.DataFrame:
    current = pd.DataFrame()
    if week is not None:
        try:

            week_int = int(week)

        except Exception:

            week_int = None

        if week_int is not None:

            current = pbp[(pbp["season"] == season) & (pbp["week"] < week_int)].copy()

    else:

        current = pbp[pbp["season"] == season].copy()



    # Include all prior seasons to stabilize per-player rates without leaking future weeks.
    history = pbp[pbp["season"] < season].copy()
    subset = pd.concat([history, current], ignore_index=True)

    return subset





def _player_defense_predictions(
    pred_games: pd.DataFrame,
    save_path: Optional[str],
    roster: pd.DataFrame,
    penalties: pd.DataFrame,
    debug: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Generate defensive front projections (sacks, hits, TFL) using PBP-derived stats.

    Returns a CSV-aligned DataFrame so Streamlit can show per-team leaders with
    optional penalty context.
    """
    if not save_path:

        return None

    if pred_games is None or pred_games.empty:

        return None



    penalties_available = penalties is not None and not penalties.empty



    pbp_path = RAW_DIR / "nfl_pbp.parquet"

    if not pbp_path.exists():

        if debug:

            print("[predict][players] PBP file missing; cannot build defensive projections")

        return None



    pbp = read_df(pbp_path)
    if pbp.empty:

        return None



    pbp = pbp.copy()

    pbp["season"] = pd.to_numeric(_column_or_default(pbp, "season"), errors="coerce")

    pbp["week"] = pd.to_numeric(_column_or_default(pbp, "week"), errors="coerce")

    pbp = pbp[pd.notna(pbp["season"]) & pd.notna(pbp["week"])]

    if pbp.empty:

        return None

    pbp["season"] = pbp["season"].astype("int64")

    pbp["week"] = pbp["week"].astype("int64")

    season_type = _column_or_default(pbp, "season_type", default_value="REG", dtype="object")

    pbp = pbp[season_type.fillna("REG").astype(str) == "REG"]

    defteam = _column_or_default(pbp, "defteam", default_value="", dtype="object")

    pbp["defteam"] = defteam.fillna("").astype(str).str.upper()

    pbp = pbp[pbp["defteam"] != ""]



    active_statuses: Optional[Set[str]] = {"ACT"} if (roster is not None and not roster.empty and "status" in roster.columns) else None

    active_cache: Dict[Tuple[int, str], Optional[Set[str]]] = {}



    stats_columns = [

        "sack",

        "sack_player_id",

        "sack_player_name",

        "half_sack_1_player_id",

        "half_sack_1_player_name",

        "half_sack_2_player_id",

        "half_sack_2_player_name",

        "qb_hit_1_player_id",

        "qb_hit_1_player_name",

        "qb_hit_2_player_id",

        "qb_hit_2_player_name",

        "tackle_for_loss_1_player_id",

        "tackle_for_loss_1_player_name",

        "tackle_for_loss_2_player_id",

        "tackle_for_loss_2_player_name",

        "punt_blocked",

        "blocked_player_id",

        "blocked_player_name",

        "penalty",

        "penalty_team",

        "penalty_player_id",

        "penalty_player_name",

        "penalty_yards",

    ]

    keep_cols = ["season", "week", "defteam", "yards_gained"] + [c for c in stats_columns if c in pbp.columns]

    pbp = pbp[keep_cols]



    results: List[Dict[str, Any]] = []

    grouped = pred_games.groupby(["season", "week"])

    for (season, week), games in grouped:

        subset = _select_pbp_window(pbp, int(season), int(week))

        penalty_stats = pd.DataFrame()

        if penalties_available:

            penalty_subset = _select_penalty_window(penalties, int(season), int(week))

            penalty_stats = _aggregate_player_penalties(penalty_subset)

            if not penalty_stats.empty:

                penalty_stats["team"] = penalty_stats["team"].astype(str).str.upper()

        if subset.empty:

            continue



        recent_threshold = int(season) - 1



        player_stats: Dict[Tuple[str, str], Dict[str, Any]] = {}



        def _update(

            player_id: Optional[str],

            player_name: Optional[str],

            team: str,

            stat: str,

            value: float,

            loss: float,

            game_key: Tuple[int, int],

        ) -> None:

            if not player_id and not player_name:

                return

            pid = str(player_id) if player_id not in (None, "", "nan") else None

            pname = str(player_name) if player_name not in (None, "", "nan") else None

            key = pid or pname

            if key is None:

                return

            entry_key = (team, key)

            entry = player_stats.setdefault(

                entry_key,

                {

                    "team": team,

                    "player_id": pid,

                    "player_name": pname,

                    "games": set(),

                    "sacks": 0.0,

                    "qb_hits": 0.0,

                    "tfl": 0.0,

                    "blocked_punts": 0.0,

                    "loss_yards": 0.0,

                },

            )

            if pid and not entry.get("player_id"):

                entry["player_id"] = pid

            if pname and not entry.get("player_name"):

                entry["player_name"] = pname

            entry["games"].add(game_key)

            if stat:

                entry[stat] += value

            if loss:

                entry["loss_yards"] += loss



        for row in subset.itertuples(index=False):

            team = getattr(row, "defteam", "")

            if not team:

                continue

            game_key = (getattr(row, "season"), getattr(row, "week"))

            yards_gained = getattr(row, "yards_gained", 0) or 0

            loss_yards = -yards_gained if yards_gained < 0 else 0.0



            if getattr(row, "sack", 0):

                _update(getattr(row, "sack_player_id", None), getattr(row, "sack_player_name", None), team, "sacks", 1.0, loss_yards, game_key)

            if getattr(row, "half_sack_1_player_id", None):

                _update(getattr(row, "half_sack_1_player_id", None), getattr(row, "half_sack_1_player_name", None), team, "sacks", 0.5, loss_yards / 2 if loss_yards else 0.0, game_key)

            if getattr(row, "half_sack_2_player_id", None):

                _update(getattr(row, "half_sack_2_player_id", None), getattr(row, "half_sack_2_player_name", None), team, "sacks", 0.5, loss_yards / 2 if loss_yards else 0.0, game_key)

            if getattr(row, "qb_hit_1_player_id", None):

                _update(getattr(row, "qb_hit_1_player_id", None), getattr(row, "qb_hit_1_player_name", None), team, "qb_hits", 1.0, 0.0, game_key)

            if getattr(row, "qb_hit_2_player_id", None):

                _update(getattr(row, "qb_hit_2_player_id", None), getattr(row, "qb_hit_2_player_name", None), team, "qb_hits", 1.0, 0.0, game_key)

            if getattr(row, "tackle_for_loss_1_player_id", None):

                _update(getattr(row, "tackle_for_loss_1_player_id", None), getattr(row, "tackle_for_loss_1_player_name", None), team, "tfl", 1.0, loss_yards, game_key)

            if getattr(row, "tackle_for_loss_2_player_id", None):

                _update(getattr(row, "tackle_for_loss_2_player_id", None), getattr(row, "tackle_for_loss_2_player_name", None), team, "tfl", 1.0, loss_yards, game_key)

            if getattr(row, "punt_blocked", 0) and getattr(row, "blocked_player_id", None):

                _update(getattr(row, "blocked_player_id", None), getattr(row, "blocked_player_name", None), team, "blocked_punts", 1.0, 0.0, game_key)



        if not player_stats:

            continue



        records = []

        for entry in player_stats.values():

            pid = entry.get("player_id")

            if not pid:

                continue

            if not entry.get("games"):

                continue

            last_season = max((g[0] for g in entry["games"]), default=None)

            if last_season is None or last_season < recent_threshold:

                continue

            team_current = entry.get("team")

            if not team_current:

                continue

            games_played = len(entry["games"]) or 1

            records.append(

                {

                    "player_id": pid,

                    "player_name": entry.get("player_name"),

                    "team": team_current,

                    "games_played": games_played,

                    "projected_sacks": entry["sacks"] / games_played,

                    "projected_qb_hits": entry["qb_hits"] / games_played,

                    "projected_tfl": entry["tfl"] / games_played,

                    "projected_blocked_punts": entry["blocked_punts"] / games_played,

                    "projected_loss_yards": entry["loss_yards"] / games_played,

                }

            )



        if not records:

            continue

        agg = pd.DataFrame(records)

        agg["season"] = season

        agg = _attach_roster_info(agg, roster, int(season), team_col="team", fill_name=True)



        for _, game in games.iterrows():

            game_id = game.get("game_id")

            for side in ("home", "away"):

                team = str(game.get(f"{side}_team", "")).upper()

                if not team:

                    continue

                players = agg[agg["team"] == team].copy()

                if roster is not None and not roster.empty:

                    key = (int(season), team)

                    if key not in active_cache:

                        active_cache[key] = _active_player_ids(roster, int(season), team, active_statuses)

                    valid_ids = active_cache.get(key)

                    if valid_ids is None:

                        if debug:

                            print(f"[predict][players][defense] missing roster data for {season} {team}, skipping")

                        continue

                    players = players[players["player_id"].isin(valid_ids)]

                if players.empty:

                    continue

                if not penalty_stats.empty:

                    team_penalties = penalty_stats[penalty_stats["team"] == team]

                    if not team_penalties.empty:

                        players = players.merge(

                            team_penalties[

                                [

                                    "player_id",

                                    "penalty_count",

                                    "penalty_yards",

                                    "penalty_games",

                                    "penalty_rate_per_game",

                                    "penalty_player_name",

                                ]

                            ],

                            on="player_id",

                            how="left",

                        )

                        if "penalty_player_name" in players.columns:

                            name_fill_mask = players["player_name"].isna() | (players["player_name"] == "")

                            if name_fill_mask.any():

                                players.loc[name_fill_mask, "player_name"] = players.loc[name_fill_mask, "penalty_player_name"]

                            players.drop(columns=["penalty_player_name"], inplace=True)

                for col, fill_value in (

                    ("penalty_count", 0),

                    ("penalty_games", 0),

                    ("penalty_yards", 0.0),

                    ("penalty_rate_per_game", 0.0),

                ):

                    if col not in players.columns:

                        players[col] = fill_value

                    else:

                        players[col] = players[col].fillna(fill_value)

                players = players.sort_values(

                    ["projected_sacks", "projected_qb_hits", "projected_tfl"],

                    ascending=[False, False, False]

                ).head(10)

                for rank, row in enumerate(players.itertuples(index=False), start=1):
                    results.append(
                        {
                            "season": season,
                            "week": week,
                            "game_id": game_id,
                            "kickoff": game.get("kickoff"),
                            "kickoff_mt": game.get("kickoff_mt"),
                            "team": team,
                            "team_side": side,
                            "player_rank": rank,
                            "player_id": row.player_id,
                            "player_name": row.player_name,
                            "jersey_number": row.jersey_number,

                            "games_sampled": row.games_played,

                            "projected_sacks": _round_or_none(row.projected_sacks, 2),

                            "projected_qb_hits": _round_or_none(row.projected_qb_hits, 2),

                            "projected_tfl": _round_or_none(row.projected_tfl, 2),

                            "projected_blocked_punts": _round_or_none(row.projected_blocked_punts, 2),

                            "projected_loss_yards": _round_or_none(row.projected_loss_yards, 2),

                            "penalties_per_game": _round_or_none(getattr(row, "penalty_rate_per_game", pd.NA), 3),

                            "penalty_count": None if pd.isna(getattr(row, "penalty_count", pd.NA)) else int(getattr(row, "penalty_count", 0)),

                            "penalty_yards": _round_or_none(getattr(row, "penalty_yards", pd.NA), 1),

                        }

                    )

    if not results:

        return None

    df_def = pd.DataFrame(results)

    _backup_existing_file(Path(save_path))

    df_def.to_csv(save_path, index=False)

    if debug:

        print(f"[predict][players] saved defensive projections -> {save_path} (rows={len(df_def)})")

    return df_def



def main():
    """CLI entry point for generating upcoming team and player predictions."""
    ap = argparse.ArgumentParser()

    ap.add_argument("--season", type=int, required=False, help="Season year to predict. Defaults to latest season in data.")

    ap.add_argument("--week", type=str, default="auto", help="Week number or 'auto' to select current/next regular-season week for the chosen season.")

    ap.add_argument("--save", type=str, default="predictions.csv")

    ap.add_argument("--dump-all", dest="dump_all", type=str, default=None, help="Optional path to save the full, uncurated predictions CSV")

    ap.add_argument("--debug", action="store_true", help="Enable verbose debug output")

    ap.add_argument(
        "--enable-shap-explanations",
        action="store_true",
        help="Enable optional per-game SHAP explanations during prediction.",
    )

    ap.add_argument("--save-players-qb", dest="save_players_qb", type=str, default="predictions_players_qb.csv", help="Path to save quarterback projections; set empty to skip")

    ap.add_argument("--save-players-offense", dest="save_players_offense", type=str, default="predictions_players_offense.csv", help="Path to save offensive skill-player projections (top 10 per team); set empty to skip")

    ap.add_argument("--save-players-defense", dest="save_players_defense", type=str, default="predictions_players_defense.csv", help="Path to save defensive projections; set empty to skip")

    ap.add_argument(

        "--volatility-artifact",

        type=str,

        default=str(DEFAULT_VOLATILITY_ARTIFACT),

        help="Path to trained volatility classifier artifact.",

    )

    ap.add_argument(

        "--volatility-threshold",

        type=float,

        default=None,

        help="Override volatility probability threshold (defaults to artifact value).",

    )

    ap.add_argument(

        "--volatility-strength",

        type=float,

        default=0.35,

        help="Shrinkage strength (0-1) towards 0.5 for high-volatility win probabilities.",

    )

    ap.add_argument(

        "--volatility-margin-strength",

        type=float,

        default=0.45,

        help="Shrinkage strength (0-1) towards 0 margin for high-volatility spreads.",

    )

    ap.add_argument(

        "--volatility-min-coverage",

        type=float,

        default=0.02,

        help="Skip volatility shrinkage when fewer than this share of the slate is labeled volatile.",

    )

    ap.add_argument(

        "--volatility-max-coverage",

        type=float,

        default=0.85,

        help="Skip volatility shrinkage when more than this share of the slate is labeled volatile.",

    )

    ap.add_argument(

        "--disable-volatility",

        action="store_true",

        help="Skip volatility-based adjustments even if an artifact is available.",

    )

    args = ap.parse_args()



    setup_logging("nfl_predictions", level="DEBUG" if args.debug else "INFO")
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

    volatility_artifact_path = Path(args.volatility_artifact) if args.volatility_artifact else DEFAULT_VOLATILITY_ARTIFACT



    sched = read_df(RAW_DIR / "espn_schedule.parquet")
    feats = read_df(PROC_DIR / "matchup_features.parquet")

    def _available_seasons(df: pd.DataFrame) -> list[int]:

        if df is None or df.empty or "season" not in df.columns:

            return []

        values = df["season"].dropna()

        if values.empty:

            return []

        return sorted({int(v) for v in values.astype(int).tolist()})

    schedule_seasons = _available_seasons(sched)

    feature_seasons = _available_seasons(feats)

    available_seasons = sorted(set(schedule_seasons) | set(feature_seasons))

    if not available_seasons:

        raise RuntimeError("No seasons found in schedule or feature data; cannot run predictions.")

    # Choose season: prefer arg, else latest available in artifacts (with fallback if missing)
    if args.season is not None:

        requested_season = int(args.season)

        if requested_season in available_seasons:

            season = requested_season

        else:

            fallback_season = available_seasons[-1]

            print(

                f"[predict][season] requested season {requested_season} not available; "

                f"falling back to {fallback_season}"

            )

            season = fallback_season

    else:

        season = available_seasons[-1]



    # Restrict to the chosen season and regular season (season_type==2) where possible

    sched_season = sched[sched["season"] == season].copy()

    if "season_type" in sched_season.columns:

        sched_regular = sched_season[sched_season["season_type"] == 2].copy()

        if sched_regular.empty and not sched_season.empty:

            if args.debug:

                print(f"[predict][schedule] no season_type==2 rows for season {season}; using all games")

            sched_regular = sched_season.copy()

    else:

        sched_regular = sched_season.copy()



    # Determine target week

    if args.week == "auto":

        # If dates available, try to find the smallest week that is upcoming or the max completed week

        week_candidates = pd.Series(dtype="Int64")

        if not sched_regular.empty and "week" in sched_regular.columns:

            week_candidates = sched_regular["week"].dropna().astype(int)

        if week_candidates.empty and not feats.empty and {"season", "week"}.issubset(feats.columns):

            feat_weeks = feats.loc[(feats["season"] == season) & feats["week"].notna(), "week"]

            if not feat_weeks.empty:

                week_candidates = feat_weeks.astype(int)

        if week_candidates.empty:

            raise RuntimeError(

                f"No valid week information found for season {season}; "

                "please rebuild schedule/features or pass --week explicitly."

            )

        if not sched_regular.empty and "date" in sched_regular.columns:

            # Parse dates and select the nearest upcoming or current week

            # Use timezone-aware UTC timestamp to match parsed schedule datetimes

            now = pd.Timestamp.now(tz="UTC")

            sched_regular["_dt"] = pd.to_datetime(sched_regular["date"], errors="coerce", utc=True)

            # Prefer next upcoming; if none, fall back to max week in this season

            upcoming_weeks = sched_regular.loc[sched_regular["_dt"] >= now, "week"].astype("Int64")

            if upcoming_weeks.notna().any():

                week = int(upcoming_weeks.min())

            else:

                week = int(week_candidates.max())

            sched_regular.drop(columns=["_dt"], inplace=True, errors="ignore")

        else:

            week = int(week_candidates.max())

    else:

        week = int(args.week)



    penalty_frame = _load_penalty_frame()



    # Build the feature slice for that season/week and compute *_diff features

    this_week = feats[(feats["season"] == season) & (feats["week"] == week)].copy()
    if this_week.empty:
        raise RuntimeError(
            f"No feature rows found for season {season}, week {week}. "
            "Run the data/feature pipeline for that season/week before predicting."
        )

    this_week = _make_feature_diffs(this_week)
    if "game_id" in this_week.columns and this_week.duplicated("game_id").any():
        dupes = this_week.loc[this_week.duplicated("game_id", keep=False), "game_id"].dropna().unique()
        print(f"[predict][schedule] dropping duplicate feature rows for game_id(s): {list(dupes[:5])}")
        this_week = this_week.drop_duplicates("game_id", keep="last").reset_index(drop=True)

    # Deduplicate columns to avoid duplicate-name DataFrame selections downstream

    try:

        this_week_nodup = this_week.loc[:, ~pd.Index(this_week.columns).duplicated()]

    except Exception:

        this_week_nodup = this_week

    # Load models if present

    preds = this_week[["season","week","home_team","away_team"]].copy()

    preds["prediction_source"] = "upcoming"

    # Optionally add game identifiers and date if available

    date_col = next((c for c in ["date", "start_date"] if c in this_week.columns), None)

    if date_col is not None and date_col not in preds.columns:

        preds["date"] = this_week[date_col]

    if "game_id" in this_week.columns and "game_id" not in preds.columns:

        preds["game_id"] = this_week["game_id"]

    if "game_uid" in this_week.columns and "game_uid" not in preds.columns:

        preds["game_uid"] = this_week["game_uid"]



    # Include weather fields if available (kickoff, roof, and all weather_* columns)

    try:

        wx_cols = [

            c for c in this_week.columns

            if (isinstance(c, str) and (c.startswith("weather_") or c in ("kickoff", "roof", "venue_lat", "venue_lon", "roof_is_dome")))

        ]

        if wx_cols:

            preds = pd.concat([preds, this_week[wx_cols].copy()], axis=1)

    except Exception:

        pass



    if "kickoff" in preds.columns and preds["kickoff"].notna().any():

        preds["kickoff_mt"] = _format_mountain_time(preds["kickoff"])

    elif "date" in preds.columns and preds["date"].notna().any():

        preds["kickoff_mt"] = _format_mountain_time(preds["date"])

    else:

        preds["kickoff_mt"] = ""



    stadium_cols = [

        "home_altitude_ft",

        "away_altitude_ft",

        "altitude_ft_diff",

        "home_crowd_noise_score",

        "away_crowd_noise_score",

        "crowd_noise_score_diff",

        "home_fan_hostility_score",

        "away_fan_hostility_score",

        "fan_hostility_score_diff",

        "home_weather_snow_index",

        "away_weather_snow_index",

        "weather_snow_index_diff",

        "home_weather_rain_index",

        "away_weather_rain_index",

        "weather_rain_index_diff",

        "home_indoor",

        "away_indoor",

        "indoor_diff",

        "rivalry_intensity",

        "rivalry_is_divisional",

        "rivalry_has_historic_component",

    ]

    context_cols = [c for c in stadium_cols if c in this_week.columns]

    if context_cols:

        preds = pd.concat([preds, this_week[context_cols].copy()], axis=1)



    try:

        win_art = joblib.load(Path("models/winprob_gb.pkl"))

        win_model = win_art.get("model", win_art)

        win_feats = win_art.get("features", [])

        feat_ranges = win_art.get("feature_ranges", {})

        logistic_model = win_art.get("logistic_model")

        logistic_feats = win_art.get("logistic_features", [])

        ensemble_weight = float(win_art.get("ensemble_weight", 1.0))

        use_cols = [c for c in win_feats if c in this_week_nodup.columns]

        if use_cols:

            Xw = this_week_nodup[use_cols].copy()

            # Clip to training ranges if provided

            if isinstance(feat_ranges, dict) and feat_ranges:

                for c in use_cols:

                    if c in feat_ranges:

                        lo, hi = feat_ranges[c]

                        Xw[c] = pd.to_numeric(Xw[c], errors="coerce").clip(lower=lo, upper=hi)

            proba_gb = win_model.predict_proba(Xw.values)[:, 1]



            proba_final = proba_gb

            if logistic_model is not None and logistic_feats:

                missing_log_feats = [c for c in logistic_feats if c not in this_week_nodup.columns]

                if not missing_log_feats:

                    X_log = this_week_nodup[logistic_feats].copy()

                    proba_log = logistic_model.predict_proba(X_log.values)[:, 1]

                    proba_final = ensemble_weight * proba_gb + (1.0 - ensemble_weight) * proba_log

                else:

                    print(f"[win_prob][predict] missing logistic features: {missing_log_feats}")



            # Per-game explanation using SHAP (best-effort)

            try:

                if not args.enable_shap_explanations:
                    raise ImportError("SHAP explanations disabled for predict_upcoming")

                import shap  # type: ignore

                # Use underlying estimator if model is calibrated

                base_est = win_model

                try:

                    cc = getattr(win_model, "calibrated_classifiers_", None)

                    if isinstance(cc, list) and len(cc) > 0 and hasattr(cc[0], "estimator"):

                        base_est = cc[0].estimator

                except Exception:

                    pass

                explainer = shap.TreeExplainer(base_est)

                shap_vals = explainer.shap_values(Xw)

                # shap can return list for classifiers; pick positive class if list

                if isinstance(shap_vals, list) and len(shap_vals) >= 2:

                    shap_mat = shap_vals[1]

                else:

                    shap_mat = shap_vals

                # Build supporting/opposing metrics strings

                supp_list = []

                opp_list = []

                for i in range(len(Xw)):

                    row = Xw.iloc[i]

                    sv = shap_mat[i]

                    pairs = list(zip(use_cols, row.values, sv))

                    # sort by absolute contribution

                    pairs.sort(key=lambda t: abs(float(t[2]) if t[2] is not None else 0.0), reverse=True)

                    support = [f"{k}={v:.3g} (Δ={float(s):+.3g})" for k, v, s in pairs if float(s) > 0][:6]

                    oppose = [f"{k}={v:.3g} (Δ={float(s):+.3g})" for k, v, s in pairs if float(s) < 0][:6]

                    supp_list.append("; ".join(support))

                    opp_list.append("; ".join(oppose))

                preds["decision_supporting_metrics"] = supp_list

                preds["decision_opposing_metrics"] = opp_list

                # Confidence explanation

                conf_expl = []

                for p in proba_gb:

                    if pd.isna(p):

                        conf_expl.append("confidence: N/A")

                    elif p >= 0.5:

                        conf_expl.append(f"confidence: {p:.1%} home")

                    else:

                        conf_expl.append(f"confidence: {(1-p):.1%} away")

                preds["decision_confidence_expl"] = conf_expl

            except Exception:

                # Fallback: simple feature-importance based explanation

                try:

                    import numpy as _np  # lightweight

                    # Use base estimator's importances when calibrated

                    base_est = win_model

                    try:

                        cc = getattr(win_model, "calibrated_classifiers_", None)

                        if isinstance(cc, list) and len(cc) > 0 and hasattr(cc[0], "estimator"):

                            base_est = cc[0].estimator

                    except Exception:

                        pass

                    imp = getattr(base_est, "feature_importances_", None)

                    if imp is not None and len(imp) == len(use_cols):

                        imp = _np.array(imp)

                        imp = imp / (imp.sum() + 1e-12)

                        supp_list = []

                        opp_list = []

                        for i in range(len(Xw)):

                            row = Xw.iloc[i].values

                            contrib = row * imp  # sign via feature value; rough heuristic

                            pairs = list(zip(use_cols, row, contrib))

                            pairs.sort(key=lambda t: abs(float(t[2])), reverse=True)

                            support = [f"{k}={v:.3g} (w={float(c):+.3g})" for k, v, c in pairs if float(c) > 0][:6]

                            oppose = [f"{k}={v:.3g} (w={float(c):+.3g})" for k, v, c in pairs if float(c) < 0][:6]

                            supp_list.append("; ".join(support))

                            opp_list.append("; ".join(oppose))

                        preds["decision_supporting_metrics"] = supp_list

                        preds["decision_opposing_metrics"] = opp_list

                        preds["decision_confidence_expl"] = [f"confidence: {p:.1%}" if pd.notna(p) else "confidence: N/A" for p in proba_gb]

                except Exception:

                    # Final fallback: use top-|diff| features to summarize

                    try:

                        supp_list = []

                        opp_list = []

                        for i in range(len(Xw)):

                            row = Xw.iloc[i]

                            pairs = list(zip(use_cols, row.values))

                            pairs.sort(key=lambda t: abs(float(t[1]) if t[1] is not None else 0.0), reverse=True)

                            support = [f"{k}={float(v):+.3g}" for k, v in pairs if float(v) > 0][:6]

                            oppose = [f"{k}={float(v):+.3g}" for k, v in pairs if float(v) < 0][:6]

                            supp_list.append("; ".join(support))

                            opp_list.append("; ".join(oppose))

                        preds["decision_supporting_metrics"] = supp_list

                        preds["decision_opposing_metrics"] = opp_list

                        preds["decision_confidence_expl"] = [

                            (f"confidence: {p:.1%} home" if pd.notna(p) and float(p) >= 0.5 else (f"confidence: {(1-float(p)):.1%} away" if pd.notna(p) else "confidence: N/A"))

                            for p in proba_gb

                        ]

                    except Exception:

                        pass

            cal_candidates = [
                Path("models/winprob_calibrator.pkl"),
                Path("models/isotonic_calibrator.pkl"),
            ]

            cal_path = next((p for p in cal_candidates if p.exists()), None)

            model_raw_proba = _coerce_probability_series(proba_final, preds.index)
            preds["home_win_prob_model_raw"] = model_raw_proba
            final_proba = model_raw_proba.copy()
            cal_art: dict[str, Any] = {}

            if cal_path is not None:

                try:

                    cal_art = joblib.load(cal_path)

                    calibrator = cal_art.get("calibrator")

                    if calibrator is not None:
                        final_proba = _apply_probability_calibrator(calibrator, final_proba)

                except Exception as ex:

                    print(f"[win_prob][predict] calibrator load failed: {ex}")

            market_caps = None

            if "sched_implied_prob_home" in preds.columns:

                market_caps = pd.to_numeric(preds["sched_implied_prob_home"], errors="coerce")

            elif "home_moneyline" in preds.columns:

                market_caps = moneyline_to_prob(preds["home_moneyline"])

            fallback_reasons: list[str] = []
            calibrated_proba = _coerce_probability_series(final_proba, preds.index)
            calibrated_proba, fallback_reason = _fallback_if_probability_collapsed(
                calibrated_proba,
                model_raw_proba,
                stage="calibration",
            )
            if fallback_reason is not None:
                fallback_used = False
                try:
                    fallback_calibrator = cal_art.get("fallback_calibrator") if cal_path is not None else None
                    fallback_method = str(cal_art.get("fallback_method", "fallback")) if cal_path is not None else "fallback"
                    if fallback_calibrator is not None:
                        fallback_proba = _apply_probability_calibrator(fallback_calibrator, model_raw_proba)
                        _, fallback_check_reason = _fallback_if_probability_collapsed(
                            fallback_proba,
                            model_raw_proba,
                            stage="calibration",
                        )
                        if fallback_check_reason is None:
                            calibrated_proba = fallback_proba
                            fallback_reasons.append(f"calibration_{fallback_method}_fallback")
                            fallback_used = True
                            print(
                                "[predict][quality] warning: calibration collapsed probability ranking; "
                                f"using {fallback_method} calibrator fallback for this slate."
                            )
                except Exception as ex:
                    if args.debug:
                        print(f"[predict][quality] calibration fallback failed: {ex}")
                if not fallback_used:
                    fallback_reasons.append(fallback_reason)
                    print(
                        "[predict][quality] warning: calibration collapsed probability ranking; "
                        "using raw model probabilities for this slate."
                    )
            preds["home_win_prob_calibrated"] = calibrated_proba
            final_proba = apply_probability_caps(calibrated_proba, market_caps)
            capped_proba = _coerce_probability_series(final_proba, preds.index)
            capped_proba, fallback_reason = _fallback_if_probability_collapsed(
                capped_proba,
                calibrated_proba,
                stage="caps",
            )
            if fallback_reason is not None:
                fallback_reasons.append(fallback_reason)
                print(
                    "[predict][quality] warning: probability caps collapsed probability ranking; "
                    "using uncapped probabilities for this slate."
                )
            preds["home_win_prob_capped"] = capped_proba
            if fallback_reasons:
                preds["home_win_prob_quality_fallback"] = ";".join(fallback_reasons)

            preds["home_win_prob"] = capped_proba



        else:

            preds["home_win_prob"] = None

    except Exception:

        preds["home_win_prob"] = None

    try:

        spread_art = joblib.load(Path("models/spread_gb.pkl"))

        spread_model = spread_art.get("model", spread_art)

        spread_feats = spread_art.get("features", [])

        spread_imputer = spread_art.get("imputer")

        quantile_models = spread_art.get("quantile_models") or {}

        spread_bias = float(spread_art.get("bias_correction", 0.0))

        use_cols = [c for c in spread_feats if c in this_week_nodup.columns]

        if use_cols:

            X_spread = this_week_nodup[use_cols].values

            if spread_imputer is not None:

                X_spread = spread_imputer.transform(X_spread)

            margin_vals = spread_model.predict(X_spread)

            if spread_bias:

                margin_vals = margin_vals - spread_bias

            preds["pred_home_margin"] = margin_vals

            if quantile_models:

                if "pred_home_margin_lo" not in preds.columns:

                    preds["pred_home_margin_lo"] = pd.NA

                if "pred_home_margin_hi" not in preds.columns:

                    preds["pred_home_margin_hi"] = pd.NA

                if 0.2 in quantile_models:

                    lo_vals = quantile_models[0.2].predict(X_spread)

                    if spread_bias:

                        lo_vals = lo_vals - spread_bias

                    preds.loc[:, "pred_home_margin_lo"] = lo_vals

                if 0.8 in quantile_models:

                    hi_vals = quantile_models[0.8].predict(X_spread)

                    if spread_bias:

                        hi_vals = hi_vals - spread_bias

                    preds.loc[:, "pred_home_margin_hi"] = hi_vals

        else:

            preds["pred_home_margin"] = None

    except Exception:

        preds["pred_home_margin"] = None



    if not args.disable_volatility:

        if volatility_artifact_path.exists():

            try:

                artifact = load_volatility_artifact(volatility_artifact_path)

                artifact_skip_reason = _volatility_artifact_skip_reason(artifact)

                if artifact_skip_reason is not None:

                    win_raw, _ = _best_probability_signal(preds, preds.index)

                    preds["home_win_prob_raw"] = win_raw

                    _append_quality_fallback(preds, "volatility_artifact_skipped")

                    raise RuntimeError(f"weak volatility artifact: {artifact_skip_reason}")

                threshold = args.volatility_threshold if args.volatility_threshold is not None else artifact.threshold

                win_raw, win_source = _best_probability_signal(preds, preds.index)

                win_series = pd.Series(pd.to_numeric(win_raw, errors="coerce"), index=preds.index, dtype=float)

                margin_series = None

                margin_raw = None

                if "pred_home_margin" in preds.columns:

                    margin_raw = preds["pred_home_margin"]

                    margin_series = pd.Series(pd.to_numeric(margin_raw, errors="coerce"), index=preds.index, dtype=float)

                volatility_input = this_week_nodup.copy()
                for col in (
                    "home_win_prob",
                    "home_win_prob_model_raw",
                    "home_win_prob_calibrated",
                    "home_win_prob_capped",
                    "pred_home_margin",
                ):
                    if col in preds.columns:
                        volatility_input[col] = preds[col].reindex(volatility_input.index)

                vol_prob, _ = score_volatility(volatility_input, artifact)

                adj_prob, adj_margin, labels = apply_volatility_shrinkage(

                    win_series,

                    margin_series,

                    vol_prob.reindex(preds.index),

                    threshold=float(threshold),

                    prob_strength=float(args.volatility_strength),

                    margin_strength=float(args.volatility_margin_strength),

                )

                adj_prob = _coerce_probability_series(adj_prob, preds.index)
                skip_shrinkage, coverage = _should_skip_volatility_shrinkage(
                    labels.reindex(preds.index),
                    min_coverage=float(args.volatility_min_coverage),
                    max_coverage=float(args.volatility_max_coverage),
                )
                if skip_shrinkage:
                    shrinkage_status = "skipped"
                    adj_prob = win_series
                    adj_margin = margin_series
                    _append_quality_fallback(preds, "volatility_coverage_skipped")
                    print(
                        "[predict][quality] warning: volatility adjustment selected "
                        f"{coverage:.1%} of this slate, outside "
                        f"{float(args.volatility_min_coverage):.1%}-{float(args.volatility_max_coverage):.1%}; "
                        f"skipping shrinkage and using {win_source}."
                    )
                else:
                    shrinkage_status = "applied"
                    signal_fallback, signal_source = _best_probability_signal(preds, preds.index)
                    restored_prob, fallback_reason = _fallback_if_probability_collapsed(
                        adj_prob,
                        signal_fallback,
                        stage="volatility",
                    )
                    if fallback_reason is not None:
                        restored_adj_prob, _, _ = apply_volatility_shrinkage(
                            restored_prob,
                            None,
                            vol_prob.reindex(preds.index),
                            threshold=float(threshold),
                            prob_strength=float(args.volatility_strength),
                            margin_strength=0.0,
                        )
                        restored_adj_prob = _coerce_probability_series(restored_adj_prob, preds.index)
                        adj_prob = restored_adj_prob if _has_probability_signal(restored_adj_prob) else restored_prob
                        win_raw = signal_fallback
                        _append_quality_fallback(preds, fallback_reason)
                        print(
                            "[predict][quality] warning: volatility adjustment collapsed probability ranking; "
                            f"using {signal_source} as the shrinkage input for this slate."
                        )

                preds["home_win_prob_raw"] = win_raw

                preds["home_win_prob"] = adj_prob

                if adj_margin is not None:

                    preds["pred_home_margin_raw"] = margin_raw if margin_raw is not None else pd.NA

                    preds["pred_home_margin"] = adj_margin

                preds["volatility_prob"] = vol_prob.reindex(preds.index)

                preds["volatility_label"] = labels.reindex(preds.index)

                if args.debug:

                    print(

                        f"[predict][volatility] {shrinkage_status} shrinkage: "

                        f"threshold={threshold:.2f} prob_strength={args.volatility_strength:.2f} "

                        f"margin_strength={args.volatility_margin_strength:.2f} coverage={coverage:.1%}"

                    )

            except Exception as exc:

                if args.debug:

                    print(f"[predict][volatility] adjustment skipped: {exc}")

        elif args.debug:

            print(f"[predict][volatility] artifact {volatility_artifact_path} not found; skipping adjustments")

    prob_series = pd.to_numeric(preds.get("home_win_prob"), errors="coerce")
    if len(prob_series.dropna()) > 1 and prob_series.dropna().nunique() <= 1:
        details = []
        for col in ("home_win_prob_model_raw", "home_win_prob_calibrated", "home_win_prob_capped"):
            if col in preds.columns:
                vals = pd.to_numeric(preds[col], errors="coerce").dropna()
                if not vals.empty:
                    details.append(f"{col} range={vals.min():.4f}-{vals.max():.4f}")
        detail_text = "; ".join(details) if details else "no intermediate probability columns available"
        print(
            "[predict][quality] warning: home_win_prob collapsed to a constant "
            f"{prob_series.dropna().iloc[0]:.4f}; {detail_text}"
        )
    for diag_col in ("home_win_prob_calibrated", "home_win_prob_capped", "home_win_prob_raw"):
        if diag_col not in preds.columns:
            continue
        diag = pd.to_numeric(preds[diag_col], errors="coerce").dropna()
        if len(diag) > 1 and diag.nunique(dropna=True) <= 1:
            print(
                f"[predict][quality] warning: {diag_col} collapsed to a constant "
                f"{diag.iloc[0]:.4f}; final home_win_prob range={prob_series.min():.4f}-{prob_series.max():.4f}"
            )



    # Build human-readable explanations for the two prediction values

    def _fmt_prob(p: float, home: str) -> str:

        if p is None or pd.isna(p):

            return "Win probability unavailable"

        return f"{home} win chance: {p:.1%} (50% = coin flip)"



    def _fmt_margin(m: float) -> str:

        if m is None or pd.isna(m):

            return "Predicted margin unavailable"

        if abs(m) < 0.25:

            return "Pick'em (~0 pts)"

        side = "Home" if m >= 0 else "Away"

        return f"{side} by {abs(m):.1f} pts"



    preds["home_win_prob_expl"] = [

        _fmt_prob(p, h) for p, h in zip(preds["home_win_prob"], preds["home_team"]) 

    ]

    preds["pred_home_margin_expl"] = [

        _fmt_margin(m) for m in preds["pred_home_margin"]

    ]



    # Additional helpful columns

    def _prob_to_american(p: float | None) -> str | None:

        if p is None or pd.isna(p):

            return None

        p = float(p)

        if p <= 0 or p >= 1:

            return None

        # American odds for home side

        if p >= 0.5:

            val = -int(round(100 * p / (1 - p)))

        else:

            val = int(round(100 * (1 - p) / p))

        sign = "+" if val > 0 else ""

        return f"{sign}{val}"



    def _prob_to_decimal(p: float | None) -> float | None:

        if p is None or pd.isna(p):

            return None

        p = float(p)

        if p <= 0:

            return None

        return round(1.0 / p, 3)



    # Pick the team label instead of HOME/AWAY for readability

    preds["home_pick"] = [

        h if (pd.notna(p) and float(p) >= 0.5) else (a if pd.notna(p) else None)

        for p, h, a in zip(preds["home_win_prob"], preds["home_team"], preds["away_team"]) 

    ]

    preds["home_confidence_pct"] = preds["home_win_prob"].apply(lambda p: round(float(p) * 100, 1) if pd.notna(p) else None)

    preds["home_odds_american"] = preds["home_win_prob"].apply(_prob_to_american)

    preds["home_odds_decimal"] = preds["home_win_prob"].apply(_prob_to_decimal)

    # Use the chosen team's win probability in the explanation

    def _pick_prob(p_home: float | None, pick: str | None, h: str | None, a: str | None) -> float | None:

        if p_home is None or pd.isna(p_home) or pick is None:

            return None

        p_home = float(p_home)

        return (p_home if pick == h else (1.0 - p_home))



    preds["pick_expl"] = [

        (f"Pick {pick}: {pp*100:.1f}% win prob; margin {m:+.1f} pts" if (pp is not None and pd.notna(m))

         else (f"Pick {pick}: {pp*100:.1f}% win prob" if pp is not None else None))

        for pick, pp, m in (

            (pick, _pick_prob(prob, pick, h, a), m)

            for pick, prob, h, a, m in zip(

                preds["home_pick"], preds["home_win_prob"], preds["home_team"], preds["away_team"], preds["pred_home_margin"]

            )

        )

    ]



    # Plain-text explanation block per game (keeps SHAP columns as well)

    def _explain_row(row: pd.Series) -> str:

        pick = row.get("home_pick")

        p_home = row.get("home_win_prob")

        margin = row.get("pred_home_margin")

        home = row.get("home_team")

        away = row.get("away_team")

        # picked team probability

        if pd.isna(p_home) or pick is None:

            p_pick = None

        else:

            p_pick = float(p_home) if pick == home else float(1 - p_home)

        # SHAP-based details if present

        supp = row.get("decision_supporting_metrics") or ""

        opp = row.get("decision_opposing_metrics") or ""

        # If missing, compute a fallback from top-magnitude *_diff features in this row

        if not supp and not opp:

            try:

                diff_cols = [c for c in row.index if isinstance(c, str) and c.endswith("_diff")]

                pairs = []

                for c in diff_cols:

                    try:

                        val = row[c]

                        if pd.isna(val):

                            continue

                        v = float(val)

                        pairs.append((c, v))

                    except Exception:

                        continue

                pairs.sort(key=lambda t: abs(t[1]), reverse=True)

                # Directional support based on pick

                if pick == home:

                    support_pairs = [(k, v) for k, v in pairs if v > 0][:6]

                    oppose_pairs = [(k, v) for k, v in pairs if v < 0][:6]

                elif pick == away:

                    support_pairs = [(k, v) for k, v in pairs if v < 0][:6]

                    oppose_pairs = [(k, v) for k, v in pairs if v > 0][:6]

                else:

                    support_pairs = pairs[:6]

                    oppose_pairs = pairs[6:12]

                supp = "; ".join([f"{k}={v:+.3g}" for k, v in support_pairs])

                opp = "; ".join([f"{k}={v:+.3g}" for k, v in oppose_pairs])

            except Exception:

                pass

        conf = row.get("decision_confidence_expl") or ""

        parts = []

        if pick and p_pick is not None:

            parts.append(f"Pick {pick}: {p_pick:.1%} win prob")

        elif pick:

            parts.append(f"Pick {pick}")

        if pd.notna(margin):

            parts.append(f"margin {float(margin):+.1f} pts")

        if conf:

            parts.append(str(conf))

        if supp:

            parts.append(f"support: {supp}")

        if opp:

            parts.append(f"oppose: {opp}")

        return " | ".join(parts)



    try:

        preds["decision_explanation"] = preds.apply(_explain_row, axis=1)

    except Exception:

        pass



    # Narrative explanation per game

    def _fmt_list(items):

        return ", ".join(items) if items else "none"



    def _narrative_row(row: pd.Series) -> str:

        home = str(row.get("home_team"))

        away = str(row.get("away_team"))

        pick = row.get("home_pick")

        p_home = row.get("home_win_prob")

        prob_pick = None

        if pd.notna(p_home) and pick:

            prob_pick = float(p_home) if pick == home else float(1 - p_home)

        margin = row.get("pred_home_margin")



        # Offense diffs (home - away): curated keys or any pass_/rush_/recv_*_diff

        curated_off_keys = [

            "pass_yds_diff", "pass_td_diff", "pass_att_diff", "rush_yds_diff", "rush_td_diff", "recv_yds_diff", "recv_td_diff"

        ]

        off_pairs = []

        for k in curated_off_keys:

            if k in row.index and pd.notna(row[k]):

                try:

                    off_pairs.append((k, float(row[k])))

                except Exception:

                    continue

        if not off_pairs:

            # scan any pass_/rush_/recv_*_diff columns

            for k in row.index:

                if isinstance(k, str) and k.endswith("_diff") and (k.startswith("pass_") or k.startswith("rush_") or k.startswith("recv_")):

                    try:

                        v = float(row[k])

                        if pd.notna(v) and abs(v) > 1e-9:

                            off_pairs.append((k, v))

                    except Exception:

                        continue

        off_pairs.sort(key=lambda x: abs(x[1]), reverse=True)

        if not off_pairs:

            # Final fallback: any *_diff column

            any_pairs = []

            for k in row.index:

                if isinstance(k, str) and k.endswith("_diff"):

                    try:

                        v = float(row[k])

                        if pd.notna(v) and abs(v) > 1e-9:

                            any_pairs.append((k, v))

                    except Exception:

                        continue

            any_pairs.sort(key=lambda x: abs(x[1]), reverse=True)

            off_pairs = any_pairs[:6]

        # Metrics that favor pick

        if pick == home:

            off_favor = [f"{k}={v:+.2g}" for k, v in off_pairs if v > 0][:3]

            off_against = [f"{k}={v:+.2g}" for k, v in off_pairs if v < 0][:3]

        else:

            off_favor = [f"{k}={v:+.2g}" for k, v in off_pairs if v < 0][:3]

            off_against = [f"{k}={v:+.2g}" for k, v in off_pairs if v > 0][:3]



        # Defense-allowed diffs (lower is better for home if diff < 0)

        def_curated = [

            "def_ypp_allowed_diff", "def_pass_ypp_allowed_diff", "def_rush_ypc_allowed_diff", "def_pass_rate_allowed_diff", "def_epa_per_play_allowed_diff"

        ]

        def_pairs = []

        for k in def_curated:

            if k in row.index and pd.notna(row[k]):

                try:

                    def_pairs.append((k, float(row[k])))

                except Exception:

                    continue

        if not def_pairs:

            for k in row.index:

                if isinstance(k, str) and k.startswith("def_") and k.endswith("_diff"):

                    try:

                        v = float(row[k])

                        if pd.notna(v) and abs(v) > 1e-9:

                            def_pairs.append((k, v))

                    except Exception:

                        continue

        def_pairs.sort(key=lambda x: abs(x[1]), reverse=True)

        if not def_pairs:

            # Final fallback: any def_*_diff if present, else any *_diff

            any_def = []

            for k in row.index:

                if isinstance(k, str) and k.startswith("def_") and k.endswith("_diff"):

                    try:

                        v = float(row[k])

                        if pd.notna(v) and abs(v) > 1e-9:

                            any_def.append((k, v))

                    except Exception:

                        continue

            if not any_def:

                for k in row.index:

                    if isinstance(k, str) and k.endswith("_diff"):

                        try:

                            v = float(row[k])

                            if pd.notna(v) and abs(v) > 1e-9:

                                any_def.append((k, v))

                        except Exception:

                            continue

            any_def.sort(key=lambda x: abs(x[1]), reverse=True)

            def_pairs = any_def[:6]

        # For home: negative diffs favorable; for away: positive diffs favorable (home worse)

        if pick == home:

            def_favor = [f"{k}={v:+.2g}" for k, v in def_pairs if v < 0][:3]

            def_against = [f"{k}={v:+.2g}" for k, v in def_pairs if v > 0][:3]

        else:

            def_favor = [f"{k}={v:+.2g}" for k, v in def_pairs if v > 0][:3]

            def_against = [f"{k}={v:+.2g}" for k, v in def_pairs if v < 0][:3]



        # Availability: starters, injuries, snaps

        starters = row.get("starters_diff")

        inj_out = row.get("inj_out_diff")

        snap_off = row.get("snaps_offense_snaps_diff")

        snap_def = row.get("snaps_defense_snaps_diff")

        avail_parts = []

        try:

            if pd.notna(starters):

                avail_parts.append(f"starters_diff={float(starters):+g}")

        except Exception:

            pass

        try:

            if pd.notna(inj_out):

                avail_parts.append(f"inj_out_diff={float(inj_out):+g}")

        except Exception:

            pass

        try:

            if pd.notna(snap_off):

                avail_parts.append(f"off_snaps_diff={float(snap_off):+g}")

        except Exception:

            pass

        try:

            if pd.notna(snap_def):

                avail_parts.append(f"def_snaps_diff={float(snap_def):+g}")

        except Exception:

            pass



        conf_str = None

        if prob_pick is not None:

            side = "home" if pick == home else "away"

            conf_str = f"confidence: {prob_pick:.1%} {side}"



        lines = []

        if pick and prob_pick is not None:

            lines.append(f"Pick {pick}: {prob_pick:.1%} win prob")

        elif pick:

            lines.append(f"Pick {pick}")

        if pd.notna(margin):

            lines.append(f"margin {float(margin):+.1f} pts")

        if conf_str:

            lines.append(conf_str)

        # Sections

        lines.append(f"Offense diffs: favor {_fmt_list(off_favor)}; oppose {_fmt_list(off_against)}")

        lines.append(f"Defense-allowed diffs: favor {_fmt_list(def_favor)}; oppose {_fmt_list(def_against)}")

        if avail_parts:

            lines.append("Availability: " + ", ".join(avail_parts))

        return " | ".join(lines)



    try:

        preds["decision_narrative"] = preds.apply(_narrative_row, axis=1)

    except Exception:

        pass



    # --- Baseline per-game stat projections from season totals (if present) ---

    # Helper to get per-game from *_home / *_away season totals

    def per_game(col_base: str):

        ch, ca = f"{col_base}_home", f"{col_base}_away"

        if ch in this_week.columns:

            preds[f"pred_home_{col_base}"] = pd.to_numeric(this_week[ch], errors="coerce") / 17.0

        else:

            preds[f"pred_home_{col_base}"] = None

        if ca in this_week.columns:

            preds[f"pred_away_{col_base}"] = pd.to_numeric(this_week[ca], errors="coerce") / 17.0

        else:

            preds[f"pred_away_{col_base}"] = None



    # Passing attempts, Rushing attempts, Touchdowns, Interceptions, First downs (receiving+rushing), 4th down

    per_game("passing_att")

    per_game("rushing_att")

    per_game("scoring_tottd")

    per_game("passing_int")

    # First downs: use downs_rec1st + downs_rush1st if available

    for side in ["home", "away"]:

        rush_col = f"downs_rush1st_{side}"

        rec_col = f"downs_rec1st_{side}"

        r1 = pd.to_numeric(this_week[rush_col], errors="coerce") if rush_col in this_week.columns else None

        e1 = pd.to_numeric(this_week[rec_col], errors="coerce") if rec_col in this_week.columns else None

        if r1 is not None or e1 is not None:

            total = (r1 if r1 is not None else 0) + (e1 if e1 is not None else 0)

            preds[f"pred_{side}_first_downs"] = total / 17.0

        else:

            preds[f"pred_{side}_first_downs"] = None

    # 4th down conversions: downs_4thmd per game (attempts also provided)

    per_game("downs_4thatt")

    per_game("downs_4thmd")



    # Punts and Field Goals are not available in current team stats scrape; set placeholders

    preds["pred_home_punts"] = None

    preds["pred_away_punts"] = None

    preds["pred_home_field_goals"] = None

    preds["pred_away_field_goals"] = None



    # Include key defensive allowed metrics and ESPN aggregates in a single concat

    copy_cols: list[str] = []

    allowed_cols = [

        "ypp_allowed", "pass_rate_allowed", "pass_ypp_allowed", "rush_ypc_allowed", "epa_per_play_allowed"

    ]

    # Ensure *_diff exists for defense metrics if components are available

    for base in allowed_cols:

        hcol = f"{base}_home"

        acol = f"{base}_away"

        dcol = f"{base}_diff"

        if hcol in this_week.columns:

            copy_cols.append(hcol)

        if acol in this_week.columns:

            copy_cols.append(acol)

        if dcol in this_week.columns:

            copy_cols.append(dcol)

        elif hcol in this_week.columns and acol in this_week.columns:

            # Compute diff on the fly for export only

            preds[dcol] = pd.to_numeric(this_week[hcol], errors="coerce") - pd.to_numeric(this_week[acol], errors="coerce")

            copy_cols.append(dcol)

    # Also dynamically include any *_allowed_* metrics that exist in features

    dyn_allowed_bases = set()

    for col in this_week.columns:

        if col.endswith("_home") and "allowed" in col and not col.startswith("espn_"):

            dyn_allowed_bases.add(col[:-5])  # strip _home

    for base in sorted(dyn_allowed_bases):

        for suffix in ("_home", "_away"):

            col = base + suffix

            if col in this_week.columns:

                copy_cols.append(col)

        dcol = base + "_diff"

        if dcol in this_week.columns:

            copy_cols.append(dcol)

        elif (base + "_home") in this_week.columns and (base + "_away") in this_week.columns:

            preds[dcol] = pd.to_numeric(this_week[base + "_home"], errors="coerce") - pd.to_numeric(this_week[base + "_away"], errors="coerce")



    # Curated ESPN metrics only

    curated_bases = [

        "espn_passing_att", "espn_passing_yds", "espn_passing_td", "espn_passing_int",

        "espn_rushing_att", "espn_rushing_yds", "espn_rushing_td",

        "espn_receiving_rec", "espn_receiving_tgts", "espn_receiving_yds", "espn_receiving_td",

    ]

    for base in curated_bases:

        for suffix in ("_home", "_away"):

            col = base + suffix

            if col in this_week.columns:

                copy_cols.append(col)

        dcol = base + "_diff"

        if dcol in this_week.columns:

            copy_cols.append(dcol)

    # Deduplicate while preserving order

    if copy_cols:

        seen = set()

        ordered_cols = [c for c in copy_cols if not (c in seen or seen.add(c))]

        existing = [c for c in ordered_cols if c in this_week.columns]

        if existing:

            preds = pd.concat([preds, this_week[existing].copy()], axis=1)

    # If no explicit '*_allowed_*' metrics exist, derive a few from opponent offense

    # def_ypp_allowed_* from opponent yards_per_play_*, def_pass_rate_allowed_* from opponent pass_rate_*

    if ("def_ypp_allowed_home" not in preds.columns) and ("yards_per_play_home" in this_week.columns and "yards_per_play_away" in this_week.columns):

        preds["def_ypp_allowed_home"] = this_week["yards_per_play_away"]

        preds["def_ypp_allowed_away"] = this_week["yards_per_play_home"]

        preds["def_ypp_allowed_diff"] = preds["def_ypp_allowed_home"] - preds["def_ypp_allowed_away"]

    if ("def_pass_rate_allowed_home" not in preds.columns) and ("pass_rate_home" in this_week.columns and "pass_rate_away" in this_week.columns):

        preds["def_pass_rate_allowed_home"] = this_week["pass_rate_away"]

        preds["def_pass_rate_allowed_away"] = this_week["pass_rate_home"]

        preds["def_pass_rate_allowed_diff"] = preds["def_pass_rate_allowed_home"] - preds["def_pass_rate_allowed_away"]

    # def_pass_ypp_allowed from passing yards per attempt if available

    if ("def_pass_ypp_allowed_home" not in preds.columns) and (

        "passing_yds_per_att_home" in this_week.columns and "passing_yds_per_att_away" in this_week.columns

    ):

        preds["def_pass_ypp_allowed_home"] = this_week["passing_yds_per_att_away"]

        preds["def_pass_ypp_allowed_away"] = this_week["passing_yds_per_att_home"]

        preds["def_pass_ypp_allowed_diff"] = preds["def_pass_ypp_allowed_home"] - preds["def_pass_ypp_allowed_away"]

    # def_rush_ypc_allowed from rushing_ypc if available

    if ("def_rush_ypc_allowed_home" not in preds.columns) and (

        "rushing_ypc_home" in this_week.columns and "rushing_ypc_away" in this_week.columns

    ):

        preds["def_rush_ypc_allowed_home"] = this_week["rushing_ypc_away"]

        preds["def_rush_ypc_allowed_away"] = this_week["rushing_ypc_home"]

        preds["def_rush_ypc_allowed_diff"] = preds["def_rush_ypc_allowed_home"] - preds["def_rush_ypc_allowed_away"]

    # def_epa_per_play_allowed if offense EPA per play is present

    if ("def_epa_per_play_allowed_home" not in preds.columns) and (

        "epa_per_play_home" in this_week.columns and "epa_per_play_away" in this_week.columns

    ):

        preds["def_epa_per_play_allowed_home"] = this_week["epa_per_play_away"]

        preds["def_epa_per_play_allowed_away"] = this_week["epa_per_play_home"]

        preds["def_epa_per_play_allowed_diff"] = preds["def_epa_per_play_allowed_home"] - preds["def_epa_per_play_allowed_away"]

    # Allowed totals from opponent offense if present

    def _flip_total(name: str):

        h, a, d = f"def_{name}_allowed_home", f"def_{name}_allowed_away", f"def_{name}_allowed_diff"

        oh, oa = f"{name}_home", f"{name}_away"

        if oh in this_week.columns and oa in this_week.columns:

            preds[h] = this_week[oa]

            preds[a] = this_week[oh]

            preds[d] = preds[h] - preds[a]



    # --- Build offense season totals from NFL.com proxies when PBP not available ---

    # plays ≈ downs_scrmplys; pass_plays ≈ pass_att; rush_plays ≈ rush_att

    # yards ≈ pass_yds + rush_yds; pass_yards ≈ pass_yds; rush_yards ≈ rush_yds

    def _ensure_offense_totals(prefix: str):

        # prefix is 'home' or 'away'

        scr = f"downs_scrmplys_{prefix}"

        pa = f"pass_att_{prefix}" if f"pass_att_{prefix}" in preds.columns else f"passing_att_{prefix}"

        ra = f"rush_att_{prefix}" if f"rush_att_{prefix}" in preds.columns else f"rushing_att_{prefix}"

        py = f"pass_yds_{prefix}" if f"pass_yds_{prefix}" in preds.columns else f"passing_passyds_{prefix}"

        ry = f"rush_yds_{prefix}" if f"rush_yds_{prefix}" in preds.columns else f"rushing_rushyds_{prefix}"



        # create on this_week so _flip_total can find them

        if f"plays_{prefix}" not in this_week.columns and scr in this_week.columns:

            this_week[f"plays_{prefix}"] = pd.to_numeric(this_week[scr], errors="coerce")

        if f"pass_plays_{prefix}" not in this_week.columns and pa in this_week.columns:

            this_week[f"pass_plays_{prefix}"] = pd.to_numeric(this_week[pa], errors="coerce")

        if f"rush_plays_{prefix}" not in this_week.columns and ra in this_week.columns:

            this_week[f"rush_plays_{prefix}"] = pd.to_numeric(this_week[ra], errors="coerce")

        # yards

        if py in this_week.columns and ry in this_week.columns and f"yards_{prefix}" not in this_week.columns:

            this_week[f"pass_yards_{prefix}"] = pd.to_numeric(this_week[py], errors="coerce")

            this_week[f"rush_yards_{prefix}"] = pd.to_numeric(this_week[ry], errors="coerce")

            this_week[f"yards_{prefix}"] = this_week[f"pass_yards_{prefix}"] + this_week[f"rush_yards_{prefix}"]

        else:

            # still ensure component naming for flip

            if py in this_week.columns and f"pass_yards_{prefix}" not in this_week.columns:

                this_week[f"pass_yards_{prefix}"] = pd.to_numeric(this_week[py], errors="coerce")

            if ry in this_week.columns and f"rush_yards_{prefix}" not in this_week.columns:

                this_week[f"rush_yards_{prefix}"] = pd.to_numeric(this_week[ry], errors="coerce")



    try:

        _ensure_offense_totals("home")

        _ensure_offense_totals("away")

    except Exception:

        pass



    for base in [

        "plays", "pass_plays", "rush_plays", "yards", "pass_yards", "rush_yards", "epa"

    ]:

        _flip_total(base)

    # Diagnostic: list available '*allowed*' columns from features

    try:

        allowed_in_feat = [c for c in this_week.columns if "allowed" in c]

        print(f"[predict] features allowed cols: {len(allowed_in_feat)} -> {sorted(allowed_in_feat)[:30]}")

    except Exception:

        pass



    # Friendly aliases for curated ESPN metrics

    alias_bases = {

        "espn_passing_att": "pass_att",

        "espn_passing_yds": "pass_yds",

        "espn_passing_td": "pass_td",

        "espn_passing_int": "pass_int",

        "espn_rushing_att": "rush_att",

        "espn_rushing_yds": "rush_yds",

        "espn_rushing_td": "rush_td",

        "espn_receiving_rec": "recv_rec",

        "espn_receiving_tgts": "recv_tgt",

        "espn_receiving_yds": "recv_yds",

        "espn_receiving_td": "recv_td",

    }

    rename_map = {}

    for src_base, dst_base in alias_bases.items():

        for suffix in ("_home", "_away", "_diff"):

            src = src_base + suffix

            dst = dst_base + suffix

            rename_map[src] = dst

    preds.rename(columns=rename_map, inplace=True)



    # Backfill curated offense metrics from NFL.com team stats if missing

    backfill_map = {

        "pass_att": "passing_att",

        "pass_yds": "passing_passyds",

        "pass_td": "passing_td",

        "pass_int": "passing_int",

        "rush_att": "rushing_att",

        "rush_yds": "rushing_rushyds",

        "rush_td": "rushing_td",

        # Receiving totals by team may be inconsistent; we skip backfill if not present reliably

        "recv_rec": None,

        "recv_tgt": None,

        "recv_yds": "receiving_yds",

        "recv_td": "receiving_td",

    }

    for friendly, src_base in backfill_map.items():

        if not src_base:

            continue

        for suffix in ("_home", "_away"):

            tgt = friendly + suffix

            src = src_base + suffix

            if tgt not in preds.columns and src in this_week.columns:

                preds[tgt] = pd.to_numeric(this_week[src], errors="coerce")

        # diff

        tgt_d = friendly + "_diff"

        if tgt_d not in preds.columns:

            h, a = friendly + "_home", friendly + "_away"

            if h in preds.columns and a in preds.columns:

                preds[tgt_d] = pd.to_numeric(preds[h], errors="coerce") - pd.to_numeric(preds[a], errors="coerce")



    # Friendly aliases for defense-allowed metrics

    def_alias = {

        "ypp_allowed": "def_ypp_allowed",

        "pass_rate_allowed": "def_pass_rate_allowed",

        "pass_ypp_allowed": "def_pass_ypp_allowed",

        "rush_ypc_allowed": "def_rush_ypc_allowed",

        "epa_per_play_allowed": "def_epa_per_play_allowed",

    }

    def_rename = {}

    for src_base, dst_base in def_alias.items():

        for suffix in ("_home", "_away", "_diff"):

            src = src_base + suffix

            dst = dst_base + suffix

            if src in preds.columns:

                def_rename[src] = dst

    preds.rename(columns=def_rename, inplace=True)



    # Prefix dynamically-discovered allowed metrics with def_ if not already prefixed

    for c in list(preds.columns):

        if (c.endswith(("_home", "_away", "_diff")) and "allowed" in c and not c.startswith(("def_", "espn_"))):

            preds.rename(columns={c: "def_" + c}, inplace=True)



    # Optional: quick debug summary to help verify columns (after renaming)

    try:

        def_found = [c for c in preds.columns if c.startswith("def_")]

        espn_found = [c for c in preds.columns if c.startswith(("pass_", "rush_", "recv_"))]

        print(f"[predict] exported defense cols: {len(def_found)}  espn cols: {len(espn_found)}")

    except Exception:

        pass



    # Recompute narrative now that curated offense/defense columns are appended

    try:

        preds["decision_narrative"] = preds.apply(_narrative_row, axis=1)

    except Exception:

        pass



    # Convert any UTC datetime columns to MST (America/Denver) for readability

    def _to_mst_str(s: pd.Series) -> pd.Series:

        try:

            dt = pd.to_datetime(s, errors="coerce", utc=True)

            dt_mst = dt.dt.tz_convert("America/Denver")

            return dt_mst.dt.strftime("%Y-%m-%d %H:%M %Z")

        except Exception:

            return s



    for col in ["date", "start_date"]:

        if col in preds.columns:

            preds[col] = _to_mst_str(preds[col])



    # === Weekly availability metrics ===

    # Injuries: inj_out, inj_doubtful, inj_questionable (from build_features aggregation)

    for base in ["inj_out", "inj_doubtful", "inj_questionable"]:

        h, a, d = f"{base}_home", f"{base}_away", f"{base}_diff"

        if h in this_week.columns:

            preds[h] = pd.to_numeric(this_week[h], errors="coerce")

        if a in this_week.columns:

            preds[a] = pd.to_numeric(this_week[a], errors="coerce")

        if h in preds.columns and a in preds.columns:

            preds[d] = pd.to_numeric(preds[h], errors="coerce") - pd.to_numeric(preds[a], errors="coerce")



    # Depth charts: count of starters if available (depth_is_starter aggregated)

    for base in ["depth_is_starter"]:

        h, a, d = f"{base}_home", f"{base}_away", f"{base}_diff"

        if h in this_week.columns:

            preds[h] = pd.to_numeric(this_week[h], errors="coerce")

        if a in this_week.columns:

            preds[a] = pd.to_numeric(this_week[a], errors="coerce")

        if h in preds.columns and a in preds.columns:

            preds[d] = pd.to_numeric(preds[h], errors="coerce") - pd.to_numeric(preds[a], errors="coerce")

    # Friendly alias for starters

    if "depth_is_starter_home" in preds.columns:

        preds.rename(columns={

            "depth_is_starter_home": "starters_home",

            "depth_is_starter_away": "starters_away",

            "depth_is_starter_diff": "starters_diff",

        }, inplace=True)



    # Injury rates: questionable_rate = inj_questionable / (inj_out + inj_doubtful + inj_questionable)

    if "inj_questionable_home" in preds.columns:

        qh = pd.to_numeric(preds["inj_questionable_home"], errors="coerce")

        qa = pd.to_numeric(preds["inj_questionable_away"], errors="coerce")

        th = (

            pd.to_numeric(preds["inj_out_home"], errors="coerce")

            + pd.to_numeric(preds["inj_doubtful_home"], errors="coerce")

            + qh

        )

        ta = (

            pd.to_numeric(preds["inj_out_away"], errors="coerce")

            + pd.to_numeric(preds["inj_doubtful_away"], errors="coerce")

            + qa

        )

        denom_h = th.replace(0, np.nan)

        denom_a = ta.replace(0, np.nan)

        preds["questionable_rate_home"] = (qh / denom_h).astype(float)

        preds["questionable_rate_away"] = (qa / denom_a).astype(float)

        preds["questionable_rate_diff"] = preds["questionable_rate_home"] - preds["questionable_rate_away"]



    # Approximate punts and field goals if still missing using team totals (very rough placeholder)

    # punts ≈ scrimmage plays − pass_att − rush_att (clipped at 0)

    for side in ("home", "away"):

        punts_col = f"pred_{side}_punts"

        if punts_col not in preds.columns or preds[punts_col].isna().all():

            plays = pd.to_numeric(this_week[f"downs_scrmplys_{side}"], errors="coerce") if f"downs_scrmplys_{side}" in this_week.columns else None

            pa = pd.to_numeric(this_week[f"passing_att_{side}"], errors="coerce") if f"passing_att_{side}" in this_week.columns else None

            ra = pd.to_numeric(this_week[f"rushing_att_{side}"], errors="coerce") if f"rushing_att_{side}" in this_week.columns else None

            if plays is not None and pa is not None and ra is not None:

                est = (plays - pa - ra).clip(lower=0).astype(float)

                preds[punts_col] = est

    for side in ("home", "away"):

        fg_col = f"pred_{side}_field_goals"

        if fg_col not in preds.columns or preds[fg_col].isna().all():

            preds[fg_col] = 0.0  # no reliable team FG data in current sources; set to 0 baseline



    odds_columns: List[str] = []

    try:

        odds_key = _load_odds_api_key()

        if odds_key:

            odds_data = fetch_odds(

                api_key=odds_key,

                sport="americanfootball_nfl",

                regions="us",

                markets=("h2h", "spreads"),

                odds_format="american",

            )

            odds_columns = _attach_odds_spreads(preds, odds_data, debug=args.debug)

        elif args.debug:

            print(f"[predict][odds] skipping odds integration: {ODDS_API_FREE_KEY_SECRET} not configured")

    except Exception as exc:

        if args.debug:

            print(f"[predict][odds] integration failed: {exc}")



    # Final cleanup: drop duplicate / empty columns so exports focus on populated predictions

    preds = _cleanup_prediction_columns(preds)



    penalty_columns: List[str] = []

    try:

        preds, penalty_columns = _attach_team_penalty_metrics(preds, penalty_frame, season, week, debug=args.debug)

    except Exception as exc:

        if args.debug:

            print(f"[predict][penalties] unable to attach team penalty metrics: {exc}")

        penalty_columns = []



    # Optionally save full dump (consolidate decision columns: keep only pick_expl)

    if args.dump_all:

        drop_cols = [

            "decision_explanation",

            "decision_narrative",

            "decision_confidence_expl",

            "decision_supporting_metrics",

            "decision_opposing_metrics",

        ]

        dump_df = preds.drop(columns=[c for c in drop_cols if c in preds.columns], errors="ignore")

        _backup_existing_file(Path(args.dump_all))

        dump_df.to_csv(args.dump_all, index=False)

        print(f"Saved full predictions to {args.dump_all} (cols={len(dump_df.columns)})")

        archived_dump = _archive_prediction_file(args.dump_all, week)

        if archived_dump and args.debug:

            print(f"[predict][archive] archived full predictions -> {archived_dump}")



    # Curated default output: highlight identity, weather, and actual prediction fields

    curated = [

        # identity

        "season", "week", "date", "game_id", "home_team", "away_team",

        # weather (if available)

        "kickoff", "kickoff_mt", "roof", "weather_temp_f", "weather_feelslike_f", "weather_wind_mph", "weather_windgust_mph", "weather_precip_prob_pct",

        "weather_temp_kickoff_f", "weather_temp_q1_f", "weather_temp_q2_f", "weather_temp_q3_f", "weather_temp_q4_f",

        # core predictions

        "home_win_prob", "home_win_prob_model_raw", "home_win_prob_calibrated", "home_win_prob_capped", "home_win_prob_raw", "home_win_prob_quality_fallback", "volatility_prob", "volatility_label", "pred_home_margin", "pred_home_margin_expl", "home_pick", "home_confidence_pct", "home_odds_american", "home_odds_decimal", "pick_expl",

        # offensive production projections

        "pred_home_passing_att", "pred_away_passing_att",

        "pred_home_rushing_att", "pred_away_rushing_att",

        "pred_home_scoring_tottd", "pred_away_scoring_tottd",

        "pred_home_passing_int", "pred_away_passing_int",

        "pred_home_first_downs", "pred_away_first_downs",

        "pred_home_downs_4thatt", "pred_away_downs_4thatt",

        "pred_home_downs_4thmd", "pred_away_downs_4thmd",

        "pred_home_punts", "pred_away_punts",

        "pred_home_field_goals", "pred_away_field_goals",

        "home_penalties_per_game", "away_penalties_per_game",

        "home_penalty_yards_per_game", "away_penalty_yards_per_game",

        "home_altitude_ft", "away_altitude_ft", "altitude_ft_diff",

        "home_crowd_noise_score", "away_crowd_noise_score", "crowd_noise_score_diff",

        "home_fan_hostility_score", "away_fan_hostility_score", "fan_hostility_score_diff",

        "home_weather_snow_index", "away_weather_snow_index", "weather_snow_index_diff",

        "home_weather_rain_index", "away_weather_rain_index", "weather_rain_index_diff",

        "home_indoor", "away_indoor", "indoor_diff",

        "rivalry_intensity", "rivalry_is_divisional", "rivalry_has_historic_component",

        # narrative / explanation

        "decision_explanation", "decision_narrative", "decision_confidence_expl", "decision_supporting_metrics", "decision_opposing_metrics",

    ]

    if odds_columns:

        for col in odds_columns:

            if col not in curated:

                curated.append(col)

    if penalty_columns:

        for col in penalty_columns:

            if col not in curated:

                curated.append(col)

    curated_cols = [c for c in curated if c in preds.columns]

    _backup_existing_file(Path(args.save))

    preds[curated_cols].to_csv(args.save, index=False)

    print(f"Saved predictions to {args.save} (season={season}, week={week}, games={len(preds)}, cols={len(curated_cols)})")

    archived_main = _archive_prediction_file(args.save, week)

    if archived_main and args.debug:

        print(f"[predict][archive] archived curated predictions -> {archived_main}")



    player_base_cols = ["season", "week", "game_id", "home_team", "away_team"]
    if "kickoff" in preds.columns:
        player_base_cols.append("kickoff")
    if "kickoff_mt" in preds.columns:
        player_base_cols.append("kickoff_mt")
    player_games = preds[player_base_cols].drop_duplicates()
    player_stats_frame = _load_player_stats_frame()

    roster_frame = _load_roster_frame()

    try:

        qb_df = _player_qb_predictions(player_games, args.save_players_qb, player_stats_frame, roster_frame, debug=args.debug)

        if qb_df is not None:

            archived_qb = _archive_prediction_file(args.save_players_qb, week)

            if archived_qb and args.debug:

                print(f"[predict][archive] archived QB predictions -> {archived_qb}")

    except Exception as exc:

        if args.debug:

            print(f"[predict][players] unable to save QB projections: {exc}")

    try:

        off_df = _player_offense_predictions(

            player_games,

            args.save_players_offense,

            player_stats_frame,

            roster_frame,

            penalty_frame,

            debug=args.debug,

        )

        if off_df is not None:

            archived_off = _archive_prediction_file(args.save_players_offense, week)

            if archived_off and args.debug:

                print(f"[predict][archive] archived offensive predictions -> {archived_off}")

    except Exception as exc:

        if args.debug:

            print(f"[predict][players] unable to save offensive projections: {exc}")

    try:

        def_df = _player_defense_predictions(

            player_games,

            args.save_players_defense,

            roster_frame,

            penalty_frame,

            debug=args.debug,

        )

        if def_df is not None:

            archived_def = _archive_prediction_file(args.save_players_defense, week)

            if archived_def and args.debug:

                print(f"[predict][archive] archived defensive predictions -> {archived_def}")

    except Exception as exc:

        if args.debug:

            print(f"[predict][players] unable to save defensive projections: {exc}")



if __name__ == "__main__":

    main()

