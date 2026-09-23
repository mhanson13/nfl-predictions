from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.player_props.labels import DEFAULT_OUTPUT_PATH as DEFAULT_LABELS_PATH
from src.utils.io import PROC_DIR, read_df, write_df
from src.utils.teams import normalize_team_abbr
from src.utils.week_filter import filter_before_week


DEFAULT_OFFENSE_OUTPUT_PATH = PROC_DIR / "player_prop_features_offense.parquet"
DEFAULT_DEFENSE_OUTPUT_PATH = PROC_DIR / "player_prop_features_defense.parquet"

OFFENSE_MARKETS = frozenset({"qb_passing_yards", "rb_rushing_yards", "wrte_receiving_yards"})
DEFENSE_MARKETS = frozenset({"def_sacks"})

ID_COLUMNS = [
    "season",
    "week",
    "game_id",
    "team",
    "opponent",
    "player_id",
    "player_name",
    "position",
    "position_group",
    "market",
    "market_family",
]

TARGET_COLUMNS = ["actual_value", "actual_over_zero"]


def _clean_team(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().upper()
    if not text or text in {"NONE", "NAN", "NA", "<NA>"}:
        return None
    try:
        return normalize_team_abbr(text)
    except Exception:
        return text


def _as_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def _prepare_labels(
    labels: pd.DataFrame,
    *,
    seasons: Sequence[int] | None = None,
    exclude_from_season: int | None = None,
    exclude_from_week: int | None = None,
) -> pd.DataFrame:
    required = {
        "season",
        "week",
        "game_id",
        "team",
        "opponent",
        "player_id",
        "market",
        "actual_value",
    }
    if missing := required - set(labels.columns):
        raise ValueError(f"Player prop labels missing required columns: {sorted(missing)}")

    out = labels.copy()
    out["season"] = pd.to_numeric(out["season"], errors="coerce")
    out["week"] = pd.to_numeric(out["week"], errors="coerce")
    out["actual_value"] = pd.to_numeric(out["actual_value"], errors="coerce")
    out = out[out["season"].notna() & out["week"].notna() & out["actual_value"].notna()]
    out["season"] = out["season"].astype(int)
    out["week"] = out["week"].astype(int)

    if seasons:
        season_set = {int(season) for season in seasons}
        out = out[out["season"].isin(season_set)]

    out = filter_before_week(out, exclude_from_season, exclude_from_week)

    for col in ["game_id", "player_id", "player_name", "position", "position_group", "market"]:
        if col not in out.columns:
            out[col] = pd.NA
        out[col] = _as_text(out[col])
    out["market"] = out["market"].str.lower()
    out["team"] = out["team"].apply(_clean_team)
    out["opponent"] = out["opponent"].apply(_clean_team)

    if "actual_over_zero" not in out.columns:
        out["actual_over_zero"] = out["actual_value"] > 0
    else:
        out["actual_over_zero"] = out["actual_over_zero"].astype(bool)

    out = out[out["market"].isin(OFFENSE_MARKETS | DEFENSE_MARKETS)]
    out = out.dropna(subset=["season", "week", "game_id", "team", "opponent", "player_id", "market"])
    out = out[out["team"] != out["opponent"]]
    out["market_family"] = out["market"].map(lambda market: "defense" if market in DEFENSE_MARKETS else "offense")
    out = out.drop_duplicates(subset=["season", "week", "game_id", "team", "player_id", "market"], keep="last")
    return out.sort_values(["season", "week", "game_id", "team", "market", "player_id"]).reset_index(drop=True)


def _grouped_shifted_expanding_mean(frame: pd.DataFrame, keys: list[str], value_col: str) -> pd.Series:
    return frame.groupby(keys, sort=False)[value_col].transform(lambda s: s.shift().expanding(min_periods=1).mean())


def _grouped_shifted_expanding_std(frame: pd.DataFrame, keys: list[str], value_col: str) -> pd.Series:
    return frame.groupby(keys, sort=False)[value_col].transform(lambda s: s.shift().expanding(min_periods=2).std())


def _grouped_shifted_rolling_mean(frame: pd.DataFrame, keys: list[str], value_col: str, window: int) -> pd.Series:
    return frame.groupby(keys, sort=False)[value_col].transform(lambda s: s.shift().rolling(window, min_periods=1).mean())


def _add_player_history_features(labels: pd.DataFrame) -> pd.DataFrame:
    out = labels.copy()
    out = out.sort_values(["player_id", "market", "season", "week", "game_id"], kind="stable")

    player_keys = ["player_id", "market"]
    season_keys = ["season", "player_id", "market"]
    out["player_games_prior"] = out.groupby(player_keys, sort=False).cumcount()
    out["player_avg_prior"] = _grouped_shifted_expanding_mean(out, player_keys, "actual_value")
    out["player_std_prior"] = _grouped_shifted_expanding_std(out, player_keys, "actual_value")
    out["player_last1_value"] = out.groupby(player_keys, sort=False)["actual_value"].shift()
    out["player_last3_avg"] = _grouped_shifted_rolling_mean(out, player_keys, "actual_value", 3)
    out["player_last5_avg"] = _grouped_shifted_rolling_mean(out, player_keys, "actual_value", 5)

    out["_actual_over_zero_float"] = out["actual_over_zero"].astype(float)
    out["player_over_zero_rate_prior"] = _grouped_shifted_expanding_mean(out, player_keys, "_actual_over_zero_float")
    out["player_season_games_prior"] = out.groupby(season_keys, sort=False).cumcount()
    out["player_season_avg_prior"] = _grouped_shifted_expanding_mean(out, season_keys, "actual_value")
    out["player_season_last3_avg"] = _grouped_shifted_rolling_mean(out, season_keys, "actual_value", 3)
    return out.drop(columns=["_actual_over_zero_float"])


def _weekly_context(labels: pd.DataFrame, *, entity_col: str, prefix: str) -> pd.DataFrame:
    weekly = (
        labels.groupby(["season", "week", entity_col, "market"], as_index=False)
        .agg(
            weekly_total=("actual_value", "sum"),
            weekly_mean=("actual_value", "mean"),
            weekly_players=("player_id", "nunique"),
        )
        .sort_values([entity_col, "market", "season", "week"], kind="stable")
    )
    keys = [entity_col, "market"]
    weekly[f"{prefix}_weekly_total_avg_prior"] = _grouped_shifted_expanding_mean(weekly, keys, "weekly_total")
    weekly[f"{prefix}_weekly_total_last3_avg"] = _grouped_shifted_rolling_mean(weekly, keys, "weekly_total", 3)
    weekly[f"{prefix}_weekly_mean_avg_prior"] = _grouped_shifted_expanding_mean(weekly, keys, "weekly_mean")
    weekly[f"{prefix}_weekly_mean_last3_avg"] = _grouped_shifted_rolling_mean(weekly, keys, "weekly_mean", 3)
    weekly[f"{prefix}_players_avg_prior"] = _grouped_shifted_expanding_mean(weekly, keys, "weekly_players")
    keep = [
        "season",
        "week",
        entity_col,
        "market",
        f"{prefix}_weekly_total_avg_prior",
        f"{prefix}_weekly_total_last3_avg",
        f"{prefix}_weekly_mean_avg_prior",
        f"{prefix}_weekly_mean_last3_avg",
        f"{prefix}_players_avg_prior",
    ]
    return weekly[keep]


def _add_context_features(labels: pd.DataFrame) -> pd.DataFrame:
    out = labels.copy()
    team_context = _weekly_context(out, entity_col="team", prefix="team_market")
    opponent_context = _weekly_context(out, entity_col="opponent", prefix="opponent_allowed")
    out = out.merge(team_context, on=["season", "week", "team", "market"], how="left")
    return out.merge(opponent_context, on=["season", "week", "opponent", "market"], how="left")


def _add_game_features(labels: pd.DataFrame) -> pd.DataFrame:
    out = labels.copy()
    game_parts = out["game_id"].astype("string").str.split("_", expand=True)
    if game_parts.shape[1] >= 4:
        away_team = game_parts[2].apply(_clean_team)
        home_team = game_parts[3].apply(_clean_team)
        out["is_home"] = out["team"].eq(home_team)
        out["is_away"] = out["team"].eq(away_team)
    else:
        out["is_home"] = pd.NA
        out["is_away"] = pd.NA
    return out


def _prepare_prediction_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    required = {
        "season",
        "week",
        "game_id",
        "team",
        "opponent",
        "player_id",
        "market",
    }
    if missing := required - set(candidates.columns):
        raise ValueError(f"Player prop prediction candidates missing required columns: {sorted(missing)}")

    out = candidates.copy()
    out["season"] = pd.to_numeric(out["season"], errors="coerce")
    out["week"] = pd.to_numeric(out["week"], errors="coerce")
    out = out[out["season"].notna() & out["week"].notna()]
    out["season"] = out["season"].astype(int)
    out["week"] = out["week"].astype(int)

    for col in ["game_id", "player_id", "player_name", "position", "position_group", "market"]:
        if col not in out.columns:
            out[col] = pd.NA
        out[col] = _as_text(out[col])
    out["market"] = out["market"].str.lower()
    out["team"] = out["team"].apply(_clean_team)
    out["opponent"] = out["opponent"].apply(_clean_team)
    out = out[out["market"].isin(OFFENSE_MARKETS | DEFENSE_MARKETS)]
    out = out.dropna(subset=["season", "week", "game_id", "team", "opponent", "player_id", "market"])
    out = out[out["team"] != out["opponent"]]
    out["market_family"] = out["market"].map(lambda market: "defense" if market in DEFENSE_MARKETS else "offense")
    out["actual_value"] = np.nan
    out["actual_over_zero"] = False
    out["played_flag"] = False
    out["active_flag"] = True
    out["value_type"] = out["market"].map(lambda market: "count" if market in DEFENSE_MARKETS else "yards")
    out["source"] = "prediction_candidate"
    out["_prediction_row"] = True
    return out.drop_duplicates(subset=["season", "week", "game_id", "team", "player_id", "market"], keep="last")


def build_player_prop_prediction_features(
    labels: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    seasons: Sequence[int] | None = None,
    exclude_from_season: int | None = None,
    exclude_from_week: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    history = _prepare_labels(
        labels,
        seasons=seasons,
        exclude_from_season=exclude_from_season,
        exclude_from_week=exclude_from_week,
    )
    candidates_prepared = _prepare_prediction_candidates(candidates)
    if candidates_prepared.empty:
        empty = pd.DataFrame(columns=ID_COLUMNS)
        return empty.copy(), empty.copy()

    history = history.copy()
    history["_prediction_row"] = False
    combined = pd.concat([history, candidates_prepared], ignore_index=True, sort=False)
    combined = combined.sort_values(["season", "week", "game_id", "team", "market", "player_id"], kind="stable")
    features = _add_player_history_features(combined)
    features = _add_context_features(features)
    features = _add_game_features(features)
    prediction_features = features[features["_prediction_row"].fillna(False)].copy()
    prediction_features = prediction_features.drop(columns=["_prediction_row"], errors="ignore")
    prediction_features = prediction_features.sort_values(
        ["season", "week", "game_id", "team", "market", "player_id"]
    ).reset_index(drop=True)
    offense = prediction_features[prediction_features["market_family"].eq("offense")].copy()
    defense = prediction_features[prediction_features["market_family"].eq("defense")].copy()
    return offense, defense


def build_player_prop_features(
    labels: pd.DataFrame,
    *,
    seasons: Sequence[int] | None = None,
    exclude_from_season: int | None = None,
    exclude_from_week: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prepared = _prepare_labels(
        labels,
        seasons=seasons,
        exclude_from_season=exclude_from_season,
        exclude_from_week=exclude_from_week,
    )
    if prepared.empty:
        empty = pd.DataFrame(columns=ID_COLUMNS + TARGET_COLUMNS)
        return empty.copy(), empty.copy()

    features = _add_player_history_features(prepared)
    features = _add_context_features(features)
    features = _add_game_features(features)
    features = features.sort_values(["season", "week", "game_id", "team", "market", "player_id"]).reset_index(drop=True)

    offense = features[features["market_family"].eq("offense")].copy()
    defense = features[features["market_family"].eq("defense")].copy()
    return offense, defense


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build split offensive and defensive player prop feature tables.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH, help="Player prop label parquet/csv.")
    parser.add_argument("--seasons", type=int, nargs="+", default=None, help="Seasons to include.")
    parser.add_argument("--offense-output", type=Path, default=DEFAULT_OFFENSE_OUTPUT_PATH, help="Offense feature output.")
    parser.add_argument("--defense-output", type=Path, default=DEFAULT_DEFENSE_OUTPUT_PATH, help="Defense feature output.")
    parser.add_argument("--exclude-from-season", type=int, default=None, help="Exclude this season/week and later rows.")
    parser.add_argument("--exclude-from-week", type=int, default=None, help="Exclude this season/week and later rows.")
    parser.add_argument("--debug", action="store_true", help="Print feature row counts.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.labels.exists():
        raise FileNotFoundError(f"Player prop labels not found: {args.labels}")

    labels = read_df(args.labels)
    offense, defense = build_player_prop_features(
        labels,
        seasons=args.seasons,
        exclude_from_season=args.exclude_from_season,
        exclude_from_week=args.exclude_from_week,
    )

    offense_path = write_df(offense, args.offense_output)
    defense_path = write_df(defense, args.defense_output)

    if args.debug:
        print("[player_props.features] row counts:")
        print(f"  offense: {len(offense)}")
        print(f"  defense: {len(defense)}")
        if not offense.empty:
            print("[player_props.features] offense markets:")
            print(offense.groupby("market").size().sort_index().to_string())
        if not defense.empty:
            print("[player_props.features] defense markets:")
            print(defense.groupby("market").size().sort_index().to_string())

    print(f"[player_props.features] wrote offense rows={len(offense)} -> {offense_path.resolve()}")
    print(f"[player_props.features] wrote defense rows={len(defense)} -> {defense_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
