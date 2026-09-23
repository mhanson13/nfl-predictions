from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.player_props.predict import _player_name_key, _player_name_prefix_key
from src.predict.utils import moneyline_to_prob
from src.utils.io import PROC_DIR, read_df, write_df


DEFAULT_LINES_PATH = PROC_DIR / "oddsapi_historical_player_prop_lines.parquet"
DEFAULT_MODEL_PREDICTIONS_PATH = Path("predictions/evaluation/player_props/model_predictions.csv")
DEFAULT_OUTPUT_PATH = PROC_DIR / "player_prop_historical_line_eval.parquet"
DEFAULT_SUMMARY_PATH = Path("predictions/evaluation/player_props/historical_line_join_summary.csv")

JOIN_OUTPUT_COLUMNS = [
    "season",
    "week",
    "game_id",
    "oddsapi_event_id",
    "requested_date",
    "snapshot_timestamp",
    "market",
    "market_key",
    "sportsbook",
    "line",
    "over_odds",
    "under_odds",
    "implied_probability",
    "over_implied_raw",
    "under_implied_raw",
    "book_hold",
    "line_balance_distance",
    "line_is_comparable",
    "line_updated_at",
    "line_player_name",
    "model_player_name",
    "player_name_key",
    "player_name_prefix_key",
    "player_id",
    "team",
    "opponent",
    "position",
    "position_group",
    "model_projection",
    "actual_value",
    "actual_over_line",
    "actual_under_line",
    "actual_push",
    "model_edge",
    "abs_model_edge",
    "model_pick",
    "model_pick_correct",
    "model_error",
    "abs_model_error",
    "line_error",
    "abs_line_error",
    "prob_over_zero",
    "actual_over_zero",
    "model_type",
    "evaluation_method",
    "train_rows",
    "feature_count",
    "match_method",
]

SUMMARY_COLUMNS = [
    "scope",
    "market",
    "sportsbook",
    "available_line_rows",
    "matched_line_rows",
    "matched_rate",
    "unique_games",
    "unique_events",
    "unique_players",
    "comparable_rows",
    "two_way_rows",
    "model_pick_rows",
    "model_pick_accuracy",
    "actual_over_rate",
    "mean_model_edge",
    "mean_abs_model_edge",
    "mean_abs_model_error",
    "mean_abs_line_error",
    "generated_at",
]


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() == ".csv":
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
    return read_df(path)


def _as_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def _prepare_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {
        "season",
        "week",
        "game_id",
        "market",
        "player_id",
        "player_name",
        "model_projection",
        "actual_value",
    }
    if missing := required - set(predictions.columns):
        raise ValueError(f"Model predictions missing required columns: {sorted(missing)}")
    if predictions.empty:
        return pd.DataFrame()

    out = predictions.copy()
    out["season"] = pd.to_numeric(out["season"], errors="coerce")
    out["week"] = pd.to_numeric(out["week"], errors="coerce")
    out["model_projection"] = pd.to_numeric(out["model_projection"], errors="coerce")
    out["actual_value"] = pd.to_numeric(out["actual_value"], errors="coerce")
    out = out[
        out["season"].notna()
        & out["week"].notna()
        & out["model_projection"].notna()
        & out["actual_value"].notna()
    ].copy()
    if out.empty:
        return pd.DataFrame()

    out["season"] = out["season"].astype(int)
    out["week"] = out["week"].astype(int)
    for col in ["game_id", "market", "player_id", "player_name"]:
        out[col] = _as_text(out[col])
    out["player_name_key"] = out["player_name"].apply(_player_name_key)
    out["player_name_prefix_key"] = out["player_name"].apply(_player_name_prefix_key)
    if "actual_over_zero" not in out.columns:
        out["actual_over_zero"] = out["actual_value"].gt(0)
    else:
        out["actual_over_zero"] = out["actual_over_zero"].astype("boolean").fillna(False).astype(bool)

    defaults = {
        "team": pd.NA,
        "opponent": pd.NA,
        "position": pd.NA,
        "position_group": pd.NA,
        "prob_over_zero": np.nan,
        "model_type": pd.NA,
        "evaluation_method": pd.NA,
        "train_rows": np.nan,
        "feature_count": np.nan,
    }
    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default

    out = out.dropna(subset=["season", "week", "game_id", "market", "player_name_key"])
    out = out.sort_values(["season", "week", "game_id", "market", "player_id"], kind="stable")
    out = out.drop_duplicates(["season", "week", "game_id", "market", "player_id"], keep="last")
    out = out.reset_index(drop=True)
    out["_prediction_row_id"] = np.arange(len(out))
    return out


def _prepare_lines(
    lines: pd.DataFrame,
    *,
    sportsbooks: Sequence[str] | None = None,
    comparable_only: bool = False,
) -> pd.DataFrame:
    required = {
        "season",
        "week",
        "game_id",
        "market",
        "sportsbook",
        "player_name",
        "line",
    }
    if missing := required - set(lines.columns):
        raise ValueError(f"Historical lines missing required columns: {sorted(missing)}")
    if lines.empty:
        return pd.DataFrame()

    out = lines.copy()
    out["season"] = pd.to_numeric(out["season"], errors="coerce")
    out["week"] = pd.to_numeric(out["week"], errors="coerce")
    out["line"] = pd.to_numeric(out["line"], errors="coerce")
    out = out[out["season"].notna() & out["week"].notna() & out["line"].notna()].copy()
    if out.empty:
        return pd.DataFrame()

    out["season"] = out["season"].astype(int)
    out["week"] = out["week"].astype(int)
    for col in ["game_id", "market", "sportsbook", "player_name"]:
        out[col] = _as_text(out[col])
    out["sportsbook"] = out["sportsbook"].str.lower()
    if sportsbooks:
        allowed = {str(book).strip().lower() for book in sportsbooks if str(book).strip()}
        if allowed:
            out = out[out["sportsbook"].isin(allowed)].copy()
    if "player_name_key" not in out.columns:
        out["player_name_key"] = out["player_name"].apply(_player_name_key)
    if "player_name_prefix_key" not in out.columns:
        out["player_name_prefix_key"] = out["player_name"].apply(_player_name_prefix_key)
    if "line_is_comparable" not in out.columns:
        out["line_is_comparable"] = False
    out["line_is_comparable"] = out["line_is_comparable"].astype("boolean").fillna(False).astype(bool)
    if comparable_only:
        out = out[out["line_is_comparable"]].copy()

    defaults = {
        "oddsapi_event_id": pd.NA,
        "requested_date": pd.NA,
        "snapshot_timestamp": pd.NA,
        "market_key": pd.NA,
        "over_odds": np.nan,
        "under_odds": np.nan,
        "implied_probability": np.nan,
        "line_balance_distance": np.nan,
        "line_updated_at": pd.NA,
    }
    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default

    out["over_odds"] = pd.to_numeric(out["over_odds"], errors="coerce")
    out["under_odds"] = pd.to_numeric(out["under_odds"], errors="coerce")
    out = out.dropna(subset=["season", "week", "game_id", "market", "player_name_key", "sportsbook"])
    out = out.sort_values(
        ["season", "week", "game_id", "market", "player_name_key", "sportsbook", "line"],
        kind="stable",
    )
    out = out.reset_index(drop=True)
    out["_line_row_id"] = np.arange(len(out))
    return out


def _add_market_metrics(joined: pd.DataFrame) -> pd.DataFrame:
    if joined.empty:
        return joined
    out = joined.copy()
    out["over_implied_raw"] = moneyline_to_prob(out["over_odds"])
    out["under_implied_raw"] = moneyline_to_prob(out["under_odds"])
    out["book_hold"] = out["over_implied_raw"] + out["under_implied_raw"] - 1.0
    out.loc[out["over_implied_raw"].isna() | out["under_implied_raw"].isna(), "book_hold"] = np.nan

    out["actual_over_line"] = out["actual_value"].gt(out["line"])
    out["actual_under_line"] = out["actual_value"].lt(out["line"])
    out["actual_push"] = out["actual_value"].eq(out["line"])
    out["model_edge"] = out["model_projection"] - out["line"]
    out["abs_model_edge"] = out["model_edge"].abs()
    out["model_pick"] = np.select(
        [out["model_edge"].gt(0), out["model_edge"].lt(0)],
        ["over", "under"],
        default="push",
    )
    out["model_pick_correct"] = np.where(
        out["model_pick"].eq("over"),
        out["actual_over_line"],
        np.where(out["model_pick"].eq("under"), out["actual_under_line"], pd.NA),
    )
    out.loc[out["actual_push"], "model_pick_correct"] = pd.NA
    out["model_error"] = out["model_projection"] - out["actual_value"]
    out["abs_model_error"] = out["model_error"].abs()
    out["line_error"] = out["line"] - out["actual_value"]
    out["abs_line_error"] = out["line_error"].abs()
    return out


def join_historical_lines(
    *,
    lines: pd.DataFrame,
    predictions: pd.DataFrame,
    sportsbooks: Sequence[str] | None = None,
    comparable_only: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prepared_lines = _prepare_lines(lines, sportsbooks=sportsbooks, comparable_only=comparable_only)
    prepared_predictions = _prepare_predictions(predictions)
    if prepared_lines.empty or prepared_predictions.empty:
        return pd.DataFrame(columns=JOIN_OUTPUT_COLUMNS), prepared_lines

    line_base = prepared_lines.rename(columns={"player_name": "line_player_name"})
    pred_base = prepared_predictions.rename(columns={"player_name": "model_player_name"})
    strict_cols = ["season", "week", "game_id", "market", "player_name_key", "player_name_prefix_key"]
    legacy_cols = ["season", "week", "game_id", "market", "player_name_key"]

    matches: list[pd.DataFrame] = []
    matched_line_ids: set[int] = set()

    strict_lines = line_base.dropna(subset=strict_cols)
    strict_predictions = pred_base.dropna(subset=strict_cols)
    if not strict_lines.empty and not strict_predictions.empty:
        strict = strict_lines.merge(pred_base, on=strict_cols, how="inner", suffixes=("_line", "_model"))
        if not strict.empty:
            strict["match_method"] = "strict_prefix"
            matches.append(strict)
            matched_line_ids.update(strict["_line_row_id"].astype(int).tolist())

    fallback_lines = line_base[~line_base["_line_row_id"].isin(matched_line_ids)].dropna(subset=legacy_cols).copy()
    fallback_predictions = pred_base.dropna(subset=legacy_cols).copy()
    if not fallback_lines.empty and not fallback_predictions.empty:
        line_prefix_counts = (
            fallback_lines.groupby(legacy_cols, dropna=False)["player_name_prefix_key"]
            .nunique(dropna=True)
            .reset_index(name="_line_prefix_count")
        )
        pred_counts = (
            fallback_predictions.groupby(legacy_cols, dropna=False)["_prediction_row_id"]
            .nunique(dropna=True)
            .reset_index(name="_prediction_count")
        )
        fallback_lines = fallback_lines.merge(line_prefix_counts, on=legacy_cols, how="left")
        fallback_predictions = fallback_predictions.merge(pred_counts, on=legacy_cols, how="left")
        fallback_lines = fallback_lines[fallback_lines["_line_prefix_count"].fillna(0).le(1)].copy()
        fallback_predictions = fallback_predictions[fallback_predictions["_prediction_count"].fillna(0).eq(1)].copy()
        fallback_lines = fallback_lines.drop(columns=["player_name_prefix_key", "_line_prefix_count"], errors="ignore")
        fallback_predictions = fallback_predictions.drop(columns=["player_name_prefix_key", "_prediction_count"], errors="ignore")
        fallback = fallback_lines.merge(
            fallback_predictions,
            on=legacy_cols,
            how="inner",
            suffixes=("_line", "_model"),
        )
        if not fallback.empty:
            fallback["match_method"] = "initial_last_unique"
            matches.append(fallback)

    if not matches:
        return pd.DataFrame(columns=JOIN_OUTPUT_COLUMNS), prepared_lines

    joined = pd.concat(matches, ignore_index=True, sort=False)
    joined = joined.sort_values(["_line_row_id", "match_method"], kind="stable")
    joined = joined.drop_duplicates("_line_row_id", keep="first")
    if "player_name_prefix_key" not in joined.columns:
        prefix_line = joined.get("player_name_prefix_key_line", pd.Series(pd.NA, index=joined.index))
        prefix_model = joined.get("player_name_prefix_key_model", pd.Series(pd.NA, index=joined.index))
        joined["player_name_prefix_key"] = prefix_line.where(prefix_line.notna(), prefix_model)
    joined = _add_market_metrics(joined)

    for col in JOIN_OUTPUT_COLUMNS:
        if col not in joined.columns:
            joined[col] = np.nan
    return joined[JOIN_OUTPUT_COLUMNS].reset_index(drop=True), prepared_lines


def _summary_for_group(lines: pd.DataFrame, joined: pd.DataFrame, *, scope: str, generated_at: str) -> dict[str, object]:
    pick_mask = joined["model_pick"].isin(["over", "under"]) & joined["model_pick_correct"].notna()
    return {
        "scope": scope,
        "market": joined["market"].iloc[0] if scope == "market" and not joined.empty else pd.NA,
        "sportsbook": joined["sportsbook"].iloc[0] if scope == "sportsbook" and not joined.empty else pd.NA,
        "available_line_rows": int(len(lines)),
        "matched_line_rows": int(len(joined)),
        "matched_rate": float(len(joined) / len(lines)) if len(lines) else np.nan,
        "unique_games": int(joined["game_id"].nunique()) if not joined.empty else 0,
        "unique_events": int(joined["oddsapi_event_id"].nunique()) if "oddsapi_event_id" in joined.columns and not joined.empty else 0,
        "unique_players": int(joined["player_id"].nunique()) if "player_id" in joined.columns and not joined.empty else 0,
        "comparable_rows": int(joined["line_is_comparable"].astype(bool).sum()) if not joined.empty else 0,
        "two_way_rows": int((joined["over_odds"].notna() & joined["under_odds"].notna()).sum()) if not joined.empty else 0,
        "model_pick_rows": int(pick_mask.sum()) if not joined.empty else 0,
        "model_pick_accuracy": float(joined.loc[pick_mask, "model_pick_correct"].astype(bool).mean()) if pick_mask.any() else np.nan,
        "actual_over_rate": float(joined["actual_over_line"].mean()) if not joined.empty else np.nan,
        "mean_model_edge": float(joined["model_edge"].mean()) if not joined.empty else np.nan,
        "mean_abs_model_edge": float(joined["abs_model_edge"].mean()) if not joined.empty else np.nan,
        "mean_abs_model_error": float(joined["abs_model_error"].mean()) if not joined.empty else np.nan,
        "mean_abs_line_error": float(joined["abs_line_error"].mean()) if not joined.empty else np.nan,
        "generated_at": generated_at,
    }


def summarize_join(lines: pd.DataFrame, joined: pd.DataFrame, *, generated_at: str | None = None) -> pd.DataFrame:
    generated_at = generated_at or datetime.now(timezone.utc).isoformat()
    if lines.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    rows = [_summary_for_group(lines, joined, scope="all", generated_at=generated_at)]

    for market, line_group in lines.groupby("market", sort=True):
        join_group = joined[joined["market"].eq(market)] if not joined.empty else pd.DataFrame(columns=joined.columns)
        row = _summary_for_group(line_group, join_group, scope="market", generated_at=generated_at)
        row["market"] = market
        rows.append(row)

    for sportsbook, line_group in lines.groupby("sportsbook", sort=True):
        join_group = joined[joined["sportsbook"].eq(sportsbook)] if not joined.empty else pd.DataFrame(columns=joined.columns)
        row = _summary_for_group(line_group, join_group, scope="sportsbook", generated_at=generated_at)
        row["sportsbook"] = sportsbook
        rows.append(row)

    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join historical sportsbook player-prop lines to walk-forward model predictions and actuals.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--lines", type=Path, default=DEFAULT_LINES_PATH)
    parser.add_argument("--model-predictions", type=Path, default=DEFAULT_MODEL_PREDICTIONS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--sportsbooks", nargs="+", default=None)
    parser.add_argument("--comparable-only", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    lines = _read_table(args.lines)
    predictions = _read_table(args.model_predictions)
    joined, prepared_lines = join_historical_lines(
        lines=lines,
        predictions=predictions,
        sportsbooks=args.sportsbooks,
        comparable_only=args.comparable_only,
    )
    summary = summarize_join(prepared_lines, joined)
    output_path = write_df(joined, args.output)
    summary_path = write_df(summary, args.summary_output)

    if args.debug:
        print("[player_props.historical_lines] summary:")
        print(summary.to_string(index=False))
    print(f"[player_props.historical_lines] wrote joined rows={len(joined)} -> {output_path.resolve()}")
    print(f"[player_props.historical_lines] wrote summary rows={len(summary)} -> {summary_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
