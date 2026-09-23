from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.player_props.predict import (
    DEFAULT_ODDS_VENDORS,
    PROPLINE_MARKET_TO_MODEL_MARKET,
    _player_name_key,
    _player_name_prefix_key,
    prepare_player_prop_odds,
)
from src.utils.io import RAW_DIR, read_df


DEFAULT_PREDICTION_DIR = Path("predictions")
DEFAULT_OUTPUT_DIR = DEFAULT_PREDICTION_DIR / "evaluation" / "player_props"

PROP_FILES: Mapping[str, str] = {
    "qb": "player_props_qb.csv",
    "offense": "player_props_offense.csv",
    "defense": "player_props_defense.csv",
}

REPORT_COLUMNS = [
    "season",
    "week",
    "scope",
    "prediction_group",
    "market",
    "sportsbook",
    "rows",
    "rows_with_lines",
    "missing_line_rows",
    "bettable_rows",
    "injury_reported_rows",
    "injury_review_rows",
    "injury_exclusion_rows",
    "roster_validated_rows",
    "roster_unknown_rows",
    "roster_mismatch_rows",
    "roster_exclusion_rows",
    "line_variance_rows",
    "avg_line_range",
    "max_line_range",
    "avg_abs_edge",
    "avg_line_book_count",
    "generated_at",
]

COVERAGE_COLUMNS = [
    "season",
    "week",
    "scope",
    "market",
    "sportsbook",
    "raw_rows",
    "raw_unmodeled_market_rows",
    "prepared_line_rows",
    "offered_game_player_pairs",
    "offered_players",
    "prediction_rows",
    "prediction_players",
    "matchable_prediction_rows",
    "published_line_rows",
    "prediction_rows_without_offered_line",
    "offered_pairs_not_in_predictions",
    "player_name_overlap_no_game",
    "name_key_collision_groups",
    "generated_at",
]


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def load_prop_predictions(prediction_dir: Path = DEFAULT_PREDICTION_DIR) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for prediction_group, filename in PROP_FILES.items():
        frame = _read_csv_if_exists(prediction_dir / filename)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["prediction_group"] = prediction_group
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _default_odds_path(season: int, week: int) -> Path:
    return RAW_DIR / f"propline_player_props_{int(season)}_wk{int(week):02d}.parquet"


def _read_data_if_exists(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    return read_df(path)


def _as_bool(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(False, index=frame.index)
    values = frame[col]
    if values.dtype == bool:
        return values.fillna(False)
    text = values.astype("string").str.strip().str.lower()
    return text.isin({"true", "1", "yes", "y"})


def _has_value(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(False, index=frame.index)
    values = frame[col]
    if pd.api.types.is_numeric_dtype(values):
        return values.notna()
    return values.astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA}).notna()


def _numeric(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(np.nan, index=frame.index)
    return pd.to_numeric(frame[col], errors="coerce")


def _summarize(
    frame: pd.DataFrame,
    *,
    season: int,
    week: int,
    scope: str,
    prediction_group: str,
    market: str,
    sportsbook: str,
    generated_at: str,
) -> dict[str, object]:
    rows = len(frame)
    line_mask = _has_value(frame, "line")
    line_range = _numeric(frame, "line_range")
    abs_edge = _numeric(frame, "edge").abs()
    line_book_count = _numeric(frame, "line_book_count")
    return {
        "season": int(season),
        "week": int(week),
        "scope": scope,
        "prediction_group": prediction_group,
        "market": market,
        "sportsbook": sportsbook,
        "rows": int(rows),
        "rows_with_lines": int(line_mask.sum()),
        "missing_line_rows": int((~line_mask).sum()),
        "bettable_rows": int(_as_bool(frame, "bettable_flag").sum()),
        "injury_reported_rows": int(_as_bool(frame, "injury_reported").sum()),
        "injury_review_rows": int(frame.get("prop_quality_flag", pd.Series("", index=frame.index)).eq("injury_review").sum()),
        "injury_exclusion_rows": int(_as_bool(frame, "injury_exclusion_flag").sum()),
        "roster_validated_rows": int(
            frame.get("roster_validation_flag", pd.Series("unknown", index=frame.index)).ne("unknown").sum()
        ),
        "roster_unknown_rows": int(
            frame.get("roster_validation_flag", pd.Series("unknown", index=frame.index)).eq("unknown").sum()
        ),
        "roster_mismatch_rows": int(
            frame.get("roster_validation_flag", pd.Series("", index=frame.index)).eq("team_mismatch").sum()
        ),
        "roster_exclusion_rows": int(
            frame.get("roster_validation_flag", pd.Series("", index=frame.index))
            .isin({"inactive_roster", "practice_squad"})
            .sum()
        ),
        "line_variance_rows": int(_as_bool(frame, "line_variance_flag").sum()),
        "avg_line_range": float(line_range[line_mask].mean()) if line_mask.any() else np.nan,
        "max_line_range": float(line_range[line_mask].max()) if line_mask.any() else np.nan,
        "avg_abs_edge": float(abs_edge[line_mask].mean()) if line_mask.any() else np.nan,
        "avg_line_book_count": float(line_book_count[line_mask].mean()) if line_mask.any() else np.nan,
        "generated_at": generated_at,
    }


def build_quality_reports(
    predictions: pd.DataFrame,
    *,
    season: int,
    week: int,
    generated_at: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    generated_at = generated_at or datetime.now(timezone.utc).isoformat()
    if predictions.empty:
        empty = pd.DataFrame(columns=REPORT_COLUMNS)
        return empty.copy(), empty.copy()

    frame = predictions.copy()
    for col in ["prediction_group", "market", "sportsbook"]:
        if col not in frame.columns:
            frame[col] = pd.NA
    frame["prediction_group"] = frame["prediction_group"].astype("string").fillna("unknown")
    frame["market"] = frame["market"].astype("string").fillna("unknown")
    frame["sportsbook"] = frame["sportsbook"].astype("string").str.strip()
    frame["sportsbook"] = frame["sportsbook"].where(frame["sportsbook"].notna() & frame["sportsbook"].ne(""), "no_line")

    detail_rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(["prediction_group", "market", "sportsbook"], dropna=False, sort=True):
        prediction_group, market, sportsbook = [str(value) for value in keys]
        detail_rows.append(
            _summarize(
                group,
                season=season,
                week=week,
                scope="market_sportsbook",
                prediction_group=prediction_group,
                market=market,
                sportsbook=sportsbook,
                generated_at=generated_at,
            )
        )

    summary_rows = [
        _summarize(
            frame,
            season=season,
            week=week,
            scope="overall",
            prediction_group="all",
            market="all",
            sportsbook="all",
            generated_at=generated_at,
        )
    ]
    for prediction_group, group in frame.groupby("prediction_group", dropna=False, sort=True):
        summary_rows.append(
            _summarize(
                group,
                season=season,
                week=week,
                scope="prediction_group",
                prediction_group=str(prediction_group),
                market="all",
                sportsbook="all",
                generated_at=generated_at,
            )
        )
    for market, group in frame.groupby("market", dropna=False, sort=True):
        summary_rows.append(
            _summarize(
                group,
                season=season,
                week=week,
                scope="market",
                prediction_group="all",
                market=str(market),
                sportsbook="all",
                generated_at=generated_at,
            )
        )

    detail = pd.DataFrame(detail_rows, columns=REPORT_COLUMNS)
    summary = pd.DataFrame(summary_rows, columns=REPORT_COLUMNS)
    return detail, summary


def _prediction_frame_for_coverage(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(
            columns=[
                "_prediction_row_id",
                "game_id",
                "market",
                "player_name_key",
                "player_name_prefix_key",
                "sportsbook",
                "line",
            ]
        )
    frame = predictions.copy()
    frame["_prediction_row_id"] = range(len(frame))
    for col in ["game_id", "market", "player_name", "sportsbook", "line"]:
        if col not in frame.columns:
            frame[col] = pd.NA
    frame["market"] = frame["market"].astype("string")
    frame["sportsbook"] = frame["sportsbook"].astype("string").str.strip().str.lower()
    frame["player_name_key"] = frame["player_name"].apply(_player_name_key)
    frame["player_name_prefix_key"] = frame["player_name"].apply(_player_name_prefix_key)
    return frame


def _raw_odds_for_coverage(odds: pd.DataFrame) -> pd.DataFrame:
    if odds.empty:
        return pd.DataFrame(columns=["market", "sportsbook", "player_name", "player_name_key"])
    frame = odds.copy()
    if "bookmaker_key" in frame.columns:
        frame["sportsbook"] = frame["bookmaker_key"].astype("string").str.strip().str.lower()
    elif "sportsbook" in frame.columns:
        frame["sportsbook"] = frame["sportsbook"].astype("string").str.strip().str.lower()
    else:
        frame["sportsbook"] = pd.NA

    if "market_key" in frame.columns:
        frame["market"] = frame["market_key"].map(PROPLINE_MARKET_TO_MODEL_MARKET)
    elif "market" not in frame.columns:
        frame["market"] = pd.NA

    if "player_name" not in frame.columns:
        frame["player_name"] = pd.NA
    frame["player_name_key"] = frame["player_name"].apply(_player_name_key)
    frame["player_name_prefix_key"] = frame["player_name"].apply(_player_name_prefix_key)
    return frame


def _prepared_odds_for_coverage(
    odds: pd.DataFrame,
    *,
    season: int,
    week: int,
    vendors: Sequence[str],
) -> pd.DataFrame:
    if odds.empty:
        return pd.DataFrame(columns=["game_id", "market", "player_name_key", "player_name_prefix_key", "sportsbook", "line"])
    if {"bookmaker_key", "market_key", "outcome_name", "player_name", "price", "point"}.issubset(odds.columns):
        return prepare_player_prop_odds(odds, season=season, week=week, vendor=vendors)
    if {"game_id", "market", "player_name_key", "sportsbook", "line"}.issubset(odds.columns):
        frame = odds.copy()
        frame["sportsbook"] = frame["sportsbook"].astype("string").str.strip().str.lower()
        return frame
        return pd.DataFrame(columns=["game_id", "market", "player_name_key", "player_name_prefix_key", "sportsbook", "line"])


def _offer_keys(frame: pd.DataFrame) -> set[tuple[str, str, str, str]]:
    required = ["game_id", "market", "player_name_key", "player_name_prefix_key"]
    if frame.empty or not set(required).issubset(frame.columns):
        return set()
    valid = frame.dropna(subset=required)
    return set(map(tuple, valid[required].astype(str).to_numpy()))


def _legacy_join_keys(frame: pd.DataFrame) -> set[tuple[str, str, str]]:
    required = ["game_id", "market", "player_name_key"]
    if frame.empty or not set(required).issubset(frame.columns):
        return set()
    valid = frame.dropna(subset=required)
    return set(map(tuple, valid[required].astype(str).to_numpy()))


def _matchable_prediction_ids(predictions: pd.DataFrame, odds: pd.DataFrame) -> set[int]:
    if predictions.empty or odds.empty:
        return set()
    strict_cols = ["game_id", "market", "player_name_key", "player_name_prefix_key"]
    legacy_cols = ["game_id", "market", "player_name_key"]
    matched_ids: set[int] = set()

    strict_odds = odds.dropna(subset=strict_cols) if set(strict_cols).issubset(odds.columns) else pd.DataFrame()
    strict_predictions = (
        predictions.dropna(subset=strict_cols) if set(strict_cols).issubset(predictions.columns) else pd.DataFrame()
    )
    if not strict_odds.empty and not strict_predictions.empty:
        strict_matches = strict_predictions.merge(strict_odds[strict_cols].drop_duplicates(), on=strict_cols, how="inner")
        matched_ids.update(strict_matches["_prediction_row_id"].dropna().astype(int).tolist())

    legacy_odds = odds.dropna(subset=legacy_cols).copy() if set(legacy_cols).issubset(odds.columns) else pd.DataFrame()
    if not legacy_odds.empty:
        prefix_counts = (
            legacy_odds.groupby(legacy_cols, dropna=False)["player_name_prefix_key"]
            .nunique(dropna=True)
            .reset_index(name="_prefix_key_count")
        )
        legacy_odds = legacy_odds.merge(prefix_counts, on=legacy_cols, how="left")
        legacy_odds = legacy_odds[legacy_odds["_prefix_key_count"].fillna(0).le(1)]
        fallback_predictions = predictions[~predictions["_prediction_row_id"].isin(matched_ids)]
        fallback_matches = fallback_predictions.merge(
            legacy_odds[legacy_cols].drop_duplicates(),
            on=legacy_cols,
            how="inner",
        )
        matched_ids.update(fallback_matches["_prediction_row_id"].dropna().astype(int).tolist())

    return matched_ids


def _name_key_collision_groups(raw: pd.DataFrame) -> int:
    required = {"market", "sportsbook", "player_name_key", "player_name"}
    if raw.empty or not required.issubset(raw.columns):
        return 0
    valid = raw.dropna(subset=["market", "sportsbook", "player_name_key", "player_name"]).copy()
    if valid.empty:
        return 0
    collisions = (
        valid.groupby(["market", "sportsbook", "player_name_key"])["player_name"]
        .nunique()
        .reset_index(name="unique_names")
    )
    return int(collisions["unique_names"].gt(1).sum())


def _coverage_row(
    *,
    predictions: pd.DataFrame,
    raw_odds: pd.DataFrame,
    prepared_odds: pd.DataFrame,
    season: int,
    week: int,
    scope: str,
    market: str,
    sportsbook: str,
    generated_at: str,
) -> dict[str, object]:
    market_filter = None if market == "all" else market
    sportsbook_filter = None if sportsbook == "all" else sportsbook

    pred = predictions.copy()
    if market_filter is not None and "market" in pred.columns:
        pred = pred[pred["market"].eq(market_filter)]

    raw = raw_odds.copy()
    if market_filter is not None and "market" in raw.columns:
        raw = raw[raw["market"].eq(market_filter)]
    if sportsbook_filter is not None and "sportsbook" in raw.columns:
        raw = raw[raw["sportsbook"].eq(sportsbook_filter)]

    odds = prepared_odds.copy()
    if market_filter is not None and "market" in odds.columns:
        odds = odds[odds["market"].eq(market_filter)]
    if sportsbook_filter is not None and "sportsbook" in odds.columns:
        odds = odds[odds["sportsbook"].eq(sportsbook_filter)]

    odds_keys = _offer_keys(odds)
    matched_ids = _matchable_prediction_ids(pred, odds)
    matchable_prediction_rows = len(matched_ids)

    line_mask = _has_value(pred, "line")
    if sportsbook_filter is not None:
        selected_sportsbook = pred.get("sportsbook", pd.Series(pd.NA, index=pred.index)).eq(sportsbook_filter)
        published_line_rows = int((line_mask & selected_sportsbook).sum())
    else:
        published_line_rows = int(line_mask.sum())

    pred_players = set(pred.get("player_name_key", pd.Series(dtype="object")).dropna().astype(str))
    odds_players = set(odds.get("player_name_key", pd.Series(dtype="object")).dropna().astype(str))
    matched_players = set(
        pred[pred["_prediction_row_id"].isin(matched_ids)]
        .get("player_name_key", pd.Series(dtype="object"))
        .dropna()
        .astype(str)
    )

    return {
        "season": int(season),
        "week": int(week),
        "scope": scope,
        "market": market,
        "sportsbook": sportsbook,
        "raw_rows": int(len(raw)),
        "raw_unmodeled_market_rows": int(raw.get("market", pd.Series(dtype="object")).isna().sum()),
        "prepared_line_rows": int(len(odds)),
        "offered_game_player_pairs": int(len(odds_keys)),
        "offered_players": int(len(odds_players)),
        "prediction_rows": int(len(pred)),
        "prediction_players": int(len(pred_players)),
        "matchable_prediction_rows": matchable_prediction_rows,
        "published_line_rows": published_line_rows,
        "prediction_rows_without_offered_line": int(max(len(pred) - matchable_prediction_rows, 0)),
        "offered_pairs_not_in_predictions": int(max(len(_legacy_join_keys(odds) - _legacy_join_keys(pred)), 0)),
        "player_name_overlap_no_game": int(len((pred_players & odds_players) - matched_players)),
        "name_key_collision_groups": _name_key_collision_groups(raw),
        "generated_at": generated_at,
    }


def build_line_coverage_report(
    predictions: pd.DataFrame,
    odds: pd.DataFrame,
    *,
    season: int,
    week: int,
    vendors: Sequence[str] = DEFAULT_ODDS_VENDORS,
    generated_at: str | None = None,
) -> pd.DataFrame:
    generated_at = generated_at or datetime.now(timezone.utc).isoformat()
    pred = _prediction_frame_for_coverage(predictions)
    raw = _raw_odds_for_coverage(odds)
    prepared = _prepared_odds_for_coverage(odds, season=season, week=week, vendors=vendors)

    markets = sorted(
        {
            str(value)
            for frame in [pred, raw, prepared]
            if "market" in frame.columns
            for value in frame["market"].dropna().unique()
        }
    )
    sportsbooks = sorted(
        {
            str(value)
            for frame in [raw, prepared]
            if "sportsbook" in frame.columns
            for value in frame["sportsbook"].dropna().unique()
        }
    )

    rows: list[dict[str, object]] = [
        _coverage_row(
            predictions=pred,
            raw_odds=raw,
            prepared_odds=prepared,
            season=season,
            week=week,
            scope="overall",
            market="all",
            sportsbook="all",
            generated_at=generated_at,
        )
    ]
    for market in markets:
        rows.append(
            _coverage_row(
                predictions=pred,
                raw_odds=raw,
                prepared_odds=prepared,
                season=season,
                week=week,
                scope="market",
                market=market,
                sportsbook="all",
                generated_at=generated_at,
            )
        )
        for sportsbook in sportsbooks:
            has_rows = (
                (not raw.empty and raw["market"].eq(market).fillna(False).any() and raw["sportsbook"].eq(sportsbook).any())
                or (
                    not prepared.empty
                    and prepared["market"].eq(market).fillna(False).any()
                    and prepared["sportsbook"].eq(sportsbook).any()
                )
            )
            if not has_rows:
                continue
            rows.append(
                _coverage_row(
                    predictions=pred,
                    raw_odds=raw,
                    prepared_odds=prepared,
                    season=season,
                    week=week,
                    scope="market_sportsbook",
                    market=market,
                    sportsbook=sportsbook,
                    generated_at=generated_at,
                )
            )

    return pd.DataFrame(rows, columns=COVERAGE_COLUMNS)


def write_quality_reports(
    *,
    prediction_dir: Path,
    output_dir: Path,
    season: int,
    week: int,
    odds_path: Path | None = None,
    generated_at: str | None = None,
) -> tuple[Path, Path, Path]:
    generated_at = generated_at or datetime.now(timezone.utc).isoformat()
    predictions = load_prop_predictions(prediction_dir)
    detail, summary = build_quality_reports(
        predictions,
        season=season,
        week=week,
        generated_at=generated_at,
    )
    odds = _read_data_if_exists(odds_path or _default_odds_path(season, week))
    coverage = build_line_coverage_report(
        predictions,
        odds,
        season=season,
        week=week,
        generated_at=generated_at,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / f"prop_quality_week_{int(week):02d}.csv"
    summary_path = output_dir / f"prop_quality_summary_week_{int(week):02d}.csv"
    coverage_path = output_dir / f"prop_line_coverage_week_{int(week):02d}.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    coverage.to_csv(coverage_path, index=False)
    return detail_path, summary_path, coverage_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build weekly quality summaries for player prop prediction CSVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--prediction-dir", type=Path, default=DEFAULT_PREDICTION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--odds-source", type=Path, default=None)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    detail_path, summary_path, coverage_path = write_quality_reports(
        prediction_dir=args.prediction_dir,
        output_dir=args.output_dir,
        season=args.season,
        week=args.week,
        odds_path=args.odds_source,
    )
    if args.debug:
        summary = pd.read_csv(summary_path)
        print("[player_props.quality] summary:")
        print(summary.to_string(index=False))
        coverage = pd.read_csv(coverage_path)
        print("[player_props.quality] line coverage:")
        print(coverage.to_string(index=False))
    print(f"[player_props.quality] wrote detail -> {detail_path.resolve()}")
    print(f"[player_props.quality] wrote summary -> {summary_path.resolve()}")
    print(f"[player_props.quality] wrote line coverage -> {coverage_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
