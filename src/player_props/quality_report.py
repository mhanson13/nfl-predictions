from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.player_props.predict import (
    ANYTIME_TD_FALLBACK_LINE_TYPE,
    DEFAULT_ODDS_VENDORS,
    DEFAULT_PLAYER_IDENTITY_PATH,
    PROPLINE_MARKET_TO_MODEL_MARKET,
    _add_anytime_td_fallback_odds,
    _game_identity_parts,
    _prepare_player_identity,
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
    "novelty": "player_props_novelty.csv",
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
    "exact_line_rows",
    "fallback_line_rows",
    "missing_line_rows",
    "bettable_rows",
    "edge_recommended_rows",
    "edge_strong_rows",
    "edge_lean_rows",
    "edge_below_threshold_rows",
    "edge_unavailable_rows",
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
    "avg_edge_probability_edge",
    "avg_edge_ev",
    "avg_line_book_count",
    "generated_at",
]

COVERAGE_COLUMNS = [
    "season",
    "week",
    "scope",
    "market",
    "sportsbook",
    "coverage_status",
    "raw_rows",
    "raw_unmodeled_market_rows",
    "prepared_line_rows",
    "exact_prepared_line_rows",
    "fallback_prepared_line_rows",
    "offered_game_player_pairs",
    "offered_players",
    "prediction_rows",
    "prediction_players",
    "matchable_prediction_rows",
    "published_line_rows",
    "exact_published_line_rows",
    "fallback_published_line_rows",
    "prediction_rows_without_offered_line",
    "offered_pairs_not_in_predictions",
    "player_name_overlap_no_game",
    "name_key_collision_groups",
    "generated_at",
]

ROSTER_LINE_AUDIT_COLUMNS = [
    "season",
    "week",
    "prediction_group",
    "game_id",
    "team",
    "current_team",
    "player_id",
    "player_name",
    "position",
    "position_group",
    "market",
    "sportsbook",
    "roster_validation_flag",
    "roster_validation_source",
    "availability_status",
    "active_current_roster",
    "prop_quality_flag",
    "injury_status",
    "injury_reported",
    "injury_exclusion_flag",
    "selected_sportsbook",
    "selected_line",
    "selected_over_odds",
    "selected_under_odds",
    "selected_line_book_count",
    "line_availability_status",
    "raw_rows",
    "prepared_line_rows",
    "matchable_line_rows",
    "published_line_flag",
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


def _fallback_line_mask(frame: pd.DataFrame) -> pd.Series:
    if frame.empty or "line_type" not in frame.columns:
        return pd.Series(False, index=frame.index)
    return frame["line_type"].astype("string").str.lower().eq(ANYTIME_TD_FALLBACK_LINE_TYPE).fillna(False)


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
    fallback_line = line_mask & _fallback_line_mask(frame)
    exact_line = line_mask & ~fallback_line
    line_range = _numeric(frame, "line_range")
    abs_edge = _numeric(frame, "edge").abs()
    edge_probability = _numeric(frame, "edge_probability_edge")
    edge_ev = _numeric(frame, "edge_ev")
    line_book_count = _numeric(frame, "line_book_count")
    edge_recommended = _as_bool(frame, "edge_recommendation_flag")
    edge_tier = frame.get("edge_recommendation_tier", pd.Series("", index=frame.index)).astype("string")
    return {
        "season": int(season),
        "week": int(week),
        "scope": scope,
        "prediction_group": prediction_group,
        "market": market,
        "sportsbook": sportsbook,
        "rows": int(rows),
        "rows_with_lines": int(line_mask.sum()),
        "exact_line_rows": int(exact_line.sum()),
        "fallback_line_rows": int(fallback_line.sum()),
        "missing_line_rows": int((~line_mask).sum()),
        "bettable_rows": int(_as_bool(frame, "bettable_flag").sum()),
        "edge_recommended_rows": int(edge_recommended.sum()),
        "edge_strong_rows": int(edge_tier.eq("strong").sum()),
        "edge_lean_rows": int(edge_tier.eq("lean").sum()),
        "edge_below_threshold_rows": int(edge_tier.eq("below_threshold").sum()),
        "edge_unavailable_rows": int(edge_tier.eq("unavailable").sum()),
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
        "avg_edge_probability_edge": float(edge_probability[line_mask].mean()) if line_mask.any() else np.nan,
        "avg_edge_ev": float(edge_ev[line_mask].mean()) if line_mask.any() else np.nan,
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
                "line_type",
            ]
        )
    frame = predictions.copy()
    frame["_prediction_row_id"] = range(len(frame))
    for col in ["game_id", "market", "player_name", "sportsbook", "line"]:
        if col not in frame.columns:
            frame[col] = pd.NA
    if "line_type" not in frame.columns:
        frame["line_type"] = pd.NA
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


def _split_csv_values(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _requested_markets_for_coverage(odds: pd.DataFrame) -> set[str]:
    if odds.empty or "requested_markets" not in odds.columns:
        return set()
    markets: set[str] = set()
    for value in odds["requested_markets"].dropna().unique():
        for market_key in _split_csv_values(value):
            market = PROPLINE_MARKET_TO_MODEL_MARKET.get(market_key)
            markets.add(market if market else market_key)
    return markets


def _requested_sportsbooks_for_coverage(odds: pd.DataFrame) -> set[str]:
    if odds.empty or "requested_bookmakers" not in odds.columns:
        return set()
    sportsbooks: set[str] = set()
    for value in odds["requested_bookmakers"].dropna().unique():
        sportsbooks.update(book.lower() for book in _split_csv_values(value))
    return sportsbooks


def _prepared_odds_for_coverage(
    odds: pd.DataFrame,
    *,
    season: int,
    week: int,
    vendors: Sequence[str],
    predictions: pd.DataFrame | None = None,
    identity_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["game_id", "market", "player_name_key", "player_name_prefix_key", "sportsbook", "line"])
    if odds.empty:
        return empty
    if {"bookmaker_key", "market_key", "outcome_name", "player_name", "price", "point"}.issubset(odds.columns):
        prepared = prepare_player_prop_odds(odds, season=season, week=week, vendor=vendors)
        return _add_anytime_td_fallback_odds(
            prepared,
            predictions if predictions is not None else pd.DataFrame(),
            identity_table=identity_table,
        )
    if {"game_id", "market", "player_name_key", "sportsbook", "line"}.issubset(odds.columns):
        frame = odds.copy()
        frame["sportsbook"] = frame["sportsbook"].astype("string").str.strip().str.lower()
        return _add_anytime_td_fallback_odds(
            frame,
            predictions if predictions is not None else pd.DataFrame(),
            identity_table=identity_table,
        )
    return empty


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


def _identity_matched_odds_legacy_keys(
    predictions: pd.DataFrame,
    odds: pd.DataFrame,
    identity_table: pd.DataFrame | None,
) -> set[tuple[str, str, str]]:
    strict_cols = ["game_id", "market", "player_name_key", "player_name_prefix_key"]
    if (
        predictions.empty
        or odds.empty
        or identity_table is None
        or identity_table.empty
        or not set(strict_cols).issubset(odds.columns)
        or not {"game_id", "market", "player_id"}.issubset(predictions.columns)
    ):
        return set()

    identity_odds = odds.dropna(subset=strict_cols).copy()
    if identity_odds.empty:
        return set()
    parts = identity_odds["game_id"].apply(_game_identity_parts)
    identity_odds["_identity_season"] = [part[0] for part in parts]
    identity_odds["_identity_week"] = [part[1] for part in parts]
    identity_odds["_identity_candidate_team"] = [part[2] for part in parts]
    identity_odds = identity_odds.explode("_identity_candidate_team").dropna(
        subset=["_identity_season", "_identity_week", "_identity_candidate_team"]
    )
    if identity_odds.empty:
        return set()
    identity_odds["_identity_season"] = identity_odds["_identity_season"].astype(int)
    identity_odds["_identity_week"] = identity_odds["_identity_week"].astype(int)

    line_identity = identity_odds.merge(
        identity_table,
        left_on=[
            "_identity_season",
            "_identity_week",
            "_identity_candidate_team",
            "player_name_key",
            "player_name_prefix_key",
        ],
        right_on=[
            "identity_season",
            "identity_week",
            "identity_team",
            "alias_name_key",
            "alias_prefix_key",
        ],
        how="inner",
    )
    if line_identity.empty:
        return set()

    pred = predictions.dropna(subset=["game_id", "market", "player_id"]).copy()
    if pred.empty:
        return set()
    pred_parts = pred["game_id"].apply(_game_identity_parts)
    pred["_prediction_season"] = [part[0] for part in pred_parts]
    pred["_prediction_week"] = [part[1] for part in pred_parts]
    pred = pred.dropna(subset=["_prediction_season", "_prediction_week"]).copy()
    if pred.empty:
        return set()
    pred["_prediction_season"] = pred["_prediction_season"].astype(int)
    pred["_prediction_week"] = pred["_prediction_week"].astype(int)
    pred["player_id"] = pred["player_id"].astype("string").str.strip()

    matches = line_identity.merge(
        pred[["_prediction_season", "_prediction_week", "game_id", "market", "player_id"]].drop_duplicates(),
        left_on=["identity_season", "identity_week", "game_id", "market", "identity_player_id"],
        right_on=["_prediction_season", "_prediction_week", "game_id", "market", "player_id"],
        how="inner",
    )
    if matches.empty:
        return set()
    return _legacy_join_keys(matches)


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


def _matchable_line_rows_for_prediction(prediction: pd.Series, odds: pd.DataFrame) -> int:
    if odds.empty:
        return 0

    strict_cols = ["game_id", "market", "player_name_key", "player_name_prefix_key"]
    legacy_cols = ["game_id", "market", "player_name_key"]

    if set(strict_cols).issubset(odds.columns):
        strict_mask = pd.Series(True, index=odds.index)
        for col in strict_cols:
            value = prediction.get(col)
            strict_mask = strict_mask & odds[col].astype("string").eq(str(value))
        strict_count = int(strict_mask.sum())
        if strict_count:
            return strict_count

    if not set(legacy_cols + ["player_name_prefix_key"]).issubset(odds.columns):
        return 0

    legacy_mask = pd.Series(True, index=odds.index)
    for col in legacy_cols:
        value = prediction.get(col)
        legacy_mask = legacy_mask & odds[col].astype("string").eq(str(value))
    legacy_odds = odds[legacy_mask].copy()
    if legacy_odds.empty:
        return 0
    prefix_count = legacy_odds["player_name_prefix_key"].nunique(dropna=True)
    if prefix_count > 1:
        return 0
    return int(len(legacy_odds))


def _published_line_prediction_ids(predictions: pd.DataFrame, sportsbook: str | None = None) -> set[int]:
    if predictions.empty or "_prediction_row_id" not in predictions.columns:
        return set()
    line_mask = _has_value(predictions, "line")
    if sportsbook is not None and "sportsbook" in predictions.columns:
        line_mask = line_mask & predictions["sportsbook"].eq(sportsbook)
    return set(predictions.loc[line_mask, "_prediction_row_id"].dropna().astype(int).tolist())


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
    identity_table: pd.DataFrame | None = None,
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

    line_mask = _has_value(pred, "line")
    fallback_line = line_mask & _fallback_line_mask(pred)
    exact_line = line_mask & ~fallback_line
    if sportsbook_filter is not None:
        selected_sportsbook = pred.get("sportsbook", pd.Series(pd.NA, index=pred.index)).eq(sportsbook_filter)
        published_line_rows = int((line_mask & selected_sportsbook).sum())
        exact_published_line_rows = int((exact_line & selected_sportsbook).sum())
        fallback_published_line_rows = int((fallback_line & selected_sportsbook).sum())
    else:
        published_line_rows = int(line_mask.sum())
        exact_published_line_rows = int(exact_line.sum())
        fallback_published_line_rows = int(fallback_line.sum())
    matched_ids.update(_published_line_prediction_ids(pred, sportsbook_filter))
    matchable_prediction_rows = len(matched_ids)

    pred_players = set(pred.get("player_name_key", pd.Series(dtype="object")).dropna().astype(str))
    odds_players = set(odds.get("player_name_key", pd.Series(dtype="object")).dropna().astype(str))
    matched_players = set(
        pred[pred["_prediction_row_id"].isin(matched_ids)]
        .get("player_name_key", pd.Series(dtype="object"))
        .dropna()
        .astype(str)
    )

    raw_rows = int(len(raw))
    prepared_line_rows = int(len(odds))
    prepared_fallback = _fallback_line_mask(odds)
    fallback_prepared_line_rows = int(prepared_fallback.sum())
    exact_prepared_line_rows = int(prepared_line_rows - fallback_prepared_line_rows)
    prediction_rows = int(len(pred))
    offered_pairs = _legacy_join_keys(odds)
    prediction_pairs = _legacy_join_keys(pred)
    identity_matched_pairs = _identity_matched_odds_legacy_keys(pred, odds, identity_table)
    if exact_published_line_rows > 0:
        coverage_status = "published"
    elif fallback_published_line_rows > 0:
        coverage_status = "fallback_only"
    elif raw_rows == 0:
        coverage_status = "vendor_no_rows"
    elif prepared_line_rows == 0:
        coverage_status = "raw_unprepared"
    elif prediction_rows == 0:
        coverage_status = "offered_no_predictions"
    elif matchable_prediction_rows == 0:
        coverage_status = "prepared_but_unmatched"
    else:
        coverage_status = "matched_not_published"

    return {
        "season": int(season),
        "week": int(week),
        "scope": scope,
        "market": market,
        "sportsbook": sportsbook,
        "coverage_status": coverage_status,
        "raw_rows": raw_rows,
        "raw_unmodeled_market_rows": int(raw.get("market", pd.Series(dtype="object")).isna().sum()),
        "prepared_line_rows": prepared_line_rows,
        "exact_prepared_line_rows": exact_prepared_line_rows,
        "fallback_prepared_line_rows": fallback_prepared_line_rows,
        "offered_game_player_pairs": int(len(odds_keys)),
        "offered_players": int(len(odds_players)),
        "prediction_rows": prediction_rows,
        "prediction_players": int(len(pred_players)),
        "matchable_prediction_rows": matchable_prediction_rows,
        "published_line_rows": published_line_rows,
        "exact_published_line_rows": exact_published_line_rows,
        "fallback_published_line_rows": fallback_published_line_rows,
        "prediction_rows_without_offered_line": int(max(len(pred) - matchable_prediction_rows, 0)),
        "offered_pairs_not_in_predictions": int(len(offered_pairs - prediction_pairs - identity_matched_pairs)),
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
    identity: pd.DataFrame | None = None,
    generated_at: str | None = None,
) -> pd.DataFrame:
    generated_at = generated_at or datetime.now(timezone.utc).isoformat()
    pred = _prediction_frame_for_coverage(predictions)
    raw = _raw_odds_for_coverage(odds)
    identity_table = _prepare_player_identity(identity)
    prepared = _prepared_odds_for_coverage(
        odds,
        season=season,
        week=week,
        vendors=vendors,
        predictions=pred,
        identity_table=identity_table,
    )
    requested_markets = _requested_markets_for_coverage(odds)
    requested_sportsbooks = _requested_sportsbooks_for_coverage(odds)

    markets = sorted(
        {
            str(value)
            for frame in [pred, raw, prepared]
            if "market" in frame.columns
            for value in frame["market"].dropna().unique()
        }
        | requested_markets
    )
    sportsbooks = sorted(
        {
            str(value)
            for frame in [raw, prepared]
            if "sportsbook" in frame.columns
            for value in frame["sportsbook"].dropna().unique()
        }
        | requested_sportsbooks
    )

    rows: list[dict[str, object]] = [
        _coverage_row(
            predictions=pred,
            raw_odds=raw,
            prepared_odds=prepared,
            identity_table=identity_table,
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
                identity_table=identity_table,
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
                or market in requested_markets
            )
            if not has_rows:
                continue
            rows.append(
                _coverage_row(
                    predictions=pred,
                    raw_odds=raw,
                    prepared_odds=prepared,
                    identity_table=identity_table,
                    season=season,
                    week=week,
                    scope="market_sportsbook",
                    market=market,
                    sportsbook=sportsbook,
                    generated_at=generated_at,
                )
            )

    return pd.DataFrame(rows, columns=COVERAGE_COLUMNS)


def build_roster_line_audit(
    predictions: pd.DataFrame,
    odds: pd.DataFrame,
    *,
    season: int,
    week: int,
    vendors: Sequence[str] = DEFAULT_ODDS_VENDORS,
    roster_flags: Sequence[str] = ("unknown",),
    identity: pd.DataFrame | None = None,
    generated_at: str | None = None,
) -> pd.DataFrame:
    generated_at = generated_at or datetime.now(timezone.utc).isoformat()
    if predictions.empty:
        return pd.DataFrame(columns=ROSTER_LINE_AUDIT_COLUMNS)

    pred = _prediction_frame_for_coverage(predictions)
    for col in ROSTER_LINE_AUDIT_COLUMNS:
        if col not in pred.columns and col not in {
            "season",
            "week",
            "sportsbook",
            "selected_sportsbook",
            "selected_line",
            "selected_over_odds",
            "selected_under_odds",
            "selected_line_book_count",
            "line_availability_status",
            "raw_rows",
            "prepared_line_rows",
            "matchable_line_rows",
            "published_line_flag",
            "generated_at",
        }:
            pred[col] = pd.NA

    roster_flag_values = {str(flag).strip().lower() for flag in roster_flags}
    pred["roster_validation_flag"] = pred["roster_validation_flag"].astype("string").fillna("unknown")
    pred = pred[pred["roster_validation_flag"].str.lower().isin(roster_flag_values)].copy()
    if pred.empty:
        return pd.DataFrame(columns=ROSTER_LINE_AUDIT_COLUMNS)

    raw = _raw_odds_for_coverage(odds)
    identity_table = _prepare_player_identity(identity)
    prepared = _prepared_odds_for_coverage(
        odds,
        season=season,
        week=week,
        vendors=vendors,
        predictions=pred,
        identity_table=identity_table,
    )
    sportsbook_values = {
        str(book).strip().lower()
        for book in vendors
        if str(book).strip()
    }
    for frame in [raw, prepared]:
        if "sportsbook" in frame.columns:
            sportsbook_values.update(str(value) for value in frame["sportsbook"].dropna().unique())
    sportsbooks = sorted(sportsbook_values) or ["all"]

    rows: list[dict[str, object]] = []
    for prediction in pred.to_dict(orient="records"):
        market = str(prediction.get("market") or "")
        selected_sportsbook = str(prediction.get("sportsbook") or "").strip().lower()
        selected_line = prediction.get("line")
        has_selected_line = pd.notna(selected_line) and str(selected_line).strip() not in {"", "nan", "None", "<NA>"}

        for sportsbook in sportsbooks:
            raw_scope = raw.copy()
            if "market" in raw_scope.columns:
                raw_scope = raw_scope[raw_scope["market"].eq(market)]
            if "sportsbook" in raw_scope.columns and sportsbook != "all":
                raw_scope = raw_scope[raw_scope["sportsbook"].eq(sportsbook)]

            prepared_scope = prepared.copy()
            if "market" in prepared_scope.columns:
                prepared_scope = prepared_scope[prepared_scope["market"].eq(market)]
            if "sportsbook" in prepared_scope.columns and sportsbook != "all":
                prepared_scope = prepared_scope[prepared_scope["sportsbook"].eq(sportsbook)]

            matchable_line_rows = _matchable_line_rows_for_prediction(pd.Series(prediction), prepared_scope)
            published_line = bool(has_selected_line and selected_sportsbook == sportsbook)
            if published_line:
                status = "published"
            elif matchable_line_rows > 0:
                status = "matchable_not_published"
            elif len(raw_scope) == 0:
                status = "vendor_no_rows"
            elif len(prepared_scope) == 0:
                status = "raw_unprepared"
            else:
                status = "no_matching_line"

            rows.append(
                {
                    "season": int(season),
                    "week": int(week),
                    "prediction_group": prediction.get("prediction_group"),
                    "game_id": prediction.get("game_id"),
                    "team": prediction.get("team"),
                    "current_team": prediction.get("current_team"),
                    "player_id": prediction.get("player_id"),
                    "player_name": prediction.get("player_name"),
                    "position": prediction.get("position"),
                    "position_group": prediction.get("position_group"),
                    "market": market,
                    "sportsbook": sportsbook,
                    "roster_validation_flag": prediction.get("roster_validation_flag"),
                    "roster_validation_source": prediction.get("roster_validation_source"),
                    "availability_status": prediction.get("availability_status"),
                    "active_current_roster": prediction.get("active_current_roster"),
                    "prop_quality_flag": prediction.get("prop_quality_flag"),
                    "injury_status": prediction.get("injury_status"),
                    "injury_reported": prediction.get("injury_reported"),
                    "injury_exclusion_flag": prediction.get("injury_exclusion_flag"),
                    "selected_sportsbook": selected_sportsbook or pd.NA,
                    "selected_line": selected_line,
                    "selected_over_odds": prediction.get("over_odds"),
                    "selected_under_odds": prediction.get("under_odds"),
                    "selected_line_book_count": prediction.get("line_book_count"),
                    "line_availability_status": status,
                    "raw_rows": int(len(raw_scope)),
                    "prepared_line_rows": int(len(prepared_scope)),
                    "matchable_line_rows": int(matchable_line_rows),
                    "published_line_flag": published_line,
                    "generated_at": generated_at,
                }
            )

    return pd.DataFrame(rows, columns=ROSTER_LINE_AUDIT_COLUMNS)


def write_quality_reports(
    *,
    prediction_dir: Path,
    output_dir: Path,
    season: int,
    week: int,
    odds_path: Path | None = None,
    player_identity_path: Path = DEFAULT_PLAYER_IDENTITY_PATH,
    generated_at: str | None = None,
) -> tuple[Path, Path, Path, Path]:
    generated_at = generated_at or datetime.now(timezone.utc).isoformat()
    predictions = load_prop_predictions(prediction_dir)
    detail, summary = build_quality_reports(
        predictions,
        season=season,
        week=week,
        generated_at=generated_at,
    )
    odds = _read_data_if_exists(odds_path or _default_odds_path(season, week))
    identity = _read_data_if_exists(player_identity_path)
    coverage = build_line_coverage_report(
        predictions,
        odds,
        season=season,
        week=week,
        identity=identity,
        generated_at=generated_at,
    )
    roster_audit = build_roster_line_audit(
        predictions,
        odds,
        season=season,
        week=week,
        identity=identity,
        generated_at=generated_at,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / f"prop_quality_week_{int(week):02d}.csv"
    summary_path = output_dir / f"prop_quality_summary_week_{int(week):02d}.csv"
    coverage_path = output_dir / f"prop_line_coverage_week_{int(week):02d}.csv"
    roster_audit_path = output_dir / f"prop_roster_line_audit_week_{int(week):02d}.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    coverage.to_csv(coverage_path, index=False)
    roster_audit.to_csv(roster_audit_path, index=False)
    return detail_path, summary_path, coverage_path, roster_audit_path


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
    parser.add_argument("--player-identity", type=Path, default=DEFAULT_PLAYER_IDENTITY_PATH)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    detail_path, summary_path, coverage_path, roster_audit_path = write_quality_reports(
        prediction_dir=args.prediction_dir,
        output_dir=args.output_dir,
        season=args.season,
        week=args.week,
        odds_path=args.odds_source,
        player_identity_path=args.player_identity,
    )
    if args.debug:
        summary = pd.read_csv(summary_path)
        print("[player_props.quality] summary:")
        print(summary.to_string(index=False))
        coverage = pd.read_csv(coverage_path)
        print("[player_props.quality] line coverage:")
        print(coverage.to_string(index=False))
        roster_audit = pd.read_csv(roster_audit_path)
        print("[player_props.quality] roster-line audit:")
        print(roster_audit.to_string(index=False))
    print(f"[player_props.quality] wrote detail -> {detail_path.resolve()}")
    print(f"[player_props.quality] wrote summary -> {summary_path.resolve()}")
    print(f"[player_props.quality] wrote line coverage -> {coverage_path.resolve()}")
    print(f"[player_props.quality] wrote roster-line audit -> {roster_audit_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
