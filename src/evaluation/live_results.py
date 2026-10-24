from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.evaluation.evaluate_predictions import _classification_metrics
from src.utils.io import read_df


DEFAULT_MTB_CSV_ROOT = Path("..") / "mtb" / "data" / "prediction-csvs"
DEFAULT_PREDICTIONS_LOG_DIR = Path("predictions_log")
DEFAULT_FEATURES_PATH = Path("data/processed/matchup_features.parquet")
DEFAULT_OUTPUT_DIR = Path("predictions/evaluation")
DEFAULT_MARKET_CLV_DETAIL_PATH = Path("analysis/market_clv_detail.csv")

WEEKLY_DETAIL_TEMPLATE = "live_prediction_vs_actual_week_{week:02d}.csv"
WEEKLY_SUMMARY_TEMPLATE = "live_prediction_vs_actual_summary_week_{week:02d}.csv"
WEEKLY_SEGMENTS_TEMPLATE = "live_prediction_vs_actual_segments_week_{week:02d}.csv"
CUMULATIVE_DETAIL_NAME = "live_prediction_vs_actual.csv"
CUMULATIVE_SUMMARY_NAME = "live_prediction_vs_actual_summary.csv"

CLV_RENAME_MAP = {
    "model_pick_side": "clv_model_pick_side",
    "model_pick_team": "clv_model_pick_team",
    "model_pick_prob": "clv_model_pick_prob",
    "model_side_pick_moneyline": "clv_model_side_pick_moneyline",
    "model_side_close_moneyline": "clv_model_side_close_moneyline",
    "model_side_pick_implied_prob_novig": "clv_model_side_pick_implied_prob_novig",
    "model_side_close_implied_prob_novig": "clv_model_side_close_implied_prob_novig",
    "model_vs_pick_prob_edge": "clv_model_vs_pick_prob_edge",
    "model_vs_close_prob_edge": "clv_model_vs_close_prob_edge",
    "pick_selected_spread": "clv_pick_selected_spread",
    "close_selected_spread": "clv_close_selected_spread",
}

CLV_OUTPUT_COLUMNS = [
    "clv_detail_available",
    "clv_status",
    "clv_model_pick_side",
    "clv_model_pick_team",
    "clv_model_pick_prob",
    "clv_model_side_pick_moneyline",
    "clv_model_side_close_moneyline",
    "clv_model_side_pick_implied_prob_novig",
    "clv_model_side_close_implied_prob_novig",
    "moneyline_clv_decimal_delta",
    "moneyline_clv_prob_delta",
    "moneyline_positive_clv",
    "clv_model_vs_pick_prob_edge",
    "clv_model_vs_close_prob_edge",
    "clv_pick_selected_spread",
    "clv_close_selected_spread",
    "spread_clv_points",
    "spread_positive_clv",
]

CLV_NULL_ON_MISMATCH_COLUMNS = [
    "clv_model_pick_prob",
    "clv_model_side_pick_moneyline",
    "clv_model_side_close_moneyline",
    "clv_model_side_pick_implied_prob_novig",
    "clv_model_side_close_implied_prob_novig",
    "moneyline_clv_decimal_delta",
    "moneyline_clv_prob_delta",
    "moneyline_positive_clv",
    "clv_model_vs_pick_prob_edge",
    "clv_model_vs_close_prob_edge",
    "clv_pick_selected_spread",
    "clv_close_selected_spread",
    "spread_clv_points",
    "spread_positive_clv",
]

PREDICTION_CONTEXT_COLUMNS = [
    "season",
    "week",
    "game_id",
    "home_team",
    "away_team",
    "kickoff",
    "kickoff_mt",
    "roof",
    "weather_temp_f",
    "weather_wind_mph",
    "home_win_prob",
    "home_win_prob_model_raw",
    "home_win_prob_calibrated",
    "home_win_prob_capped",
    "home_win_prob_raw",
    "volatility_prob",
    "volatility_label",
    "pred_home_margin",
    "home_pick",
    "home_confidence_pct",
    "home_odds_american",
    "home_odds_decimal",
    "market_policy_primary_signal",
    "market_policy_tier",
    "market_policy_status",
    "market_model_pick",
    "market_vegas_pick",
    "market_home_moneyline",
    "market_away_moneyline",
    "market_home_implied_prob_novig",
    "market_away_implied_prob_novig",
    "market_moneyline_favorite_disagreement",
    "market_moneyline_prob_edge",
    "market_moneyline_edge_bin",
    "market_moneyline_policy_tier",
    "market_moneyline_policy_status",
    "market_moneyline_policy_recommendation_flag",
    "market_home_spread",
    "market_away_spread",
    "market_home_margin",
    "market_spread_edge_points",
    "market_spread_abs_edge_points",
    "market_spread_selected_side",
    "market_spread_edge_bin",
    "market_spread_policy_tier",
    "market_spread_policy_status",
    "market_spread_policy_recommendation_flag",
    "decision_explanation",
    "decision_narrative",
]

NUMERIC_PREDICTION_COLUMNS = [
    "season",
    "week",
    "home_win_prob",
    "home_win_prob_model_raw",
    "home_win_prob_calibrated",
    "home_win_prob_capped",
    "home_win_prob_raw",
    "volatility_prob",
    "pred_home_margin",
    "home_confidence_pct",
    "home_odds_american",
    "home_odds_decimal",
    "market_home_moneyline",
    "market_away_moneyline",
    "market_home_implied_prob_novig",
    "market_away_implied_prob_novig",
    "market_moneyline_prob_edge",
    "market_home_spread",
    "market_away_spread",
    "market_home_margin",
    "market_spread_edge_points",
    "market_spread_abs_edge_points",
]

SEGMENT_COLUMNS = [
    "prediction_source_type",
    "volatility_label",
    "market_policy_tier",
    "market_policy_status",
    "market_moneyline_edge_bin",
    "market_moneyline_policy_tier",
    "market_moneyline_policy_status",
    "market_spread_edge_bin",
    "market_spread_policy_tier",
    "market_spread_policy_status",
    "model_pick_team",
    "actual_winner",
]


@dataclass(frozen=True)
class PredictionSource:
    source_type: str
    path: Path
    frame: pd.DataFrame
    last_modified_utc: str


@dataclass(frozen=True)
class LiveResultArtifacts:
    detail_path: Path
    cumulative_detail_path: Path
    summary_path: Path
    cumulative_summary_path: Path
    segments_path: Path | None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join immutable live prediction snapshots to finalized game results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--features-path", type=Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--market-clv-detail", type=Path, default=DEFAULT_MARKET_CLV_DETAIL_PATH)
    parser.add_argument("--mtb-csv-root", type=Path, default=DEFAULT_MTB_CSV_ROOT)
    parser.add_argument("--predictions-log-dir", type=Path, default=DEFAULT_PREDICTIONS_LOG_DIR)
    parser.add_argument(
        "--prediction-file",
        type=Path,
        default=None,
        help="Explicit immutable prediction CSV to grade. If omitted, source order is used.",
    )
    parser.add_argument(
        "--source-order",
        nargs="+",
        default=["mtb_archive", "locked_log"],
        choices=["mtb_archive", "locked_log"],
        help="Prediction snapshot lookup order when --prediction-file is not supplied.",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Write empty artifacts instead of failing when no finalized games can be joined.",
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args(argv)


def archived_prediction_csv_path(mtb_csv_root: Path, season: int, week: int) -> Path:
    return mtb_csv_root / "seasons" / str(season) / f"week_{week:02d}" / "predictions.csv"


def locked_prediction_csv_path(predictions_log_dir: Path, season: int, week: int) -> Path:
    return predictions_log_dir / str(season) / f"week_{week:02d}_predictions.csv"


def _last_modified_utc(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _read_prediction_source(path: Path, source_type: str, season: int, week: int) -> PredictionSource:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    prepared = prepare_prediction_frame(frame, season=season, week=week)
    return PredictionSource(
        source_type=source_type,
        path=path.resolve(),
        frame=prepared,
        last_modified_utc=_last_modified_utc(path),
    )


def load_prediction_source(
    *,
    season: int,
    week: int,
    mtb_csv_root: Path = DEFAULT_MTB_CSV_ROOT,
    predictions_log_dir: Path = DEFAULT_PREDICTIONS_LOG_DIR,
    prediction_file: Path | None = None,
    source_order: Iterable[str] = ("mtb_archive", "locked_log"),
) -> PredictionSource:
    if prediction_file is not None:
        return _read_prediction_source(prediction_file, "explicit_file", season, week)

    candidates = {
        "mtb_archive": archived_prediction_csv_path(mtb_csv_root, season, week),
        "locked_log": locked_prediction_csv_path(predictions_log_dir, season, week),
    }
    missing: list[str] = []
    for source_type in source_order:
        path = candidates[source_type]
        if path.exists():
            return _read_prediction_source(path, source_type, season, week)
        missing.append(f"{source_type}={path}")
    raise FileNotFoundError(
        "No immutable prediction snapshot found. Checked: " + "; ".join(missing)
    )


def prepare_prediction_frame(frame: pd.DataFrame, *, season: int, week: int) -> pd.DataFrame:
    out = frame.copy()
    if "season" not in out.columns:
        out["season"] = season
    if "week" not in out.columns:
        out["week"] = week

    out["season"] = pd.to_numeric(out["season"], errors="coerce").fillna(season).astype(int)
    out["week"] = pd.to_numeric(out["week"], errors="coerce").fillna(week).astype(int)
    out = out[(out["season"] == season) & (out["week"] == week)].copy()

    required = {"season", "week", "game_id", "home_team", "away_team", "home_win_prob"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Prediction snapshot missing required columns: {sorted(missing)}")
    if out.empty:
        raise ValueError(f"Prediction snapshot has no rows for season {season}, week {week}")
    _raise_on_duplicate_keys(out, ["season", "week", "game_id"], "prediction snapshot")

    if "pred_home_margin" not in out.columns:
        out["pred_home_margin"] = np.nan

    for col in NUMERIC_PREDICTION_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    keep = [col for col in PREDICTION_CONTEXT_COLUMNS if col in out.columns]
    return out[keep].copy()


def load_finalized_actuals(features_path: Path, *, season: int, week: int) -> pd.DataFrame:
    cols = [
        "season",
        "week",
        "game_id",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "home_margin",
    ]
    actuals = read_df(features_path, columns=cols)
    actuals["season"] = pd.to_numeric(actuals["season"], errors="coerce")
    actuals["week"] = pd.to_numeric(actuals["week"], errors="coerce")
    actuals = actuals[(actuals["season"] == season) & (actuals["week"] == week)].copy()
    if actuals.empty:
        return actuals

    for col in ("home_score", "away_score", "home_margin"):
        actuals[col] = pd.to_numeric(actuals[col], errors="coerce")
    actuals = actuals.dropna(subset=["home_score", "away_score", "home_margin"])
    actuals = actuals.drop_duplicates(subset=["season", "week", "game_id"], keep="last")
    _raise_on_duplicate_keys(actuals, ["season", "week", "game_id"], "actual results")
    actuals["actual_home_win"] = np.select(
        [actuals["home_margin"] > 0.0, actuals["home_margin"] < 0.0],
        [1.0, 0.0],
        default=np.nan,
    )
    actuals["actual_winner"] = np.select(
        [actuals["home_margin"] > 0.0, actuals["home_margin"] < 0.0],
        [actuals["home_team"], actuals["away_team"]],
        default="TIE",
    )
    return actuals


def load_market_clv_detail(path: Path, *, season: int, week: int) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    detail = pd.read_csv(path)
    required = {"season", "week", "game_id"}
    if detail.empty or not required.issubset(detail.columns):
        return pd.DataFrame()
    detail["season"] = pd.to_numeric(detail["season"], errors="coerce")
    detail["week"] = pd.to_numeric(detail["week"], errors="coerce")
    detail = detail[(detail["season"] == season) & (detail["week"] == week)].copy()
    if detail.empty:
        return detail
    _raise_on_duplicate_keys(detail, ["season", "week", "game_id"], "market CLV detail")
    return detail


def build_prediction_vs_actual(
    predictions: PredictionSource,
    actuals: pd.DataFrame,
    *,
    clv_detail: pd.DataFrame | None = None,
) -> pd.DataFrame:
    pred = predictions.frame.copy()
    if actuals.empty:
        empty = _empty_detail_frame(pred.columns)
        return attach_market_clv_detail(empty, clv_detail if clv_detail is not None else pd.DataFrame())

    merged = pred.merge(
        actuals,
        on=["season", "week", "game_id"],
        how="inner",
        suffixes=("", "_actual"),
        validate="one_to_one",
    )
    if merged.empty:
        return _empty_detail_frame(pred.columns)

    merged["prediction_source_type"] = predictions.source_type
    merged["prediction_source_file"] = str(predictions.path)
    merged["prediction_source_last_modified_utc"] = predictions.last_modified_utc
    merged["graded_at_utc"] = datetime.now(timezone.utc).isoformat()

    merged["home_win_prob"] = pd.to_numeric(merged["home_win_prob"], errors="coerce").clip(0.0, 1.0)
    merged["pred_home_margin"] = pd.to_numeric(merged["pred_home_margin"], errors="coerce")
    merged["predicted_home_win"] = np.where(
        merged["home_win_prob"].notna(),
        merged["home_win_prob"].ge(0.5),
        np.nan,
    )
    merged["model_pick_team"] = np.where(
        merged["predicted_home_win"].fillna(False),
        merged["home_team"],
        merged["away_team"],
    )
    merged.loc[merged["predicted_home_win"].isna(), "model_pick_team"] = pd.NA
    merged["model_pick_prob"] = np.where(
        merged["predicted_home_win"].fillna(False),
        merged["home_win_prob"],
        1.0 - merged["home_win_prob"],
    )
    merged.loc[merged["home_win_prob"].isna(), "model_pick_prob"] = np.nan

    merged["su_correct"] = np.where(
        merged["actual_home_win"].isna() | merged["predicted_home_win"].isna(),
        np.nan,
        merged["predicted_home_win"].astype(float).eq(merged["actual_home_win"]),
    )
    merged["su_result"] = np.select(
        [
            merged["actual_home_win"].isna(),
            merged["su_correct"].eq(True),
            merged["su_correct"].eq(False),
        ],
        ["push", "win", "loss"],
        default="ungraded",
    )

    eps = 1e-9
    merged["brier_component"] = (merged["home_win_prob"] - merged["actual_home_win"]) ** 2
    merged["log_loss_component"] = -(
        merged["actual_home_win"] * np.log(np.clip(merged["home_win_prob"], eps, 1.0 - eps))
        + (1.0 - merged["actual_home_win"])
        * np.log(np.clip(1.0 - merged["home_win_prob"], eps, 1.0 - eps))
    )
    merged["margin_error"] = merged["pred_home_margin"] - merged["home_margin"]
    merged["abs_margin_error"] = merged["margin_error"].abs()
    merged["sq_margin_error"] = merged["margin_error"] ** 2
    merged["misclassified"] = merged["su_correct"].eq(False)

    _add_moneyline_grades(merged)
    _add_spread_grades(merged)
    merged = attach_market_clv_detail(merged, clv_detail if clv_detail is not None else pd.DataFrame())

    return _order_detail_columns(merged)


def attach_market_clv_detail(frame: pd.DataFrame, clv_detail: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    _ensure_clv_columns(out)
    if out.empty or clv_detail.empty:
        return out

    required = {"season", "week", "game_id"}
    if not required.issubset(clv_detail.columns):
        return out

    source_cols = [
        col
        for col in [
            "season",
            "week",
            "game_id",
            *CLV_RENAME_MAP.keys(),
            "moneyline_clv_decimal_delta",
            "moneyline_clv_prob_delta",
            "moneyline_positive_clv",
            "spread_clv_points",
            "spread_positive_clv",
            "clv_status",
        ]
        if col in clv_detail.columns
    ]
    detail = clv_detail[source_cols].rename(columns=CLV_RENAME_MAP).copy()
    detail["clv_detail_available"] = True
    _raise_on_duplicate_keys(detail, ["season", "week", "game_id"], "market CLV detail")

    out = out.drop(columns=[col for col in CLV_OUTPUT_COLUMNS if col in out.columns])
    out = out.merge(
        detail,
        on=["season", "week", "game_id"],
        how="left",
        validate="one_to_one",
    )
    _ensure_clv_columns(out)
    out["clv_detail_available"] = out["clv_detail_available"].fillna(False).astype(bool)
    out["clv_status"] = out["clv_status"].fillna("unavailable_missing_clv_detail")

    live_side = np.where(out["predicted_home_win"].fillna(False), "home", "away")
    live_side = pd.Series(live_side, index=out.index, dtype="object")
    live_side[out["predicted_home_win"].isna()] = pd.NA
    clv_side = out["clv_model_pick_side"].astype("string")
    side_mismatch = (
        out["clv_detail_available"]
        & clv_side.notna()
        & live_side.astype("string").notna()
        & clv_side.ne(live_side.astype("string"))
    )

    live_team = out.get("model_pick_team", pd.Series(pd.NA, index=out.index)).astype("string")
    clv_team = out["clv_model_pick_team"].astype("string")
    team_mismatch = (
        out["clv_detail_available"]
        & clv_team.notna()
        & live_team.notna()
        & clv_team.ne(live_team)
    )
    mismatch = side_mismatch | team_mismatch
    if mismatch.any():
        out.loc[mismatch, "clv_status"] = "unavailable_model_side_mismatch"
        for col in CLV_NULL_ON_MISMATCH_COLUMNS:
            if col in out.columns:
                out.loc[mismatch, col] = np.nan
    return out


def summarize_detail(detail: pd.DataFrame) -> pd.DataFrame:
    records = [
        _summary_record(
            detail,
            scope="overall",
            segment_column="overall",
            segment_value="all",
        )
    ]
    return pd.DataFrame.from_records(records)


def summarize_segments(detail: pd.DataFrame, *, min_games: int = 1) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    if detail.empty:
        return pd.DataFrame()

    for segment_column in SEGMENT_COLUMNS:
        if segment_column not in detail.columns:
            continue
        for segment_value, group in detail.groupby(segment_column, dropna=False):
            if len(group) < min_games:
                continue
            records.append(
                _summary_record(
                    group,
                    scope="segment",
                    segment_column=segment_column,
                    segment_value=str(segment_value),
                )
            )
    return pd.DataFrame.from_records(records)


def write_artifacts(
    *,
    detail: pd.DataFrame,
    summary: pd.DataFrame,
    segments: pd.DataFrame,
    output_dir: Path,
    season: int,
    week: int,
) -> LiveResultArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / WEEKLY_DETAIL_TEMPLATE.format(week=week)
    summary_path = output_dir / WEEKLY_SUMMARY_TEMPLATE.format(week=week)
    segments_path = output_dir / WEEKLY_SEGMENTS_TEMPLATE.format(week=week)
    cumulative_detail_path = output_dir / CUMULATIVE_DETAIL_NAME
    cumulative_summary_path = output_dir / CUMULATIVE_SUMMARY_NAME

    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    if not segments.empty:
        segments.to_csv(segments_path, index=False)
        segments_output: Path | None = segments_path
    else:
        segments_output = None

    _write_cumulative(
        detail,
        cumulative_detail_path,
        keys=["season", "week", "game_id", "prediction_source_type"],
    )
    _write_cumulative(
        summary.assign(season=season, week=week),
        cumulative_summary_path,
        keys=["season", "week", "scope", "segment_column", "segment_value"],
    )

    return LiveResultArtifacts(
        detail_path=detail_path,
        cumulative_detail_path=cumulative_detail_path,
        summary_path=summary_path,
        cumulative_summary_path=cumulative_summary_path,
        segments_path=segments_output,
    )


def build_and_write_live_results(
    *,
    season: int,
    week: int,
    features_path: Path = DEFAULT_FEATURES_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    market_clv_detail_path: Path = DEFAULT_MARKET_CLV_DETAIL_PATH,
    mtb_csv_root: Path = DEFAULT_MTB_CSV_ROOT,
    predictions_log_dir: Path = DEFAULT_PREDICTIONS_LOG_DIR,
    prediction_file: Path | None = None,
    source_order: Iterable[str] = ("mtb_archive", "locked_log"),
    allow_empty: bool = False,
) -> LiveResultArtifacts:
    prediction_source = load_prediction_source(
        season=season,
        week=week,
        mtb_csv_root=mtb_csv_root,
        predictions_log_dir=predictions_log_dir,
        prediction_file=prediction_file,
        source_order=source_order,
    )
    actuals = load_finalized_actuals(features_path, season=season, week=week)
    clv_detail = load_market_clv_detail(market_clv_detail_path, season=season, week=week)
    detail = build_prediction_vs_actual(prediction_source, actuals, clv_detail=clv_detail)
    if detail.empty and not allow_empty:
        raise RuntimeError(
            f"No finalized games joined for season {season}, week {week}. "
            "Run after final scores are available or pass --allow-empty."
        )
    summary = summarize_detail(detail)
    segments = summarize_segments(detail)
    return write_artifacts(
        detail=detail,
        summary=summary,
        segments=segments,
        output_dir=output_dir,
        season=season,
        week=week,
    )


def _add_moneyline_grades(frame: pd.DataFrame) -> None:
    frame["moneyline_result"] = frame["su_result"]
    home_odds = _numeric_series(frame, "market_home_moneyline")
    away_odds = _numeric_series(frame, "market_away_moneyline")
    frame["model_pick_moneyline"] = np.where(
        frame["predicted_home_win"].fillna(False),
        home_odds,
        away_odds,
    )
    frame.loc[frame["predicted_home_win"].isna(), "model_pick_moneyline"] = np.nan
    decimal = frame["model_pick_moneyline"].apply(_american_to_decimal)
    frame["moneyline_profit_unit"] = np.select(
        [
            frame["moneyline_result"].eq("win") & decimal.notna(),
            frame["moneyline_result"].eq("loss") & decimal.notna(),
            frame["moneyline_result"].eq("push") & decimal.notna(),
        ],
        [decimal - 1.0, -1.0, 0.0],
        default=np.nan,
    )


def _add_spread_grades(frame: pd.DataFrame) -> None:
    if "market_home_margin" not in frame.columns:
        if "market_home_spread" in frame.columns:
            frame["market_home_margin"] = -pd.to_numeric(frame["market_home_spread"], errors="coerce")
        else:
            frame["market_home_margin"] = np.nan
    else:
        frame["market_home_margin"] = pd.to_numeric(frame["market_home_margin"], errors="coerce")

    frame["spread_result_points"] = frame["home_margin"] - frame["market_home_margin"]
    frame["home_cover"] = frame["spread_result_points"] > 0.0
    frame["away_cover"] = frame["spread_result_points"] < 0.0
    frame["spread_push"] = frame["spread_result_points"] == 0.0
    selected_side = frame.get("market_spread_selected_side")
    if selected_side is None:
        edge = _numeric_series(frame, "market_spread_edge_points")
        selected_side = pd.Series(
            np.where(edge >= 0.0, "home", "away"),
            index=frame.index,
            dtype="object",
        )
        selected_side[edge.isna()] = "unavailable"
        frame["market_spread_selected_side"] = selected_side
    else:
        frame["market_spread_selected_side"] = selected_side.fillna("unavailable")

    selected_cover = np.where(
        frame["market_spread_selected_side"].eq("home"),
        frame["home_cover"],
        np.where(frame["market_spread_selected_side"].eq("away"), frame["away_cover"], np.nan),
    )
    frame["selected_spread_cover"] = selected_cover
    frame["selected_spread_result"] = np.select(
        [
            frame["market_home_margin"].isna() | frame["market_spread_selected_side"].eq("unavailable"),
            frame["spread_push"],
            pd.Series(selected_cover, index=frame.index).eq(True),
            pd.Series(selected_cover, index=frame.index).eq(False),
        ],
        ["unavailable", "push", "win", "loss"],
        default="ungraded",
    )


def _summary_record(
    frame: pd.DataFrame,
    *,
    scope: str,
    segment_column: str,
    segment_value: str,
) -> dict[str, object]:
    record: dict[str, object] = {
        "scope": scope,
        "segment_column": segment_column,
        "segment_value": segment_value,
        "n_games": int(len(frame)),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if frame.empty:
        return {
            **record,
            "accuracy": np.nan,
            "f1": np.nan,
            "auc": np.nan,
            "brier": np.nan,
            "log_loss": np.nan,
            "mae_margin": np.nan,
            "rmse_margin": np.nan,
            "ats_graded": 0,
            "ats_hit_rate": np.nan,
            "moneyline_graded": 0,
            "moneyline_roi": np.nan,
            "clv_detail_games": 0,
            "moneyline_clv_tracked": 0,
            "avg_moneyline_clv_prob_delta": np.nan,
            "positive_moneyline_clv_rate": np.nan,
            "spread_clv_tracked": 0,
            "avg_spread_clv_points": np.nan,
            "positive_spread_clv_rate": np.nan,
        }

    graded = frame[frame["actual_home_win"].notna()].copy()
    if not graded.empty:
        graded["predicted_home_win"] = graded["predicted_home_win"].astype(bool).astype(int)
        class_metrics = _classification_metrics(graded)
        record.update(class_metrics)
        record["accuracy"] = float(graded["su_correct"].astype(float).mean())
        record["auc"] = (
            float(roc_auc_score(graded["actual_home_win"], graded["home_win_prob"]))
            if graded["actual_home_win"].nunique() == 2
            else np.nan
        )
        record["brier"] = float(graded["brier_component"].mean())
        record["log_loss"] = float(graded["log_loss_component"].mean())
    else:
        record.update(_classification_metrics(pd.DataFrame()))
        record.update({"accuracy": np.nan, "auc": np.nan, "brier": np.nan, "log_loss": np.nan})

    margin = frame.dropna(subset=["abs_margin_error", "sq_margin_error"])
    record["mae_margin"] = float(margin["abs_margin_error"].mean()) if not margin.empty else np.nan
    record["rmse_margin"] = (
        float(math.sqrt(margin["sq_margin_error"].mean())) if not margin.empty else np.nan
    )

    ats = frame[frame["selected_spread_result"].isin(["win", "loss", "push"])]
    ats_decisions = ats[ats["selected_spread_result"].isin(["win", "loss"])]
    record["ats_graded"] = int(len(ats))
    record["ats_wins"] = int(ats["selected_spread_result"].eq("win").sum())
    record["ats_losses"] = int(ats["selected_spread_result"].eq("loss").sum())
    record["ats_pushes"] = int(ats["selected_spread_result"].eq("push").sum())
    record["ats_hit_rate"] = (
        float(ats_decisions["selected_spread_result"].eq("win").mean())
        if not ats_decisions.empty
        else np.nan
    )

    ml = frame[frame["moneyline_result"].isin(["win", "loss", "push"])]
    ml_decisions = ml[ml["moneyline_result"].isin(["win", "loss"])]
    ml_profit = pd.to_numeric(ml["moneyline_profit_unit"], errors="coerce").dropna()
    record["moneyline_graded"] = int(len(ml))
    record["moneyline_wins"] = int(ml["moneyline_result"].eq("win").sum())
    record["moneyline_losses"] = int(ml["moneyline_result"].eq("loss").sum())
    record["moneyline_pushes"] = int(ml["moneyline_result"].eq("push").sum())
    record["moneyline_hit_rate"] = (
        float(ml_decisions["moneyline_result"].eq("win").mean()) if not ml_decisions.empty else np.nan
    )
    record["moneyline_roi"] = float(ml_profit.mean()) if not ml_profit.empty else np.nan

    clv_detail_available = _boolish_series(frame, "clv_detail_available")
    record["clv_detail_games"] = int(clv_detail_available.sum())
    moneyline_clv = pd.to_numeric(
        frame.get("moneyline_clv_prob_delta", pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    )
    moneyline_clv = moneyline_clv.dropna()
    record["moneyline_clv_tracked"] = int(len(moneyline_clv))
    record["avg_moneyline_clv_prob_delta"] = (
        float(moneyline_clv.mean()) if not moneyline_clv.empty else np.nan
    )
    moneyline_positive = _boolish_series(frame, "moneyline_positive_clv")
    moneyline_positive = moneyline_positive[
        pd.to_numeric(
            frame.get("moneyline_clv_prob_delta", pd.Series(np.nan, index=frame.index)),
            errors="coerce",
        ).notna()
    ]
    record["positive_moneyline_clv_rate"] = (
        float(moneyline_positive.mean()) if not moneyline_positive.empty else np.nan
    )
    spread_clv = pd.to_numeric(
        frame.get("spread_clv_points", pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    ).dropna()
    record["spread_clv_tracked"] = int(len(spread_clv))
    record["avg_spread_clv_points"] = float(spread_clv.mean()) if not spread_clv.empty else np.nan
    spread_positive = _boolish_series(frame, "spread_positive_clv")
    spread_positive = spread_positive[
        pd.to_numeric(
            frame.get("spread_clv_points", pd.Series(np.nan, index=frame.index)),
            errors="coerce",
        ).notna()
    ]
    record["positive_spread_clv_rate"] = (
        float(spread_positive.mean()) if not spread_positive.empty else np.nan
    )
    return record


def _write_cumulative(frame: pd.DataFrame, path: Path, *, keys: list[str]) -> None:
    if path.exists():
        existing = pd.read_csv(path)
        combined = pd.concat([existing, frame], ignore_index=True)
    else:
        combined = frame.copy()
    if not combined.empty and set(keys).issubset(combined.columns):
        combined = combined.drop_duplicates(subset=keys, keep="last")
        sort_cols = [col for col in ["season", "week", "game_id", "scope", "segment_column"] if col in combined.columns]
        if sort_cols:
            combined = combined.sort_values(sort_cols)
    combined.to_csv(path, index=False)


def _raise_on_duplicate_keys(frame: pd.DataFrame, keys: list[str], label: str) -> None:
    dupes = frame[frame.duplicated(keys, keep=False)]
    if dupes.empty:
        return
    sample = dupes[keys].drop_duplicates().head(10).to_dict("records")
    raise ValueError(f"Duplicate {label} rows for keys {keys}: {sample}")


def _american_to_decimal(value: object) -> float:
    try:
        odds = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if np.isnan(odds) or odds == 0.0:
        return float("nan")
    if odds > 0:
        return 1.0 + odds / 100.0
    return 1.0 + 100.0 / abs(odds)


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _boolish_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    series = frame[column]
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "1.0", "yes", "y"})


def _ensure_clv_columns(frame: pd.DataFrame) -> None:
    if "clv_detail_available" not in frame.columns:
        frame["clv_detail_available"] = False
    if "clv_status" not in frame.columns:
        frame["clv_status"] = "unavailable_missing_clv_detail"
    for col in CLV_OUTPUT_COLUMNS:
        if col not in frame.columns:
            frame[col] = np.nan
    frame["clv_detail_available"] = frame["clv_detail_available"].fillna(False)
    frame["clv_status"] = frame["clv_status"].fillna("unavailable_missing_clv_detail")


def _empty_detail_frame(prediction_columns: Iterable[str]) -> pd.DataFrame:
    computed = [
        "prediction_source_type",
        "prediction_source_file",
        "prediction_source_last_modified_utc",
        "graded_at_utc",
        "home_score",
        "away_score",
        "home_margin",
        "actual_home_win",
        "actual_winner",
        "predicted_home_win",
        "model_pick_team",
        "model_pick_prob",
        "su_correct",
        "su_result",
        "brier_component",
        "log_loss_component",
        "margin_error",
        "abs_margin_error",
        "moneyline_result",
        "moneyline_profit_unit",
        "selected_spread_result",
    ]
    columns = list(dict.fromkeys([*prediction_columns, *computed]))
    return pd.DataFrame(columns=columns)


def _order_detail_columns(frame: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "season",
        "week",
        "game_id",
        "prediction_source_type",
        "prediction_source_file",
        "prediction_source_last_modified_utc",
        "graded_at_utc",
        "home_team",
        "away_team",
        "home_team_actual",
        "away_team_actual",
        "home_score",
        "away_score",
        "home_margin",
        "actual_winner",
        "actual_home_win",
        "home_win_prob",
        "predicted_home_win",
        "model_pick_team",
        "model_pick_prob",
        "su_correct",
        "su_result",
        "brier_component",
        "log_loss_component",
        "pred_home_margin",
        "margin_error",
        "abs_margin_error",
        "sq_margin_error",
        "moneyline_result",
        "model_pick_moneyline",
        "moneyline_profit_unit",
        "market_home_margin",
        "spread_result_points",
        "market_spread_selected_side",
        "selected_spread_cover",
        "selected_spread_result",
        "clv_detail_available",
        "clv_status",
        "clv_model_pick_side",
        "clv_model_pick_team",
        "clv_model_side_pick_moneyline",
        "clv_model_side_close_moneyline",
        "moneyline_clv_decimal_delta",
        "moneyline_clv_prob_delta",
        "moneyline_positive_clv",
        "spread_clv_points",
        "spread_positive_clv",
    ]
    ordered = [col for col in preferred if col in frame.columns]
    ordered.extend(col for col in frame.columns if col not in ordered)
    return frame[ordered].copy()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    artifacts = build_and_write_live_results(
        season=args.season,
        week=args.week,
        features_path=args.features_path,
        output_dir=args.output_dir,
        market_clv_detail_path=args.market_clv_detail,
        mtb_csv_root=args.mtb_csv_root,
        predictions_log_dir=args.predictions_log_dir,
        prediction_file=args.prediction_file,
        source_order=args.source_order,
        allow_empty=args.allow_empty,
    )
    if args.debug:
        print(f"[live_results] wrote detail -> {artifacts.detail_path.resolve()}")
        print(f"[live_results] wrote cumulative detail -> {artifacts.cumulative_detail_path.resolve()}")
        print(f"[live_results] wrote summary -> {artifacts.summary_path.resolve()}")
        print(f"[live_results] wrote cumulative summary -> {artifacts.cumulative_summary_path.resolve()}")
        if artifacts.segments_path is not None:
            print(f"[live_results] wrote segments -> {artifacts.segments_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
