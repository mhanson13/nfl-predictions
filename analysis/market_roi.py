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

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from math import erf, sqrt
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.utils.week_filter import filter_before_week


ANALYSIS_DIR = Path("analysis")
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

MERGED_PREDICTIONS_PATH = Path("predictions/evaluation/merged_predictions_actuals.csv")
MATCHUP_FEATURES_PATH = Path("data/processed/matchup_features.parquet")

SPREAD_EDGE_BINS = [
    (0.0, 1.0),
    (1.0, 2.0),
    (2.0, 3.0),
    (3.0, 5.0),
    (5.0, 7.0),
    (7.0, 10.0),
    (10.0, np.inf),
]

SPREAD_KEY_NUMBERS = [3.0, 6.0, 7.0, 10.0]
SPREAD_KEY_NUMBER_TOLERANCE = 0.5

MARKET_PROB_EDGE_BINS = [
    (-np.inf, 0.0, "<0%"),
    (0.0, 0.02, "0-2%"),
    (0.02, 0.05, "2-5%"),
    (0.05, 0.08, "5-8%"),
    (0.08, 0.10, "8-10%"),
    (0.10, 0.15, "10-15%"),
    (0.15, np.inf, "15%+"),
]

MARKET_MODEL_CONFIDENCE_BINS = [
    (0.50, 0.55, "50-55%"),
    (0.55, 0.60, "55-60%"),
    (0.60, 0.65, "60-65%"),
    (0.65, 0.70, "65-70%"),
    (0.70, np.inf, "70%+"),
]

MARKET_MONEYLINE_BINS = [
    (-np.inf, -150.0, "favorite <=-150"),
    (-150.0, -110.0, "favorite -150 to -110"),
    (-110.0, 110.0, "near pick"),
    (110.0, 200.0, "underdog +110 to +200"),
    (200.0, np.inf, "underdog +200+"),
]

MARKET_SELECTED_PROBABILITY_BINS = [
    (0.0, 0.35, "<35%"),
    (0.35, 0.40, "35-40%"),
    (0.40, 0.45, "40-45%"),
    (0.45, 0.50, "45-50%"),
    (0.50, 0.55, "50-55%"),
    (0.55, 0.60, "55-60%"),
    (0.60, 0.65, "60-65%"),
    (0.65, 0.70, "65-70%"),
    (0.70, np.inf, "70%+"),
]

SPREAD_COVER_PROB_BINS = [
    (0.50, 0.55),
    (0.55, 0.60),
    (0.60, 0.65),
    (0.65, 0.70),
    (0.70, 0.75),
    (0.75, 0.80),
    (0.80, np.inf),
]

ALLOWED_SPREAD_POLICY_EDGE_BINS = {"2-3"}
SPREAD_COVER_CALIBRATION_MIN_BETS = 50
SPREAD_COVER_CALIBRATION_MAX_ABS_ERROR = 0.05
SPREAD_COVER_CALIBRATION_CONFIDENCE_MIN_PROB = 0.65
MARKET_CLV_SUMMARY_PATH = ANALYSIS_DIR / "market_clv_summary.csv"
MARKET_CLV_DETAIL_PATH = ANALYSIS_DIR / "market_clv_detail.csv"
MARKET_CLV_MIN_TRACKED_GAMES = 10
MARKET_CLV_MIN_POSITIVE_RATE = 0.52
MARKET_CLV_MIN_AVG_PROB_DELTA = 0.0

MARKET_BENCHMARK_CLV_COLUMNS = [
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

MARKET_DISAGREEMENT_SEGMENT_COLUMNS = [
    "scope",
    "season",
    "segment_family",
    "segment",
    "n_bets",
    "wins",
    "losses",
    "pushes",
    "hit_rate",
    "roi",
    "total_profit",
    "avg_model_pick_prob",
    "avg_vegas_model_side_prob",
    "avg_prob_edge_model_side",
    "avg_model_side_moneyline",
    "avg_model_side_decimal",
    "clv_games",
    "avg_moneyline_clv_prob_delta",
    "positive_moneyline_clv_rate",
]

MARKET_EDGE_POLICY_COLUMNS = [
    "signal",
    "recommended",
    "recommendation_tier",
    "selection",
    "threshold",
    "edge_bin",
    "bet_side",
    "n_bets",
    "wins",
    "losses",
    "pushes",
    "hit_rate",
    "roi",
    "avg_prob_edge_model_side",
    "avg_edge_points",
    "avg_abs_edge_points",
    "avg_expected_profit",
    "policy_min_bets",
    "policy_min_roi",
    "policy_min_probability_edge",
    "policy_min_spread_edge",
    "stability_pass",
    "stability_status",
    "stability_qualified_seasons",
    "stability_positive_season_rate",
    "stability_recent_positive_season_rate",
    "stability_severe_negative_season_rate",
    "stability_min_season_bets",
    "stability_min_qualified_seasons",
    "clv_pass",
    "clv_status",
    "clv_threshold",
    "clv_tracked_games",
    "clv_avg_prob_delta",
    "clv_positive_rate",
    "clv_min_tracked_games",
    "clv_min_positive_rate",
    "clv_min_avg_prob_delta",
    "spread_cover_calibration_pass",
    "spread_cover_calibration_status",
    "spread_cover_calibration_evaluated_bins",
    "spread_cover_calibration_max_abs_error",
    "spread_cover_calibration_worst_bin",
    "spread_cover_calibration_worst_error",
    "spread_cover_calibration_min_bets",
    "spread_cover_calibration_max_abs_error_threshold",
    "spread_cover_calibration_confidence_min_prob",
    "source_artifact",
    "generated_at",
]


def american_to_decimal(odds: float) -> float:
    if pd.isna(odds):
        return np.nan
    odds = float(odds)
    if odds > 0:
        return 1.0 + odds / 100.0
    if odds < 0:
        return 1.0 + 100.0 / abs(odds)
    return 1.0


def american_to_implied_prob(odds: float) -> float:
    if pd.isna(odds):
        return np.nan
    odds = float(odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    if odds < 0:
        return -odds / (-odds + 100.0)
    return 0.5


def standard_normal_cdf(value: pd.Series | np.ndarray | float) -> np.ndarray:
    scale = sqrt(2.0)
    arr = np.asarray(value, dtype=float)
    erf_vec = np.vectorize(erf, otypes=[float])
    return 0.5 * (1.0 + erf_vec(arr / scale))


def load_predictions(
    start_season: int | None,
    end_season: int | None,
    exclude_from_season: int | None = None,
    exclude_from_week: int | None = None,
) -> pd.DataFrame:
    if not MERGED_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"{MERGED_PREDICTIONS_PATH} not found. Run evaluation before market ROI analysis."
        )
    df = pd.read_csv(MERGED_PREDICTIONS_PATH)
    if start_season is not None:
        df = df[df["season"] >= start_season]
    if end_season is not None:
        df = df[df["season"] <= end_season]
    df = filter_before_week(df, exclude_from_season, exclude_from_week)
    df = df.dropna(subset=["home_win_prob", "pred_home_margin", "home_margin"])
    df["home_margin"] = df["home_margin"].astype(float)
    return df


def load_market_lines(columns: Iterable[str]) -> pd.DataFrame:
    if not MATCHUP_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"{MATCHUP_FEATURES_PATH} not found. Build features before market ROI analysis."
        )
    try:
        df = pd.read_parquet(MATCHUP_FEATURES_PATH, columns=list(columns))
    except Exception:
        df = pd.read_parquet(MATCHUP_FEATURES_PATH, columns=list(columns), engine="fastparquet")
    return df


def prepare_dataset(pred_df: pd.DataFrame) -> pd.DataFrame:
    odds_columns = [
        "game_id",
        "home_moneyline",
        "away_moneyline",
        "spread_line",
        "home_spread_odds",
        "away_spread_odds",
    ]
    odds_df = load_market_lines(odds_columns)
    if odds_df.duplicated("game_id").any():
        dupes = odds_df.loc[odds_df.duplicated("game_id", keep=False), "game_id"].unique()
        print(f"[market_roi] warning: dropping duplicate odds rows for game_id(s) {dupes[:5]!r}")
        odds_df = odds_df.drop_duplicates("game_id", keep="last")
    merged = pred_df.merge(odds_df, on="game_id", how="left", validate="one_to_one")
    for col in ["home_moneyline", "away_moneyline", "spread_line", "home_spread_odds", "away_spread_odds"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    merged = merged.dropna(subset=["home_moneyline", "away_moneyline", "spread_line"])

    merged["away_win_prob"] = 1.0 - merged["home_win_prob"]
    merged["home_decimal"] = merged["home_moneyline"].apply(american_to_decimal)
    merged["away_decimal"] = merged["away_moneyline"].apply(american_to_decimal)
    merged["home_implied_prob"] = merged["home_moneyline"].apply(american_to_implied_prob)
    merged["away_implied_prob"] = merged["away_moneyline"].apply(american_to_implied_prob)
    implied_sum = merged["home_implied_prob"] + merged["away_implied_prob"]
    merged["home_implied_prob_novig"] = np.where(
        implied_sum > 0.0,
        merged["home_implied_prob"] / implied_sum,
        np.nan,
    )
    merged["away_implied_prob_novig"] = np.where(
        implied_sum > 0.0,
        merged["away_implied_prob"] / implied_sum,
        np.nan,
    )
    merged["home_tie"] = merged["home_margin"] == 0.0
    merged["home_actual_win"] = (merged["home_margin"] > 0.0).astype(int)
    merged["away_actual_win"] = (merged["home_margin"] < 0.0).astype(int)

    merged["home_spread_decimal"] = merged["home_spread_odds"].apply(american_to_decimal)
    merged["away_spread_decimal"] = merged["away_spread_odds"].apply(american_to_decimal)
    # matchup_features normalizes spread_line as market expected home margin:
    # +6.5 means the home team is favored by 6.5, -3.0 means the away team is favored by 3.
    merged["market_home_margin"] = merged["spread_line"]
    merged["spread_edge_points"] = merged["pred_home_margin"] - merged["market_home_margin"]
    merged["spread_result_points"] = merged["home_margin"] - merged["market_home_margin"]
    merged["home_cover"] = merged["spread_result_points"] > 0.0
    merged["away_cover"] = merged["spread_result_points"] < 0.0
    merged["spread_push"] = merged["spread_result_points"] == 0.0
    merged["margin_error"] = merged["pred_home_margin"] - merged["home_margin"]
    return merged


def summarize_moneyline(df: pd.DataFrame, thresholds: Iterable[float]) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []
    for side in ("home", "away"):
        prob_col = "home_win_prob" if side == "home" else "away_win_prob"
        decimal_col = f"{side}_decimal"
        implied_col = f"{side}_implied_prob"
        actual_col = f"{side}_actual_win"
        tie_col = "home_tie"
        moneyline_col = f"{side}_moneyline"

        base = df.dropna(subset=[prob_col, decimal_col, implied_col, moneyline_col]).copy()
        base["expected_profit"] = base[prob_col] * (base[decimal_col] - 1.0) - (1.0 - base[prob_col])

        for threshold in thresholds:
            selection = base[base["expected_profit"] > threshold]
            n_bets = int(len(selection))
            if n_bets == 0:
                records.append(
                    {
                        "bet_type": "moneyline",
                        "bet_side": side,
                        "threshold": float(threshold),
                        "n_bets": 0,
                        "wins": 0,
                        "losses": 0,
                        "pushes": 0,
                        "hit_rate": np.nan,
                        "avg_model_prob": np.nan,
                        "avg_implied_prob": np.nan,
                        "avg_expected_profit": np.nan,
                        "avg_edge_prob": np.nan,
                        "avg_odds_decimal": np.nan,
                        "avg_odds_american": np.nan,
                        "total_profit": 0.0,
                        "roi": np.nan,
                    }
                )
                continue

            pushes = int(selection[tie_col].sum())
            wins = int(selection.loc[~selection[tie_col], actual_col].sum())
            losses = n_bets - wins - pushes
            profits = np.where(
                selection[tie_col],
                0.0,
                np.where(
                    selection[actual_col] == 1,
                    selection[decimal_col] - 1.0,
                    -1.0,
                ),
            )
            total_profit = float(np.sum(profits))
            denom = wins + losses
            hit_rate = float(wins / denom) if denom > 0 else np.nan
            roi = float(total_profit / n_bets)

            avg_model_prob = float(selection[prob_col].mean())
            avg_implied = float(selection[implied_col].mean())
            records.append(
                {
                    "bet_type": "moneyline",
                    "bet_side": side,
                    "threshold": float(threshold),
                    "n_bets": n_bets,
                    "wins": wins,
                    "losses": losses,
                    "pushes": pushes,
                    "hit_rate": hit_rate,
                    "avg_model_prob": avg_model_prob,
                    "avg_implied_prob": avg_implied,
                    "avg_expected_profit": float(selection["expected_profit"].mean()),
                    "avg_edge_prob": float(avg_model_prob - avg_implied),
                    "avg_odds_decimal": float(selection[decimal_col].mean()),
                    "avg_odds_american": float(selection[moneyline_col].mean()),
                    "total_profit": total_profit,
                    "roi": roi,
                }
            )
    return pd.DataFrame.from_records(records)


def build_market_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    """Build per-game model-vs-market benchmark rows."""
    benchmark = df.dropna(
        subset=[
            "home_win_prob",
            "away_win_prob",
            "home_implied_prob_novig",
            "away_implied_prob_novig",
            "home_decimal",
            "away_decimal",
            "home_margin",
        ]
    ).copy()
    if benchmark.empty:
        return benchmark

    benchmark["model_home_pick"] = benchmark["home_win_prob"] >= 0.5
    benchmark["vegas_home_pick"] = benchmark["home_implied_prob_novig"] >= 0.5
    benchmark["favorite_agreement"] = benchmark["model_home_pick"] == benchmark["vegas_home_pick"]
    benchmark["actual_home_win_bool"] = benchmark["home_margin"] > 0.0

    benchmark["model_correct"] = np.where(
        benchmark["home_tie"],
        np.nan,
        benchmark["model_home_pick"] == benchmark["actual_home_win_bool"],
    )
    benchmark["vegas_correct"] = np.where(
        benchmark["home_tie"],
        np.nan,
        benchmark["vegas_home_pick"] == benchmark["actual_home_win_bool"],
    )

    benchmark["model_pick"] = np.where(
        benchmark["model_home_pick"],
        benchmark.get("home_team", "HOME"),
        benchmark.get("away_team", "AWAY"),
    )
    benchmark["vegas_pick"] = np.where(
        benchmark["vegas_home_pick"],
        benchmark.get("home_team", "HOME"),
        benchmark.get("away_team", "AWAY"),
    )
    benchmark["actual_winner"] = np.select(
        [benchmark["home_margin"] > 0.0, benchmark["home_margin"] < 0.0],
        [benchmark.get("home_team", "HOME"), benchmark.get("away_team", "AWAY")],
        default="TIE",
    )

    benchmark["model_pick_prob"] = np.where(
        benchmark["model_home_pick"],
        benchmark["home_win_prob"],
        benchmark["away_win_prob"],
    )
    benchmark["vegas_pick_prob"] = np.where(
        benchmark["vegas_home_pick"],
        benchmark["home_implied_prob_novig"],
        benchmark["away_implied_prob_novig"],
    )
    benchmark["vegas_model_side_prob"] = np.where(
        benchmark["model_home_pick"],
        benchmark["home_implied_prob_novig"],
        benchmark["away_implied_prob_novig"],
    )
    benchmark["prob_edge_model_side"] = benchmark["model_pick_prob"] - benchmark["vegas_model_side_prob"]

    benchmark["model_side_moneyline"] = np.where(
        benchmark["model_home_pick"],
        benchmark["home_moneyline"],
        benchmark["away_moneyline"],
    )
    benchmark["model_side_decimal"] = np.where(
        benchmark["model_home_pick"],
        benchmark["home_decimal"],
        benchmark["away_decimal"],
    )
    benchmark["model_side_actual_win"] = np.where(
        benchmark["model_home_pick"],
        benchmark["home_actual_win"],
        benchmark["away_actual_win"],
    )
    benchmark["model_side_profit"] = np.where(
        benchmark["home_tie"],
        0.0,
        np.where(
            benchmark["model_side_actual_win"] == 1,
            benchmark["model_side_decimal"] - 1.0,
            -1.0,
        ),
    )

    benchmark["vegas_side_moneyline"] = np.where(
        benchmark["vegas_home_pick"],
        benchmark["home_moneyline"],
        benchmark["away_moneyline"],
    )
    benchmark["vegas_side_decimal"] = np.where(
        benchmark["vegas_home_pick"],
        benchmark["home_decimal"],
        benchmark["away_decimal"],
    )
    benchmark["vegas_side_actual_win"] = np.where(
        benchmark["vegas_home_pick"],
        benchmark["home_actual_win"],
        benchmark["away_actual_win"],
    )
    benchmark["vegas_side_profit"] = np.where(
        benchmark["home_tie"],
        0.0,
        np.where(
            benchmark["vegas_side_actual_win"] == 1,
            benchmark["vegas_side_decimal"] - 1.0,
            -1.0,
        ),
    )
    return benchmark


def load_market_clv_detail(path: Path) -> pd.DataFrame:
    """Load optional per-game CLV detail for benchmark enrichment."""
    if not path.exists():
        return pd.DataFrame()
    detail = pd.read_csv(path)
    if "game_id" not in detail.columns:
        raise ValueError(f"{path} must include game_id")
    for col in ["season", "week"]:
        if col in detail.columns:
            detail[col] = pd.to_numeric(detail[col], errors="coerce").astype("Int64")
    return detail


def attach_market_clv_detail(benchmark: pd.DataFrame, clv_detail: pd.DataFrame) -> pd.DataFrame:
    """Attach reliable pick-time-vs-close CLV fields to per-game benchmark rows."""
    enriched = benchmark.copy()
    if enriched.empty:
        for col in MARKET_BENCHMARK_CLV_COLUMNS:
            enriched[col] = pd.Series(dtype=object)
        return enriched

    if clv_detail.empty:
        enriched["clv_detail_available"] = False
        enriched["clv_status"] = "unavailable_missing_clv_detail"
        for col in MARKET_BENCHMARK_CLV_COLUMNS:
            if col not in enriched.columns:
                enriched[col] = np.nan
        return enriched

    rename_map = {
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
    passthrough_cols = [
        "clv_status",
        "moneyline_clv_decimal_delta",
        "moneyline_clv_prob_delta",
        "moneyline_positive_clv",
        "spread_clv_points",
        "spread_positive_clv",
    ]
    source_cols = [
        col
        for col in ["season", "week", "game_id", *rename_map.keys(), *passthrough_cols]
        if col in clv_detail.columns
    ]
    detail = clv_detail[source_cols].rename(columns=rename_map).copy()
    merge_keys = ["game_id"]
    if {"season", "week"}.issubset(enriched.columns) and {"season", "week"}.issubset(detail.columns):
        for frame in [enriched, detail]:
            frame["season"] = pd.to_numeric(frame["season"], errors="coerce").astype("Int64")
            frame["week"] = pd.to_numeric(frame["week"], errors="coerce").astype("Int64")
        merge_keys = ["season", "week", "game_id"]
    detail = detail.drop_duplicates(merge_keys, keep="last")

    enriched = enriched.merge(detail, on=merge_keys, how="left", validate="many_to_one")
    enriched["clv_detail_available"] = enriched["clv_status"].notna()
    enriched["clv_status"] = enriched["clv_status"].fillna("unavailable_missing_clv_detail")

    expected_side = pd.Series(
        np.where(enriched["model_home_pick"].fillna(False).astype(bool), "home", "away"),
        index=enriched.index,
    )
    detail_side = enriched.get("clv_model_pick_side", pd.Series(pd.NA, index=enriched.index)).astype("string")
    side_mismatch = detail_side.notna() & detail_side.ne(expected_side)
    if side_mismatch.any():
        invalid_cols = [
            col
            for col in MARKET_BENCHMARK_CLV_COLUMNS
            if col not in {"clv_detail_available", "clv_status", "clv_model_pick_side", "clv_model_pick_team"}
        ]
        for col in invalid_cols:
            if col in enriched.columns:
                enriched.loc[side_mismatch, col] = np.nan
        enriched.loc[side_mismatch, "clv_status"] = "unavailable_model_side_mismatch"

    for col in MARKET_BENCHMARK_CLV_COLUMNS:
        if col not in enriched.columns:
            enriched[col] = np.nan
    return enriched


def _benchmark_record(
    *,
    scope: str,
    season: int | str,
    segment: str,
    selection: pd.DataFrame,
) -> dict[str, float | int | str]:
    n_games = int(len(selection))
    if n_games == 0:
        return {
            "scope": scope,
            "season": season,
            "segment": segment,
            "n_games": 0,
            "model_accuracy": np.nan,
            "vegas_accuracy": np.nan,
            "favorite_agreement_rate": np.nan,
            "model_side_roi": np.nan,
            "vegas_side_roi": np.nan,
            "avg_model_pick_prob": np.nan,
            "avg_vegas_pick_prob": np.nan,
            "avg_vegas_model_side_prob": np.nan,
            "avg_prob_edge_model_side": np.nan,
        }

    return {
        "scope": scope,
        "season": season,
        "segment": segment,
        "n_games": n_games,
        "model_accuracy": float(selection["model_correct"].mean()),
        "vegas_accuracy": float(selection["vegas_correct"].mean()),
        "favorite_agreement_rate": float(selection["favorite_agreement"].mean()),
        "model_side_roi": float(selection["model_side_profit"].sum() / n_games),
        "vegas_side_roi": float(selection["vegas_side_profit"].sum() / n_games),
        "avg_model_pick_prob": float(selection["model_pick_prob"].mean()),
        "avg_vegas_pick_prob": float(selection["vegas_pick_prob"].mean()),
        "avg_vegas_model_side_prob": float(selection["vegas_model_side_prob"].mean()),
        "avg_prob_edge_model_side": float(selection["prob_edge_model_side"].mean()),
    }


def summarize_market_benchmark(benchmark: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []
    if benchmark.empty:
        return pd.DataFrame.from_records(records)

    def add_group(scope: str, season: int | str, group: pd.DataFrame) -> None:
        segments = [
            ("all", group),
            ("agree", group[group["favorite_agreement"]]),
            ("disagree", group[~group["favorite_agreement"]]),
        ]
        for segment, selection in segments:
            if not selection.empty:
                records.append(
                    _benchmark_record(
                        scope=scope,
                        season=season,
                        segment=segment,
                        selection=selection,
                    )
                )

    add_group("overall", "all", benchmark)
    for season, season_df in benchmark.groupby("season", sort=True):
        add_group("season", int(season), season_df)

    return pd.DataFrame.from_records(records)


def summarize_disagreement_moneyline_roi(
    benchmark: pd.DataFrame,
    thresholds: Iterable[float],
) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []
    base = benchmark[(~benchmark["favorite_agreement"]) & benchmark["prob_edge_model_side"].notna()].copy()

    for threshold in thresholds:
        selection = base[base["prob_edge_model_side"] >= threshold]
        n_bets = int(len(selection))
        if n_bets == 0:
            records.append(
                {
                    "bet_type": "moneyline_disagreement",
                    "threshold": float(threshold),
                    "n_bets": 0,
                    "wins": 0,
                    "losses": 0,
                    "pushes": 0,
                    "hit_rate": np.nan,
                    "avg_prob_edge_model_side": np.nan,
                    "avg_model_pick_prob": np.nan,
                    "avg_vegas_model_side_prob": np.nan,
                    "avg_odds_decimal": np.nan,
                    "avg_odds_american": np.nan,
                    "total_profit": 0.0,
                    "roi": np.nan,
                }
            )
            continue

        pushes = int(selection["home_tie"].sum())
        wins = int(selection.loc[~selection["home_tie"], "model_side_actual_win"].sum())
        losses = n_bets - wins - pushes
        total_profit = float(selection["model_side_profit"].sum())
        denom = wins + losses

        records.append(
            {
                "bet_type": "moneyline_disagreement",
                "threshold": float(threshold),
                "n_bets": n_bets,
                "wins": wins,
                "losses": losses,
                "pushes": pushes,
                "hit_rate": float(wins / denom) if denom > 0 else np.nan,
                "avg_prob_edge_model_side": float(selection["prob_edge_model_side"].mean()),
                "avg_model_pick_prob": float(selection["model_pick_prob"].mean()),
                "avg_vegas_model_side_prob": float(selection["vegas_model_side_prob"].mean()),
                "avg_odds_decimal": float(selection["model_side_decimal"].mean()),
                "avg_odds_american": float(selection["model_side_moneyline"].mean()),
                "total_profit": total_profit,
                "roi": float(total_profit / n_bets),
            }
        )

    return pd.DataFrame.from_records(records)


def _empty_moneyline_stability_row(
    *,
    threshold: float,
    min_season_bets: int,
    min_qualified_seasons: int,
    min_positive_season_rate: float,
    severe_negative_roi: float,
    max_severe_negative_season_rate: float,
    recent_seasons: int,
    min_recent_qualified_seasons: int,
    min_recent_positive_season_rate: float,
) -> dict[str, object]:
    return {
        "bet_type": "moneyline_disagreement",
        "threshold": float(threshold),
        "n_bets": 0,
        "wins": 0,
        "losses": 0,
        "pushes": 0,
        "roi": np.nan,
        "qualified_seasons": 0,
        "positive_qualified_seasons": 0,
        "severe_negative_qualified_seasons": 0,
        "positive_qualified_season_rate": np.nan,
        "severe_negative_qualified_season_rate": np.nan,
        "recent_qualified_seasons": 0,
        "recent_positive_qualified_seasons": 0,
        "recent_positive_qualified_season_rate": np.nan,
        "stability_pass": False,
        "stability_status": "insufficient_qualified_seasons",
        "min_season_bets": int(min_season_bets),
        "min_qualified_seasons": int(min_qualified_seasons),
        "min_positive_season_rate": float(min_positive_season_rate),
        "severe_negative_roi": float(severe_negative_roi),
        "max_severe_negative_season_rate": float(max_severe_negative_season_rate),
        "recent_seasons": int(recent_seasons),
        "min_recent_qualified_seasons": int(min_recent_qualified_seasons),
        "min_recent_positive_season_rate": float(min_recent_positive_season_rate),
    }


def _moneyline_stability_status(
    *,
    qualified_seasons: int,
    positive_qualified_season_rate: float,
    severe_negative_qualified_season_rate: float,
    recent_qualified_seasons: int,
    recent_positive_qualified_season_rate: float,
    min_qualified_seasons: int,
    min_positive_season_rate: float,
    max_severe_negative_season_rate: float,
    min_recent_qualified_seasons: int,
    min_recent_positive_season_rate: float,
) -> tuple[bool, str]:
    if qualified_seasons < int(min_qualified_seasons):
        return False, "insufficient_qualified_seasons"
    if positive_qualified_season_rate < float(min_positive_season_rate):
        return False, "weak_positive_season_rate"
    if severe_negative_qualified_season_rate > float(max_severe_negative_season_rate):
        return False, "severe_negative_season_rate"
    if recent_qualified_seasons < int(min_recent_qualified_seasons):
        return False, "insufficient_recent_qualified_seasons"
    if recent_positive_qualified_season_rate < float(min_recent_positive_season_rate):
        return False, "weak_recent_positive_season_rate"
    return True, "stable"


def summarize_disagreement_moneyline_season_stability(
    benchmark: pd.DataFrame,
    thresholds: Iterable[float],
    *,
    min_season_bets: int = 20,
    min_qualified_seasons: int = 5,
    min_positive_season_rate: float = 0.55,
    severe_negative_roi: float = -0.15,
    max_severe_negative_season_rate: float = 0.25,
    recent_seasons: int = 3,
    min_recent_qualified_seasons: int = 2,
    min_recent_positive_season_rate: float = 0.50,
) -> pd.DataFrame:
    """Evaluate whether moneyline disagreement ROI is stable across seasons."""
    columns = list(
        _empty_moneyline_stability_row(
            threshold=0.0,
            min_season_bets=min_season_bets,
            min_qualified_seasons=min_qualified_seasons,
            min_positive_season_rate=min_positive_season_rate,
            severe_negative_roi=severe_negative_roi,
            max_severe_negative_season_rate=max_severe_negative_season_rate,
            recent_seasons=recent_seasons,
            min_recent_qualified_seasons=min_recent_qualified_seasons,
            min_recent_positive_season_rate=min_recent_positive_season_rate,
        ).keys()
    )
    if benchmark.empty:
        return pd.DataFrame(columns=columns)

    base = benchmark[(~benchmark["favorite_agreement"]) & benchmark["prob_edge_model_side"].notna()].copy()
    seasons = sorted(pd.to_numeric(benchmark["season"], errors="coerce").dropna().astype(int).unique())
    recent_season_values = set(seasons[-int(recent_seasons) :]) if recent_seasons > 0 else set()
    records: list[dict[str, object]] = []

    for threshold in thresholds:
        selection = base[base["prob_edge_model_side"] >= float(threshold)].copy()
        if selection.empty:
            records.append(
                _empty_moneyline_stability_row(
                    threshold=float(threshold),
                    min_season_bets=min_season_bets,
                    min_qualified_seasons=min_qualified_seasons,
                    min_positive_season_rate=min_positive_season_rate,
                    severe_negative_roi=severe_negative_roi,
                    max_severe_negative_season_rate=max_severe_negative_season_rate,
                    recent_seasons=recent_seasons,
                    min_recent_qualified_seasons=min_recent_qualified_seasons,
                    min_recent_positive_season_rate=min_recent_positive_season_rate,
                )
            )
            continue

        pushes = int(selection["home_tie"].sum())
        wins = int(selection.loc[~selection["home_tie"], "model_side_actual_win"].sum())
        losses = int(len(selection) - wins - pushes)
        total_profit = float(selection["model_side_profit"].sum())
        season_records: list[dict[str, object]] = []

        for season, season_df in selection.groupby("season", sort=True):
            season_n = int(len(season_df))
            season_profit = float(season_df["model_side_profit"].sum())
            season_records.append(
                {
                    "season": int(season),
                    "n_bets": season_n,
                    "roi": float(season_profit / season_n) if season_n else np.nan,
                }
            )

        season_frame = pd.DataFrame.from_records(season_records)
        qualified = season_frame[season_frame["n_bets"].ge(int(min_season_bets))].copy()
        recent_qualified = qualified[qualified["season"].isin(recent_season_values)].copy()

        qualified_seasons = int(len(qualified))
        positive_qualified_seasons = int(qualified["roi"].gt(0.0).sum()) if qualified_seasons else 0
        severe_negative_qualified_seasons = (
            int(qualified["roi"].le(float(severe_negative_roi)).sum()) if qualified_seasons else 0
        )
        positive_rate = (
            float(positive_qualified_seasons / qualified_seasons) if qualified_seasons else np.nan
        )
        severe_rate = (
            float(severe_negative_qualified_seasons / qualified_seasons) if qualified_seasons else np.nan
        )
        recent_qualified_seasons = int(len(recent_qualified))
        recent_positive_qualified_seasons = (
            int(recent_qualified["roi"].gt(0.0).sum()) if recent_qualified_seasons else 0
        )
        recent_positive_rate = (
            float(recent_positive_qualified_seasons / recent_qualified_seasons)
            if recent_qualified_seasons
            else np.nan
        )

        stability_pass, stability_status = _moneyline_stability_status(
            qualified_seasons=qualified_seasons,
            positive_qualified_season_rate=0.0 if np.isnan(positive_rate) else positive_rate,
            severe_negative_qualified_season_rate=0.0 if np.isnan(severe_rate) else severe_rate,
            recent_qualified_seasons=recent_qualified_seasons,
            recent_positive_qualified_season_rate=0.0 if np.isnan(recent_positive_rate) else recent_positive_rate,
            min_qualified_seasons=min_qualified_seasons,
            min_positive_season_rate=min_positive_season_rate,
            max_severe_negative_season_rate=max_severe_negative_season_rate,
            min_recent_qualified_seasons=min_recent_qualified_seasons,
            min_recent_positive_season_rate=min_recent_positive_season_rate,
        )
        records.append(
            {
                "bet_type": "moneyline_disagreement",
                "threshold": float(threshold),
                "n_bets": int(len(selection)),
                "wins": wins,
                "losses": losses,
                "pushes": pushes,
                "roi": float(total_profit / len(selection)),
                "qualified_seasons": qualified_seasons,
                "positive_qualified_seasons": positive_qualified_seasons,
                "severe_negative_qualified_seasons": severe_negative_qualified_seasons,
                "positive_qualified_season_rate": positive_rate,
                "severe_negative_qualified_season_rate": severe_rate,
                "recent_qualified_seasons": recent_qualified_seasons,
                "recent_positive_qualified_seasons": recent_positive_qualified_seasons,
                "recent_positive_qualified_season_rate": recent_positive_rate,
                "stability_pass": bool(stability_pass),
                "stability_status": stability_status,
                "min_season_bets": int(min_season_bets),
                "min_qualified_seasons": int(min_qualified_seasons),
                "min_positive_season_rate": float(min_positive_season_rate),
                "severe_negative_roi": float(severe_negative_roi),
                "max_severe_negative_season_rate": float(max_severe_negative_season_rate),
                "recent_seasons": int(recent_seasons),
                "min_recent_qualified_seasons": int(min_recent_qualified_seasons),
                "min_recent_positive_season_rate": float(min_recent_positive_season_rate),
            }
        )

    return pd.DataFrame.from_records(records, columns=columns)


def _moneyline_disagreement_bin_record(
    *,
    scope: str,
    season: int | str,
    edge_bin: str,
    min_edge: float,
    max_edge: float,
    selection: pd.DataFrame,
) -> dict[str, float | int | str]:
    n_bets = int(len(selection))
    max_value = np.nan if np.isinf(max_edge) else float(max_edge)
    if n_bets == 0:
        return {
            "scope": scope,
            "season": season,
            "edge_bin": edge_bin,
            "min_edge": float(min_edge),
            "max_edge": max_value,
            "n_bets": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "hit_rate": np.nan,
            "avg_prob_edge_model_side": np.nan,
            "avg_model_pick_prob": np.nan,
            "avg_vegas_model_side_prob": np.nan,
            "avg_odds_decimal": np.nan,
            "avg_odds_american": np.nan,
            "total_profit": 0.0,
            "roi": np.nan,
        }

    pushes = int(selection["home_tie"].sum())
    wins = int(selection.loc[~selection["home_tie"], "model_side_actual_win"].sum())
    losses = n_bets - wins - pushes
    graded = wins + losses
    total_profit = float(selection["model_side_profit"].sum())
    return {
        "scope": scope,
        "season": season,
        "edge_bin": edge_bin,
        "min_edge": float(min_edge),
        "max_edge": max_value,
        "n_bets": n_bets,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "hit_rate": float(wins / graded) if graded > 0 else np.nan,
        "avg_prob_edge_model_side": float(selection["prob_edge_model_side"].mean()),
        "avg_model_pick_prob": float(selection["model_pick_prob"].mean()),
        "avg_vegas_model_side_prob": float(selection["vegas_model_side_prob"].mean()),
        "avg_odds_decimal": float(selection["model_side_decimal"].mean()),
        "avg_odds_american": float(selection["model_side_moneyline"].mean()),
        "total_profit": total_profit,
        "roi": float(total_profit / n_bets),
    }


def summarize_disagreement_moneyline_edge_bins(
    benchmark: pd.DataFrame,
    bins: Iterable[tuple[float, float, str]] = MARKET_PROB_EDGE_BINS,
) -> pd.DataFrame:
    """Summarize model-vs-Vegas disagreement ROI in bounded probability-edge bins."""
    columns = [
        "scope",
        "season",
        "edge_bin",
        "min_edge",
        "max_edge",
        "n_bets",
        "wins",
        "losses",
        "pushes",
        "hit_rate",
        "avg_prob_edge_model_side",
        "avg_model_pick_prob",
        "avg_vegas_model_side_prob",
        "avg_odds_decimal",
        "avg_odds_american",
        "total_profit",
        "roi",
    ]
    if benchmark.empty:
        return pd.DataFrame(columns=columns)

    base = benchmark[(~benchmark["favorite_agreement"]) & benchmark["prob_edge_model_side"].notna()].copy()
    records: list[dict[str, float | int | str]] = []

    def add_rows(scope: str, season: int | str, group: pd.DataFrame) -> None:
        for min_edge, max_edge, edge_bin in bins:
            if np.isneginf(min_edge):
                selection = group[group["prob_edge_model_side"] < max_edge]
            elif np.isinf(max_edge):
                selection = group[group["prob_edge_model_side"] >= min_edge]
            else:
                selection = group[
                    (group["prob_edge_model_side"] >= min_edge)
                    & (group["prob_edge_model_side"] < max_edge)
                ]
            records.append(
                _moneyline_disagreement_bin_record(
                    scope=scope,
                    season=season,
                    edge_bin=edge_bin,
                    min_edge=min_edge,
                    max_edge=max_edge,
                    selection=selection,
                )
            )

    add_rows("overall", "all", base)
    for season, season_df in base.groupby("season", sort=True):
        add_rows("season", int(season), season_df)

    return pd.DataFrame.from_records(records, columns=columns)


def _moneyline_market_status(value: float) -> str | None:
    if pd.isna(value):
        return None
    value = float(value)
    if value < -110.0:
        return "favorite"
    if value > 110.0:
        return "underdog"
    return "near_pick"


def _coerce_bool(value: object) -> bool | None:
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
        return None
    return bool(value)


def _moneyline_disagreement_segment_record(
    *,
    scope: str,
    season: int | str,
    segment_family: str,
    segment: str,
    selection: pd.DataFrame,
) -> dict[str, float | int | str]:
    n_bets = int(len(selection))
    if n_bets == 0:
        return {
            "scope": scope,
            "season": season,
            "segment_family": segment_family,
            "segment": segment,
            "n_bets": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "hit_rate": np.nan,
            "roi": np.nan,
            "total_profit": 0.0,
            "avg_model_pick_prob": np.nan,
            "avg_vegas_model_side_prob": np.nan,
            "avg_prob_edge_model_side": np.nan,
            "avg_model_side_moneyline": np.nan,
            "avg_model_side_decimal": np.nan,
            "clv_games": 0,
            "avg_moneyline_clv_prob_delta": np.nan,
            "positive_moneyline_clv_rate": np.nan,
        }

    pushes = int(selection["home_tie"].sum())
    wins = int(selection.loc[~selection["home_tie"], "model_side_actual_win"].sum())
    losses = n_bets - wins - pushes
    graded = wins + losses
    total_profit = float(selection["model_side_profit"].sum())

    if "moneyline_clv_prob_delta" in selection.columns:
        clv_delta = pd.to_numeric(selection["moneyline_clv_prob_delta"], errors="coerce")
    else:
        clv_delta = pd.Series(np.nan, index=selection.index)
    clv_mask = clv_delta.notna()
    clv_games = int(clv_mask.sum())

    positive_clv_rate = np.nan
    if clv_games and "moneyline_positive_clv" in selection.columns:
        positive_clv = selection.loc[clv_mask, "moneyline_positive_clv"].apply(_coerce_bool).dropna()
        if not positive_clv.empty:
            positive_clv_rate = float(positive_clv.astype(bool).mean())

    return {
        "scope": scope,
        "season": season,
        "segment_family": segment_family,
        "segment": segment,
        "n_bets": n_bets,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "hit_rate": float(wins / graded) if graded > 0 else np.nan,
        "roi": float(total_profit / n_bets),
        "total_profit": total_profit,
        "avg_model_pick_prob": float(selection["model_pick_prob"].mean()),
        "avg_vegas_model_side_prob": float(selection["vegas_model_side_prob"].mean()),
        "avg_prob_edge_model_side": float(selection["prob_edge_model_side"].mean()),
        "avg_model_side_moneyline": float(selection["model_side_moneyline"].mean()),
        "avg_model_side_decimal": float(selection["model_side_decimal"].mean())
        if "model_side_decimal" in selection.columns
        else np.nan,
        "clv_games": clv_games,
        "avg_moneyline_clv_prob_delta": float(clv_delta.mean()) if clv_games else np.nan,
        "positive_moneyline_clv_rate": positive_clv_rate,
    }


def summarize_disagreement_moneyline_segments(benchmark: pd.DataFrame) -> pd.DataFrame:
    """Slice favorite-disagreement moneyline ROI into actionable diagnostic segments."""
    if benchmark.empty or "favorite_agreement" not in benchmark.columns:
        return pd.DataFrame(columns=MARKET_DISAGREEMENT_SEGMENT_COLUMNS)

    base = benchmark[benchmark["favorite_agreement"].eq(False)].copy()
    if base.empty:
        return pd.DataFrame(columns=MARKET_DISAGREEMENT_SEGMENT_COLUMNS)

    base["model_pick_side"] = np.where(base["model_home_pick"].fillna(False).astype(bool), "home", "away")
    base["model_side_moneyline_status"] = base["model_side_moneyline"].apply(_moneyline_market_status)
    base["model_confidence_bin"] = base["model_pick_prob"].apply(
        lambda value: _bin_label(value, MARKET_MODEL_CONFIDENCE_BINS)
    )
    base["prob_edge_bin"] = base["prob_edge_model_side"].apply(
        lambda value: _bin_label(value, MARKET_PROB_EDGE_BINS)
    )
    base["model_side_moneyline_bin"] = base["model_side_moneyline"].apply(
        lambda value: _bin_label(value, MARKET_MONEYLINE_BINS)
    )

    segment_specs = [
        ("model_pick_side", "model_pick_side"),
        ("model_side_moneyline_status", "model_side_moneyline_status"),
        ("model_confidence", "model_confidence_bin"),
        ("probability_edge", "prob_edge_bin"),
        ("model_side_moneyline", "model_side_moneyline_bin"),
    ]
    if "clv_status" in base.columns:
        base["clv_status_segment"] = base["clv_status"].fillna("missing")
        segment_specs.append(("clv_status", "clv_status_segment"))
    if {"moneyline_clv_prob_delta", "moneyline_positive_clv"}.issubset(base.columns):
        clv_delta = pd.to_numeric(base["moneyline_clv_prob_delta"], errors="coerce")
        clv_positive = base["moneyline_positive_clv"].apply(_coerce_bool)
        base["moneyline_clv_result"] = pd.NA
        base.loc[clv_delta.notna() & clv_positive.eq(True), "moneyline_clv_result"] = "positive_clv"
        base.loc[clv_delta.notna() & clv_positive.eq(False), "moneyline_clv_result"] = "negative_or_flat_clv"
        segment_specs.append(("moneyline_clv_result", "moneyline_clv_result"))

    records: list[dict[str, float | int | str]] = []

    def add_segments(scope: str, season: int | str, group: pd.DataFrame) -> None:
        records.append(
            _moneyline_disagreement_segment_record(
                scope=scope,
                season=season,
                segment_family="all",
                segment="all",
                selection=group,
            )
        )
        for family, column in segment_specs:
            for segment, selection in group.dropna(subset=[column]).groupby(column, sort=False):
                records.append(
                    _moneyline_disagreement_segment_record(
                        scope=scope,
                        season=season,
                        segment_family=family,
                        segment=str(segment),
                        selection=selection,
                    )
                )

    add_segments("overall", "all", base)
    for season, season_df in base.groupby("season", sort=True):
        add_segments("season", int(season), season_df)

    return pd.DataFrame.from_records(records, columns=MARKET_DISAGREEMENT_SEGMENT_COLUMNS)


def _probability_edge_calibration_record(
    *,
    scope: str,
    season: int | str,
    bin_family: str,
    bin_label: str,
    min_value: float,
    max_value: float,
    selection: pd.DataFrame,
) -> dict[str, float | int | str]:
    n_games = int(len(selection))
    min_output = np.nan if np.isinf(min_value) else float(min_value)
    max_output = np.nan if np.isinf(max_value) else float(max_value)
    if n_games == 0:
        return {
            "scope": scope,
            "season": season,
            "bin_family": bin_family,
            "bin_label": bin_label,
            "min_value": min_output,
            "max_value": max_output,
            "n_games": 0,
            "n_graded": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "actual_model_side_win_rate": np.nan,
            "favorite_agreement_rate": np.nan,
            "avg_model_pick_prob": np.nan,
            "avg_vegas_model_side_prob": np.nan,
            "avg_prob_edge_model_side": np.nan,
            "model_calibration_error": np.nan,
            "vegas_calibration_error": np.nan,
            "model_abs_calibration_error": np.nan,
            "vegas_abs_calibration_error": np.nan,
            "abs_calibration_delta_model_minus_vegas": np.nan,
            "model_brier": np.nan,
            "vegas_brier": np.nan,
            "brier_delta_model_minus_vegas": np.nan,
            "model_side_roi": np.nan,
            "avg_model_side_moneyline": np.nan,
        }

    pushes = int(selection["home_tie"].sum())
    graded = selection[~selection["home_tie"]].copy()
    n_graded = int(len(graded))
    wins = int(graded["model_side_actual_win"].sum()) if n_graded else 0
    losses = n_graded - wins
    actual_rate = float(wins / n_graded) if n_graded else np.nan
    avg_model_prob = float(selection["model_pick_prob"].mean())
    avg_vegas_prob = float(selection["vegas_model_side_prob"].mean())
    model_error = avg_model_prob - actual_rate if not np.isnan(actual_rate) else np.nan
    vegas_error = avg_vegas_prob - actual_rate if not np.isnan(actual_rate) else np.nan
    if n_graded:
        outcome = graded["model_side_actual_win"].astype(float)
        model_brier = float(np.mean(np.square(graded["model_pick_prob"].astype(float) - outcome)))
        vegas_brier = float(np.mean(np.square(graded["vegas_model_side_prob"].astype(float) - outcome)))
    else:
        model_brier = np.nan
        vegas_brier = np.nan

    return {
        "scope": scope,
        "season": season,
        "bin_family": bin_family,
        "bin_label": bin_label,
        "min_value": min_output,
        "max_value": max_output,
        "n_games": n_games,
        "n_graded": n_graded,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "actual_model_side_win_rate": actual_rate,
        "favorite_agreement_rate": float(selection["favorite_agreement"].mean()),
        "avg_model_pick_prob": avg_model_prob,
        "avg_vegas_model_side_prob": avg_vegas_prob,
        "avg_prob_edge_model_side": float(selection["prob_edge_model_side"].mean()),
        "model_calibration_error": model_error,
        "vegas_calibration_error": vegas_error,
        "model_abs_calibration_error": abs(model_error) if not np.isnan(model_error) else np.nan,
        "vegas_abs_calibration_error": abs(vegas_error) if not np.isnan(vegas_error) else np.nan,
        "abs_calibration_delta_model_minus_vegas": (
            abs(model_error) - abs(vegas_error)
            if not np.isnan(model_error) and not np.isnan(vegas_error)
            else np.nan
        ),
        "model_brier": model_brier,
        "vegas_brier": vegas_brier,
        "brier_delta_model_minus_vegas": (
            model_brier - vegas_brier if not np.isnan(model_brier) and not np.isnan(vegas_brier) else np.nan
        ),
        "model_side_roi": float(selection["model_side_profit"].sum() / n_games),
        "avg_model_side_moneyline": float(selection["model_side_moneyline"].mean()),
    }


def summarize_market_probability_edge_calibration(benchmark: pd.DataFrame) -> pd.DataFrame:
    """Compare model-selected-side probabilities against no-vig Vegas probabilities by bin."""
    columns = [
        "scope",
        "season",
        "bin_family",
        "bin_label",
        "min_value",
        "max_value",
        "n_games",
        "n_graded",
        "wins",
        "losses",
        "pushes",
        "actual_model_side_win_rate",
        "favorite_agreement_rate",
        "avg_model_pick_prob",
        "avg_vegas_model_side_prob",
        "avg_prob_edge_model_side",
        "model_calibration_error",
        "vegas_calibration_error",
        "model_abs_calibration_error",
        "vegas_abs_calibration_error",
        "abs_calibration_delta_model_minus_vegas",
        "model_brier",
        "vegas_brier",
        "brier_delta_model_minus_vegas",
        "model_side_roi",
        "avg_model_side_moneyline",
    ]
    if benchmark.empty:
        return pd.DataFrame(columns=columns)

    base = benchmark[
        benchmark["model_pick_prob"].notna()
        & benchmark["vegas_model_side_prob"].notna()
        & benchmark["prob_edge_model_side"].notna()
    ].copy()
    records: list[dict[str, float | int | str]] = []

    bin_specs = [
        ("all", "model_pick_prob", [(-np.inf, np.inf, "all")]),
        ("model_probability", "model_pick_prob", MARKET_SELECTED_PROBABILITY_BINS),
        ("vegas_same_side_probability", "vegas_model_side_prob", MARKET_SELECTED_PROBABILITY_BINS),
        ("model_minus_vegas_probability_edge", "prob_edge_model_side", MARKET_PROB_EDGE_BINS),
    ]

    def add_rows(scope: str, season: int | str, group: pd.DataFrame) -> None:
        for bin_family, source_col, bins in bin_specs:
            for min_value, max_value, bin_label in bins:
                values = group[source_col]
                if np.isneginf(min_value) and np.isposinf(max_value):
                    selection = group
                elif np.isneginf(min_value):
                    selection = group[values < max_value]
                elif np.isposinf(max_value):
                    selection = group[values >= min_value]
                else:
                    selection = group[(values >= min_value) & (values < max_value)]
                records.append(
                    _probability_edge_calibration_record(
                        scope=scope,
                        season=season,
                        bin_family=bin_family,
                        bin_label=bin_label,
                        min_value=min_value,
                        max_value=max_value,
                        selection=selection,
                    )
                )

    add_rows("overall", "all", base)
    for season, season_df in base.groupby("season", sort=True):
        add_rows("season", int(season), season_df)

    return pd.DataFrame.from_records(records, columns=columns)


def _market_segment_record(
    *,
    scope: str,
    season: int | str,
    segment_family: str,
    segment: str,
    selection: pd.DataFrame,
) -> dict[str, float | int | str]:
    n_bets = int(len(selection))
    if n_bets == 0:
        return {
            "scope": scope,
            "season": season,
            "segment_family": segment_family,
            "segment": segment,
            "n_bets": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "hit_rate": np.nan,
            "model_accuracy": np.nan,
            "vegas_accuracy": np.nan,
            "model_side_roi": np.nan,
            "vegas_side_roi": np.nan,
            "avg_model_pick_prob": np.nan,
            "avg_vegas_pick_prob": np.nan,
            "avg_vegas_model_side_prob": np.nan,
            "avg_prob_edge_model_side": np.nan,
            "avg_model_side_moneyline": np.nan,
        }

    pushes = int(selection["home_tie"].sum())
    wins = int(selection.loc[~selection["home_tie"], "model_side_actual_win"].sum())
    losses = n_bets - wins - pushes
    graded = wins + losses
    return {
        "scope": scope,
        "season": season,
        "segment_family": segment_family,
        "segment": segment,
        "n_bets": n_bets,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "hit_rate": float(wins / graded) if graded > 0 else np.nan,
        "model_accuracy": float(selection["model_correct"].mean()),
        "vegas_accuracy": float(selection["vegas_correct"].mean()),
        "model_side_roi": float(selection["model_side_profit"].sum() / n_bets),
        "vegas_side_roi": float(selection["vegas_side_profit"].sum() / n_bets),
        "avg_model_pick_prob": float(selection["model_pick_prob"].mean()),
        "avg_vegas_pick_prob": float(selection["vegas_pick_prob"].mean()),
        "avg_vegas_model_side_prob": float(selection["vegas_model_side_prob"].mean()),
        "avg_prob_edge_model_side": float(selection["prob_edge_model_side"].mean()),
        "avg_model_side_moneyline": float(selection["model_side_moneyline"].mean()),
    }


def _bin_label(value: float, bins: Iterable[tuple[float, float, str]]) -> str | None:
    if pd.isna(value):
        return None
    value = float(value)
    for lower, upper, label in bins:
        lower_match = value >= lower
        upper_match = value < upper if not np.isinf(upper) else value >= lower
        if lower_match and upper_match:
            return label
    return None


def summarize_market_utility_segments(benchmark: pd.DataFrame) -> pd.DataFrame:
    """Summarize where model-vs-market picks have or lack historical utility."""
    columns = [
        "scope",
        "season",
        "segment_family",
        "segment",
        "n_bets",
        "wins",
        "losses",
        "pushes",
        "hit_rate",
        "model_accuracy",
        "vegas_accuracy",
        "model_side_roi",
        "vegas_side_roi",
        "avg_model_pick_prob",
        "avg_vegas_pick_prob",
        "avg_vegas_model_side_prob",
        "avg_prob_edge_model_side",
        "avg_model_side_moneyline",
    ]
    if benchmark.empty:
        return pd.DataFrame(columns=columns)

    base = benchmark.copy()
    base["agreement_segment"] = np.where(base["favorite_agreement"], "agree", "disagree")
    base["model_confidence_bin"] = base["model_pick_prob"].apply(
        lambda value: _bin_label(value, MARKET_MODEL_CONFIDENCE_BINS)
    )
    base["prob_edge_bin"] = base["prob_edge_model_side"].apply(
        lambda value: _bin_label(value, MARKET_PROB_EDGE_BINS)
    )
    base["model_side_moneyline_bin"] = base["model_side_moneyline"].apply(
        lambda value: _bin_label(value, MARKET_MONEYLINE_BINS)
    )

    records: list[dict[str, float | int | str]] = []

    def add_segments(scope: str, season: int | str, group: pd.DataFrame) -> None:
        segment_specs = [
            ("agreement", "agreement_segment"),
            ("model_confidence", "model_confidence_bin"),
            ("probability_edge", "prob_edge_bin"),
            ("model_side_moneyline", "model_side_moneyline_bin"),
        ]
        records.append(
            _market_segment_record(
                scope=scope,
                season=season,
                segment_family="all",
                segment="all",
                selection=group,
            )
        )
        for family, column in segment_specs:
            for segment, selection in group.dropna(subset=[column]).groupby(column, sort=False):
                records.append(
                    _market_segment_record(
                        scope=scope,
                        season=season,
                        segment_family=family,
                        segment=str(segment),
                        selection=selection,
                    )
                )

    add_segments("overall", "all", base)
    for season, season_df in base.groupby("season", sort=True):
        add_segments("season", int(season), season_df)

    return pd.DataFrame.from_records(records, columns=columns)


def summarize_spread(
    df: pd.DataFrame,
    thresholds: Iterable[float],
    sigma_margin_error: float,
) -> pd.DataFrame:
    valid = df.dropna(
        subset=[
            "spread_line",
            "pred_home_margin",
            "home_spread_odds",
            "away_spread_odds",
            "home_spread_decimal",
            "away_spread_decimal",
        ]
    ).copy()
    if valid.empty:
        return pd.DataFrame(
            columns=[
                "bet_type",
                "bet_side",
                "threshold",
                "n_bets",
                "wins",
                "losses",
                "pushes",
                "hit_rate",
                "avg_cover_prob",
                "avg_expected_profit",
                "avg_edge_points",
                "avg_odds_decimal",
                "avg_odds_american",
                "total_profit",
                "roi",
            ]
        )

    sigma = max(sigma_margin_error, 1e-6)
    valid["home_cover_prob"] = standard_normal_cdf(valid["spread_edge_points"] / sigma)
    valid["away_cover_prob"] = standard_normal_cdf(-valid["spread_edge_points"] / sigma)
    valid["home_expected_profit"] = (
        valid["home_cover_prob"] * (valid["home_spread_decimal"] - 1.0)
        - (1.0 - valid["home_cover_prob"])
    )
    valid["away_expected_profit"] = (
        valid["away_cover_prob"] * (valid["away_spread_decimal"] - 1.0)
        - (1.0 - valid["away_cover_prob"])
    )

    records: list[dict[str, float | int | str]] = []
    for side, prob_col, decimal_col, odds_col, expected_col, cover_col in [
        ("home", "home_cover_prob", "home_spread_decimal", "home_spread_odds", "home_expected_profit", "home_cover"),
        ("away", "away_cover_prob", "away_spread_decimal", "away_spread_odds", "away_expected_profit", "away_cover"),
    ]:
        edge = valid["spread_edge_points"] if side == "home" else -valid["spread_edge_points"]
        base = valid.assign(edge_points=edge)

        for threshold in thresholds:
            selection = base[edge > threshold]
            n_bets = int(len(selection))
            if n_bets == 0:
                records.append(
                    {
                        "bet_type": "spread",
                        "bet_side": side,
                        "threshold": float(threshold),
                        "n_bets": 0,
                        "wins": 0,
                        "losses": 0,
                        "pushes": 0,
                        "hit_rate": np.nan,
                        "avg_cover_prob": np.nan,
                        "avg_expected_profit": np.nan,
                        "avg_edge_points": np.nan,
                        "avg_odds_decimal": np.nan,
                        "avg_odds_american": np.nan,
                        "total_profit": 0.0,
                        "roi": np.nan,
                    }
                )
                continue

            pushes = int(selection["spread_push"].sum())
            wins = int(selection.loc[~selection["spread_push"], cover_col].sum())
            losses = n_bets - wins - pushes
            profits = np.where(
                selection["spread_push"],
                0.0,
                np.where(
                    selection[cover_col],
                    selection[decimal_col] - 1.0,
                    -1.0,
                ),
            )
            total_profit = float(np.sum(profits))
            denom = wins + losses
            hit_rate = float(wins / denom) if denom > 0 else np.nan

            records.append(
                {
                    "bet_type": "spread",
                    "bet_side": side,
                    "threshold": float(threshold),
                    "n_bets": n_bets,
                    "wins": wins,
                    "losses": losses,
                    "pushes": pushes,
                    "hit_rate": hit_rate,
                    "avg_cover_prob": float(selection[prob_col].mean()),
                    "avg_expected_profit": float(selection[expected_col].mean()),
                    "avg_edge_points": float(selection["edge_points"].mean()),
                    "avg_odds_decimal": float(selection[decimal_col].mean()),
                    "avg_odds_american": float(selection[odds_col].mean()),
                    "total_profit": total_profit,
                    "roi": float(total_profit / n_bets),
                }
            )
    return pd.DataFrame.from_records(records)


def _spread_calibration_record(
    *,
    scope: str,
    season: int | str,
    bet_side: str,
    cover_prob_bin: str,
    min_cover_prob: float,
    max_cover_prob: float,
    selection: pd.DataFrame,
) -> dict[str, float | int | str]:
    n_bets = int(len(selection))
    max_value = np.nan if np.isinf(max_cover_prob) else float(max_cover_prob)
    if n_bets == 0:
        return {
            "scope": scope,
            "season": season,
            "bet_side": bet_side,
            "cover_prob_bin": cover_prob_bin,
            "min_cover_prob": float(min_cover_prob),
            "max_cover_prob": max_value,
            "n_bets": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "actual_cover_rate": np.nan,
            "avg_cover_prob": np.nan,
            "calibration_error": np.nan,
            "avg_edge_points": np.nan,
            "avg_abs_edge_points": np.nan,
            "total_profit": 0.0,
            "roi": np.nan,
        }

    pushes = int(selection["spread_push"].sum())
    wins = int(selection.loc[~selection["spread_push"], "selected_cover"].sum())
    losses = n_bets - wins - pushes
    graded = wins + losses
    total_profit = float(selection["selected_profit"].sum())
    actual_cover_rate = float(wins / graded) if graded > 0 else np.nan
    avg_cover_prob = float(selection["cover_prob"].mean())
    return {
        "scope": scope,
        "season": season,
        "bet_side": bet_side,
        "cover_prob_bin": cover_prob_bin,
        "min_cover_prob": float(min_cover_prob),
        "max_cover_prob": max_value,
        "n_bets": n_bets,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "actual_cover_rate": actual_cover_rate,
        "avg_cover_prob": avg_cover_prob,
        "calibration_error": float(avg_cover_prob - actual_cover_rate) if not pd.isna(actual_cover_rate) else np.nan,
        "avg_edge_points": float(selection["edge_points"].mean()),
        "avg_abs_edge_points": float(selection["edge_points"].abs().mean()),
        "total_profit": total_profit,
        "roi": float(total_profit / n_bets),
    }


def summarize_spread_cover_calibration(
    df: pd.DataFrame,
    sigma_margin_error: float,
    bins: Iterable[tuple[float, float]] = SPREAD_COVER_PROB_BINS,
) -> pd.DataFrame:
    """Compare modeled spread cover probabilities to empirical cover rates."""
    columns = [
        "scope",
        "season",
        "bet_side",
        "cover_prob_bin",
        "min_cover_prob",
        "max_cover_prob",
        "n_bets",
        "wins",
        "losses",
        "pushes",
        "actual_cover_rate",
        "avg_cover_prob",
        "calibration_error",
        "avg_edge_points",
        "avg_abs_edge_points",
        "total_profit",
        "roi",
    ]
    valid = df.dropna(
        subset=[
            "spread_edge_points",
            "home_cover",
            "away_cover",
            "spread_push",
            "home_spread_decimal",
            "away_spread_decimal",
        ]
    ).copy()
    if valid.empty:
        return pd.DataFrame(columns=columns)

    sigma = max(sigma_margin_error, 1e-6)
    home = pd.DataFrame(
        {
            "season": valid["season"],
            "bet_side": "home",
            "cover_prob": standard_normal_cdf(valid["spread_edge_points"] / sigma),
            "edge_points": valid["spread_edge_points"],
            "selected_cover": valid["home_cover"],
            "spread_push": valid["spread_push"],
            "selected_profit": np.where(
                valid["spread_push"],
                0.0,
                np.where(valid["home_cover"], valid["home_spread_decimal"] - 1.0, -1.0),
            ),
        }
    )
    away = pd.DataFrame(
        {
            "season": valid["season"],
            "bet_side": "away",
            "cover_prob": standard_normal_cdf(-valid["spread_edge_points"] / sigma),
            "edge_points": -valid["spread_edge_points"],
            "selected_cover": valid["away_cover"],
            "spread_push": valid["spread_push"],
            "selected_profit": np.where(
                valid["spread_push"],
                0.0,
                np.where(valid["away_cover"], valid["away_spread_decimal"] - 1.0, -1.0),
            ),
        }
    )
    long = pd.concat([home, away], ignore_index=True)
    long = long[long["cover_prob"] >= 0.50].copy()

    records: list[dict[str, float | int | str]] = []

    def add_rows(scope: str, season: int | str, bet_side: str, group: pd.DataFrame) -> None:
        for min_prob, max_prob in bins:
            cover_prob_bin = f"{min_prob:.0%}+" if np.isinf(max_prob) else f"{min_prob:.0%}-{max_prob:.0%}"
            if np.isinf(max_prob):
                selection = group[group["cover_prob"] >= min_prob]
            else:
                selection = group[(group["cover_prob"] >= min_prob) & (group["cover_prob"] < max_prob)]
            records.append(
                _spread_calibration_record(
                    scope=scope,
                    season=season,
                    bet_side=bet_side,
                    cover_prob_bin=cover_prob_bin,
                    min_cover_prob=min_prob,
                    max_cover_prob=max_prob,
                    selection=selection,
                )
            )

    add_rows("overall", "all", "all", long)
    for side, side_df in long.groupby("bet_side", sort=True):
        add_rows("overall", "all", str(side), side_df)
    for season, season_df in long.groupby("season", sort=True):
        add_rows("season", int(season), "all", season_df)

    return pd.DataFrame.from_records(records, columns=columns)


def summarize_spread_cover_recalibration(
    spread_calibration: pd.DataFrame,
    *,
    min_bets: int = SPREAD_COVER_CALIBRATION_MIN_BETS,
    max_abs_calibration_error: float = SPREAD_COVER_CALIBRATION_MAX_ABS_ERROR,
    confidence_min_cover_prob: float = SPREAD_COVER_CALIBRATION_CONFIDENCE_MIN_PROB,
) -> pd.DataFrame:
    """Convert raw spread cover probabilities into empirical calibration gates."""
    columns = [
        "scope",
        "season",
        "bet_side",
        "cover_prob_bin",
        "min_cover_prob",
        "max_cover_prob",
        "n_bets",
        "actual_cover_rate",
        "avg_cover_prob",
        "recalibrated_cover_prob",
        "calibration_error",
        "abs_calibration_error",
        "raw_minus_recalibrated_cover_prob",
        "recommended_for_confidence",
        "calibration_status",
        "policy_min_bets",
        "policy_max_abs_calibration_error",
        "policy_confidence_min_cover_prob",
    ]
    if spread_calibration.empty:
        return pd.DataFrame(columns=columns)

    frame = spread_calibration.copy()
    for col in [
        "min_cover_prob",
        "max_cover_prob",
        "n_bets",
        "actual_cover_rate",
        "avg_cover_prob",
        "calibration_error",
    ]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        else:
            frame[col] = np.nan

    frame["recalibrated_cover_prob"] = frame["actual_cover_rate"]
    frame["abs_calibration_error"] = frame["calibration_error"].abs()
    frame["raw_minus_recalibrated_cover_prob"] = frame["avg_cover_prob"] - frame["recalibrated_cover_prob"]
    enough_sample = frame["n_bets"].ge(int(min_bets))
    calibrated = enough_sample & frame["abs_calibration_error"].le(float(max_abs_calibration_error))
    confidence_bin = frame["min_cover_prob"].ge(float(confidence_min_cover_prob))
    frame["recommended_for_confidence"] = calibrated & confidence_bin
    frame["calibration_status"] = np.select(
        [
            frame["n_bets"].isna() | frame["n_bets"].lt(int(min_bets)),
            frame["calibration_error"].gt(float(max_abs_calibration_error)),
            frame["calibration_error"].lt(-float(max_abs_calibration_error)),
            confidence_bin & calibrated,
        ],
        [
            "insufficient_sample",
            "overconfident",
            "underconfident",
            "calibrated_for_confidence",
        ],
        default="diagnostic_only",
    )
    frame["policy_min_bets"] = int(min_bets)
    frame["policy_max_abs_calibration_error"] = float(max_abs_calibration_error)
    frame["policy_confidence_min_cover_prob"] = float(confidence_min_cover_prob)
    return frame[columns]


def summarize_spread_cover_calibration_gate(
    spread_recalibration: pd.DataFrame | None,
    *,
    min_bets: int = SPREAD_COVER_CALIBRATION_MIN_BETS,
    max_abs_calibration_error: float = SPREAD_COVER_CALIBRATION_MAX_ABS_ERROR,
    confidence_min_cover_prob: float = SPREAD_COVER_CALIBRATION_CONFIDENCE_MIN_PROB,
) -> dict[str, object]:
    """Summarize whether high raw spread cover probabilities can drive policy."""
    result: dict[str, object] = {
        "pass": False,
        "status": "unavailable",
        "evaluated_bins": 0,
        "max_abs_error": np.nan,
        "worst_bin": np.nan,
        "worst_error": np.nan,
        "min_bets": int(min_bets),
        "max_abs_error_threshold": float(max_abs_calibration_error),
        "confidence_min_prob": float(confidence_min_cover_prob),
    }
    if spread_recalibration is None or spread_recalibration.empty:
        return result

    frame = spread_recalibration.copy()
    required = {"scope", "season", "bet_side", "min_cover_prob", "n_bets", "calibration_error"}
    if not required.issubset(frame.columns):
        result["status"] = "missing_required_columns"
        return result

    frame["min_cover_prob"] = pd.to_numeric(frame["min_cover_prob"], errors="coerce")
    frame["n_bets"] = pd.to_numeric(frame["n_bets"], errors="coerce")
    frame["calibration_error"] = pd.to_numeric(frame["calibration_error"], errors="coerce")
    frame["abs_calibration_error"] = frame["calibration_error"].abs()
    high_confidence = frame[
        frame["scope"].eq("overall")
        & frame["season"].astype(str).eq("all")
        & frame["bet_side"].eq("all")
        & frame["min_cover_prob"].ge(float(confidence_min_cover_prob))
    ].copy()
    sampled = high_confidence[high_confidence["n_bets"].ge(int(min_bets))].copy()
    if sampled.empty:
        result["status"] = "insufficient_high_confidence_sample"
        return result

    worst = sampled.loc[sampled["abs_calibration_error"].idxmax()]
    max_abs_error = float(worst["abs_calibration_error"])
    worst_error = float(worst["calibration_error"])
    result.update(
        {
            "evaluated_bins": int(len(sampled)),
            "max_abs_error": max_abs_error,
            "worst_bin": worst.get("cover_prob_bin", np.nan),
            "worst_error": worst_error,
        }
    )
    if max_abs_error <= float(max_abs_calibration_error):
        result["pass"] = True
        result["status"] = "calibrated"
    elif worst_error > 0:
        result["status"] = "blocked_spread_cover_overconfidence"
    else:
        result["status"] = "blocked_spread_cover_underconfidence"
    return result


def summarize_spread_edge_bins(
    df: pd.DataFrame,
    bins: Iterable[tuple[float, float]] = SPREAD_EDGE_BINS,
) -> pd.DataFrame:
    valid = df.dropna(
        subset=[
            "spread_edge_points",
            "home_cover",
            "away_cover",
            "spread_push",
            "home_spread_decimal",
            "away_spread_decimal",
            "home_spread_odds",
            "away_spread_odds",
        ]
    ).copy()
    if valid.empty:
        return pd.DataFrame(
            columns=[
                "scope",
                "season",
                "edge_bin",
                "min_edge",
                "max_edge",
                "n_bets",
                "home_bets",
                "away_bets",
                "wins",
                "losses",
                "pushes",
                "hit_rate",
                "avg_abs_edge_points",
                "avg_signed_edge_points",
                "avg_odds_decimal",
                "avg_odds_american",
                "total_profit",
                "roi",
            ]
        )

    valid["abs_edge_points"] = valid["spread_edge_points"].abs()
    valid["selected_side"] = np.where(valid["spread_edge_points"] >= 0.0, "home", "away")
    valid["selected_cover"] = np.where(valid["selected_side"] == "home", valid["home_cover"], valid["away_cover"])
    valid["selected_decimal"] = np.where(
        valid["selected_side"] == "home",
        valid["home_spread_decimal"],
        valid["away_spread_decimal"],
    )
    valid["selected_odds"] = np.where(
        valid["selected_side"] == "home",
        valid["home_spread_odds"],
        valid["away_spread_odds"],
    )
    valid["selected_profit"] = np.where(
        valid["spread_push"],
        0.0,
        np.where(valid["selected_cover"], valid["selected_decimal"] - 1.0, -1.0),
    )

    records: list[dict[str, float | int | str]] = []

    def add_rows(scope: str, season: int | str, group: pd.DataFrame) -> None:
        for min_edge, max_edge in bins:
            if np.isinf(max_edge):
                selection = group[group["abs_edge_points"] >= min_edge]
                edge_bin = f"{min_edge:g}+"
                max_value = np.nan
            else:
                selection = group[(group["abs_edge_points"] >= min_edge) & (group["abs_edge_points"] < max_edge)]
                edge_bin = f"{min_edge:g}-{max_edge:g}"
                max_value = float(max_edge)

            n_bets = int(len(selection))
            if n_bets == 0:
                records.append(
                    {
                        "scope": scope,
                        "season": season,
                        "edge_bin": edge_bin,
                        "min_edge": float(min_edge),
                        "max_edge": max_value,
                        "n_bets": 0,
                        "home_bets": 0,
                        "away_bets": 0,
                        "wins": 0,
                        "losses": 0,
                        "pushes": 0,
                        "hit_rate": np.nan,
                        "avg_abs_edge_points": np.nan,
                        "avg_signed_edge_points": np.nan,
                        "avg_odds_decimal": np.nan,
                        "avg_odds_american": np.nan,
                        "total_profit": 0.0,
                        "roi": np.nan,
                    }
                )
                continue

            pushes = int(selection["spread_push"].sum())
            wins = int(selection.loc[~selection["spread_push"], "selected_cover"].sum())
            losses = n_bets - wins - pushes
            denom = wins + losses
            total_profit = float(selection["selected_profit"].sum())

            records.append(
                {
                    "scope": scope,
                    "season": season,
                    "edge_bin": edge_bin,
                    "min_edge": float(min_edge),
                    "max_edge": max_value,
                    "n_bets": n_bets,
                    "home_bets": int((selection["selected_side"] == "home").sum()),
                    "away_bets": int((selection["selected_side"] == "away").sum()),
                    "wins": wins,
                    "losses": losses,
                    "pushes": pushes,
                    "hit_rate": float(wins / denom) if denom > 0 else np.nan,
                    "avg_abs_edge_points": float(selection["abs_edge_points"].mean()),
                    "avg_signed_edge_points": float(selection["spread_edge_points"].mean()),
                    "avg_odds_decimal": float(selection["selected_decimal"].mean()),
                    "avg_odds_american": float(selection["selected_odds"].mean()),
                    "total_profit": total_profit,
                    "roi": float(total_profit / n_bets),
                }
            )

    add_rows("overall", "all", valid)
    for season, season_df in valid.groupby("season", sort=True):
        add_rows("season", int(season), season_df)

    return pd.DataFrame.from_records(records)


def _spread_edge_bin_label(value: float, bins: Iterable[tuple[float, float]] = SPREAD_EDGE_BINS) -> str | None:
    if pd.isna(value):
        return None
    value = float(value)
    for min_edge, max_edge in bins:
        if np.isinf(max_edge):
            if value >= min_edge:
                return f"{min_edge:g}+"
        elif value >= min_edge and value < max_edge:
            return f"{min_edge:g}-{max_edge:g}"
    return None


def _selected_spread_frame(df: pd.DataFrame) -> pd.DataFrame:
    valid = df.dropna(
        subset=[
            "season",
            "spread_edge_points",
            "pred_home_margin",
            "market_home_margin",
            "home_cover",
            "away_cover",
            "spread_push",
            "home_spread_decimal",
            "away_spread_decimal",
            "home_spread_odds",
            "away_spread_odds",
        ]
    ).copy()
    if valid.empty:
        return valid

    valid["abs_edge_points"] = valid["spread_edge_points"].abs()
    valid["edge_bucket"] = valid["abs_edge_points"].apply(_spread_edge_bin_label)
    valid["selected_side"] = np.where(valid["spread_edge_points"] >= 0.0, "home", "away")
    valid["selected_cover"] = np.where(valid["selected_side"] == "home", valid["home_cover"], valid["away_cover"])
    valid["selected_decimal"] = np.where(
        valid["selected_side"] == "home",
        valid["home_spread_decimal"],
        valid["away_spread_decimal"],
    )
    valid["selected_odds"] = np.where(
        valid["selected_side"] == "home",
        valid["home_spread_odds"],
        valid["away_spread_odds"],
    )
    valid["selected_profit"] = np.where(
        valid["spread_push"],
        0.0,
        np.where(valid["selected_cover"], valid["selected_decimal"] - 1.0, -1.0),
    )
    valid["market_selected_margin"] = np.where(
        valid["selected_side"] == "home",
        valid["market_home_margin"],
        -valid["market_home_margin"],
    )
    valid["model_selected_margin"] = np.where(
        valid["selected_side"] == "home",
        valid["pred_home_margin"],
        -valid["pred_home_margin"],
    )
    valid["market_abs_line"] = valid["market_selected_margin"].abs()
    valid["model_abs_line"] = valid["model_selected_margin"].abs()
    return valid


def _spread_key_number_record(
    *,
    scope: str,
    season: int | str,
    key_number: float,
    edge_bucket: str,
    segment: str,
    selection: pd.DataFrame,
) -> dict[str, float | int | str]:
    n_bets = int(len(selection))
    if n_bets == 0:
        return {
            "scope": scope,
            "season": season,
            "key_number": float(key_number),
            "edge_bucket": edge_bucket,
            "segment": segment,
            "n_bets": 0,
            "home_bets": 0,
            "away_bets": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "hit_rate": np.nan,
            "avg_abs_edge_points": np.nan,
            "avg_signed_edge_points": np.nan,
            "avg_market_abs_line": np.nan,
            "avg_model_abs_line": np.nan,
            "avg_market_selected_margin": np.nan,
            "avg_model_selected_margin": np.nan,
            "avg_odds_decimal": np.nan,
            "avg_odds_american": np.nan,
            "total_profit": 0.0,
            "roi": np.nan,
        }

    pushes = int(selection["spread_push"].sum())
    wins = int(selection.loc[~selection["spread_push"], "selected_cover"].sum())
    losses = n_bets - wins - pushes
    graded = wins + losses
    total_profit = float(selection["selected_profit"].sum())
    return {
        "scope": scope,
        "season": season,
        "key_number": float(key_number),
        "edge_bucket": edge_bucket,
        "segment": segment,
        "n_bets": n_bets,
        "home_bets": int((selection["selected_side"] == "home").sum()),
        "away_bets": int((selection["selected_side"] == "away").sum()),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "hit_rate": float(wins / graded) if graded > 0 else np.nan,
        "avg_abs_edge_points": float(selection["abs_edge_points"].mean()),
        "avg_signed_edge_points": float(selection["spread_edge_points"].mean()),
        "avg_market_abs_line": float(selection["market_abs_line"].mean()),
        "avg_model_abs_line": float(selection["model_abs_line"].mean()),
        "avg_market_selected_margin": float(selection["market_selected_margin"].mean()),
        "avg_model_selected_margin": float(selection["model_selected_margin"].mean()),
        "avg_odds_decimal": float(selection["selected_decimal"].mean()),
        "avg_odds_american": float(selection["selected_odds"].mean()),
        "total_profit": total_profit,
        "roi": float(total_profit / n_bets),
    }


def summarize_spread_key_numbers(
    df: pd.DataFrame,
    *,
    key_numbers: Iterable[float] = SPREAD_KEY_NUMBERS,
    key_tolerance: float = SPREAD_KEY_NUMBER_TOLERANCE,
) -> pd.DataFrame:
    columns = [
        "scope",
        "season",
        "key_number",
        "edge_bucket",
        "segment",
        "n_bets",
        "home_bets",
        "away_bets",
        "wins",
        "losses",
        "pushes",
        "hit_rate",
        "avg_abs_edge_points",
        "avg_signed_edge_points",
        "avg_market_abs_line",
        "avg_model_abs_line",
        "avg_market_selected_margin",
        "avg_model_selected_margin",
        "avg_odds_decimal",
        "avg_odds_american",
        "total_profit",
        "roi",
    ]
    selected = _selected_spread_frame(df)
    if selected.empty:
        return pd.DataFrame(columns=columns)

    keys = [float(key_number) for key_number in key_numbers]
    records: list[dict[str, float | int | str]] = []

    def add_rows(scope: str, season: int | str, group: pd.DataFrame) -> None:
        bucket_groups: list[tuple[str, pd.DataFrame]] = [("all", group)]
        for edge_bucket, bucket_df in group.dropna(subset=["edge_bucket"]).groupby("edge_bucket", sort=False):
            bucket_groups.append((str(edge_bucket), bucket_df))

        for edge_bucket, bucket_df in bucket_groups:
            for key in keys:
                market_near = (bucket_df["market_abs_line"] - key).abs() <= float(key_tolerance)
                model_near = (bucket_df["model_abs_line"] - key).abs() <= float(key_tolerance)
                low = np.minimum(bucket_df["market_abs_line"], bucket_df["model_abs_line"])
                high = np.maximum(bucket_df["market_abs_line"], bucket_df["model_abs_line"])
                crosses = (low <= key) & (high >= key)
                segment_specs = [
                    ("all", pd.Series(True, index=bucket_df.index)),
                    ("market_near_key", market_near),
                    ("model_near_key", model_near),
                    ("edge_crosses_key", crosses),
                    ("key_window", market_near | model_near | crosses),
                    ("not_key_window", ~(market_near | model_near | crosses)),
                ]
                for segment, mask in segment_specs:
                    records.append(
                        _spread_key_number_record(
                            scope=scope,
                            season=season,
                            key_number=key,
                            edge_bucket=edge_bucket,
                            segment=segment,
                            selection=bucket_df[mask],
                        )
                    )

    add_rows("overall", "all", selected)
    for season, season_df in selected.groupby("season", sort=True):
        add_rows("season", int(season), season_df)

    return pd.DataFrame.from_records(records, columns=columns)


def _best_policy_candidate(
    frame: pd.DataFrame,
    *,
    min_bets: int,
    min_roi: float,
    threshold_col: str | None = None,
    min_threshold: float | None = None,
    max_threshold: float | None = None,
    allowed_values_col: str | None = None,
    allowed_values: set[str] | None = None,
) -> tuple[pd.Series | None, bool]:
    if frame.empty:
        return None, False
    candidates = frame.copy()
    for col in ["n_bets", "roi"]:
        if col in candidates.columns:
            candidates[col] = pd.to_numeric(candidates[col], errors="coerce")
    candidates = candidates[candidates["n_bets"].ge(int(min_bets)) & candidates["roi"].notna()]
    if threshold_col and threshold_col in candidates.columns and min_threshold is not None:
        candidates[threshold_col] = pd.to_numeric(candidates[threshold_col], errors="coerce")
        candidates = candidates[candidates[threshold_col].ge(float(min_threshold))]
    if threshold_col and threshold_col in candidates.columns and max_threshold is not None:
        candidates[threshold_col] = pd.to_numeric(candidates[threshold_col], errors="coerce")
        candidates = candidates[candidates[threshold_col].le(float(max_threshold))]
    if allowed_values_col and allowed_values_col in candidates.columns and allowed_values is not None:
        candidates = candidates[candidates[allowed_values_col].astype(str).isin(allowed_values)]
    if candidates.empty:
        return None, False

    eligible = candidates[candidates["roi"].gt(float(min_roi))]
    if not eligible.empty:
        ordered = eligible.sort_values(["roi", "n_bets"], ascending=[False, False], kind="stable")
        return ordered.iloc[0], True

    ordered = candidates.sort_values(["roi", "n_bets"], ascending=[False, False], kind="stable")
    return ordered.iloc[0], False


def _policy_tier(row: pd.Series | None, recommended: bool, recommendation_tier: str | None = None) -> str:
    if recommendation_tier:
        return recommendation_tier
    if row is None:
        return "unavailable"
    if not recommended:
        return "no_bet"
    return "strong" if float(row.get("roi", 0.0) or 0.0) >= 0.05 else "lean"


def _policy_value(row: pd.Series | None, key: str) -> object:
    if row is None or key not in row:
        return np.nan
    return row.get(key)


def _market_policy_record(
    *,
    signal: str,
    row: pd.Series | None,
    recommended: bool,
    selection: str,
    source_artifact: str,
    generated_at: str,
    min_bets: int,
    min_roi: float,
    min_probability_edge: float | None,
    min_spread_edge: float | None,
    recommendation_tier: str | None = None,
    stability: pd.Series | None = None,
    moneyline_clv: dict[str, object] | None = None,
    spread_cover_calibration: dict[str, object] | None = None,
) -> dict[str, object]:
    moneyline_clv = moneyline_clv or {}
    spread_cover_calibration = spread_cover_calibration or {}
    return {
        "signal": signal,
        "recommended": bool(recommended),
        "recommendation_tier": _policy_tier(row, recommended, recommendation_tier),
        "selection": selection,
        "threshold": _policy_value(row, "threshold"),
        "edge_bin": _policy_value(row, "edge_bin"),
        "bet_side": _policy_value(row, "bet_side"),
        "n_bets": _policy_value(row, "n_bets") if row is not None else 0,
        "wins": _policy_value(row, "wins") if row is not None else 0,
        "losses": _policy_value(row, "losses") if row is not None else 0,
        "pushes": _policy_value(row, "pushes") if row is not None else 0,
        "hit_rate": _policy_value(row, "hit_rate"),
        "roi": _policy_value(row, "roi"),
        "avg_prob_edge_model_side": _policy_value(row, "avg_prob_edge_model_side"),
        "avg_edge_points": _policy_value(row, "avg_edge_points"),
        "avg_abs_edge_points": _policy_value(row, "avg_abs_edge_points"),
        "avg_expected_profit": _policy_value(row, "avg_expected_profit"),
        "policy_min_bets": int(min_bets),
        "policy_min_roi": float(min_roi),
        "policy_min_probability_edge": np.nan if min_probability_edge is None else float(min_probability_edge),
        "policy_min_spread_edge": np.nan if min_spread_edge is None else float(min_spread_edge),
        "stability_pass": _policy_value(stability, "stability_pass"),
        "stability_status": _policy_value(stability, "stability_status"),
        "stability_qualified_seasons": _policy_value(stability, "qualified_seasons"),
        "stability_positive_season_rate": _policy_value(stability, "positive_qualified_season_rate"),
        "stability_recent_positive_season_rate": _policy_value(stability, "recent_positive_qualified_season_rate"),
        "stability_severe_negative_season_rate": _policy_value(stability, "severe_negative_qualified_season_rate"),
        "stability_min_season_bets": _policy_value(stability, "min_season_bets"),
        "stability_min_qualified_seasons": _policy_value(stability, "min_qualified_seasons"),
        "clv_pass": moneyline_clv.get("pass", np.nan),
        "clv_status": moneyline_clv.get("status", np.nan),
        "clv_threshold": moneyline_clv.get("threshold", np.nan),
        "clv_tracked_games": moneyline_clv.get("tracked_games", np.nan),
        "clv_avg_prob_delta": moneyline_clv.get("avg_prob_delta", np.nan),
        "clv_positive_rate": moneyline_clv.get("positive_rate", np.nan),
        "clv_min_tracked_games": moneyline_clv.get("min_tracked_games", np.nan),
        "clv_min_positive_rate": moneyline_clv.get("min_positive_rate", np.nan),
        "clv_min_avg_prob_delta": moneyline_clv.get("min_avg_prob_delta", np.nan),
        "spread_cover_calibration_pass": spread_cover_calibration.get("pass", np.nan),
        "spread_cover_calibration_status": spread_cover_calibration.get("status", np.nan),
        "spread_cover_calibration_evaluated_bins": spread_cover_calibration.get("evaluated_bins", np.nan),
        "spread_cover_calibration_max_abs_error": spread_cover_calibration.get("max_abs_error", np.nan),
        "spread_cover_calibration_worst_bin": spread_cover_calibration.get("worst_bin", np.nan),
        "spread_cover_calibration_worst_error": spread_cover_calibration.get("worst_error", np.nan),
        "spread_cover_calibration_min_bets": spread_cover_calibration.get("min_bets", np.nan),
        "spread_cover_calibration_max_abs_error_threshold": spread_cover_calibration.get(
            "max_abs_error_threshold", np.nan
        ),
        "spread_cover_calibration_confidence_min_prob": spread_cover_calibration.get(
            "confidence_min_prob", np.nan
        ),
        "source_artifact": source_artifact,
        "generated_at": generated_at,
    }


def _matching_moneyline_stability_row(
    moneyline_stability: pd.DataFrame | None,
    moneyline_row: pd.Series | None,
) -> pd.Series | None:
    if moneyline_stability is None or moneyline_stability.empty or moneyline_row is None:
        return None
    if "threshold" not in moneyline_stability.columns or "threshold" not in moneyline_row:
        return None
    threshold = pd.to_numeric(pd.Series([moneyline_row["threshold"]]), errors="coerce").iloc[0]
    if pd.isna(threshold):
        return None
    frame = moneyline_stability.copy()
    frame["threshold"] = pd.to_numeric(frame["threshold"], errors="coerce")
    match = frame[np.isclose(frame["threshold"], float(threshold), atol=1e-9, equal_nan=False)]
    if match.empty:
        return None
    return match.iloc[0]


def _matching_moneyline_clv_row(
    moneyline_clv_summary: pd.DataFrame | None,
    moneyline_row: pd.Series | None,
) -> pd.Series | None:
    if moneyline_clv_summary is None or moneyline_clv_summary.empty or moneyline_row is None:
        return None
    if "threshold" not in moneyline_clv_summary.columns or "threshold" not in moneyline_row:
        return None
    threshold = pd.to_numeric(pd.Series([moneyline_row["threshold"]]), errors="coerce").iloc[0]
    if pd.isna(threshold):
        return None
    frame = moneyline_clv_summary.copy()
    if "scope" in frame.columns:
        frame = frame[frame["scope"].astype(str).eq("moneyline_threshold")].copy()
    frame["threshold"] = pd.to_numeric(frame["threshold"], errors="coerce")
    match = frame[np.isclose(frame["threshold"], float(threshold), atol=1e-9, equal_nan=False)]
    if match.empty:
        return None
    return match.iloc[0]


def summarize_moneyline_clv_gate(
    moneyline_clv_summary: pd.DataFrame | None,
    moneyline_row: pd.Series | None,
    *,
    min_tracked_games: int = MARKET_CLV_MIN_TRACKED_GAMES,
    min_positive_rate: float = MARKET_CLV_MIN_POSITIVE_RATE,
    min_avg_prob_delta: float = MARKET_CLV_MIN_AVG_PROB_DELTA,
) -> dict[str, object]:
    """Summarize whether the selected moneyline threshold has enough positive CLV support."""
    threshold = np.nan
    if moneyline_row is not None and "threshold" in moneyline_row:
        threshold = pd.to_numeric(pd.Series([moneyline_row["threshold"]]), errors="coerce").iloc[0]
    result: dict[str, object] = {
        "pass": False,
        "status": "unavailable",
        "threshold": threshold,
        "tracked_games": 0,
        "avg_prob_delta": np.nan,
        "positive_rate": np.nan,
        "min_tracked_games": int(min_tracked_games),
        "min_positive_rate": float(min_positive_rate),
        "min_avg_prob_delta": float(min_avg_prob_delta),
    }
    if moneyline_row is None:
        result["status"] = "no_policy_threshold"
        return result
    if moneyline_clv_summary is None or moneyline_clv_summary.empty:
        return result

    row = _matching_moneyline_clv_row(moneyline_clv_summary, moneyline_row)
    if row is None:
        result["status"] = "no_matching_threshold_clv"
        return result

    tracked = pd.to_numeric(pd.Series([row.get("moneyline_clv_available", np.nan)]), errors="coerce").iloc[0]
    avg_delta = pd.to_numeric(pd.Series([row.get("avg_moneyline_clv_prob_delta", np.nan)]), errors="coerce").iloc[0]
    positive_rate = pd.to_numeric(pd.Series([row.get("positive_moneyline_clv_rate", np.nan)]), errors="coerce").iloc[0]
    result.update(
        {
            "tracked_games": int(tracked) if pd.notna(tracked) else 0,
            "avg_prob_delta": float(avg_delta) if pd.notna(avg_delta) else np.nan,
            "positive_rate": float(positive_rate) if pd.notna(positive_rate) else np.nan,
            "threshold": float(row.get("threshold", threshold)) if pd.notna(row.get("threshold", threshold)) else threshold,
        }
    )

    if result["tracked_games"] < int(min_tracked_games):
        result["status"] = "insufficient_tracked_games"
        return result
    if pd.isna(avg_delta) or float(avg_delta) <= float(min_avg_prob_delta):
        result["status"] = "negative_avg_clv"
        return result
    if pd.isna(positive_rate) or float(positive_rate) < float(min_positive_rate):
        result["status"] = "weak_positive_clv_rate"
        return result

    result["pass"] = True
    result["status"] = "clv_pass"
    return result


def build_market_edge_policy(
    disagreement_roi: pd.DataFrame,
    spread_roi: pd.DataFrame,
    spread_edge_bins: pd.DataFrame,
    *,
    moneyline_stability: pd.DataFrame | None = None,
    moneyline_clv_summary: pd.DataFrame | None = None,
    spread_cover_recalibration: pd.DataFrame | None = None,
    min_bets: int = 100,
    min_roi: float = 0.0,
    min_probability_edge: float = 0.02,
    min_spread_edge: float = 2.0,
    clv_min_tracked_games: int = MARKET_CLV_MIN_TRACKED_GAMES,
    clv_min_positive_rate: float = MARKET_CLV_MIN_POSITIVE_RATE,
    clv_min_avg_prob_delta: float = MARKET_CLV_MIN_AVG_PROB_DELTA,
    spread_calibration_min_bets: int = SPREAD_COVER_CALIBRATION_MIN_BETS,
    spread_calibration_max_abs_error: float = SPREAD_COVER_CALIBRATION_MAX_ABS_ERROR,
    spread_calibration_confidence_min_prob: float = SPREAD_COVER_CALIBRATION_CONFIDENCE_MIN_PROB,
    generated_at: str | None = None,
) -> pd.DataFrame:
    """Build conservative market-edge recommendations from historical ROI artifacts."""
    generated_at = generated_at or datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, object]] = []

    moneyline_stability_required = moneyline_stability is not None and not moneyline_stability.empty
    moneyline_frame = disagreement_roi
    if moneyline_stability_required:
        if "stability_pass" in moneyline_stability.columns:
            stable_mask = moneyline_stability["stability_pass"].astype(bool)
        else:
            stable_mask = pd.Series(False, index=moneyline_stability.index)
        stable_thresholds = moneyline_stability[stable_mask].copy()
        if not stable_thresholds.empty and "threshold" in stable_thresholds.columns:
            stable_values = pd.to_numeric(stable_thresholds["threshold"], errors="coerce").dropna().to_numpy()
            frame = disagreement_roi.copy()
            frame["threshold"] = pd.to_numeric(frame.get("threshold"), errors="coerce")
            moneyline_frame = frame[
                np.isclose(
                    frame["threshold"].to_numpy(dtype=float)[:, None],
                    stable_values[None, :],
                    atol=1e-9,
                ).any(axis=1)
            ].copy()
        else:
            moneyline_frame = disagreement_roi.iloc[0:0].copy()

    moneyline_row, moneyline_recommended = _best_policy_candidate(
        moneyline_frame,
        min_bets=min_bets,
        min_roi=min_roi,
        threshold_col="threshold",
        min_threshold=min_probability_edge,
        max_threshold=0.10,
    )
    moneyline_review_row: pd.Series | None = None
    moneyline_review_recommended = False
    if moneyline_row is None and moneyline_stability_required:
        moneyline_review_row, moneyline_review_recommended = _best_policy_candidate(
            disagreement_roi,
            min_bets=min_bets,
            min_roi=min_roi,
            threshold_col="threshold",
            min_threshold=min_probability_edge,
            max_threshold=0.10,
        )
        moneyline_row = moneyline_review_row
        moneyline_recommended = False
    elif moneyline_stability_required:
        moneyline_review_row, moneyline_review_recommended = _best_policy_candidate(
            disagreement_roi,
            min_bets=min_bets,
            min_roi=min_roi,
            threshold_col="threshold",
            min_threshold=min_probability_edge,
            max_threshold=0.10,
        )

    moneyline_stability_row = _matching_moneyline_stability_row(moneyline_stability, moneyline_row)
    moneyline_stability_pass = True
    if moneyline_stability_required:
        moneyline_stability_pass = bool(
            moneyline_stability_row is not None and moneyline_stability_row.get("stability_pass", False)
        )
    moneyline_tier_override = None
    if moneyline_stability_required and not moneyline_stability_pass and moneyline_review_recommended:
        moneyline_tier_override = "review"
    moneyline_recommended = bool(moneyline_recommended and moneyline_stability_pass)
    moneyline_clv_gate_required = moneyline_clv_summary is not None
    moneyline_clv = summarize_moneyline_clv_gate(
        moneyline_clv_summary,
        moneyline_row,
        min_tracked_games=clv_min_tracked_games,
        min_positive_rate=clv_min_positive_rate,
        min_avg_prob_delta=clv_min_avg_prob_delta,
    )
    if moneyline_clv_gate_required and moneyline_row is not None:
        moneyline_clv_pass = bool(moneyline_clv.get("pass", False))
        if moneyline_recommended and not moneyline_clv_pass:
            clv_status = str(moneyline_clv.get("status", "unavailable") or "unavailable")
            if clv_status in {"unavailable", "no_matching_threshold_clv", "insufficient_tracked_games"}:
                moneyline_tier_override = "review"
            else:
                moneyline_tier_override = "no_bet"
            moneyline_recommended = False
    moneyline_selection = "no recommended moneyline disagreement threshold"
    if moneyline_row is not None:
        moneyline_selection = f"prob_edge >= {float(moneyline_row['threshold']):.2f}"
    rows.append(
        _market_policy_record(
            signal="moneyline_disagreement",
            row=moneyline_row,
            recommended=moneyline_recommended,
            selection=moneyline_selection,
            source_artifact="analysis/market_disagreement_roi.csv",
            generated_at=generated_at,
            min_bets=min_bets,
            min_roi=min_roi,
            min_probability_edge=min_probability_edge,
            min_spread_edge=None,
            recommendation_tier=moneyline_tier_override,
            stability=moneyline_stability_row,
            moneyline_clv=moneyline_clv if moneyline_clv_gate_required else None,
        )
    )

    spread_row, _spread_recommended = _best_policy_candidate(
        spread_roi,
        min_bets=min_bets,
        min_roi=min_roi,
        threshold_col="threshold",
        min_threshold=min_spread_edge,
    )
    spread_recommended = False
    spread_selection = "no recommended spread threshold"
    if spread_row is not None:
        spread_selection = f"{str(spread_row.get('bet_side', '')).lower()} edge > {float(spread_row['threshold']):g} pts"
    rows.append(
        _market_policy_record(
            signal="spread_threshold",
            row=spread_row,
            recommended=spread_recommended,
            selection=spread_selection,
            source_artifact="analysis/market_roi_spread.csv",
            generated_at=generated_at,
            min_bets=min_bets,
            min_roi=min_roi,
            min_probability_edge=None,
            min_spread_edge=min_spread_edge,
        )
    )

    edge_bins = spread_edge_bins.copy()
    if not edge_bins.empty and "scope" in edge_bins.columns:
        edge_bins = edge_bins[edge_bins["scope"].eq("overall")].copy()
    edge_bin_row, edge_bin_recommended = _best_policy_candidate(
        edge_bins,
        min_bets=min_bets,
        min_roi=min_roi,
        threshold_col="min_edge",
        min_threshold=min_spread_edge,
        allowed_values_col="edge_bin",
        allowed_values=ALLOWED_SPREAD_POLICY_EDGE_BINS,
    )
    edge_bin_selection = "no recommended spread edge bin"
    if edge_bin_row is not None:
        edge_bin_selection = f"{edge_bin_row.get('edge_bin')} pts"
    spread_calibration_summary = summarize_spread_cover_calibration_gate(
        spread_cover_recalibration,
        min_bets=spread_calibration_min_bets,
        max_abs_calibration_error=spread_calibration_max_abs_error,
        confidence_min_cover_prob=spread_calibration_confidence_min_prob,
    )
    spread_calibration_gate_applies = spread_cover_recalibration is not None and not spread_cover_recalibration.empty
    spread_calibration_pass = bool(spread_calibration_summary.get("pass", False))
    if spread_calibration_gate_applies:
        edge_bin_recommended = bool(edge_bin_recommended and spread_calibration_pass)
        if edge_bin_row is not None and not spread_calibration_pass:
            edge_bin_tier = "review"
        else:
            edge_bin_tier = None
    else:
        edge_bin_tier = None
    rows.append(
        _market_policy_record(
            signal="spread_edge_bin",
            row=edge_bin_row,
            recommended=edge_bin_recommended,
            selection=edge_bin_selection,
            source_artifact="analysis/market_roi_spread_edge_bins.csv",
            generated_at=generated_at,
            min_bets=min_bets,
            min_roi=min_roi,
            min_probability_edge=None,
            min_spread_edge=min_spread_edge,
            recommendation_tier=edge_bin_tier,
            spread_cover_calibration=spread_calibration_summary,
        )
    )

    out = pd.DataFrame.from_records(rows)
    for col in MARKET_EDGE_POLICY_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out[MARKET_EDGE_POLICY_COLUMNS]


def _json_safe_value(value: object) -> object:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _json_safe_record(row: pd.Series) -> dict[str, object]:
    return {key: _json_safe_value(value) for key, value in row.items()}


def run(args: argparse.Namespace) -> None:
    predictions = load_predictions(
        args.start_season,
        args.end_season,
        args.exclude_from_season,
        args.exclude_from_week,
    )
    if args.debug:
        print(
            f"[market_roi] Loaded {len(predictions)} evaluated games "
            f"({predictions['season'].min()}-{predictions['season'].max()})"
        )
    dataset = prepare_dataset(predictions)
    if args.debug:
        missing = len(predictions) - len(dataset)
        print(f"[market_roi] Prepared dataset with {len(dataset)} games ({missing} missing odds filtered out)")

    sigma = float(np.sqrt(np.mean(np.square(dataset["margin_error"])))) if not dataset.empty else float("nan")
    if args.debug and not np.isnan(sigma):
        print(f"[market_roi] Estimated margin RMSE sigma={sigma:.3f}")

    moneyline_df = summarize_moneyline(dataset, args.moneyline_thresholds)
    spread_df = summarize_spread(dataset, args.spread_thresholds, sigma)
    benchmark_df = build_market_benchmark(dataset)
    if args.market_clv_detail.exists():
        clv_detail_df = load_market_clv_detail(args.market_clv_detail)
    else:
        clv_detail_df = pd.DataFrame()
        if args.debug:
            print(f"[market_roi] CLV detail not found at {args.market_clv_detail}; benchmark CLV fields unavailable.")
    benchmark_df = attach_market_clv_detail(benchmark_df, clv_detail_df)
    benchmark_summary_df = summarize_market_benchmark(benchmark_df)
    disagreement_roi_df = summarize_disagreement_moneyline_roi(benchmark_df, args.benchmark_edge_thresholds)
    disagreement_stability_df = summarize_disagreement_moneyline_season_stability(
        benchmark_df,
        args.benchmark_edge_thresholds,
        min_season_bets=args.stability_min_season_bets,
        min_qualified_seasons=args.stability_min_qualified_seasons,
        min_positive_season_rate=args.stability_min_positive_season_rate,
        severe_negative_roi=args.stability_severe_negative_roi,
        max_severe_negative_season_rate=args.stability_max_severe_negative_season_rate,
        recent_seasons=args.stability_recent_seasons,
        min_recent_qualified_seasons=args.stability_min_recent_qualified_seasons,
        min_recent_positive_season_rate=args.stability_min_recent_positive_season_rate,
    )
    disagreement_edge_bins_df = summarize_disagreement_moneyline_edge_bins(benchmark_df)
    disagreement_segments_df = summarize_disagreement_moneyline_segments(benchmark_df)
    probability_edge_calibration_df = summarize_market_probability_edge_calibration(benchmark_df)
    spread_edge_bins_df = summarize_spread_edge_bins(dataset)
    spread_key_numbers_df = summarize_spread_key_numbers(dataset)
    spread_calibration_df = summarize_spread_cover_calibration(dataset, sigma)
    spread_recalibration_df = summarize_spread_cover_recalibration(
        spread_calibration_df,
        min_bets=args.spread_calibration_min_bets,
        max_abs_calibration_error=args.spread_calibration_max_abs_error,
        confidence_min_cover_prob=args.spread_calibration_confidence_min_prob,
    )
    market_segments_df = summarize_market_utility_segments(benchmark_df)
    if args.market_clv_summary.exists():
        market_clv_summary_df = pd.read_csv(args.market_clv_summary)
    else:
        market_clv_summary_df = pd.DataFrame()
        if args.debug:
            print(f"[market_roi] CLV summary not found at {args.market_clv_summary}; moneyline policy will be review-only.")
    market_policy_df = build_market_edge_policy(
        disagreement_roi_df,
        spread_df,
        spread_edge_bins_df,
        moneyline_stability=disagreement_stability_df,
        moneyline_clv_summary=market_clv_summary_df,
        spread_cover_recalibration=spread_recalibration_df,
        min_bets=args.policy_min_bets,
        min_roi=args.policy_min_roi,
        min_probability_edge=args.policy_min_probability_edge,
        min_spread_edge=args.policy_min_spread_edge,
        clv_min_tracked_games=args.clv_min_tracked_games,
        clv_min_positive_rate=args.clv_min_positive_rate,
        clv_min_avg_prob_delta=args.clv_min_avg_prob_delta,
        spread_calibration_min_bets=args.spread_calibration_min_bets,
        spread_calibration_max_abs_error=args.spread_calibration_max_abs_error,
        spread_calibration_confidence_min_prob=args.spread_calibration_confidence_min_prob,
    )

    moneyline_path = ANALYSIS_DIR / "market_roi_moneyline.csv"
    spread_path = ANALYSIS_DIR / "market_roi_spread.csv"
    benchmark_path = ANALYSIS_DIR / "market_benchmark.csv"
    benchmark_summary_path = ANALYSIS_DIR / "market_benchmark_summary.csv"
    disagreement_roi_path = ANALYSIS_DIR / "market_disagreement_roi.csv"
    disagreement_stability_path = ANALYSIS_DIR / "market_disagreement_season_stability.csv"
    disagreement_edge_bins_path = ANALYSIS_DIR / "market_disagreement_edge_bins.csv"
    disagreement_segments_path = ANALYSIS_DIR / "market_disagreement_segments.csv"
    probability_edge_calibration_path = ANALYSIS_DIR / "market_probability_edge_calibration.csv"
    spread_edge_bins_path = ANALYSIS_DIR / "market_roi_spread_edge_bins.csv"
    spread_key_numbers_path = ANALYSIS_DIR / "market_spread_key_numbers.csv"
    spread_calibration_path = ANALYSIS_DIR / "market_spread_calibration.csv"
    spread_recalibration_path = ANALYSIS_DIR / "market_spread_cover_recalibration.csv"
    market_segments_path = ANALYSIS_DIR / "market_utility_segments.csv"
    market_policy_path = ANALYSIS_DIR / "market_edge_policy.csv"
    summary_path = ANALYSIS_DIR / "market_roi_summary.json"

    moneyline_df.sort_values(["bet_side", "threshold"]).to_csv(moneyline_path, index=False)
    spread_df.sort_values(["bet_side", "threshold"]).to_csv(spread_path, index=False)

    benchmark_columns = [
        "season",
        "week",
        "game_id",
        "home_team",
        "away_team",
        "home_win_prob",
        "away_win_prob",
        "home_implied_prob",
        "away_implied_prob",
        "home_implied_prob_novig",
        "away_implied_prob_novig",
        "home_moneyline",
        "away_moneyline",
        "model_home_pick",
        "vegas_home_pick",
        "favorite_agreement",
        "model_pick",
        "vegas_pick",
        "actual_winner",
        "model_correct",
        "vegas_correct",
        "model_pick_prob",
        "vegas_pick_prob",
        "vegas_model_side_prob",
        "prob_edge_model_side",
        "model_side_moneyline",
        "model_side_profit",
        "vegas_side_moneyline",
        "vegas_side_profit",
        *MARKET_BENCHMARK_CLV_COLUMNS,
        "pred_home_margin",
        "market_home_margin",
        "spread_edge_points",
        "home_margin",
        "spread_result_points",
    ]
    benchmark_columns = [col for col in benchmark_columns if col in benchmark_df.columns]
    benchmark_df.sort_values(["season", "week", "game_id"])[benchmark_columns].to_csv(benchmark_path, index=False)
    benchmark_summary_df.to_csv(benchmark_summary_path, index=False)
    disagreement_roi_df.sort_values(["threshold"]).to_csv(disagreement_roi_path, index=False)
    disagreement_stability_df.sort_values(["threshold"]).to_csv(disagreement_stability_path, index=False)
    disagreement_edge_bins_df.to_csv(disagreement_edge_bins_path, index=False)
    disagreement_segments_df.to_csv(disagreement_segments_path, index=False)
    probability_edge_calibration_df.to_csv(probability_edge_calibration_path, index=False)
    spread_edge_bins_df.to_csv(spread_edge_bins_path, index=False)
    spread_key_numbers_df.to_csv(spread_key_numbers_path, index=False)
    spread_calibration_df.to_csv(spread_calibration_path, index=False)
    spread_recalibration_df.to_csv(spread_recalibration_path, index=False)
    market_segments_df.to_csv(market_segments_path, index=False)
    market_policy_df.to_csv(market_policy_path, index=False)

    summary: dict[str, object] = {
        "settings": {
            "start_season": args.start_season,
            "end_season": args.end_season,
            "exclude_from_season": args.exclude_from_season,
            "exclude_from_week": args.exclude_from_week,
            "moneyline_thresholds": list(map(float, args.moneyline_thresholds)),
            "spread_thresholds": list(map(float, args.spread_thresholds)),
            "benchmark_edge_thresholds": list(map(float, args.benchmark_edge_thresholds)),
            "min_moneyline_edge_bin_bets": int(args.min_moneyline_edge_bin_bets),
            "min_probability_calibration_bin_games": int(args.min_probability_calibration_bin_games),
            "policy_min_bets": int(args.policy_min_bets),
            "policy_min_roi": float(args.policy_min_roi),
            "policy_min_probability_edge": float(args.policy_min_probability_edge),
            "policy_min_spread_edge": float(args.policy_min_spread_edge),
            "market_clv_detail": str(args.market_clv_detail),
            "market_clv_summary": str(args.market_clv_summary),
            "clv_min_tracked_games": int(args.clv_min_tracked_games),
            "clv_min_positive_rate": float(args.clv_min_positive_rate),
            "clv_min_avg_prob_delta": float(args.clv_min_avg_prob_delta),
            "stability_min_season_bets": int(args.stability_min_season_bets),
            "stability_min_qualified_seasons": int(args.stability_min_qualified_seasons),
            "stability_min_positive_season_rate": float(args.stability_min_positive_season_rate),
            "stability_severe_negative_roi": float(args.stability_severe_negative_roi),
            "stability_max_severe_negative_season_rate": float(args.stability_max_severe_negative_season_rate),
            "stability_recent_seasons": int(args.stability_recent_seasons),
            "stability_min_recent_qualified_seasons": int(args.stability_min_recent_qualified_seasons),
            "stability_min_recent_positive_season_rate": float(
                args.stability_min_recent_positive_season_rate
            ),
            "spread_key_numbers": list(map(float, SPREAD_KEY_NUMBERS)),
            "spread_key_number_tolerance": float(SPREAD_KEY_NUMBER_TOLERANCE),
            "spread_calibration_min_bets": int(args.spread_calibration_min_bets),
            "spread_calibration_max_abs_error": float(args.spread_calibration_max_abs_error),
            "spread_calibration_confidence_min_prob": float(args.spread_calibration_confidence_min_prob),
            "margin_error_sigma": float(sigma),
            "spread_line_convention": "market_home_margin",
        },
        "moneyline_best": {},
        "spread_best": {},
        "market_benchmark_overall": {},
        "market_benchmark_disagreement": {},
        "moneyline_disagreement_best": {},
        "moneyline_disagreement_stability": {},
        "moneyline_disagreement_clv_gate": {},
        "moneyline_disagreement_edge_bin_best": {},
        "moneyline_disagreement_segment_best": {},
        "market_probability_edge_calibration_overall": {},
        "market_probability_edge_calibration_model_best": {},
        "spread_edge_bin_best": {},
        "spread_key_number_policy_bucket": {},
        "spread_cover_calibration_gate": {},
        "market_edge_policy": [],
    }
    valid_ml = moneyline_df[(moneyline_df["n_bets"] > 0) & moneyline_df["roi"].notna()]
    if not valid_ml.empty:
        best_ml = valid_ml.loc[valid_ml["roi"].idxmax()]
        summary["moneyline_best"] = _json_safe_record(best_ml)
    valid_spread = spread_df[(spread_df["n_bets"] > 0) & spread_df["roi"].notna()]
    if not valid_spread.empty:
        best_spread = valid_spread.loc[valid_spread["roi"].idxmax()]
        summary["spread_best"] = _json_safe_record(best_spread)
    if not benchmark_summary_df.empty:
        overall = benchmark_summary_df[
            (benchmark_summary_df["scope"] == "overall") & (benchmark_summary_df["segment"] == "all")
        ]
        if not overall.empty:
            summary["market_benchmark_overall"] = _json_safe_record(overall.iloc[0])
        disagreement = benchmark_summary_df[
            (benchmark_summary_df["scope"] == "overall") & (benchmark_summary_df["segment"] == "disagree")
        ]
        if not disagreement.empty:
            summary["market_benchmark_disagreement"] = _json_safe_record(disagreement.iloc[0])
    valid_disagreement = disagreement_roi_df[
        (disagreement_roi_df["n_bets"] > 0) & disagreement_roi_df["roi"].notna()
    ]
    if not valid_disagreement.empty:
        best_disagreement = valid_disagreement.loc[valid_disagreement["roi"].idxmax()]
        summary["moneyline_disagreement_best"] = _json_safe_record(best_disagreement)
    policy_moneyline = (
        market_policy_df[market_policy_df["signal"].eq("moneyline_disagreement")]
        if not market_policy_df.empty and "signal" in market_policy_df.columns
        else pd.DataFrame()
    )
    if not policy_moneyline.empty and not disagreement_stability_df.empty:
        threshold = pd.to_numeric(policy_moneyline.iloc[0].get("threshold"), errors="coerce")
        if pd.notna(threshold):
            stability_match = disagreement_stability_df[
                np.isclose(
                    pd.to_numeric(disagreement_stability_df["threshold"], errors="coerce"),
                    float(threshold),
                    atol=1e-9,
                    equal_nan=False,
                )
            ]
            if not stability_match.empty:
                summary["moneyline_disagreement_stability"] = _json_safe_record(stability_match.iloc[0])
    if not policy_moneyline.empty:
        moneyline_policy_row = policy_moneyline.iloc[0]
        summary["moneyline_disagreement_clv_gate"] = {
            key: _json_safe_value(moneyline_policy_row.get(key))
            for key in [
                "clv_pass",
                "clv_status",
                "clv_threshold",
                "clv_tracked_games",
                "clv_avg_prob_delta",
                "clv_positive_rate",
                "clv_min_tracked_games",
                "clv_min_positive_rate",
                "clv_min_avg_prob_delta",
            ]
        }
    valid_disagreement_bins = disagreement_edge_bins_df[
        (disagreement_edge_bins_df["scope"] == "overall")
        & (disagreement_edge_bins_df["n_bets"] >= args.min_moneyline_edge_bin_bets)
        & disagreement_edge_bins_df["roi"].notna()
    ]
    if not valid_disagreement_bins.empty:
        best_disagreement_bin = valid_disagreement_bins.loc[valid_disagreement_bins["roi"].idxmax()]
        summary["moneyline_disagreement_edge_bin_best"] = _json_safe_record(best_disagreement_bin)
    valid_disagreement_segments = disagreement_segments_df[
        (disagreement_segments_df["scope"] == "overall")
        & disagreement_segments_df["segment_family"].ne("all")
        & (disagreement_segments_df["n_bets"] >= args.min_moneyline_edge_bin_bets)
        & disagreement_segments_df["roi"].notna()
    ]
    if not valid_disagreement_segments.empty:
        best_disagreement_segment = valid_disagreement_segments.loc[valid_disagreement_segments["roi"].idxmax()]
        summary["moneyline_disagreement_segment_best"] = _json_safe_record(best_disagreement_segment)
    overall_probability_calibration = probability_edge_calibration_df[
        (probability_edge_calibration_df["scope"] == "overall")
        & (probability_edge_calibration_df["bin_family"] == "all")
        & (probability_edge_calibration_df["bin_label"] == "all")
    ]
    if not overall_probability_calibration.empty:
        summary["market_probability_edge_calibration_overall"] = _json_safe_record(
            overall_probability_calibration.iloc[0]
        )
    valid_model_probability_bins = probability_edge_calibration_df[
        (probability_edge_calibration_df["scope"] == "overall")
        & (probability_edge_calibration_df["bin_family"] == "model_probability")
        & (probability_edge_calibration_df["n_games"] >= args.min_probability_calibration_bin_games)
        & probability_edge_calibration_df["brier_delta_model_minus_vegas"].notna()
    ]
    if not valid_model_probability_bins.empty:
        best_probability_bin = valid_model_probability_bins.loc[
            valid_model_probability_bins["brier_delta_model_minus_vegas"].idxmin()
        ]
        summary["market_probability_edge_calibration_model_best"] = _json_safe_record(best_probability_bin)
    valid_edge_bins = spread_edge_bins_df[
        (spread_edge_bins_df["scope"] == "overall")
        & (spread_edge_bins_df["n_bets"] >= args.min_spread_edge_bin_bets)
        & spread_edge_bins_df["roi"].notna()
    ]
    if not valid_edge_bins.empty:
        best_edge_bin = valid_edge_bins.loc[valid_edge_bins["roi"].idxmax()]
        summary["spread_edge_bin_best"] = _json_safe_record(best_edge_bin)
    policy_key_rows = spread_key_numbers_df[
        (spread_key_numbers_df["scope"] == "overall")
        & (spread_key_numbers_df["edge_bucket"] == "2-3")
        & (spread_key_numbers_df["segment"] == "key_window")
        & (spread_key_numbers_df["n_bets"] > 0)
    ]
    if not policy_key_rows.empty:
        summary["spread_key_number_policy_bucket"] = [
            _json_safe_record(row) for _, row in policy_key_rows.sort_values("key_number").iterrows()
        ]
    spread_calibration_gate = summarize_spread_cover_calibration_gate(
        spread_recalibration_df,
        min_bets=args.spread_calibration_min_bets,
        max_abs_calibration_error=args.spread_calibration_max_abs_error,
        confidence_min_cover_prob=args.spread_calibration_confidence_min_prob,
    )
    summary["spread_cover_calibration_gate"] = {
        key: _json_safe_value(value) for key, value in spread_calibration_gate.items()
    }
    if not market_policy_df.empty:
        summary["market_edge_policy"] = [_json_safe_record(row) for _, row in market_policy_df.iterrows()]

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=== Model vs Vegas benchmark ===")
    if benchmark_summary_df.empty:
        print("No benchmark rows matched the criteria.")
    else:
        display_cols = [
            "scope",
            "season",
            "segment",
            "n_games",
            "model_accuracy",
            "vegas_accuracy",
            "favorite_agreement_rate",
            "model_side_roi",
            "avg_prob_edge_model_side",
        ]
        display = benchmark_summary_df[
            (benchmark_summary_df["scope"] == "overall")
            | ((benchmark_summary_df["scope"] == "season") & (benchmark_summary_df["segment"] == "all"))
        ]
        print(display[display_cols].to_string(index=False))

    print("=== Moneyline ROI summary ===")
    if moneyline_df.empty:
        print("No moneyline bets matched the criteria.")
    else:
        display_cols = [
            "bet_side",
            "threshold",
            "n_bets",
            "wins",
            "losses",
            "pushes",
            "roi",
            "avg_model_prob",
            "avg_implied_prob",
            "avg_expected_profit",
        ]
        print(moneyline_df[display_cols].sort_values(["bet_side", "threshold"]).to_string(index=False))

    print("\n=== Moneyline disagreement ROI summary ===")
    if disagreement_roi_df.empty:
        print("No disagreement moneyline bets matched the criteria.")
    else:
        display_cols = [
            "threshold",
            "n_bets",
            "wins",
            "losses",
            "pushes",
            "roi",
            "hit_rate",
            "avg_prob_edge_model_side",
        ]
        print(disagreement_roi_df[display_cols].sort_values(["threshold"]).to_string(index=False))

    print("\n=== Moneyline disagreement season-stability summary ===")
    if disagreement_stability_df.empty:
        print("No disagreement stability rows matched the criteria.")
    else:
        display_cols = [
            "threshold",
            "n_bets",
            "roi",
            "qualified_seasons",
            "positive_qualified_season_rate",
            "recent_qualified_seasons",
            "recent_positive_qualified_season_rate",
            "severe_negative_qualified_season_rate",
            "stability_pass",
            "stability_status",
        ]
        print(disagreement_stability_df[display_cols].sort_values(["threshold"]).to_string(index=False))

    print("\n=== Moneyline disagreement bounded edge-bin ROI summary ===")
    if disagreement_edge_bins_df.empty:
        print("No disagreement edge-bin rows matched the criteria.")
    else:
        display_cols = [
            "scope",
            "season",
            "edge_bin",
            "n_bets",
            "wins",
            "losses",
            "pushes",
            "roi",
            "hit_rate",
            "avg_prob_edge_model_side",
        ]
        display = disagreement_edge_bins_df[disagreement_edge_bins_df["scope"] == "overall"]
        print(display[display_cols].to_string(index=False))

    print("\n=== Moneyline disagreement segment summary ===")
    if disagreement_segments_df.empty:
        print("No disagreement segment rows matched the criteria.")
    else:
        display_cols = [
            "segment_family",
            "segment",
            "n_bets",
            "wins",
            "losses",
            "pushes",
            "roi",
            "hit_rate",
            "avg_prob_edge_model_side",
            "clv_games",
            "positive_moneyline_clv_rate",
        ]
        display = disagreement_segments_df[
            (disagreement_segments_df["scope"] == "overall")
            & (
                disagreement_segments_df["segment_family"].eq("all")
                | (disagreement_segments_df["n_bets"] >= args.min_moneyline_edge_bin_bets)
            )
        ]
        print(display[display_cols].to_string(index=False))

    print("\n=== Model vs no-vig Vegas probability calibration summary ===")
    if probability_edge_calibration_df.empty:
        print("No probability calibration rows matched the criteria.")
    else:
        display_cols = [
            "bin_family",
            "bin_label",
            "n_games",
            "actual_model_side_win_rate",
            "avg_model_pick_prob",
            "avg_vegas_model_side_prob",
            "model_calibration_error",
            "vegas_calibration_error",
            "model_brier",
            "vegas_brier",
            "brier_delta_model_minus_vegas",
            "model_side_roi",
        ]
        display = probability_edge_calibration_df[
            (probability_edge_calibration_df["scope"] == "overall")
            & (
                probability_edge_calibration_df["bin_family"].isin(
                    ["all", "model_probability", "model_minus_vegas_probability_edge"]
                )
            )
        ]
        print(display[display_cols].to_string(index=False))

    print("\n=== Spread ROI summary ===")
    if spread_df.empty:
        print("No spread bets matched the criteria.")
    else:
        display_cols = [
            "bet_side",
            "threshold",
            "n_bets",
            "wins",
            "losses",
            "pushes",
            "roi",
            "avg_cover_prob",
            "avg_expected_profit",
            "avg_edge_points",
        ]
        print(spread_df[display_cols].sort_values(["bet_side", "threshold"]).to_string(index=False))

    print("\n=== Spread edge-bin ROI summary ===")
    if spread_edge_bins_df.empty:
        print("No spread edge-bin bets matched the criteria.")
    else:
        display_cols = [
            "scope",
            "season",
            "edge_bin",
            "n_bets",
            "home_bets",
            "away_bets",
            "roi",
            "hit_rate",
            "avg_abs_edge_points",
        ]
        display = spread_edge_bins_df[spread_edge_bins_df["scope"] == "overall"]
        print(display[display_cols].to_string(index=False))

    print("\n=== Spread key-number diagnostics ===")
    if spread_key_numbers_df.empty:
        print("No spread key-number rows matched the criteria.")
    else:
        display_cols = [
            "key_number",
            "edge_bucket",
            "segment",
            "n_bets",
            "home_bets",
            "away_bets",
            "roi",
            "hit_rate",
            "avg_abs_edge_points",
            "avg_market_abs_line",
            "avg_model_abs_line",
        ]
        display = spread_key_numbers_df[
            (spread_key_numbers_df["scope"] == "overall")
            & (spread_key_numbers_df["edge_bucket"].isin(["all", "2-3", "3-5", "5-7", "7-10", "10+"]))
            & (spread_key_numbers_df["segment"].isin(["edge_crosses_key", "key_window"]))
        ]
        print(display[display_cols].to_string(index=False))

    print("\n=== Spread cover-probability calibration summary ===")
    if spread_calibration_df.empty:
        print("No spread calibration rows matched the criteria.")
    else:
        display_cols = [
            "scope",
            "season",
            "bet_side",
            "cover_prob_bin",
            "n_bets",
            "actual_cover_rate",
            "avg_cover_prob",
            "calibration_error",
            "roi",
        ]
        display = spread_calibration_df[
            (spread_calibration_df["scope"] == "overall") & (spread_calibration_df["bet_side"] == "all")
        ]
        print(display[display_cols].to_string(index=False))

    print("\n=== Spread cover-probability recalibration gate ===")
    if spread_recalibration_df.empty:
        print("No spread recalibration rows matched the criteria.")
    else:
        display_cols = [
            "cover_prob_bin",
            "n_bets",
            "actual_cover_rate",
            "avg_cover_prob",
            "recalibrated_cover_prob",
            "calibration_error",
            "calibration_status",
            "recommended_for_confidence",
        ]
        display = spread_recalibration_df[
            (spread_recalibration_df["scope"] == "overall") & (spread_recalibration_df["bet_side"] == "all")
        ]
        print(display[display_cols].to_string(index=False))

    print("\n=== Market edge policy ===")
    if market_policy_df.empty:
        print("No market edge policy rows generated.")
    else:
        display_cols = [
            "signal",
            "recommended",
            "recommendation_tier",
            "selection",
            "n_bets",
            "roi",
            "hit_rate",
            "clv_status",
            "clv_tracked_games",
            "clv_avg_prob_delta",
            "clv_positive_rate",
        ]
        print(market_policy_df[display_cols].to_string(index=False))

    print(f"\nBenchmark rows written to {benchmark_path.resolve()}")
    print(f"Benchmark summary written to {benchmark_summary_path.resolve()}")
    print(f"Disagreement ROI results written to {disagreement_roi_path.resolve()}")
    print(f"Disagreement season-stability results written to {disagreement_stability_path.resolve()}")
    print(f"Disagreement edge-bin results written to {disagreement_edge_bins_path.resolve()}")
    print(f"Disagreement segment results written to {disagreement_segments_path.resolve()}")
    print(f"Probability edge calibration results written to {probability_edge_calibration_path.resolve()}")
    print(f"Market utility segments written to {market_segments_path.resolve()}")
    print(f"Market edge policy written to {market_policy_path.resolve()}")
    print(f"\nMoneyline results written to {moneyline_path.resolve()}")
    print(f"Spread results written to {spread_path.resolve()}")
    print(f"Spread edge-bin results written to {spread_edge_bins_path.resolve()}")
    print(f"Spread key-number diagnostics written to {spread_key_numbers_path.resolve()}")
    print(f"Spread calibration results written to {spread_calibration_path.resolve()}")
    print(f"Spread cover recalibration results written to {spread_recalibration_path.resolve()}")
    print(f"Summary written to {summary_path.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate market ROI for model predictions.")
    parser.add_argument("--start-season", type=int, default=2016, help="Lower bound season for analysis (inclusive).")
    parser.add_argument("--end-season", type=int, default=None, help="Upper bound season for analysis (inclusive).")
    parser.add_argument("--exclude-from-season", type=int, default=None, help="Exclude this season/week and later rows from ROI analysis.")
    parser.add_argument("--exclude-from-week", type=int, default=None, help="Exclude this season/week and later rows from ROI analysis.")
    parser.add_argument("--debug", action="store_true", help="Enable verbose logging.")
    parser.add_argument(
        "--moneyline-thresholds",
        type=float,
        nargs="+",
        default=[0.0, 0.02, 0.05],
        help="Expected value thresholds (per unit) for moneyline bets.",
    )
    parser.add_argument(
        "--spread-thresholds",
        type=float,
        nargs="+",
        default=[0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0],
        help="Predicted margin edge thresholds (points) for spread bets.",
    )
    parser.add_argument(
        "--benchmark-edge-thresholds",
        type=float,
        nargs="+",
        default=[0.0, 0.02, 0.05, 0.08, 0.10, 0.15],
        help="Model-vs-no-vig-Vegas probability edge thresholds for disagreement moneyline ROI.",
    )
    parser.add_argument(
        "--min-spread-edge-bin-bets",
        type=int,
        default=50,
        help="Minimum overall bets required before a spread edge bin can be selected as best in the JSON summary.",
    )
    parser.add_argument(
        "--min-moneyline-edge-bin-bets",
        type=int,
        default=25,
        help="Minimum overall bets required before a bounded moneyline disagreement edge bin can be selected as best in the JSON summary.",
    )
    parser.add_argument(
        "--min-probability-calibration-bin-games",
        type=int,
        default=50,
        help="Minimum games required before a probability calibration bin can be selected as best in the JSON summary.",
    )
    parser.add_argument(
        "--policy-min-bets",
        type=int,
        default=100,
        help="Minimum historical bets required before a market edge policy row can be recommended.",
    )
    parser.add_argument(
        "--policy-min-roi",
        type=float,
        default=0.0,
        help="Minimum historical ROI required before a market edge policy row can be recommended.",
    )
    parser.add_argument(
        "--policy-min-probability-edge",
        type=float,
        default=0.02,
        help="Minimum model-vs-no-vig-market probability edge for disagreement moneyline policy candidates.",
    )
    parser.add_argument(
        "--policy-min-spread-edge",
        type=float,
        default=2.0,
        help="Minimum predicted margin edge in points for spread policy candidates.",
    )
    parser.add_argument(
        "--market-clv-detail",
        type=Path,
        default=MARKET_CLV_DETAIL_PATH,
        help="Optional weekly per-game CLV detail merged into market_benchmark.csv when available.",
    )
    parser.add_argument(
        "--market-clv-summary",
        type=Path,
        default=MARKET_CLV_SUMMARY_PATH,
        help="Weekly CLV summary used to gate live moneyline disagreement policy recommendations.",
    )
    parser.add_argument(
        "--clv-min-tracked-games",
        type=int,
        default=MARKET_CLV_MIN_TRACKED_GAMES,
        help="Minimum current-week tracked CLV games required before a moneyline policy can be strong.",
    )
    parser.add_argument(
        "--clv-min-positive-rate",
        type=float,
        default=MARKET_CLV_MIN_POSITIVE_RATE,
        help="Minimum positive CLV rate required before a moneyline policy can be strong.",
    )
    parser.add_argument(
        "--clv-min-avg-prob-delta",
        type=float,
        default=MARKET_CLV_MIN_AVG_PROB_DELTA,
        help="Minimum average no-vig implied-probability CLV delta required before a moneyline policy can be strong.",
    )
    parser.add_argument(
        "--spread-calibration-min-bets",
        type=int,
        default=SPREAD_COVER_CALIBRATION_MIN_BETS,
        help="Minimum bets in a spread cover-probability bin before it can pass the confidence gate.",
    )
    parser.add_argument(
        "--spread-calibration-max-abs-error",
        type=float,
        default=SPREAD_COVER_CALIBRATION_MAX_ABS_ERROR,
        help="Maximum absolute spread cover-probability calibration error allowed for confidence use.",
    )
    parser.add_argument(
        "--spread-calibration-confidence-min-prob",
        type=float,
        default=SPREAD_COVER_CALIBRATION_CONFIDENCE_MIN_PROB,
        help="Minimum raw spread cover probability treated as high-confidence for the calibration gate.",
    )
    parser.add_argument(
        "--stability-min-season-bets",
        type=int,
        default=20,
        help="Minimum bets in a season before moneyline disagreement stability includes that season.",
    )
    parser.add_argument(
        "--stability-min-qualified-seasons",
        type=int,
        default=5,
        help="Minimum qualified seasons required before moneyline disagreement policy can be recommended.",
    )
    parser.add_argument(
        "--stability-min-positive-season-rate",
        type=float,
        default=0.55,
        help="Minimum share of qualified seasons with positive ROI for moneyline disagreement policy.",
    )
    parser.add_argument(
        "--stability-severe-negative-roi",
        type=float,
        default=-0.15,
        help="ROI cutoff that counts as a severe negative season for moneyline disagreement stability.",
    )
    parser.add_argument(
        "--stability-max-severe-negative-season-rate",
        type=float,
        default=0.25,
        help="Maximum share of qualified seasons allowed at or below the severe negative ROI cutoff.",
    )
    parser.add_argument(
        "--stability-recent-seasons",
        type=int,
        default=3,
        help="Number of most recent seasons considered by the recency branch of the stability gate.",
    )
    parser.add_argument(
        "--stability-min-recent-qualified-seasons",
        type=int,
        default=2,
        help="Minimum recent qualified seasons required before moneyline disagreement policy can be recommended.",
    )
    parser.add_argument(
        "--stability-min-recent-positive-season-rate",
        type=float,
        default=0.50,
        help="Minimum share of recent qualified seasons with positive ROI for moneyline disagreement policy.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
