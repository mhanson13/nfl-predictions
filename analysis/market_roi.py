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
    benchmark_summary_df = summarize_market_benchmark(benchmark_df)
    disagreement_roi_df = summarize_disagreement_moneyline_roi(benchmark_df, args.benchmark_edge_thresholds)
    spread_edge_bins_df = summarize_spread_edge_bins(dataset)

    moneyline_path = ANALYSIS_DIR / "market_roi_moneyline.csv"
    spread_path = ANALYSIS_DIR / "market_roi_spread.csv"
    benchmark_path = ANALYSIS_DIR / "market_benchmark.csv"
    benchmark_summary_path = ANALYSIS_DIR / "market_benchmark_summary.csv"
    disagreement_roi_path = ANALYSIS_DIR / "market_disagreement_roi.csv"
    spread_edge_bins_path = ANALYSIS_DIR / "market_roi_spread_edge_bins.csv"
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
    spread_edge_bins_df.to_csv(spread_edge_bins_path, index=False)

    summary: dict[str, object] = {
        "settings": {
            "start_season": args.start_season,
            "end_season": args.end_season,
            "exclude_from_season": args.exclude_from_season,
            "exclude_from_week": args.exclude_from_week,
            "moneyline_thresholds": list(map(float, args.moneyline_thresholds)),
            "spread_thresholds": list(map(float, args.spread_thresholds)),
            "benchmark_edge_thresholds": list(map(float, args.benchmark_edge_thresholds)),
            "margin_error_sigma": float(sigma),
            "spread_line_convention": "market_home_margin",
        },
        "moneyline_best": {},
        "spread_best": {},
        "market_benchmark_overall": {},
        "market_benchmark_disagreement": {},
        "moneyline_disagreement_best": {},
        "spread_edge_bin_best": {},
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
    valid_edge_bins = spread_edge_bins_df[
        (spread_edge_bins_df["scope"] == "overall")
        & (spread_edge_bins_df["n_bets"] >= args.min_spread_edge_bin_bets)
        & spread_edge_bins_df["roi"].notna()
    ]
    if not valid_edge_bins.empty:
        best_edge_bin = valid_edge_bins.loc[valid_edge_bins["roi"].idxmax()]
        summary["spread_edge_bin_best"] = _json_safe_record(best_edge_bin)

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

    print(f"\nBenchmark rows written to {benchmark_path.resolve()}")
    print(f"Benchmark summary written to {benchmark_summary_path.resolve()}")
    print(f"Disagreement ROI results written to {disagreement_roi_path.resolve()}")
    print(f"\nMoneyline results written to {moneyline_path.resolve()}")
    print(f"Spread results written to {spread_path.resolve()}")
    print(f"Spread edge-bin results written to {spread_edge_bins_path.resolve()}")
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
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
