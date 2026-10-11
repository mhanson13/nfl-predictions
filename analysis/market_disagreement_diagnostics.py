# Copyright (c) 2025 Matt Hanson
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Favorite-disagreement diagnostics for model-vs-market decisions.

This module is intentionally diagnostic-only. It studies games where the
model-selected winner disagrees with the no-vig Vegas favorite, then summarizes
which slices are helping or hurting before any live recommendation policy is
changed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ANALYSIS_DIR = Path("analysis")
DEFAULT_BENCHMARK_PATH = ANALYSIS_DIR / "market_benchmark.csv"
DEFAULT_BLEND_PATH = ANALYSIS_DIR / "market_blend_backtest.csv"
DEFAULT_GAMES_OUTPUT = ANALYSIS_DIR / "market_disagreement_games.csv"
DEFAULT_DIAGNOSTICS_OUTPUT = ANALYSIS_DIR / "market_disagreement_diagnostics.csv"
DEFAULT_RECOMMENDATIONS_OUTPUT = ANALYSIS_DIR / "market_disagreement_recommendations.csv"

CONFIDENCE_BINS = [
    (0.50, 0.55, "50-55%"),
    (0.55, 0.60, "55-60%"),
    (0.60, 0.65, "60-65%"),
    (0.65, 0.70, "65-70%"),
    (0.70, np.inf, "70%+"),
]

PROB_EDGE_BINS = [
    (0.00, 0.02, "0-2%"),
    (0.02, 0.05, "2-5%"),
    (0.05, 0.08, "5-8%"),
    (0.08, 0.10, "8-10%"),
    (0.10, 0.15, "10-15%"),
    (0.15, np.inf, "15%+"),
]

MARKET_SIDE_PROB_BINS = [
    (0.00, 0.35, "<35%"),
    (0.35, 0.40, "35-40%"),
    (0.40, 0.45, "40-45%"),
    (0.45, 0.50, "45-50%"),
    (0.50, np.inf, "50%+"),
]

SPREAD_EDGE_BINS = [
    (0.00, 2.00, "0-2 pts"),
    (2.00, 3.00, "2-3 pts"),
    (3.00, 5.00, "3-5 pts"),
    (5.00, np.inf, "5+ pts"),
]

SEGMENT_COLUMNS = {
    "model_side": "model_side",
    "model_side_odds_bucket": "model_side_odds_bucket",
    "model_confidence_bin": "model_confidence_bin",
    "prob_edge_bin": "prob_edge_bin",
    "vegas_model_side_prob_bin": "vegas_model_side_prob_bin",
    "spread_edge_bin": "spread_edge_bin",
    "spread_edge_alignment": "spread_edge_alignment",
    "moneyline_clv_segment": "moneyline_clv_segment",
    "blend_followed": "blend_followed",
}

GAME_COLUMNS = [
    "season",
    "week",
    "game_id",
    "home_team",
    "away_team",
    "model_team",
    "vegas_team",
    "actual_winner",
    "model_side",
    "model_correct",
    "vegas_correct",
    "model_pick_prob",
    "vegas_model_side_prob",
    "prob_edge_model_side",
    "model_side_moneyline",
    "model_side_profit",
    "vegas_side_profit",
    "model_side_odds_bucket",
    "model_confidence_bin",
    "prob_edge_bin",
    "vegas_model_side_prob_bin",
    "spread_edge_points",
    "abs_spread_edge_points",
    "spread_edge_bin",
    "spread_edge_alignment",
    "moneyline_clv_prob_delta",
    "moneyline_positive_clv",
    "moneyline_clv_segment",
    "blend_available",
    "blend_followed",
    "blend_correct",
    "blend_side_profit",
    "blend_pick_prob",
    "prob_edge_blend_side",
]


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).ne(0)
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "1.0", "yes"})


def _bucket(value: float, bins: list[tuple[float, float, str]], *, fallback: str = "unavailable") -> str:
    if pd.isna(value):
        return fallback
    value = float(value)
    for lower, upper, label in bins:
        if value >= lower and (value < upper if not np.isinf(upper) else True):
            return label
    return fallback


def _odds_bucket(odds: float) -> str:
    if pd.isna(odds):
        return "unavailable"
    odds = float(odds)
    if odds <= -200:
        return "<=-200"
    if odds <= -150:
        return "-199 to -150"
    if odds <= -120:
        return "-149 to -120"
    if odds < 100:
        return "-119 to +99"
    if odds <= 120:
        return "+100 to +120"
    if odds <= 200:
        return "+121 to +200"
    return "+201+"


def _spread_alignment(model_home_pick: bool, spread_edge_points: float) -> str:
    if pd.isna(spread_edge_points):
        return "unavailable"
    edge = float(spread_edge_points)
    if abs(edge) < 1e-9:
        return "neutral"
    if (bool(model_home_pick) and edge > 0.0) or ((not bool(model_home_pick)) and edge < 0.0):
        return "aligns_model"
    return "opposes_model"


def _clv_segment(row: pd.Series) -> str:
    available = row.get("clv_detail_available", False)
    if pd.isna(available) or not bool(available):
        return "clv_unavailable"
    positive = row.get("moneyline_positive_clv", np.nan)
    if pd.isna(positive):
        return "clv_unavailable"
    return "positive_clv" if bool(positive) else "negative_clv"


def _mean_bool(series: pd.Series) -> float:
    if series.empty:
        return np.nan
    return float(series.astype(bool).mean())


def load_benchmark(path: Path = DEFAULT_BENCHMARK_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run analysis.market_roi first.")
    return pd.read_csv(path)


def load_blend_detail(path: Path = DEFAULT_BLEND_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def prepare_disagreement_games(benchmark: pd.DataFrame, blend_detail: pd.DataFrame | None = None) -> pd.DataFrame:
    required = {
        "season",
        "week",
        "game_id",
        "home_team",
        "away_team",
        "home_win_prob",
        "home_implied_prob_novig",
        "home_margin",
        "home_moneyline",
        "away_moneyline",
        "model_side_profit",
        "vegas_side_profit",
        "model_pick_prob",
        "vegas_model_side_prob",
        "prob_edge_model_side",
    }
    missing = sorted(required.difference(benchmark.columns))
    if missing:
        raise ValueError(f"market benchmark is missing required columns: {missing}")

    df = benchmark.copy()
    numeric_cols = [
        "season",
        "week",
        "home_win_prob",
        "home_implied_prob_novig",
        "home_margin",
        "home_moneyline",
        "away_moneyline",
        "model_side_profit",
        "vegas_side_profit",
        "model_pick_prob",
        "vegas_model_side_prob",
        "prob_edge_model_side",
        "model_side_moneyline",
        "moneyline_clv_prob_delta",
        "spread_edge_points",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["home_margin"].ne(0.0)].copy()
    df["home_win_actual"] = df["home_margin"].gt(0.0)
    df["model_home_pick"] = _as_bool(df["model_home_pick"]) if "model_home_pick" in df else df["home_win_prob"].ge(0.5)
    df["vegas_home_pick"] = (
        _as_bool(df["vegas_home_pick"]) if "vegas_home_pick" in df else df["home_implied_prob_novig"].ge(0.5)
    )
    df["favorite_agreement"] = (
        _as_bool(df["favorite_agreement"])
        if "favorite_agreement" in df
        else df["model_home_pick"].eq(df["vegas_home_pick"])
    )
    df["model_correct"] = (
        _as_bool(df["model_correct"]) if "model_correct" in df else df["model_home_pick"].eq(df["home_win_actual"])
    )
    df["vegas_correct"] = (
        _as_bool(df["vegas_correct"]) if "vegas_correct" in df else df["vegas_home_pick"].eq(df["home_win_actual"])
    )
    if "model_side_moneyline" not in df:
        df["model_side_moneyline"] = np.where(df["model_home_pick"], df["home_moneyline"], df["away_moneyline"])
    if "actual_winner" not in df:
        df["actual_winner"] = np.where(df["home_win_actual"], df["home_team"], df["away_team"])

    df = df[~df["favorite_agreement"]].copy()
    df["season"] = df["season"].astype(int)
    df["week"] = df["week"].astype(int)
    df["model_side"] = np.where(df["model_home_pick"], "home", "away")
    df["model_team"] = np.where(df["model_home_pick"], df["home_team"], df["away_team"])
    df["vegas_team"] = np.where(df["vegas_home_pick"], df["home_team"], df["away_team"])
    df["model_side_odds_bucket"] = df["model_side_moneyline"].apply(_odds_bucket)
    df["model_confidence_bin"] = df["model_pick_prob"].apply(lambda value: _bucket(value, CONFIDENCE_BINS))
    df["prob_edge_bin"] = df["prob_edge_model_side"].apply(lambda value: _bucket(value, PROB_EDGE_BINS))
    df["vegas_model_side_prob_bin"] = df["vegas_model_side_prob"].apply(
        lambda value: _bucket(value, MARKET_SIDE_PROB_BINS)
    )
    df["abs_spread_edge_points"] = df.get("spread_edge_points", pd.Series(np.nan, index=df.index)).abs()
    df["spread_edge_bin"] = df["abs_spread_edge_points"].apply(lambda value: _bucket(value, SPREAD_EDGE_BINS))
    spread_edge_series = df.get("spread_edge_points", pd.Series(np.nan, index=df.index))
    df["spread_edge_alignment"] = [
        _spread_alignment(model_home_pick, spread_edge)
        for model_home_pick, spread_edge in zip(df["model_home_pick"], spread_edge_series)
    ]
    if "clv_detail_available" in df:
        df["clv_detail_available"] = _as_bool(df["clv_detail_available"])
    else:
        df["clv_detail_available"] = False
    if "moneyline_positive_clv" in df:
        df["moneyline_positive_clv"] = _as_bool(df["moneyline_positive_clv"])
    else:
        df["moneyline_positive_clv"] = np.nan
    df["moneyline_clv_segment"] = df.apply(_clv_segment, axis=1)

    blend = pd.DataFrame() if blend_detail is None else blend_detail.copy()
    if not blend.empty and {"season", "week", "game_id"}.issubset(blend.columns):
        blend_cols = [
            "season",
            "week",
            "game_id",
            "blend_home_pick",
            "blend_correct",
            "blend_side_profit",
            "blend_pick_prob",
            "prob_edge_blend_side",
            "blend_side_moneyline",
        ]
        available_cols = [col for col in blend_cols if col in blend.columns]
        blend = blend[available_cols].copy()
        for col in ["season", "week"]:
            if col in blend:
                blend[col] = pd.to_numeric(blend[col], errors="coerce").astype("Int64")
        for col in ["blend_side_profit", "blend_pick_prob", "prob_edge_blend_side", "blend_side_moneyline"]:
            if col in blend:
                blend[col] = pd.to_numeric(blend[col], errors="coerce")
        if "blend_home_pick" in blend:
            blend["blend_home_pick"] = _as_bool(blend["blend_home_pick"])
        if "blend_correct" in blend:
            blend["blend_correct"] = _as_bool(blend["blend_correct"])
        df = df.merge(blend, on=["season", "week", "game_id"], how="left")

    df["blend_available"] = df.get("blend_home_pick", pd.Series(np.nan, index=df.index)).notna()
    if "blend_home_pick" in df:
        df["blend_followed"] = np.select(
            [
                df["blend_home_pick"].eq(df["model_home_pick"]),
                df["blend_home_pick"].eq(df["vegas_home_pick"]),
            ],
            ["model", "vegas"],
            default="other",
        )
        df.loc[~df["blend_available"], "blend_followed"] = "unavailable"
    else:
        df["blend_followed"] = "unavailable"

    for col in GAME_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[GAME_COLUMNS].copy()


def _summary_record(*, scope: str, season: int | str, segment_family: str, segment: str, group: pd.DataFrame) -> dict:
    n_games = int(len(group))
    model_roi = float(group["model_side_profit"].mean()) if n_games else np.nan
    vegas_roi = float(group["vegas_side_profit"].mean()) if n_games else np.nan
    model_accuracy = _mean_bool(group["model_correct"]) if n_games else np.nan
    vegas_accuracy = _mean_bool(group["vegas_correct"]) if n_games else np.nan
    clv = group[group["moneyline_clv_segment"].isin(["positive_clv", "negative_clv"])]
    blend = group[group["blend_available"].fillna(False).astype(bool)]
    record = {
        "scope": scope,
        "season": season,
        "segment_family": segment_family,
        "segment": segment,
        "n_games": n_games,
        "model_wins": int(group["model_correct"].sum()) if n_games else 0,
        "vegas_wins": int(group["vegas_correct"].sum()) if n_games else 0,
        "model_accuracy": model_accuracy,
        "vegas_accuracy": vegas_accuracy,
        "model_accuracy_delta_vs_vegas": model_accuracy - vegas_accuracy,
        "model_roi": model_roi,
        "vegas_roi": vegas_roi,
        "model_roi_delta_vs_vegas": model_roi - vegas_roi,
        "avg_model_side_moneyline": float(group["model_side_moneyline"].mean()) if n_games else np.nan,
        "avg_model_pick_prob": float(group["model_pick_prob"].mean()) if n_games else np.nan,
        "avg_vegas_model_side_prob": float(group["vegas_model_side_prob"].mean()) if n_games else np.nan,
        "avg_prob_edge_model_side": float(group["prob_edge_model_side"].mean()) if n_games else np.nan,
        "avg_abs_spread_edge_points": float(group["abs_spread_edge_points"].mean()) if n_games else np.nan,
        "clv_tracked_games": int(len(clv)),
        "clv_avg_prob_delta": float(clv["moneyline_clv_prob_delta"].mean()) if not clv.empty else np.nan,
        "clv_positive_rate": _mean_bool(clv["moneyline_positive_clv"]) if not clv.empty else np.nan,
        "blend_n_games": int(len(blend)),
        "blend_accuracy": _mean_bool(blend["blend_correct"]) if not blend.empty else np.nan,
        "blend_roi": float(blend["blend_side_profit"].mean()) if not blend.empty else np.nan,
        "blend_roi_delta_vs_model": (
            float(blend["blend_side_profit"].mean() - blend["model_side_profit"].mean()) if not blend.empty else np.nan
        ),
        "blend_roi_delta_vs_vegas": (
            float(blend["blend_side_profit"].mean() - blend["vegas_side_profit"].mean()) if not blend.empty else np.nan
        ),
        "blend_follow_model_rate": float(blend["blend_followed"].eq("model").mean()) if not blend.empty else np.nan,
        "blend_follow_vegas_rate": float(blend["blend_followed"].eq("vegas").mean()) if not blend.empty else np.nan,
    }
    return record


def summarize_disagreement_diagnostics(games: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    if games.empty:
        return pd.DataFrame.from_records(records)

    def add(scope: str, season: int | str, family: str, segment: str, group: pd.DataFrame) -> None:
        records.append(_summary_record(scope=scope, season=season, segment_family=family, segment=segment, group=group))

    add("overall", "all", "all", "all", games)
    for season, season_group in games.groupby("season", sort=True):
        add("season", int(season), "all", "all", season_group)

    for family, col in SEGMENT_COLUMNS.items():
        if col not in games.columns:
            continue
        for segment, group in games.groupby(col, dropna=False, sort=True):
            label = "unavailable" if pd.isna(segment) else str(segment)
            add("overall", "all", family, label, group)
        for season, season_group in games.groupby("season", sort=True):
            for segment, group in season_group.groupby(col, dropna=False, sort=True):
                label = "unavailable" if pd.isna(segment) else str(segment)
                add("season", int(season), family, label, group)

    return pd.DataFrame.from_records(records)


def build_disagreement_recommendations(
    diagnostics: pd.DataFrame,
    *,
    min_games: int = 25,
    min_season_games: int = 5,
    min_qualified_seasons: int = 2,
    min_positive_season_rate: float = 0.55,
    min_model_roi: float = 0.0,
    min_model_roi_delta_vs_vegas: float = 0.0,
    generated_at: str | None = None,
) -> pd.DataFrame:
    generated_at = generated_at or datetime.now(timezone.utc).isoformat()
    if diagnostics.empty:
        return pd.DataFrame()

    overall = diagnostics[
        diagnostics["scope"].eq("overall")
        & pd.to_numeric(diagnostics["n_games"], errors="coerce").ge(int(min_games))
    ].copy()
    records: list[dict] = []
    for _, row in overall.iterrows():
        family = str(row["segment_family"])
        segment = str(row["segment"])
        if family in {"blend_followed", "moneyline_clv_segment"}:
            continue
        season_rows = diagnostics[
            diagnostics["scope"].eq("season")
            & diagnostics["segment_family"].eq(family)
            & diagnostics["segment"].eq(segment)
            & pd.to_numeric(diagnostics["n_games"], errors="coerce").ge(int(min_season_games))
        ].copy()
        qualified = int(len(season_rows))
        positive_model_rate = (
            float(season_rows["model_roi"].gt(float(min_model_roi)).mean()) if qualified else np.nan
        )
        positive_delta_rate = (
            float(season_rows["model_roi_delta_vs_vegas"].gt(float(min_model_roi_delta_vs_vegas)).mean())
            if qualified
            else np.nan
        )
        model_candidate = bool(
            qualified >= int(min_qualified_seasons)
            and positive_model_rate >= float(min_positive_season_rate)
            and positive_delta_rate >= float(min_positive_season_rate)
            and float(row["model_roi"]) > float(min_model_roi)
            and float(row["model_roi_delta_vs_vegas"]) > float(min_model_roi_delta_vs_vegas)
        )
        blend_candidate = bool(
            pd.notna(row.get("blend_roi"))
            and int(row.get("blend_n_games", 0)) >= int(min_games)
            and float(row["blend_roi"]) > 0.0
            and float(row["blend_roi_delta_vs_model"]) > 0.0
            and float(row["blend_roi_delta_vs_vegas"]) > 0.0
        )
        avoid_model = bool(float(row["model_roi"]) < 0.0 and float(row["model_roi_delta_vs_vegas"]) < 0.0)
        if model_candidate:
            action = "candidate_model_side_segment"
            tier = "research"
            score = float(row["model_roi"]) + float(row["model_roi_delta_vs_vegas"])
        elif blend_candidate:
            action = "candidate_market_blend_segment"
            tier = "research"
            score = float(row["blend_roi"]) + float(row["blend_roi_delta_vs_model"])
        elif avoid_model:
            action = "avoid_model_side_segment"
            tier = "block"
            score = abs(float(row["model_roi"])) + abs(float(row["model_roi_delta_vs_vegas"]))
        else:
            action = "monitor"
            tier = "monitor"
            score = float(row["model_roi_delta_vs_vegas"]) if pd.notna(row["model_roi_delta_vs_vegas"]) else 0.0

        records.append(
            {
                "action": action,
                "recommendation_tier": tier,
                "segment_family": family,
                "segment": segment,
                "n_games": int(row["n_games"]),
                "model_accuracy": row["model_accuracy"],
                "vegas_accuracy": row["vegas_accuracy"],
                "model_accuracy_delta_vs_vegas": row["model_accuracy_delta_vs_vegas"],
                "model_roi": row["model_roi"],
                "vegas_roi": row["vegas_roi"],
                "model_roi_delta_vs_vegas": row["model_roi_delta_vs_vegas"],
                "blend_n_games": int(row.get("blend_n_games", 0)),
                "blend_roi": row.get("blend_roi", np.nan),
                "blend_roi_delta_vs_model": row.get("blend_roi_delta_vs_model", np.nan),
                "qualified_seasons": qualified,
                "positive_model_roi_season_rate": positive_model_rate,
                "positive_model_vs_vegas_delta_season_rate": positive_delta_rate,
                "min_games": int(min_games),
                "min_season_games": int(min_season_games),
                "min_qualified_seasons": int(min_qualified_seasons),
                "rank_score": score,
                "source_artifact": "analysis/market_disagreement_diagnostics.csv",
                "generated_at": generated_at,
            }
        )

    out = pd.DataFrame.from_records(records)
    if out.empty:
        return out
    priority = {
        "candidate_model_side_segment": 0,
        "candidate_market_blend_segment": 1,
        "avoid_model_side_segment": 2,
        "monitor": 3,
    }
    out["_priority"] = out["action"].map(priority).fillna(9)
    out = out.sort_values(["_priority", "rank_score", "n_games"], ascending=[True, False, False]).drop(
        columns=["_priority"]
    )
    return out.reset_index(drop=True)


def run(args: argparse.Namespace) -> None:
    benchmark = load_benchmark(args.benchmark)
    blend = load_blend_detail(args.blend_detail) if args.blend_detail else pd.DataFrame()
    games = prepare_disagreement_games(benchmark, blend)
    diagnostics = summarize_disagreement_diagnostics(games)
    recommendations = build_disagreement_recommendations(
        diagnostics,
        min_games=args.min_games,
        min_season_games=args.min_season_games,
        min_qualified_seasons=args.min_qualified_seasons,
        min_positive_season_rate=args.min_positive_season_rate,
    )

    args.games_output.parent.mkdir(parents=True, exist_ok=True)
    games.to_csv(args.games_output, index=False)
    diagnostics.to_csv(args.diagnostics_output, index=False)
    recommendations.to_csv(args.recommendations_output, index=False)

    print("=== Favorite-disagreement diagnostics ===")
    if diagnostics.empty:
        print("No favorite-disagreement rows were available.")
    else:
        display = diagnostics[
            diagnostics["scope"].eq("overall")
            & diagnostics["segment_family"].isin(["all", "model_side", "prob_edge_bin", "blend_followed"])
        ]
        cols = [
            "segment_family",
            "segment",
            "n_games",
            "model_accuracy",
            "vegas_accuracy",
            "model_roi",
            "vegas_roi",
            "model_roi_delta_vs_vegas",
            "blend_roi",
        ]
        print(display[cols].to_string(index=False))

    print("\n=== Favorite-disagreement research recommendations ===")
    if recommendations.empty:
        print("No recommendation rows met the minimum sample filter.")
    else:
        cols = [
            "action",
            "segment_family",
            "segment",
            "n_games",
            "model_roi",
            "vegas_roi",
            "model_roi_delta_vs_vegas",
            "qualified_seasons",
        ]
        print(recommendations.head(20)[cols].to_string(index=False))

    print(f"\nDisagreement game detail written to {args.games_output.resolve()}")
    print(f"Disagreement diagnostics written to {args.diagnostics_output.resolve()}")
    print(f"Disagreement recommendations written to {args.recommendations_output.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze model-vs-Vegas favorite-disagreement segments.")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK_PATH)
    parser.add_argument("--blend-detail", type=Path, default=DEFAULT_BLEND_PATH)
    parser.add_argument("--games-output", type=Path, default=DEFAULT_GAMES_OUTPUT)
    parser.add_argument("--diagnostics-output", type=Path, default=DEFAULT_DIAGNOSTICS_OUTPUT)
    parser.add_argument("--recommendations-output", type=Path, default=DEFAULT_RECOMMENDATIONS_OUTPUT)
    parser.add_argument("--min-games", type=int, default=25)
    parser.add_argument("--min-season-games", type=int, default=5)
    parser.add_argument("--min-qualified-seasons", type=int, default=2)
    parser.add_argument("--min-positive-season-rate", type=float, default=0.55)
    parser.add_argument("--debug", action="store_true", help="Accepted for pipeline consistency.")
    return parser


def main() -> None:
    parser = build_parser()
    run(parser.parse_args())


if __name__ == "__main__":
    main()
