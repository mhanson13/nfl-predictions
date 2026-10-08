# Copyright (c) 2025 Matt Hanson
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Walk-forward market-aware blend backtest.

The blend intentionally uses only prior seasons for each held-out season. It
compares raw model probabilities, no-vig Vegas probabilities, and a simple
logistic meta-model trained on model/market disagreement features.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ANALYSIS_DIR = Path("analysis")
DEFAULT_BENCHMARK_PATH = ANALYSIS_DIR / "market_benchmark.csv"
DEFAULT_DETAIL_OUTPUT = ANALYSIS_DIR / "market_blend_backtest.csv"
DEFAULT_SUMMARY_OUTPUT = ANALYSIS_DIR / "market_blend_summary.csv"
DEFAULT_POLICY_OUTPUT = ANALYSIS_DIR / "market_blend_policy.csv"

FEATURE_COLUMNS = [
    "home_win_prob",
    "home_implied_prob_novig",
    "model_market_home_prob_edge",
    "abs_model_market_home_prob_edge",
    "model_home_confidence",
    "vegas_home_confidence",
    "pred_home_margin",
    "market_home_margin",
    "spread_edge_points",
    "abs_spread_edge_points",
    "favorite_agreement_flag",
]

DETAIL_COLUMNS = [
    "season",
    "week",
    "game_id",
    "home_team",
    "away_team",
    "fold_train_start_season",
    "fold_train_end_season",
    "fold_train_games",
    "home_win_actual",
    "home_win_prob",
    "home_implied_prob_novig",
    "blend_home_win_prob",
    "model_home_pick",
    "vegas_home_pick",
    "blend_home_pick",
    "model_correct",
    "vegas_correct",
    "blend_correct",
    "favorite_agreement",
    "blend_vegas_agreement",
    "blend_model_agreement",
    "model_side_profit",
    "vegas_side_profit",
    "blend_side_profit",
    "model_pick_prob",
    "vegas_model_side_prob",
    "blend_pick_prob",
    "vegas_blend_side_prob",
    "prob_edge_model_side",
    "prob_edge_blend_side",
    "model_side_moneyline",
    "vegas_side_moneyline",
    "blend_side_moneyline",
    "pred_home_margin",
    "market_home_margin",
    "spread_edge_points",
]

INTERNAL_COLUMNS = [
    "away_implied_prob_novig",
    "home_moneyline",
    "away_moneyline",
    "home_decimal",
    "away_decimal",
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


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes"})


def _safe_auc(y_true: pd.Series, y_prob: pd.Series) -> float:
    if y_true.nunique(dropna=True) < 2:
        return np.nan
    return float(roc_auc_score(y_true.astype(int), y_prob.astype(float)))


def _safe_log_loss(y_true: pd.Series, y_prob: pd.Series) -> float:
    clipped = np.clip(y_prob.astype(float), 1e-6, 1.0 - 1e-6)
    return float(log_loss(y_true.astype(int), clipped, labels=[0, 1]))


def load_benchmark(path: Path = DEFAULT_BENCHMARK_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run analysis.market_roi first.")
    return pd.read_csv(path)


def prepare_blend_dataset(benchmark: pd.DataFrame) -> pd.DataFrame:
    required = {
        "season",
        "week",
        "game_id",
        "home_team",
        "away_team",
        "home_win_prob",
        "home_implied_prob_novig",
        "away_implied_prob_novig",
        "home_moneyline",
        "away_moneyline",
        "home_margin",
        "pred_home_margin",
        "market_home_margin",
        "spread_edge_points",
        "model_side_profit",
        "vegas_side_profit",
        "model_side_moneyline",
        "vegas_side_moneyline",
        "model_pick_prob",
        "vegas_model_side_prob",
        "prob_edge_model_side",
    }
    missing = sorted(required.difference(benchmark.columns))
    if missing:
        raise ValueError(f"market benchmark is missing required columns: {missing}")

    df = benchmark.copy()
    numeric_columns = [
        "season",
        "week",
        "home_win_prob",
        "home_implied_prob_novig",
        "away_implied_prob_novig",
        "home_moneyline",
        "away_moneyline",
        "home_margin",
        "pred_home_margin",
        "market_home_margin",
        "spread_edge_points",
        "model_side_profit",
        "vegas_side_profit",
        "model_side_moneyline",
        "vegas_side_moneyline",
        "model_pick_prob",
        "vegas_model_side_prob",
        "prob_edge_model_side",
    ]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["home_margin"].ne(0.0)].copy()
    df["home_win_actual"] = (df["home_margin"] > 0.0).astype(int)
    df["model_home_pick"] = df["home_win_prob"].ge(0.5)
    df["vegas_home_pick"] = df["home_implied_prob_novig"].ge(0.5)
    if "favorite_agreement" in df.columns:
        df["favorite_agreement"] = _as_bool(df["favorite_agreement"])
    else:
        df["favorite_agreement"] = df["model_home_pick"].eq(df["vegas_home_pick"])
    df["model_correct"] = df["model_home_pick"].eq(df["home_win_actual"].astype(bool))
    df["vegas_correct"] = df["vegas_home_pick"].eq(df["home_win_actual"].astype(bool))

    df["model_market_home_prob_edge"] = df["home_win_prob"] - df["home_implied_prob_novig"]
    df["abs_model_market_home_prob_edge"] = df["model_market_home_prob_edge"].abs()
    df["model_home_confidence"] = (df["home_win_prob"] - 0.5).abs()
    df["vegas_home_confidence"] = (df["home_implied_prob_novig"] - 0.5).abs()
    df["abs_spread_edge_points"] = df["spread_edge_points"].abs()
    df["favorite_agreement_flag"] = df["favorite_agreement"].astype(int)
    df["home_decimal"] = df["home_moneyline"].apply(american_to_decimal)
    df["away_decimal"] = df["away_moneyline"].apply(american_to_decimal)

    keep = [col for col in DETAIL_COLUMNS if col in df.columns]
    keep.extend([col for col in FEATURE_COLUMNS if col not in keep])
    keep.extend([col for col in INTERNAL_COLUMNS if col not in keep])
    keep = list(dict.fromkeys(keep))
    df = df.dropna(subset=FEATURE_COLUMNS + ["home_win_actual", "home_decimal", "away_decimal"]).copy()
    df["season"] = df["season"].astype(int)
    df["week"] = df["week"].astype(int)
    return df[keep].copy()


def walk_forward_blend(
    dataset: pd.DataFrame,
    *,
    min_train_seasons: int = 3,
    blend_c: float = 1.0,
) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    seasons = sorted(dataset["season"].dropna().astype(int).unique())
    for season in seasons:
        train = dataset[dataset["season"] < season].copy()
        test = dataset[dataset["season"] == season].copy()
        train_seasons = sorted(train["season"].dropna().astype(int).unique())
        if len(train_seasons) < int(min_train_seasons) or test.empty:
            continue
        if train["home_win_actual"].nunique() < 2:
            continue

        model = Pipeline(
            steps=[
                ("scale", StandardScaler()),
                ("logit", LogisticRegression(C=float(blend_c), max_iter=1000, solver="lbfgs")),
            ]
        )
        model.fit(train[FEATURE_COLUMNS], train["home_win_actual"].astype(int))
        scored = test.copy()
        scored["blend_home_win_prob"] = model.predict_proba(scored[FEATURE_COLUMNS])[:, 1]
        scored["fold_train_start_season"] = int(min(train_seasons))
        scored["fold_train_end_season"] = int(max(train_seasons))
        scored["fold_train_games"] = int(len(train))
        records.append(scored)

    if not records:
        return pd.DataFrame(columns=DETAIL_COLUMNS)
    out = pd.concat(records, ignore_index=True)
    return score_blend_predictions(out)


def score_blend_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(columns=DETAIL_COLUMNS)
    df = predictions.copy()
    df["blend_home_pick"] = df["blend_home_win_prob"].ge(0.5)
    actual_home = df["home_win_actual"].astype(bool)
    df["blend_correct"] = df["blend_home_pick"].eq(actual_home)
    df["blend_vegas_agreement"] = df["blend_home_pick"].eq(df["vegas_home_pick"])
    df["blend_model_agreement"] = df["blend_home_pick"].eq(df["model_home_pick"])
    df["blend_pick_prob"] = np.where(df["blend_home_pick"], df["blend_home_win_prob"], 1.0 - df["blend_home_win_prob"])
    df["vegas_blend_side_prob"] = np.where(
        df["blend_home_pick"],
        df["home_implied_prob_novig"],
        df["away_implied_prob_novig"],
    )
    df["prob_edge_blend_side"] = df["blend_pick_prob"] - df["vegas_blend_side_prob"]
    df["blend_side_moneyline"] = np.where(df["blend_home_pick"], df["home_moneyline"], df["away_moneyline"])
    df["blend_side_decimal"] = np.where(df["blend_home_pick"], df["home_decimal"], df["away_decimal"])
    df["blend_side_actual_win"] = np.where(df["blend_home_pick"], actual_home, ~actual_home).astype(int)
    df["blend_side_profit"] = np.where(
        df["blend_side_actual_win"].eq(1),
        df["blend_side_decimal"] - 1.0,
        -1.0,
    )
    for col in DETAIL_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[DETAIL_COLUMNS].copy()


def _summary_record(
    *,
    scope: str,
    season: int | str,
    segment: str,
    selection: pd.DataFrame,
) -> dict[str, object]:
    n_games = int(len(selection))
    if n_games == 0:
        return {
            "scope": scope,
            "season": season,
            "segment": segment,
            "n_games": 0,
        }

    y = selection["home_win_actual"].astype(int)
    model_prob = selection["home_win_prob"].astype(float)
    vegas_prob = selection["home_implied_prob_novig"].astype(float)
    blend_prob = selection["blend_home_win_prob"].astype(float)
    model_brier = float(brier_score_loss(y, model_prob))
    vegas_brier = float(brier_score_loss(y, vegas_prob))
    blend_brier = float(brier_score_loss(y, blend_prob))
    model_roi = float(selection["model_side_profit"].sum() / n_games)
    vegas_roi = float(selection["vegas_side_profit"].sum() / n_games)
    blend_roi = float(selection["blend_side_profit"].sum() / n_games)

    return {
        "scope": scope,
        "season": season,
        "segment": segment,
        "n_games": n_games,
        "model_accuracy": float(selection["model_correct"].mean()),
        "vegas_accuracy": float(selection["vegas_correct"].mean()),
        "blend_accuracy": float(selection["blend_correct"].mean()),
        "model_brier": model_brier,
        "vegas_brier": vegas_brier,
        "blend_brier": blend_brier,
        "blend_brier_delta_vs_model": blend_brier - model_brier,
        "blend_brier_delta_vs_vegas": blend_brier - vegas_brier,
        "model_log_loss": _safe_log_loss(y, model_prob),
        "vegas_log_loss": _safe_log_loss(y, vegas_prob),
        "blend_log_loss": _safe_log_loss(y, blend_prob),
        "model_auc": _safe_auc(y, model_prob),
        "vegas_auc": _safe_auc(y, vegas_prob),
        "blend_auc": _safe_auc(y, blend_prob),
        "model_side_roi": model_roi,
        "vegas_side_roi": vegas_roi,
        "blend_side_roi": blend_roi,
        "blend_roi_delta_vs_model": blend_roi - model_roi,
        "blend_roi_delta_vs_vegas": blend_roi - vegas_roi,
        "model_vegas_agreement_rate": float(selection["favorite_agreement"].mean()),
        "blend_vegas_agreement_rate": float(selection["blend_vegas_agreement"].mean()),
        "blend_model_agreement_rate": float(selection["blend_model_agreement"].mean()),
        "avg_model_home_prob": float(model_prob.mean()),
        "avg_vegas_home_prob": float(vegas_prob.mean()),
        "avg_blend_home_prob": float(blend_prob.mean()),
        "avg_prob_edge_blend_side": float(selection["prob_edge_blend_side"].mean()),
    }


def summarize_blend_backtest(scored: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    if scored.empty:
        return pd.DataFrame.from_records(records)

    def add_group(scope: str, season: int | str, group: pd.DataFrame) -> None:
        segments = [
            ("all", group),
            ("model_vegas_agree", group[group["favorite_agreement"]]),
            ("model_vegas_disagree", group[~group["favorite_agreement"]]),
            ("blend_vegas_agree", group[group["blend_vegas_agreement"]]),
            ("blend_vegas_disagree", group[~group["blend_vegas_agreement"]]),
            ("blend_model_agree", group[group["blend_model_agreement"]]),
            ("blend_model_disagree", group[~group["blend_model_agreement"]]),
        ]
        for segment, selection in segments:
            if not selection.empty:
                records.append(
                    _summary_record(
                        scope=scope,
                        season=season,
                        segment=segment,
                        selection=selection,
                    )
                )

    add_group("overall", "all", scored)
    for season, season_df in scored.groupby("season", sort=True):
        add_group("season", int(season), season_df)
    return pd.DataFrame.from_records(records)


def build_blend_policy(
    summary: pd.DataFrame,
    *,
    min_games: int = 250,
    generated_at: str | None = None,
) -> pd.DataFrame:
    generated_at = generated_at or datetime.now(timezone.utc).isoformat()
    if summary.empty:
        overall = None
    else:
        rows = summary[
            summary["scope"].eq("overall")
            & summary["segment"].eq("all")
            & pd.to_numeric(summary["n_games"], errors="coerce").ge(int(min_games))
        ]
        overall = rows.iloc[0] if not rows.empty else None

    def value(key: str) -> object:
        if overall is None or key not in overall:
            return np.nan
        return overall.get(key)

    n_games = int(value("n_games")) if overall is not None and pd.notna(value("n_games")) else 0
    brier_recommended = bool(
        overall is not None
        and float(value("blend_brier_delta_vs_model")) < 0.0
        and float(value("blend_brier_delta_vs_vegas")) < 0.0
    )
    roi_recommended = bool(
        overall is not None
        and float(value("blend_side_roi")) > 0.0
        and float(value("blend_roi_delta_vs_model")) > 0.0
        and float(value("blend_roi_delta_vs_vegas")) > 0.0
    )
    records = [
        {
            "signal": "market_blend_probability",
            "recommended": brier_recommended,
            "recommendation_tier": "strong" if brier_recommended else "no_bet",
            "selection": "blend home_win_prob",
            "n_games": n_games,
            "blend_metric": value("blend_brier"),
            "model_metric": value("model_brier"),
            "vegas_metric": value("vegas_brier"),
            "delta_vs_model": value("blend_brier_delta_vs_model"),
            "delta_vs_vegas": value("blend_brier_delta_vs_vegas"),
            "policy_min_games": int(min_games),
            "source_artifact": "analysis/market_blend_summary.csv",
            "generated_at": generated_at,
        },
        {
            "signal": "market_blend_moneyline",
            "recommended": roi_recommended,
            "recommendation_tier": "strong" if roi_recommended else "no_bet",
            "selection": "blend selected side",
            "n_games": n_games,
            "blend_metric": value("blend_side_roi"),
            "model_metric": value("model_side_roi"),
            "vegas_metric": value("vegas_side_roi"),
            "delta_vs_model": value("blend_roi_delta_vs_model"),
            "delta_vs_vegas": value("blend_roi_delta_vs_vegas"),
            "policy_min_games": int(min_games),
            "source_artifact": "analysis/market_blend_summary.csv",
            "generated_at": generated_at,
        },
    ]
    return pd.DataFrame.from_records(records)


def run(args: argparse.Namespace) -> None:
    benchmark = load_benchmark(args.benchmark)
    dataset = prepare_blend_dataset(benchmark)
    scored = walk_forward_blend(
        dataset,
        min_train_seasons=args.min_train_seasons,
        blend_c=args.blend_c,
    )
    summary = summarize_blend_backtest(scored)
    policy = build_blend_policy(summary, min_games=args.policy_min_games)

    args.detail_output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.detail_output, index=False)
    summary.to_csv(args.summary_output, index=False)
    policy.to_csv(args.policy_output, index=False)

    print("=== Market blend summary ===")
    if summary.empty:
        print("No market blend rows were scored.")
    else:
        display_cols = [
            "scope",
            "season",
            "segment",
            "n_games",
            "model_brier",
            "vegas_brier",
            "blend_brier",
            "blend_brier_delta_vs_vegas",
            "model_side_roi",
            "vegas_side_roi",
            "blend_side_roi",
        ]
        display = summary[(summary["scope"] == "overall") | (summary["segment"] == "all")]
        print(display[display_cols].to_string(index=False))

    print("\n=== Market blend policy ===")
    print(policy.to_string(index=False))
    print(f"\nBlend detail written to {args.detail_output.resolve()}")
    print(f"Blend summary written to {args.summary_output.resolve()}")
    print(f"Blend policy written to {args.policy_output.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backtest a walk-forward market-aware probability blend.")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK_PATH)
    parser.add_argument("--detail-output", type=Path, default=DEFAULT_DETAIL_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--policy-output", type=Path, default=DEFAULT_POLICY_OUTPUT)
    parser.add_argument("--min-train-seasons", type=int, default=3)
    parser.add_argument("--blend-c", type=float, default=1.0)
    parser.add_argument("--policy-min-games", type=int, default=250)
    parser.add_argument("--debug", action="store_true", help="Accepted for pipeline consistency.")
    return parser


def main() -> None:
    parser = build_parser()
    run(parser.parse_args())


if __name__ == "__main__":
    main()
