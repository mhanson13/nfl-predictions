# Copyright (c) 2025 Matt Hanson
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Walk-forward confidence recalibration strategy backtest.

This module intentionally does not alter live predictions. It compares the
current team win probability against market and empirical confidence-shrink
alternatives using only prior seasons for each held-out season.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


ANALYSIS_DIR = Path("analysis")
DEFAULT_BENCHMARK_PATH = ANALYSIS_DIR / "market_benchmark.csv"
DEFAULT_DETAIL_OUTPUT = ANALYSIS_DIR / "market_confidence_recalibration_detail.csv"
DEFAULT_SUMMARY_OUTPUT = ANALYSIS_DIR / "market_confidence_recalibration_backtest.csv"
DEFAULT_DECISION_OUTPUT = ANALYSIS_DIR / "market_confidence_recalibration_decision.csv"

CONFIDENCE_BINS = [
    (0.50, 0.55, "50-55%"),
    (0.55, 0.60, "55-60%"),
    (0.60, 0.65, "60-65%"),
    (0.65, 0.70, "65-70%"),
    (0.70, np.inf, "70%+"),
]

STRATEGY_LABELS = {
    "current_model": "Current model probability",
    "vegas_novig": "No-vig Vegas probability",
    "vegas_blend_50_50": "50/50 model and no-vig Vegas blend",
    "confidence_shrink": "Prior empirical confidence-bin shrinkage",
    "disagreement_shrink": "Prior empirical disagreement-only shrinkage",
}

DETAIL_COLUMNS = [
    "season",
    "week",
    "game_id",
    "home_team",
    "away_team",
    "strategy",
    "strategy_label",
    "fold_train_start_season",
    "fold_train_end_season",
    "fold_train_games",
    "home_win_actual",
    "home_win_prob",
    "strategy_home_win_prob",
    "home_implied_prob_novig",
    "model_home_pick",
    "vegas_home_pick",
    "strategy_home_pick",
    "favorite_agreement",
    "strategy_vegas_agreement",
    "strategy_model_agreement",
    "model_confidence_bin",
    "calibration_lookup_segment",
    "calibration_lookup_bin",
    "calibration_source",
    "calibration_sample_size",
    "calibration_actual_rate",
    "strategy_correct",
    "strategy_pick_prob",
    "strategy_side_moneyline",
    "strategy_side_profit",
    "strategy_side_actual_win",
    "model_side_profit",
    "vegas_side_profit",
    "prob_edge_model_side",
    "pred_home_margin",
    "market_home_margin",
    "spread_edge_points",
]

SUMMARY_COLUMNS = [
    "scope",
    "season",
    "segment",
    "strategy",
    "strategy_label",
    "n_games",
    "accuracy",
    "brier",
    "log_loss",
    "auc",
    "selected_side_roi",
    "avg_home_win_prob",
    "avg_selected_side_prob",
    "actual_selected_side_win_rate",
    "selected_side_calibration_error",
    "vegas_agreement_rate",
    "model_agreement_rate",
    "avg_model_side_profit",
    "avg_vegas_side_profit",
]

DECISION_COLUMNS = [
    "candidate_strategy",
    "candidate_label",
    "decision",
    "recommended_layer",
    "overall_n_games",
    "overall_brier_delta_vs_current",
    "overall_log_loss_delta_vs_current",
    "overall_auc_delta_vs_current",
    "overall_accuracy_delta_vs_current",
    "overall_roi_delta_vs_current",
    "disagreement_n_games",
    "disagreement_brier_delta_vs_current",
    "disagreement_log_loss_delta_vs_current",
    "disagreement_auc_delta_vs_current",
    "disagreement_accuracy_delta_vs_current",
    "disagreement_roi_delta_vs_current",
    "passes_overall_probability_gate",
    "passes_disagreement_policy_gate",
    "source_artifact",
    "generated_at",
]


def _safe_auc(y_true: pd.Series, y_prob: pd.Series) -> float:
    if y_true.nunique(dropna=True) < 2:
        return np.nan
    return float(roc_auc_score(y_true.astype(int), y_prob.astype(float)))


def _safe_log_loss(y_true: pd.Series, y_prob: pd.Series) -> float:
    clipped = np.clip(y_prob.astype(float), 1e-6, 1.0 - 1e-6)
    return float(log_loss(y_true.astype(int), clipped, labels=[0, 1]))


def _confidence_bin(value: float) -> str:
    if pd.isna(value):
        return "unavailable"
    value = float(value)
    for lower, upper, label in CONFIDENCE_BINS:
        upper_match = value < upper if not np.isinf(upper) else value >= lower
        if value >= lower and upper_match:
            return label
    return "unavailable"


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes"})


def _selected_side_profit(
    home_pick: pd.Series,
    home_win_actual: pd.Series,
    home_moneyline: pd.Series,
    away_moneyline: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    side_moneyline = pd.Series(np.where(home_pick, home_moneyline, away_moneyline), index=home_pick.index, dtype=float)
    side_actual_win = pd.Series(np.where(home_pick, home_win_actual, 1 - home_win_actual), index=home_pick.index)
    decimal = pd.Series(
        np.where(side_moneyline > 0.0, 1.0 + side_moneyline / 100.0, 1.0 + 100.0 / side_moneyline.abs()),
        index=home_pick.index,
        dtype=float,
    )
    profit = pd.Series(np.where(side_actual_win.astype(int).eq(1), decimal - 1.0, -1.0), index=home_pick.index)
    return side_moneyline, side_actual_win.astype(int), profit.astype(float)


def load_benchmark(path: Path = DEFAULT_BENCHMARK_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run analysis.market_roi first.")
    return pd.read_csv(path)


def prepare_dataset(benchmark: pd.DataFrame) -> pd.DataFrame:
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
        "model_side_profit",
        "vegas_side_profit",
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
        "away_implied_prob_novig",
        "home_moneyline",
        "away_moneyline",
        "home_margin",
        "model_side_profit",
        "vegas_side_profit",
        "prob_edge_model_side",
        "pred_home_margin",
        "market_home_margin",
        "spread_edge_points",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=[
            "season",
            "week",
            "home_win_prob",
            "home_implied_prob_novig",
            "away_implied_prob_novig",
            "home_moneyline",
            "away_moneyline",
            "home_margin",
        ]
    ).copy()
    df = df[df["home_margin"].ne(0.0)].copy()
    df["season"] = df["season"].astype(int)
    df["week"] = df["week"].astype(int)
    df["home_win_actual"] = (df["home_margin"] > 0.0).astype(int)
    df["model_home_pick"] = df["home_win_prob"].ge(0.5)
    df["vegas_home_pick"] = df["home_implied_prob_novig"].ge(0.5)
    if "favorite_agreement" in df.columns:
        df["favorite_agreement"] = _as_bool(df["favorite_agreement"])
    else:
        df["favorite_agreement"] = df["model_home_pick"].eq(df["vegas_home_pick"])
    df["model_pick_prob"] = np.where(df["model_home_pick"], df["home_win_prob"], 1.0 - df["home_win_prob"])
    df["model_confidence_bin"] = df["model_pick_prob"].apply(_confidence_bin)
    df["agreement_segment"] = np.where(df["favorite_agreement"], "agree", "disagree")
    df["model_side_actual_win"] = np.where(
        df["model_home_pick"],
        df["home_win_actual"],
        1 - df["home_win_actual"],
    ).astype(int)
    return df


def _build_empirical_lookup(train: pd.DataFrame, *, min_bin_games: int) -> tuple[dict[tuple[str, str], dict[str, object]], dict[str, dict[str, object]], dict[str, object]]:
    segment_lookup: dict[tuple[str, str], dict[str, object]] = {}
    bin_lookup: dict[str, dict[str, object]] = {}

    grouped = train.groupby(["agreement_segment", "model_confidence_bin"], dropna=False)
    for (segment, confidence_bin), group in grouped:
        n_games = int(len(group))
        if n_games >= int(min_bin_games):
            segment_lookup[(str(segment), str(confidence_bin))] = {
                "actual_rate": float(group["model_side_actual_win"].mean()),
                "sample_size": n_games,
            }

    for confidence_bin, group in train.groupby("model_confidence_bin", dropna=False):
        n_games = int(len(group))
        if n_games >= int(min_bin_games):
            bin_lookup[str(confidence_bin)] = {
                "actual_rate": float(group["model_side_actual_win"].mean()),
                "sample_size": n_games,
            }

    fallback = {
        "actual_rate": float(train["model_side_actual_win"].mean()) if not train.empty else np.nan,
        "sample_size": int(len(train)),
    }
    return segment_lookup, bin_lookup, fallback


def _empirical_pick_prob(
    row: pd.Series,
    segment_lookup: dict[tuple[str, str], dict[str, object]],
    bin_lookup: dict[str, dict[str, object]],
    fallback: dict[str, object],
) -> tuple[float, str, int, float]:
    segment = str(row["agreement_segment"])
    confidence_bin = str(row["model_confidence_bin"])
    segment_key = (segment, confidence_bin)
    if segment_key in segment_lookup:
        record = segment_lookup[segment_key]
        return float(record["actual_rate"]), "segment_confidence_bin", int(record["sample_size"]), float(record["actual_rate"])
    if confidence_bin in bin_lookup:
        record = bin_lookup[confidence_bin]
        return float(record["actual_rate"]), "confidence_bin", int(record["sample_size"]), float(record["actual_rate"])
    if pd.notna(fallback.get("actual_rate", np.nan)):
        rate = float(fallback["actual_rate"])
        return rate, "overall_model_side_rate", int(fallback.get("sample_size", 0)), rate
    return float(row["model_pick_prob"]), "current_model_fallback", 0, np.nan


def _pick_prob_to_home_prob(model_home_pick: pd.Series, pick_prob: pd.Series) -> pd.Series:
    return pd.Series(np.where(model_home_pick.astype(bool), pick_prob, 1.0 - pick_prob), index=model_home_pick.index)


def _score_strategy_frame(base: pd.DataFrame, *, strategy: str, home_prob: pd.Series) -> pd.DataFrame:
    out = base.copy()
    out["strategy"] = strategy
    out["strategy_label"] = STRATEGY_LABELS.get(strategy, strategy)
    out["strategy_home_win_prob"] = pd.Series(home_prob, index=out.index, dtype=float).clip(1e-6, 1.0 - 1e-6)
    out["strategy_home_pick"] = out["strategy_home_win_prob"].ge(0.5)
    out["strategy_correct"] = out["strategy_home_pick"].eq(out["home_win_actual"].astype(bool))
    out["strategy_vegas_agreement"] = out["strategy_home_pick"].eq(out["vegas_home_pick"])
    out["strategy_model_agreement"] = out["strategy_home_pick"].eq(out["model_home_pick"])
    out["strategy_pick_prob"] = np.where(
        out["strategy_home_pick"],
        out["strategy_home_win_prob"],
        1.0 - out["strategy_home_win_prob"],
    )
    side_moneyline, side_actual_win, side_profit = _selected_side_profit(
        out["strategy_home_pick"],
        out["home_win_actual"],
        out["home_moneyline"],
        out["away_moneyline"],
    )
    out["strategy_side_moneyline"] = side_moneyline
    out["strategy_side_actual_win"] = side_actual_win
    out["strategy_side_profit"] = side_profit
    for col in DETAIL_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out[DETAIL_COLUMNS].copy()


def walk_forward_recalibration_backtest(
    dataset: pd.DataFrame,
    *,
    min_train_seasons: int = 3,
    min_bin_games: int = 25,
    vegas_blend_weight: float = 0.5,
) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    seasons = sorted(dataset["season"].dropna().astype(int).unique())
    for season in seasons:
        train = dataset[dataset["season"] < season].copy()
        test = dataset[dataset["season"] == season].copy()
        train_seasons = sorted(train["season"].dropna().astype(int).unique())
        if len(train_seasons) < int(min_train_seasons) or test.empty:
            continue

        segment_lookup, bin_lookup, fallback = _build_empirical_lookup(train, min_bin_games=min_bin_games)
        scored = test.copy()
        scored["fold_train_start_season"] = int(min(train_seasons))
        scored["fold_train_end_season"] = int(max(train_seasons))
        scored["fold_train_games"] = int(len(train))

        empirical_records = scored.apply(
            lambda row: _empirical_pick_prob(row, segment_lookup, bin_lookup, fallback),
            axis=1,
            result_type="expand",
        )
        empirical_records.columns = [
            "empirical_pick_prob",
            "empirical_calibration_source",
            "empirical_calibration_sample_size",
            "empirical_actual_rate",
        ]
        scored = pd.concat([scored, empirical_records], axis=1)
        empirical_home_prob = _pick_prob_to_home_prob(scored["model_home_pick"], scored["empirical_pick_prob"])

        base_meta = scored.copy()
        base_meta["calibration_lookup_segment"] = scored["agreement_segment"]
        base_meta["calibration_lookup_bin"] = scored["model_confidence_bin"]
        base_meta["calibration_source"] = "current_model"
        base_meta["calibration_sample_size"] = 0
        base_meta["calibration_actual_rate"] = np.nan
        records.append(_score_strategy_frame(base_meta, strategy="current_model", home_prob=scored["home_win_prob"]))

        vegas_meta = base_meta.copy()
        vegas_meta["calibration_source"] = "no_vig_market"
        records.append(
            _score_strategy_frame(vegas_meta, strategy="vegas_novig", home_prob=scored["home_implied_prob_novig"])
        )

        blend_weight = float(np.clip(vegas_blend_weight, 0.0, 1.0))
        blend_meta = base_meta.copy()
        blend_meta["calibration_source"] = f"fixed_vegas_blend_{blend_weight:.2f}"
        blend_home_prob = ((1.0 - blend_weight) * scored["home_win_prob"]) + (
            blend_weight * scored["home_implied_prob_novig"]
        )
        records.append(_score_strategy_frame(blend_meta, strategy="vegas_blend_50_50", home_prob=blend_home_prob))

        confidence_meta = scored.copy()
        confidence_meta["calibration_lookup_segment"] = scored["agreement_segment"]
        confidence_meta["calibration_lookup_bin"] = scored["model_confidence_bin"]
        confidence_meta["calibration_source"] = scored["empirical_calibration_source"]
        confidence_meta["calibration_sample_size"] = scored["empirical_calibration_sample_size"].astype(int)
        confidence_meta["calibration_actual_rate"] = scored["empirical_actual_rate"]
        records.append(
            _score_strategy_frame(confidence_meta, strategy="confidence_shrink", home_prob=empirical_home_prob)
        )

        disagreement_home_prob = scored["home_win_prob"].copy()
        disagreement_home_prob.loc[~scored["favorite_agreement"].astype(bool)] = empirical_home_prob.loc[
            ~scored["favorite_agreement"].astype(bool)
        ]
        disagreement_meta = confidence_meta.copy()
        disagreement_meta.loc[disagreement_meta["favorite_agreement"].astype(bool), "calibration_source"] = (
            "current_model_agreement_row"
        )
        disagreement_meta.loc[disagreement_meta["favorite_agreement"].astype(bool), "calibration_sample_size"] = 0
        disagreement_meta.loc[disagreement_meta["favorite_agreement"].astype(bool), "calibration_actual_rate"] = np.nan
        records.append(
            _score_strategy_frame(
                disagreement_meta,
                strategy="disagreement_shrink",
                home_prob=disagreement_home_prob,
            )
        )

    if not records:
        return pd.DataFrame(columns=DETAIL_COLUMNS)
    return pd.concat(records, ignore_index=True)


def _summary_record(*, scope: str, season: int | str, segment: str, strategy: str, selection: pd.DataFrame) -> dict[str, object]:
    n_games = int(len(selection))
    if n_games == 0:
        return {
            "scope": scope,
            "season": season,
            "segment": segment,
            "strategy": strategy,
            "strategy_label": STRATEGY_LABELS.get(strategy, strategy),
            "n_games": 0,
        }

    y = selection["home_win_actual"].astype(int)
    prob = selection["strategy_home_win_prob"].astype(float)
    actual_selected_rate = float(selection["strategy_side_actual_win"].mean())
    avg_selected_prob = float(selection["strategy_pick_prob"].mean())
    return {
        "scope": scope,
        "season": season,
        "segment": segment,
        "strategy": strategy,
        "strategy_label": STRATEGY_LABELS.get(strategy, strategy),
        "n_games": n_games,
        "accuracy": float(selection["strategy_correct"].mean()),
        "brier": float(brier_score_loss(y, prob)),
        "log_loss": _safe_log_loss(y, prob),
        "auc": _safe_auc(y, prob),
        "selected_side_roi": float(selection["strategy_side_profit"].sum() / n_games),
        "avg_home_win_prob": float(prob.mean()),
        "avg_selected_side_prob": avg_selected_prob,
        "actual_selected_side_win_rate": actual_selected_rate,
        "selected_side_calibration_error": avg_selected_prob - actual_selected_rate,
        "vegas_agreement_rate": float(selection["strategy_vegas_agreement"].mean()),
        "model_agreement_rate": float(selection["strategy_model_agreement"].mean()),
        "avg_model_side_profit": float(selection["model_side_profit"].mean()),
        "avg_vegas_side_profit": float(selection["vegas_side_profit"].mean()),
    }


def summarize_recalibration_backtest(detail: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    if detail.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    def add_group(scope: str, season: int | str, group: pd.DataFrame) -> None:
        segments = [
            ("all", group),
            ("model_vegas_agree", group[group["favorite_agreement"].astype(bool)]),
            ("model_vegas_disagree", group[~group["favorite_agreement"].astype(bool)]),
        ]
        for segment, segment_df in segments:
            if segment_df.empty:
                continue
            for strategy, strategy_df in segment_df.groupby("strategy", sort=False):
                records.append(
                    _summary_record(
                        scope=scope,
                        season=season,
                        segment=segment,
                        strategy=str(strategy),
                        selection=strategy_df,
                    )
                )

    add_group("overall", "all", detail)
    for season, season_df in detail.groupby("season", sort=True):
        add_group("season", int(season), season_df)

    out = pd.DataFrame.from_records(records)
    for col in SUMMARY_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out[SUMMARY_COLUMNS]


def _summary_lookup(summary: pd.DataFrame, *, segment: str, strategy: str) -> pd.Series | None:
    rows = summary[
        summary["scope"].astype(str).eq("overall")
        & summary["season"].astype(str).eq("all")
        & summary["segment"].astype(str).eq(segment)
        & summary["strategy"].astype(str).eq(strategy)
    ]
    if rows.empty:
        return None
    return rows.iloc[0]


def _delta(candidate: pd.Series | None, baseline: pd.Series | None, key: str) -> float:
    if candidate is None or baseline is None:
        return np.nan
    cand = pd.to_numeric(pd.Series([candidate.get(key, np.nan)]), errors="coerce").iloc[0]
    base = pd.to_numeric(pd.Series([baseline.get(key, np.nan)]), errors="coerce").iloc[0]
    if pd.isna(cand) or pd.isna(base):
        return np.nan
    return float(cand - base)


def build_recalibration_decision(
    summary: pd.DataFrame,
    *,
    generated_at: str | None = None,
    min_games: int = 250,
    min_disagreement_games: int = 50,
) -> pd.DataFrame:
    generated_at = generated_at or datetime.now(timezone.utc).isoformat()
    records: list[dict[str, object]] = []
    baseline_overall = _summary_lookup(summary, segment="all", strategy="current_model")
    baseline_disagree = _summary_lookup(summary, segment="model_vegas_disagree", strategy="current_model")

    for strategy in ["vegas_novig", "vegas_blend_50_50", "confidence_shrink", "disagreement_shrink"]:
        overall = _summary_lookup(summary, segment="all", strategy=strategy)
        disagree = _summary_lookup(summary, segment="model_vegas_disagree", strategy=strategy)

        overall_n = int(overall.get("n_games", 0)) if overall is not None and pd.notna(overall.get("n_games")) else 0
        disagree_n = (
            int(disagree.get("n_games", 0)) if disagree is not None and pd.notna(disagree.get("n_games")) else 0
        )

        overall_brier_delta = _delta(overall, baseline_overall, "brier")
        overall_log_delta = _delta(overall, baseline_overall, "log_loss")
        overall_auc_delta = _delta(overall, baseline_overall, "auc")
        overall_acc_delta = _delta(overall, baseline_overall, "accuracy")
        overall_roi_delta = _delta(overall, baseline_overall, "selected_side_roi")
        disagree_brier_delta = _delta(disagree, baseline_disagree, "brier")
        disagree_log_delta = _delta(disagree, baseline_disagree, "log_loss")
        disagree_auc_delta = _delta(disagree, baseline_disagree, "auc")
        disagree_acc_delta = _delta(disagree, baseline_disagree, "accuracy")
        disagree_roi_delta = _delta(disagree, baseline_disagree, "selected_side_roi")

        probability_lift = bool(
            overall_n >= int(min_games)
            and pd.notna(overall_brier_delta)
            and pd.notna(overall_log_delta)
            and overall_brier_delta < 0.0
            and overall_log_delta < 0.0
        )
        accuracy_safe = bool(pd.isna(overall_acc_delta) or overall_acc_delta >= -0.005)
        disagreement_roi_safe = bool(pd.isna(disagree_roi_delta) or disagree_roi_delta >= 0.0)
        passes_overall_probability_gate = bool(probability_lift and accuracy_safe and disagreement_roi_safe)
        passes_disagreement_policy_gate = bool(
            disagree_n >= int(min_disagreement_games)
            and pd.notna(disagree_brier_delta)
            and pd.notna(disagree_log_delta)
            and pd.notna(disagree_roi_delta)
            and disagree_brier_delta < 0.0
            and disagree_log_delta < 0.0
            and disagree_roi_delta > 0.0
        )

        market_reference_strategy = strategy == "vegas_novig"
        market_blend_strategy = strategy == "vegas_blend_50_50"
        shrink_strategy = strategy in {"confidence_shrink", "disagreement_shrink"}

        if market_reference_strategy and probability_lift:
            decision = "market_reference_beats_model"
            recommended_layer = "benchmark_only"
        elif market_blend_strategy and passes_overall_probability_gate and passes_disagreement_policy_gate:
            decision = "market_blend_probability_and_policy_candidate"
            recommended_layer = "market_blend"
        elif market_blend_strategy and probability_lift:
            decision = "market_blend_probability_research_only"
            recommended_layer = "research"
        elif shrink_strategy and passes_overall_probability_gate and passes_disagreement_policy_gate:
            decision = "base_and_market_candidate"
            recommended_layer = "both"
        elif shrink_strategy and passes_overall_probability_gate:
            decision = "base_probability_candidate"
            recommended_layer = "base_model"
        elif shrink_strategy and probability_lift:
            decision = "base_probability_research_only"
            recommended_layer = "research"
        elif passes_disagreement_policy_gate:
            decision = "market_policy_only_candidate"
            recommended_layer = "market_policy"
        else:
            decision = "no_live_change"
            recommended_layer = "none"

        records.append(
            {
                "candidate_strategy": strategy,
                "candidate_label": STRATEGY_LABELS.get(strategy, strategy),
                "decision": decision,
                "recommended_layer": recommended_layer,
                "overall_n_games": overall_n,
                "overall_brier_delta_vs_current": overall_brier_delta,
                "overall_log_loss_delta_vs_current": overall_log_delta,
                "overall_auc_delta_vs_current": overall_auc_delta,
                "overall_accuracy_delta_vs_current": overall_acc_delta,
                "overall_roi_delta_vs_current": overall_roi_delta,
                "disagreement_n_games": disagree_n,
                "disagreement_brier_delta_vs_current": disagree_brier_delta,
                "disagreement_log_loss_delta_vs_current": disagree_log_delta,
                "disagreement_auc_delta_vs_current": disagree_auc_delta,
                "disagreement_accuracy_delta_vs_current": disagree_acc_delta,
                "disagreement_roi_delta_vs_current": disagree_roi_delta,
                "passes_overall_probability_gate": passes_overall_probability_gate,
                "passes_disagreement_policy_gate": passes_disagreement_policy_gate,
                "source_artifact": str(DEFAULT_SUMMARY_OUTPUT),
                "generated_at": generated_at,
            }
        )

    out = pd.DataFrame.from_records(records)
    for col in DECISION_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out[DECISION_COLUMNS]


def run(args: argparse.Namespace) -> None:
    benchmark = load_benchmark(args.benchmark)
    dataset = prepare_dataset(benchmark)
    detail = walk_forward_recalibration_backtest(
        dataset,
        min_train_seasons=args.min_train_seasons,
        min_bin_games=args.min_bin_games,
        vegas_blend_weight=args.vegas_blend_weight,
    )
    summary = summarize_recalibration_backtest(detail)
    decision = build_recalibration_decision(
        summary,
        min_games=args.decision_min_games,
        min_disagreement_games=args.decision_min_disagreement_games,
    )

    args.detail_output.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.detail_output, index=False)
    summary.to_csv(args.summary_output, index=False)
    decision.to_csv(args.decision_output, index=False)

    print("=== Market confidence recalibration backtest ===")
    if summary.empty:
        print("No confidence recalibration strategy rows were scored.")
    else:
        display_cols = [
            "scope",
            "season",
            "segment",
            "strategy",
            "n_games",
            "accuracy",
            "brier",
            "log_loss",
            "auc",
            "selected_side_roi",
            "selected_side_calibration_error",
        ]
        display = summary[
            summary["scope"].eq("overall")
            & summary["segment"].isin(["all", "model_vegas_disagree"])
        ].copy()
        print(display[display_cols].to_string(index=False))

    print("\n=== Recalibration decision ===")
    if decision.empty:
        print("No recalibration decision rows were generated.")
    else:
        display_cols = [
            "candidate_strategy",
            "decision",
            "recommended_layer",
            "overall_brier_delta_vs_current",
            "overall_log_loss_delta_vs_current",
            "disagreement_brier_delta_vs_current",
            "disagreement_log_loss_delta_vs_current",
            "disagreement_roi_delta_vs_current",
        ]
        print(decision[display_cols].to_string(index=False))

    print(f"\nConfidence recalibration detail written to {args.detail_output.resolve()}")
    print(f"Confidence recalibration summary written to {args.summary_output.resolve()}")
    print(f"Confidence recalibration decision written to {args.decision_output.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backtest model-confidence recalibration strategies.")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK_PATH)
    parser.add_argument("--detail-output", type=Path, default=DEFAULT_DETAIL_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--decision-output", type=Path, default=DEFAULT_DECISION_OUTPUT)
    parser.add_argument("--min-train-seasons", type=int, default=3)
    parser.add_argument("--min-bin-games", type=int, default=25)
    parser.add_argument("--vegas-blend-weight", type=float, default=0.5)
    parser.add_argument("--decision-min-games", type=int, default=250)
    parser.add_argument("--decision-min-disagreement-games", type=int, default=50)
    parser.add_argument("--debug", action="store_true", help="Accepted for pipeline consistency.")
    return parser


def main() -> None:
    parser = build_parser()
    run(parser.parse_args())


if __name__ == "__main__":
    main()
