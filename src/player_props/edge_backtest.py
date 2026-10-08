from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.utils.io import PROC_DIR, read_df, write_df


DEFAULT_LINES_PATH = PROC_DIR / "player_prop_historical_line_eval.parquet"
DEFAULT_PROBABILITIES_PATH = PROC_DIR / "player_prop_over_probability_eval.parquet"
DEFAULT_DETAIL_PATH = Path("predictions/evaluation/player_props/edge_backtest_detail.csv")
DEFAULT_SUMMARY_PATH = Path("predictions/evaluation/player_props/edge_backtest_summary.csv")
DEFAULT_MARKET_THRESHOLDS_PATH = Path("predictions/evaluation/player_props/edge_backtest_market_thresholds.csv")
DEFAULT_POLICY_PATH = Path("predictions/evaluation/player_props/edge_recommendation_policy.csv")

DEFAULT_EV_THRESHOLDS = (0.0, 0.02, 0.05, 0.10)
DEFAULT_PROB_EDGE_THRESHOLDS = (0.0, 0.02, 0.05, 0.10)
DEFAULT_POLICY_MIN_BETS = 100
DEFAULT_POLICY_MIN_ROI = 0.0
DEFAULT_POLICY_MIN_THRESHOLD = 0.02

BACKTEST_COLUMNS = [
    "selection_scope",
    "strategy",
    "threshold",
    "season",
    "week",
    "game_id",
    "market",
    "sportsbook",
    "player_id",
    "model_player_name",
    "line_player_name",
    "team",
    "opponent",
    "position",
    "line",
    "bet_side",
    "odds",
    "break_even_probability",
    "model_probability",
    "market_probability",
    "prob_edge",
    "edge_ev",
    "selection_metric",
    "model_projection",
    "actual_value",
    "actual_over_line",
    "actual_under_line",
    "actual_push",
    "bet_result",
    "stake",
    "profit",
    "roi",
    "prob_over_model",
    "prob_over_market",
    "prob_edge_vs_market",
    "calibration_method",
    "calibration_sample_size",
    "line_is_comparable",
    "line_updated_at",
    "match_method",
    "generated_at",
]

SUMMARY_COLUMNS = [
    "selection_scope",
    "strategy",
    "threshold",
    "scope",
    "market",
    "sportsbook",
    "season",
    "n_bets",
    "wins",
    "losses",
    "pushes",
    "win_rate",
    "avg_odds",
    "avg_model_probability",
    "avg_market_probability",
    "avg_prob_edge",
    "avg_edge_ev",
    "avg_selection_metric",
    "total_staked",
    "total_profit",
    "roi",
    "avg_profit_per_bet",
    "generated_at",
]

POLICY_COLUMNS = [
    "selection_scope",
    "strategy",
    "scope",
    "market",
    "sportsbook",
    "threshold",
    "n_bets",
    "wins",
    "losses",
    "pushes",
    "win_rate",
    "avg_odds",
    "avg_model_probability",
    "avg_market_probability",
    "avg_prob_edge",
    "avg_edge_ev",
    "roi",
    "avg_profit_per_bet",
    "recommendation_tier",
    "policy_min_bets",
    "policy_min_roi",
    "policy_min_threshold",
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


def _american_profit_per_unit(values: pd.Series | np.ndarray) -> np.ndarray:
    odds = np.asarray(pd.to_numeric(values, errors="coerce"), dtype=float)
    profit = np.full_like(odds, np.nan, dtype=float)
    positive = odds > 0
    negative = odds < 0
    profit[positive] = odds[positive] / 100.0
    profit[negative] = 100.0 / np.abs(odds[negative])
    return profit


def _american_break_even(values: pd.Series | np.ndarray) -> np.ndarray:
    odds = np.asarray(pd.to_numeric(values, errors="coerce"), dtype=float)
    probability = np.full_like(odds, np.nan, dtype=float)
    positive = odds > 0
    negative = odds < 0
    probability[positive] = 100.0 / (odds[positive] + 100.0)
    probability[negative] = np.abs(odds[negative]) / (np.abs(odds[negative]) + 100.0)
    return probability


def _as_bool(series: pd.Series, default: bool = False) -> pd.Series:
    return series.astype("boolean").fillna(default).astype(bool)


def _normalize_join_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in ["season", "week"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    if "line" in out.columns:
        out["line"] = pd.to_numeric(out["line"], errors="coerce")
        out["_line_key"] = out["line"].round(3)
    for col in ["game_id", "market", "sportsbook", "player_id"]:
        if col in out.columns:
            out[col] = out[col].astype("string").str.strip()
    if "sportsbook" in out.columns:
        out["sportsbook"] = out["sportsbook"].str.lower()
    return out


def _merge_probability_columns(lines: pd.DataFrame, probabilities: pd.DataFrame) -> pd.DataFrame:
    if lines.empty or probabilities.empty:
        return pd.DataFrame()

    required_lines = {"season", "week", "game_id", "market", "sportsbook", "player_id", "line"}
    required_probabilities = required_lines | {"prob_over_model", "prob_over_market"}
    if missing := required_lines - set(lines.columns):
        raise ValueError(f"Historical lines missing required columns: {sorted(missing)}")
    if missing := required_probabilities - set(probabilities.columns):
        raise ValueError(f"Over-probability rows missing required columns: {sorted(missing)}")

    lines_norm = _normalize_join_keys(lines)
    probabilities_norm = _normalize_join_keys(probabilities)
    join_cols = ["season", "week", "game_id", "market", "sportsbook", "player_id", "_line_key"]
    probability_cols = [
        *join_cols,
        "prob_over_model",
        "prob_over_model_raw",
        "prob_over_residual",
        "prob_over_market",
        "prob_edge_vs_market",
        "market_blend_weight",
        "residual_brier_prior",
        "market_brier_prior",
        "calibration_method",
        "calibration_sample_size",
    ]
    probability_cols = [col for col in probability_cols if col in probabilities_norm.columns]
    probability_lookup = (
        probabilities_norm[probability_cols]
        .sort_values(join_cols, kind="stable")
        .drop_duplicates(join_cols, keep="last")
    )
    joined = lines_norm.merge(probability_lookup, on=join_cols, how="inner", suffixes=("", "_probability"))
    return joined


def prepare_backtest_frame(lines: pd.DataFrame, probabilities: pd.DataFrame) -> pd.DataFrame:
    joined = _merge_probability_columns(lines, probabilities)
    if joined.empty:
        return pd.DataFrame()

    numeric_cols = [
        "line",
        "over_odds",
        "under_odds",
        "model_projection",
        "actual_value",
        "prob_over_model",
        "prob_over_market",
        "prob_edge_vs_market",
        "calibration_sample_size",
    ]
    for col in numeric_cols:
        if col in joined.columns:
            joined[col] = pd.to_numeric(joined[col], errors="coerce")

    if "actual_push" not in joined.columns:
        joined["actual_push"] = joined["actual_value"].eq(joined["line"])
    joined["actual_push"] = _as_bool(joined["actual_push"])

    if "actual_over_line" not in joined.columns:
        joined["actual_over_line"] = joined["actual_value"].gt(joined["line"])
    joined["actual_over_line"] = _as_bool(joined["actual_over_line"])

    if "actual_under_line" not in joined.columns:
        joined["actual_under_line"] = joined["actual_value"].lt(joined["line"])
    joined["actual_under_line"] = _as_bool(joined["actual_under_line"])

    joined = joined[
        joined["prob_over_model"].notna()
        & joined["prob_over_market"].notna()
        & joined["actual_value"].notna()
        & ~joined["actual_push"]
    ].copy()
    if joined.empty:
        return pd.DataFrame()

    joined["prob_under_model"] = 1.0 - joined["prob_over_model"]
    joined["prob_under_market"] = 1.0 - joined["prob_over_market"]
    return joined.reset_index(drop=True)


def _side_candidates(frame: pd.DataFrame, *, generated_at: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=BACKTEST_COLUMNS)

    candidates: list[pd.DataFrame] = []
    side_specs = [
        ("over", "over_odds", "actual_over_line", "prob_over_model", "prob_over_market"),
        ("under", "under_odds", "actual_under_line", "prob_under_model", "prob_under_market"),
    ]
    base_cols = [
        "season",
        "week",
        "game_id",
        "market",
        "sportsbook",
        "player_id",
        "model_player_name",
        "line_player_name",
        "team",
        "opponent",
        "position",
        "line",
        "model_projection",
        "actual_value",
        "actual_over_line",
        "actual_under_line",
        "actual_push",
        "prob_over_model",
        "prob_over_market",
        "prob_edge_vs_market",
        "calibration_method",
        "calibration_sample_size",
        "line_is_comparable",
        "line_updated_at",
        "match_method",
    ]
    base_cols = [col for col in base_cols if col in frame.columns]

    for side, odds_col, actual_col, model_prob_col, market_prob_col in side_specs:
        if odds_col not in frame.columns:
            continue
        side_frame = frame[base_cols].copy()
        side_frame["bet_side"] = side
        side_frame["odds"] = pd.to_numeric(frame[odds_col], errors="coerce")
        side_frame["model_probability"] = pd.to_numeric(frame[model_prob_col], errors="coerce")
        side_frame["market_probability"] = pd.to_numeric(frame[market_prob_col], errors="coerce")
        side_frame["prob_edge"] = side_frame["model_probability"] - side_frame["market_probability"]
        win_profit = _american_profit_per_unit(side_frame["odds"])
        side_frame["break_even_probability"] = _american_break_even(side_frame["odds"])
        side_frame["edge_ev"] = (side_frame["model_probability"] * win_profit) - (
            1.0 - side_frame["model_probability"]
        )

        won = _as_bool(frame[actual_col])
        push = _as_bool(frame["actual_push"])
        side_frame["bet_result"] = np.where(push, "push", np.where(won, "win", "loss"))
        side_frame["stake"] = 1.0
        side_frame["profit"] = np.where(push, 0.0, np.where(won, win_profit, -1.0))
        side_frame["roi"] = side_frame["profit"] / side_frame["stake"]
        side_frame["generated_at"] = generated_at
        candidates.append(side_frame)

    if not candidates:
        return pd.DataFrame(columns=BACKTEST_COLUMNS)

    out = pd.concat(candidates, ignore_index=True, sort=False)
    out = out[out["odds"].notna() & out["model_probability"].notna() & out["market_probability"].notna()].copy()
    for col in BACKTEST_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out[BACKTEST_COLUMNS]


def _threshold_candidates(
    candidates: pd.DataFrame,
    *,
    ev_thresholds: Iterable[float],
    prob_edge_thresholds: Iterable[float],
) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(columns=BACKTEST_COLUMNS)

    selected: list[pd.DataFrame] = []
    threshold_specs = [
        ("ev", "edge_ev", ev_thresholds),
        ("prob_edge", "prob_edge", prob_edge_thresholds),
    ]
    for strategy, metric_col, thresholds in threshold_specs:
        metric = pd.to_numeric(candidates[metric_col], errors="coerce")
        for threshold in thresholds:
            threshold_value = float(threshold)
            pick = candidates[metric.gt(threshold_value)].copy()
            if pick.empty:
                continue
            pick["strategy"] = strategy
            pick["threshold"] = threshold_value
            pick["selection_metric"] = pd.to_numeric(pick[metric_col], errors="coerce")
            selected.append(pick)

    if not selected:
        return pd.DataFrame(columns=BACKTEST_COLUMNS)
    return pd.concat(selected, ignore_index=True, sort=False)


def _best_price_rows(selected: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame(columns=BACKTEST_COLUMNS)

    group_cols = ["strategy", "threshold", "season", "week", "game_id", "market"]
    if "player_id" in selected.columns:
        group_cols.append("player_id")
    elif "model_player_name" in selected.columns:
        group_cols.append("model_player_name")

    sort_cols = [*group_cols, "selection_metric", "edge_ev", "prob_edge"]
    ascending = [True] * len(group_cols) + [False, False, False]
    best = (
        selected.sort_values(sort_cols, ascending=ascending, kind="stable")
        .drop_duplicates(group_cols, keep="first")
        .copy()
    )
    best["selection_scope"] = "best_price"
    return best


def build_edge_backtest(
    lines: pd.DataFrame,
    probabilities: pd.DataFrame,
    *,
    ev_thresholds: Iterable[float] = DEFAULT_EV_THRESHOLDS,
    prob_edge_thresholds: Iterable[float] = DEFAULT_PROB_EDGE_THRESHOLDS,
    generated_at: str | None = None,
) -> pd.DataFrame:
    generated_at = generated_at or datetime.now(timezone.utc).isoformat()
    prepared = prepare_backtest_frame(lines, probabilities)
    if prepared.empty:
        return pd.DataFrame(columns=BACKTEST_COLUMNS)

    candidates = _side_candidates(prepared, generated_at=generated_at)
    selected = _threshold_candidates(
        candidates,
        ev_thresholds=ev_thresholds,
        prob_edge_thresholds=prob_edge_thresholds,
    )
    if selected.empty:
        return pd.DataFrame(columns=BACKTEST_COLUMNS)

    per_book = selected.copy()
    per_book["selection_scope"] = "per_book"
    best_price = _best_price_rows(per_book)
    out = pd.concat([per_book, best_price], ignore_index=True, sort=False)
    for col in BACKTEST_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out[BACKTEST_COLUMNS].sort_values(
        ["selection_scope", "strategy", "threshold", "season", "week", "game_id", "market", "sportsbook"],
        kind="stable",
    )


def _summary_record(
    group: pd.DataFrame,
    *,
    scope: str,
    market: object = pd.NA,
    sportsbook: object = pd.NA,
    season: object = pd.NA,
    generated_at: str,
) -> dict[str, object]:
    n_bets = int(len(group))
    wins = int(group["bet_result"].eq("win").sum())
    losses = int(group["bet_result"].eq("loss").sum())
    pushes = int(group["bet_result"].eq("push").sum())
    graded = wins + losses
    total_staked = float(pd.to_numeric(group["stake"], errors="coerce").sum())
    total_profit = float(pd.to_numeric(group["profit"], errors="coerce").sum())
    return {
        "selection_scope": group["selection_scope"].iloc[0],
        "strategy": group["strategy"].iloc[0],
        "threshold": float(group["threshold"].iloc[0]),
        "scope": scope,
        "market": market,
        "sportsbook": sportsbook,
        "season": season,
        "n_bets": n_bets,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate": float(wins / graded) if graded else float("nan"),
        "avg_odds": float(pd.to_numeric(group["odds"], errors="coerce").mean()),
        "avg_model_probability": float(pd.to_numeric(group["model_probability"], errors="coerce").mean()),
        "avg_market_probability": float(pd.to_numeric(group["market_probability"], errors="coerce").mean()),
        "avg_prob_edge": float(pd.to_numeric(group["prob_edge"], errors="coerce").mean()),
        "avg_edge_ev": float(pd.to_numeric(group["edge_ev"], errors="coerce").mean()),
        "avg_selection_metric": float(pd.to_numeric(group["selection_metric"], errors="coerce").mean()),
        "total_staked": total_staked,
        "total_profit": total_profit,
        "roi": float(total_profit / total_staked) if total_staked else float("nan"),
        "avg_profit_per_bet": float(total_profit / n_bets) if n_bets else float("nan"),
        "generated_at": generated_at,
    }


def summarize_edge_backtest(detail: pd.DataFrame, *, generated_at: str | None = None) -> pd.DataFrame:
    generated_at = generated_at or datetime.now(timezone.utc).isoformat()
    if detail.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    rows: list[dict[str, object]] = []
    base_cols = ["selection_scope", "strategy", "threshold"]
    for keys, group in detail.groupby(base_cols, sort=True):
        rows.append(_summary_record(group, scope="all", generated_at=generated_at))

    group_specs = [
        ("market", [*base_cols, "market"]),
        ("sportsbook", [*base_cols, "sportsbook"]),
        ("market_sportsbook", [*base_cols, "market", "sportsbook"]),
        ("season", [*base_cols, "season"]),
        ("market_season", [*base_cols, "market", "season"]),
    ]
    for scope, group_cols in group_specs:
        for keys, group in detail.groupby(group_cols, sort=True):
            key_map = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
            rows.append(
                _summary_record(
                    group,
                    scope=scope,
                    market=key_map.get("market", pd.NA),
                    sportsbook=key_map.get("sportsbook", pd.NA),
                    season=key_map.get("season", pd.NA),
                    generated_at=generated_at,
                )
            )

    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def build_market_threshold_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    out = summary[summary["scope"].isin(["all", "market", "market_sportsbook"])].copy()
    return out.sort_values(["selection_scope", "strategy", "scope", "market", "sportsbook", "threshold"], kind="stable")


def build_edge_recommendation_policy(
    summary: pd.DataFrame,
    *,
    min_bets: int = DEFAULT_POLICY_MIN_BETS,
    min_roi: float = DEFAULT_POLICY_MIN_ROI,
    min_threshold: float = DEFAULT_POLICY_MIN_THRESHOLD,
) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame(columns=POLICY_COLUMNS)

    out = summary[
        summary["selection_scope"].eq("best_price")
        & summary["strategy"].eq("prob_edge")
        & summary["scope"].isin(["market_sportsbook", "market"])
    ].copy()
    if out.empty:
        return pd.DataFrame(columns=POLICY_COLUMNS)

    out["n_bets"] = pd.to_numeric(out["n_bets"], errors="coerce")
    out["roi"] = pd.to_numeric(out["roi"], errors="coerce")
    out["threshold"] = pd.to_numeric(out["threshold"], errors="coerce")
    out = out[
        out["market"].notna()
        & out["n_bets"].ge(int(min_bets))
        & out["roi"].gt(float(min_roi))
        & out["threshold"].ge(float(min_threshold))
    ].copy()
    if out.empty:
        return pd.DataFrame(columns=POLICY_COLUMNS)

    out["recommendation_tier"] = np.where(out["roi"].ge(0.05), "strong", "lean")
    out["policy_min_bets"] = int(min_bets)
    out["policy_min_roi"] = float(min_roi)
    out["policy_min_threshold"] = float(min_threshold)
    out["_scope_rank"] = np.where(out["scope"].eq("market_sportsbook"), 0, 1)
    out = out.sort_values(
        ["_scope_rank", "scope", "market", "sportsbook", "threshold", "roi", "n_bets"],
        ascending=[True, True, True, True, False, False, False],
        kind="stable",
    )
    out = out.drop_duplicates(["scope", "market", "sportsbook"], keep="first")
    for col in POLICY_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out[POLICY_COLUMNS].sort_values(["scope", "market", "sportsbook"], kind="stable").reset_index(drop=True)


def _float_sequence(values: Sequence[str] | None, default: Sequence[float]) -> list[float]:
    if not values:
        return list(default)
    return [float(value) for value in values]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest player-prop over/under betting edges from historical lines and calibrated probabilities.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--lines", type=Path, default=DEFAULT_LINES_PATH)
    parser.add_argument("--probabilities", type=Path, default=DEFAULT_PROBABILITIES_PATH)
    parser.add_argument("--detail-output", type=Path, default=DEFAULT_DETAIL_PATH)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--market-threshold-output", type=Path, default=DEFAULT_MARKET_THRESHOLDS_PATH)
    parser.add_argument("--policy-output", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--ev-thresholds", nargs="*", default=None)
    parser.add_argument("--prob-edge-thresholds", nargs="*", default=None)
    parser.add_argument("--policy-min-bets", type=int, default=DEFAULT_POLICY_MIN_BETS)
    parser.add_argument("--policy-min-roi", type=float, default=DEFAULT_POLICY_MIN_ROI)
    parser.add_argument("--policy-min-threshold", type=float, default=DEFAULT_POLICY_MIN_THRESHOLD)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    generated_at = datetime.now(timezone.utc).isoformat()
    lines = _read_table(args.lines)
    probabilities = _read_table(args.probabilities)
    detail = build_edge_backtest(
        lines,
        probabilities,
        ev_thresholds=_float_sequence(args.ev_thresholds, DEFAULT_EV_THRESHOLDS),
        prob_edge_thresholds=_float_sequence(args.prob_edge_thresholds, DEFAULT_PROB_EDGE_THRESHOLDS),
        generated_at=generated_at,
    )
    summary = summarize_edge_backtest(detail, generated_at=generated_at)
    market_thresholds = build_market_threshold_summary(summary)
    policy = build_edge_recommendation_policy(
        summary,
        min_bets=args.policy_min_bets,
        min_roi=args.policy_min_roi,
        min_threshold=args.policy_min_threshold,
    )

    detail_path = write_df(detail, args.detail_output)
    summary_path = write_df(summary, args.summary_output)
    market_threshold_path = write_df(market_thresholds, args.market_threshold_output)
    policy_path = write_df(policy, args.policy_output)

    if args.debug:
        display = summary[summary["scope"].eq("all")].copy()
        if not display.empty:
            print("[player_props.edge_backtest] overall summary:")
            print(
                display[
                    [
                        "selection_scope",
                        "strategy",
                        "threshold",
                        "n_bets",
                        "win_rate",
                        "total_profit",
                        "roi",
                        "avg_edge_ev",
                    ]
                ].to_string(index=False)
            )
        if not policy.empty:
            print("[player_props.edge_backtest] recommendation policy:")
            print(policy[["scope", "market", "sportsbook", "threshold", "n_bets", "roi", "recommendation_tier"]].to_string(index=False))
    print(f"[player_props.edge_backtest] wrote detail rows={len(detail)} -> {detail_path.resolve()}")
    print(f"[player_props.edge_backtest] wrote summary rows={len(summary)} -> {summary_path.resolve()}")
    print(f"[player_props.edge_backtest] wrote market thresholds rows={len(market_thresholds)} -> {market_threshold_path.resolve()}")
    print(f"[player_props.edge_backtest] wrote recommendation policy rows={len(policy)} -> {policy_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
