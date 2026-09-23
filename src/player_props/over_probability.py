from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from src.utils.io import PROC_DIR, read_df, write_df
from src.utils.week_filter import filter_before_week


DEFAULT_INPUT_PATH = PROC_DIR / "player_prop_historical_line_eval.parquet"
DEFAULT_OUTPUT_PATH = PROC_DIR / "player_prop_over_probability_eval.parquet"
DEFAULT_METRICS_PATH = Path("predictions/evaluation/player_props/over_probability_metrics.csv")
DEFAULT_BINS_PATH = Path("predictions/evaluation/player_props/over_probability_bins.csv")

PROBABILITY_OUTPUT_COLUMNS = [
    "season",
    "week",
    "game_id",
    "oddsapi_event_id",
    "market",
    "sportsbook",
    "player_id",
    "model_player_name",
    "line_player_name",
    "team",
    "opponent",
    "position",
    "model_projection",
    "line",
    "actual_value",
    "actual_over_line",
    "implied_probability",
    "prob_over_model",
    "prob_over_model_raw",
    "prob_over_market",
    "prob_edge_vs_market",
    "calibration_sample_size",
    "calibration_method",
    "model_edge",
    "abs_model_edge",
    "model_pick",
    "model_pick_correct",
    "probability_pick",
    "probability_pick_correct",
    "model_error",
    "abs_model_error",
    "line_error",
    "abs_line_error",
    "line_is_comparable",
    "line_updated_at",
    "match_method",
]

METRIC_COLUMNS = [
    "scope",
    "market",
    "n",
    "actual_over_rate",
    "model_brier",
    "market_brier",
    "model_logloss",
    "market_logloss",
    "model_auc",
    "market_auc",
    "probability_pick_accuracy",
    "model_projection_pick_accuracy",
    "market_pick_accuracy",
    "mean_prob_over_model",
    "mean_prob_over_market",
    "mean_prob_edge_vs_market",
    "residual_cdf_rows",
    "market_implied_fallback_rows",
    "base_rate_fallback_rows",
    "mean_calibration_sample_size",
    "generated_at",
]

BIN_COLUMNS = [
    "scope",
    "market",
    "bin_low",
    "bin_high",
    "n",
    "mean_prob_over_model",
    "actual_over_rate",
    "brier",
    "generated_at",
]


def _clip_probability(values: pd.Series | np.ndarray, floor: float, cap: float) -> np.ndarray:
    arr = np.asarray(pd.to_numeric(values, errors="coerce"), dtype=float)
    return np.clip(arr, floor, cap)


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() == ".csv":
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
    return read_df(path)


def _prediction_key_columns(frame: pd.DataFrame) -> list[str]:
    preferred = ["season", "week", "game_id", "market", "player_id"]
    if all(col in frame.columns for col in preferred):
        return preferred
    fallback = ["season", "week", "game_id", "market", "model_player_name"]
    return [col for col in fallback if col in frame.columns]


def _prepare_scoring_rows(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"market", "line"}
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame()
    out = frame.copy()
    if "model_projection" not in out.columns and "projection" in out.columns:
        out["model_projection"] = out["projection"]
    if "implied_probability" not in out.columns:
        out["implied_probability"] = np.nan
    for col in ["line", "model_projection", "implied_probability"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["market"] = out["market"].astype("string").str.strip()
    out = out[out["market"].notna() & out["line"].notna() & out["model_projection"].notna()].copy()
    if not out.empty:
        out["model_edge"] = out["model_projection"] - out["line"]
    return out


def prepare_probability_input(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"season", "week", "market", "line", "model_projection", "actual_value"}
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame()

    out = frame.copy()
    for col in ["season", "week", "line", "model_projection", "actual_value", "implied_probability"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out[
        out["season"].notna()
        & out["week"].notna()
        & out["line"].notna()
        & out["model_projection"].notna()
        & out["actual_value"].notna()
    ].copy()
    if out.empty:
        return pd.DataFrame()

    out["season"] = out["season"].astype(int)
    out["week"] = out["week"].astype(int)
    out["market"] = out["market"].astype("string").str.strip()
    out["residual"] = out["actual_value"] - out["model_projection"]
    if "actual_over_line" not in out.columns:
        out["actual_over_line"] = out["actual_value"].gt(out["line"])
    else:
        out["actual_over_line"] = out["actual_over_line"].astype("boolean")
    if "actual_push" not in out.columns:
        out["actual_push"] = out["actual_value"].eq(out["line"])
    else:
        out["actual_push"] = out["actual_push"].astype("boolean").fillna(False).astype(bool)
    out = out[~out["actual_push"].fillna(False).astype(bool)].copy()
    if out.empty:
        return pd.DataFrame()
    out["actual_over_line"] = out["actual_over_line"].fillna(False).astype(bool)
    out["model_edge"] = out["model_projection"] - out["line"]
    out["abs_model_edge"] = out["model_edge"].abs()
    return out.sort_values(["season", "week", "game_id", "market"], kind="stable").reset_index(drop=True)


def _unique_residual_rows(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame(columns=["season", "week", "market", "residual"])
    key_cols = _prediction_key_columns(history)
    residuals = history.sort_values(key_cols, kind="stable").drop_duplicates(key_cols, keep="last")
    return residuals[["season", "week", "market", "residual"]].copy()


def _prior_base_rate(history: pd.DataFrame, market: str) -> float:
    prior = history[history["market"].eq(market)] if not history.empty else history
    if prior.empty:
        return 0.5
    rate = pd.to_numeric(prior["actual_over_line"], errors="coerce").mean()
    return float(rate) if np.isfinite(rate) else 0.5


def _score_rows_against_history(
    rows: pd.DataFrame,
    residual_history: pd.DataFrame,
    *,
    base_rate: float,
    min_samples: int,
    shrinkage: float,
    probability_floor: float,
    probability_cap: float,
) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()

    out = rows.copy()
    out["prob_over_market"] = _clip_probability(out["implied_probability"], probability_floor, probability_cap)
    out["prob_over_model_raw"] = np.nan
    out["prob_over_model"] = np.nan
    out["calibration_sample_size"] = int(len(residual_history))
    out["calibration_method"] = "base_rate_fallback"

    if len(residual_history) >= int(min_samples):
        residuals = pd.to_numeric(residual_history["residual"], errors="coerce").dropna().to_numpy(dtype=float)
        if residuals.size >= int(min_samples):
            thresholds = -pd.to_numeric(out["model_edge"], errors="coerce").to_numpy(dtype=float)
            raw = (residuals[:, None] > thresholds[None, :]).mean(axis=0)
            weight = residuals.size / (residuals.size + max(float(shrinkage), 0.0))
            model_prob = (weight * raw) + ((1.0 - weight) * float(base_rate))
            out["prob_over_model_raw"] = np.clip(raw, probability_floor, probability_cap)
            out["prob_over_model"] = np.clip(model_prob, probability_floor, probability_cap)
            out["calibration_method"] = "residual_cdf"
            return out

    implied = pd.to_numeric(out["implied_probability"], errors="coerce")
    implied_valid = implied.notna()
    out.loc[implied_valid, "prob_over_model"] = np.clip(
        implied.loc[implied_valid].astype(float),
        probability_floor,
        probability_cap,
    )
    out.loc[implied_valid, "calibration_method"] = "market_implied_fallback"
    out.loc[~implied_valid, "prob_over_model"] = np.clip(float(base_rate), probability_floor, probability_cap)
    return out


def add_walk_forward_over_probabilities(
    frame: pd.DataFrame,
    *,
    min_samples: int = 100,
    shrinkage: float = 50.0,
    probability_floor: float = 0.02,
    probability_cap: float = 0.98,
) -> pd.DataFrame:
    prepared = prepare_probability_input(frame)
    if prepared.empty:
        return pd.DataFrame(columns=PROBABILITY_OUTPUT_COLUMNS)

    scored_groups: list[pd.DataFrame] = []
    for market, market_rows in prepared.groupby("market", sort=True):
        market_rows = market_rows.sort_values(["season", "week", "game_id"], kind="stable").copy()
        for (season, week), scoring_rows in market_rows.groupby(["season", "week"], sort=True):
            prior_rows = market_rows[
                (market_rows["season"] < int(season))
                | ((market_rows["season"] == int(season)) & (market_rows["week"] < int(week)))
            ].copy()
            residuals = _unique_residual_rows(prior_rows)
            scored = _score_rows_against_history(
                scoring_rows,
                residuals,
                base_rate=_prior_base_rate(prior_rows, str(market)),
                min_samples=min_samples,
                shrinkage=shrinkage,
                probability_floor=probability_floor,
                probability_cap=probability_cap,
            )
            scored_groups.append(scored)

    if not scored_groups:
        return pd.DataFrame(columns=PROBABILITY_OUTPUT_COLUMNS)

    out = pd.concat(scored_groups, ignore_index=True, sort=False)
    out["prob_edge_vs_market"] = out["prob_over_model"] - out["prob_over_market"]
    out["prob_over"] = out["prob_over_model"]
    out["prob_edge"] = out["prob_edge_vs_market"]
    out["probability_pick"] = np.where(out["prob_over_model"].ge(0.5), "over", "under")
    out["probability_pick_correct"] = np.where(
        out["probability_pick"].eq("over"),
        out["actual_over_line"],
        ~out["actual_over_line"],
    )
    for col in PROBABILITY_OUTPUT_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out[PROBABILITY_OUTPUT_COLUMNS].sort_values(
        ["season", "week", "game_id", "market", "sportsbook"],
        kind="stable",
    )


def attach_current_over_probabilities(
    predictions: pd.DataFrame,
    historical_lines: pd.DataFrame,
    *,
    season: int | None = None,
    week: int | None = None,
    min_samples: int = 100,
    shrinkage: float = 50.0,
    probability_floor: float = 0.02,
    probability_cap: float = 0.98,
) -> pd.DataFrame:
    if predictions.empty or historical_lines.empty:
        return predictions

    scoring_rows = _prepare_scoring_rows(predictions)
    if scoring_rows.empty:
        return predictions

    history = prepare_probability_input(historical_lines)
    history = filter_before_week(history, season, week)
    if history.empty:
        return predictions

    out = predictions.copy()
    for col in ["prob_over_raw", "prob_edge", "prob_over_sample_size"]:
        if col not in out.columns:
            out[col] = np.nan
    if "prob_over_method" not in out.columns:
        out["prob_over_method"] = pd.Series(pd.NA, index=out.index, dtype="object")
    else:
        out["prob_over_method"] = out["prob_over_method"].astype("object")

    scored_parts: list[pd.DataFrame] = []
    for market, rows in scoring_rows.groupby("market", sort=True):
        prior_rows = history[history["market"].eq(market)].copy()
        residuals = _unique_residual_rows(prior_rows)
        scored = _score_rows_against_history(
            rows,
            residuals,
            base_rate=_prior_base_rate(prior_rows, str(market)),
            min_samples=min_samples,
            shrinkage=shrinkage,
            probability_floor=probability_floor,
            probability_cap=probability_cap,
        )
        scored_parts.append(scored)

    if not scored_parts:
        return out

    scored_all = pd.concat(scored_parts, ignore_index=False, sort=False)
    target_index = scored_all.index
    out.loc[target_index, "prob_over"] = scored_all["prob_over_model"]
    out.loc[target_index, "prob_over_raw"] = scored_all["prob_over_model_raw"]
    out.loc[target_index, "prob_edge"] = scored_all["prob_over_model"] - scored_all["prob_over_market"]
    out.loc[target_index, "prob_over_method"] = scored_all["calibration_method"]
    out.loc[target_index, "prob_over_sample_size"] = scored_all["calibration_sample_size"]
    return out


def _binary_metric_values(actual: pd.Series, prob: pd.Series) -> tuple[float, float, float]:
    valid = pd.DataFrame({"actual": actual, "prob": prob}).dropna()
    if valid.empty:
        return (float("nan"), float("nan"), float("nan"))
    y_true = valid["actual"].astype(bool).astype(int)
    y_prob = valid["prob"].clip(1e-6, 1.0 - 1e-6)
    brier = float(brier_score_loss(y_true, y_prob))
    logloss_value = float(log_loss(y_true, y_prob, labels=[0, 1]))
    auc = float(roc_auc_score(y_true, y_prob)) if y_true.nunique() == 2 else float("nan")
    return (brier, logloss_value, auc)


def _pick_accuracy(actual: pd.Series, prob: pd.Series) -> float:
    valid = pd.DataFrame({"actual": actual, "prob": prob}).dropna()
    if valid.empty:
        return float("nan")
    picks = valid["prob"].ge(0.5)
    return float(picks.eq(valid["actual"].astype(bool)).mean())


def _projection_pick_accuracy(group: pd.DataFrame) -> float:
    if "model_pick_correct" in group.columns and group["model_pick_correct"].notna().any():
        return float(group["model_pick_correct"].dropna().astype(bool).mean())
    if "model_edge" not in group.columns:
        return float("nan")
    valid = group[group["model_edge"].ne(0)].copy()
    if valid.empty:
        return float("nan")
    pick_over = valid["model_edge"].gt(0)
    return float(pick_over.eq(valid["actual_over_line"].astype(bool)).mean())


def _metrics_for_group(
    group: pd.DataFrame,
    *,
    scope: str,
    market: str | None,
    generated_at: str,
) -> dict[str, object]:
    actual = group["actual_over_line"].astype(bool)
    model_brier, model_logloss, model_auc = _binary_metric_values(actual, group["prob_over_model"])
    market_brier, market_logloss, market_auc = _binary_metric_values(actual, group["prob_over_market"])
    return {
        "scope": scope,
        "market": market if market is not None else pd.NA,
        "n": int(len(group)),
        "actual_over_rate": float(actual.mean()) if len(group) else float("nan"),
        "model_brier": model_brier,
        "market_brier": market_brier,
        "model_logloss": model_logloss,
        "market_logloss": market_logloss,
        "model_auc": model_auc,
        "market_auc": market_auc,
        "probability_pick_accuracy": _pick_accuracy(actual, group["prob_over_model"]),
        "model_projection_pick_accuracy": _projection_pick_accuracy(group),
        "market_pick_accuracy": _pick_accuracy(actual, group["prob_over_market"]),
        "mean_prob_over_model": float(group["prob_over_model"].mean()),
        "mean_prob_over_market": float(group["prob_over_market"].mean()),
        "mean_prob_edge_vs_market": float(group["prob_edge_vs_market"].mean()),
        "residual_cdf_rows": int(group["calibration_method"].eq("residual_cdf").sum()),
        "market_implied_fallback_rows": int(group["calibration_method"].eq("market_implied_fallback").sum()),
        "base_rate_fallback_rows": int(group["calibration_method"].eq("base_rate_fallback").sum()),
        "mean_calibration_sample_size": float(group["calibration_sample_size"].mean()),
        "generated_at": generated_at,
    }


def summarize_over_probability(scored: pd.DataFrame, *, generated_at: str | None = None) -> pd.DataFrame:
    generated_at = generated_at or datetime.now(timezone.utc).isoformat()
    if scored.empty:
        return pd.DataFrame(columns=METRIC_COLUMNS)
    rows = [_metrics_for_group(scored, scope="all", market=None, generated_at=generated_at)]
    for market, group in scored.groupby("market", sort=True):
        rows.append(_metrics_for_group(group, scope="market", market=str(market), generated_at=generated_at))
    return pd.DataFrame(rows, columns=METRIC_COLUMNS)


def build_calibration_bins(scored: pd.DataFrame, *, generated_at: str | None = None) -> pd.DataFrame:
    generated_at = generated_at or datetime.now(timezone.utc).isoformat()
    if scored.empty:
        return pd.DataFrame(columns=BIN_COLUMNS)

    rows: list[dict[str, object]] = []
    bin_edges = np.arange(0.0, 1.01, 0.1)
    groups: list[tuple[str, str | None, pd.DataFrame]] = [("all", None, scored)]
    groups.extend(("market", str(market), group) for market, group in scored.groupby("market", sort=True))
    for scope, market, group in groups:
        valid = group[group["prob_over_model"].notna()].copy()
        if valid.empty:
            continue
        valid["_bin"] = pd.cut(valid["prob_over_model"], bins=bin_edges, include_lowest=True, right=True)
        for interval, bin_group in valid.groupby("_bin", observed=True, sort=True):
            if bin_group.empty:
                continue
            rows.append(
                {
                    "scope": scope,
                    "market": market if market is not None else pd.NA,
                    "bin_low": float(interval.left),
                    "bin_high": float(interval.right),
                    "n": int(len(bin_group)),
                    "mean_prob_over_model": float(bin_group["prob_over_model"].mean()),
                    "actual_over_rate": float(bin_group["actual_over_line"].astype(bool).mean()),
                    "brier": float(
                        brier_score_loss(
                            bin_group["actual_over_line"].astype(bool).astype(int),
                            bin_group["prob_over_model"].clip(1e-6, 1.0 - 1e-6),
                        )
                    ),
                    "generated_at": generated_at,
                }
            )
    return pd.DataFrame(rows, columns=BIN_COLUMNS)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate and evaluate per-market player-prop over probabilities.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--bins-output", type=Path, default=DEFAULT_BINS_PATH)
    parser.add_argument("--min-samples", type=int, default=100)
    parser.add_argument("--shrinkage", type=float, default=50.0)
    parser.add_argument("--probability-floor", type=float, default=0.02)
    parser.add_argument("--probability-cap", type=float, default=0.98)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    lines = _read_table(args.input)
    scored = add_walk_forward_over_probabilities(
        lines,
        min_samples=args.min_samples,
        shrinkage=args.shrinkage,
        probability_floor=args.probability_floor,
        probability_cap=args.probability_cap,
    )
    generated_at = datetime.now(timezone.utc).isoformat()
    metrics = summarize_over_probability(scored, generated_at=generated_at)
    bins = build_calibration_bins(scored, generated_at=generated_at)
    output_path = write_df(scored, args.output)
    metrics_path = write_df(metrics, args.metrics_output)
    bins_path = write_df(bins, args.bins_output)
    if args.debug:
        print("[player_props.over_probability] metrics:")
        print(metrics.to_string(index=False))
    print(f"[player_props.over_probability] wrote scored rows={len(scored)} -> {output_path.resolve()}")
    print(f"[player_props.over_probability] wrote metrics rows={len(metrics)} -> {metrics_path.resolve()}")
    print(f"[player_props.over_probability] wrote calibration bins rows={len(bins)} -> {bins_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
