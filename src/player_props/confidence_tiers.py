from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, mean_squared_error, roc_auc_score

from src.player_props.features import (
    DEFAULT_DEFENSE_OUTPUT_PATH,
    DEFAULT_NOVELTY_OUTPUT_PATH,
    DEFAULT_OFFENSE_OUTPUT_PATH,
)
from src.player_props.historical_lines import DEFAULT_OUTPUT_PATH as DEFAULT_LINE_EVAL_PATH
from src.player_props.train import DEFAULT_OUTPUT_DIR, MODEL_PREDICTIONS_NAME
from src.utils.io import read_df


DEFAULT_MODEL_PREDICTIONS_PATH = DEFAULT_OUTPUT_DIR / MODEL_PREDICTIONS_NAME
DEFAULT_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "confidence_tier_metrics.csv"

CONFIDENCE_TIER_ORDER = {"low": 0, "medium": 1, "high": 2}
KEY_COLUMNS = ["season", "week", "game_id", "team", "player_id", "market"]
METRIC_COLUMNS = [
    "evaluation_scope",
    "scope",
    "market",
    "confidence_tier",
    "n_rows",
    "avg_sample_size",
    "min_sample_size",
    "max_sample_size",
    "mae",
    "rmse",
    "brier_over_zero",
    "logloss_over_zero",
    "auc_over_zero",
    "actual_over_zero_rate",
    "prob_over_zero_mean",
    "model_pick_accuracy",
    "mean_abs_model_edge",
    "mean_abs_model_error",
    "mean_abs_line_error",
    "generated_at",
]


def _read_table(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        return read_df(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _confidence_tier(sample_size: object) -> str:
    try:
        sample = int(float(sample_size))
    except (TypeError, ValueError):
        sample = 0
    if sample >= 20:
        return "high"
    if sample >= 5:
        return "medium"
    return "low"


def _feature_sample_lookup(feature_frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for frame in feature_frames:
        if frame.empty or "player_games_prior" not in frame.columns:
            continue
        if missing := set(KEY_COLUMNS) - set(frame.columns):
            continue
        sample = frame[KEY_COLUMNS + ["player_games_prior"]].copy()
        sample["sample_size"] = pd.to_numeric(sample["player_games_prior"], errors="coerce").fillna(0).astype(int)
        frames.append(sample.drop(columns=["player_games_prior"]))
    if not frames:
        return pd.DataFrame(columns=KEY_COLUMNS + ["sample_size"])
    out = pd.concat(frames, ignore_index=True, sort=False)
    out["season"] = pd.to_numeric(out["season"], errors="coerce")
    out["week"] = pd.to_numeric(out["week"], errors="coerce")
    out = out[out["season"].notna() & out["week"].notna()]
    out["season"] = out["season"].astype(int)
    out["week"] = out["week"].astype(int)
    for col in ["game_id", "team", "player_id", "market"]:
        out[col] = out[col].astype("string").str.strip()
    out = out.sort_values(KEY_COLUMNS, kind="stable")
    return out.drop_duplicates(KEY_COLUMNS, keep="last").reset_index(drop=True)


def attach_confidence_tiers(frame: pd.DataFrame, sample_lookup: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    for col in KEY_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out["season"] = pd.to_numeric(out["season"], errors="coerce")
    out["week"] = pd.to_numeric(out["week"], errors="coerce")
    out = out[out["season"].notna() & out["week"].notna()].copy()
    if out.empty:
        return out
    out["season"] = out["season"].astype(int)
    out["week"] = out["week"].astype(int)
    for col in ["game_id", "team", "player_id", "market"]:
        out[col] = out[col].astype("string").str.strip()

    existing_sample = (
        pd.to_numeric(out["sample_size"], errors="coerce")
        if "sample_size" in out.columns
        else pd.Series(np.nan, index=out.index)
    )
    out = out.drop(columns=["sample_size"], errors="ignore")
    if not sample_lookup.empty:
        out = out.merge(sample_lookup, on=KEY_COLUMNS, how="left")
    else:
        out["sample_size"] = np.nan
    out["sample_size"] = pd.to_numeric(out["sample_size"], errors="coerce").where(
        pd.to_numeric(out["sample_size"], errors="coerce").notna(),
        existing_sample.to_numpy(),
    )
    out["sample_size"] = out["sample_size"].fillna(0).astype(int)
    out["confidence_tier"] = out["sample_size"].apply(_confidence_tier)
    return out


def _binary_metrics(group: pd.DataFrame) -> tuple[float, float, float, float, float]:
    if "actual_over_zero" not in group.columns or "prob_over_zero" not in group.columns:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)
    valid = group.copy()
    valid["actual_over_zero"] = valid["actual_over_zero"].astype("boolean")
    valid["prob_over_zero"] = pd.to_numeric(valid["prob_over_zero"], errors="coerce")
    valid = valid[valid["actual_over_zero"].notna() & valid["prob_over_zero"].notna()].copy()
    if valid.empty:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)
    y_true = valid["actual_over_zero"].astype(bool).astype(int)
    y_prob = valid["prob_over_zero"].clip(1e-6, 1 - 1e-6)
    brier = float(brier_score_loss(y_true, y_prob))
    logloss_value = float(log_loss(y_true, y_prob, labels=[0, 1]))
    auc = float(roc_auc_score(y_true, y_prob)) if y_true.nunique() == 2 else np.nan
    return (brier, logloss_value, auc, float(y_true.mean()), float(y_prob.mean()))


def _summary_row(
    group: pd.DataFrame,
    *,
    evaluation_scope: str,
    scope: str,
    market: str,
    confidence_tier: str,
    generated_at: str,
) -> dict[str, object]:
    actual = pd.to_numeric(group.get("actual_value", pd.Series(np.nan, index=group.index)), errors="coerce")
    projection = pd.to_numeric(group.get("model_projection", pd.Series(np.nan, index=group.index)), errors="coerce")
    valid_projection = actual.notna() & projection.notna()
    error = projection[valid_projection] - actual[valid_projection]
    brier, logloss_value, auc, actual_rate, prob_mean = _binary_metrics(group)

    model_pick = group.get("model_pick_correct", pd.Series(pd.NA, index=group.index)).astype("boolean")
    model_pick_accuracy = float(model_pick.dropna().astype(bool).mean()) if model_pick.notna().any() else np.nan
    sample_size = pd.to_numeric(group.get("sample_size", pd.Series(np.nan, index=group.index)), errors="coerce")
    return {
        "evaluation_scope": evaluation_scope,
        "scope": scope,
        "market": market,
        "confidence_tier": confidence_tier,
        "n_rows": int(len(group)),
        "avg_sample_size": float(sample_size.mean()) if sample_size.notna().any() else np.nan,
        "min_sample_size": float(sample_size.min()) if sample_size.notna().any() else np.nan,
        "max_sample_size": float(sample_size.max()) if sample_size.notna().any() else np.nan,
        "mae": float(error.abs().mean()) if len(error) else np.nan,
        "rmse": float(mean_squared_error(actual[valid_projection], projection[valid_projection]) ** 0.5)
        if valid_projection.any()
        else np.nan,
        "brier_over_zero": brier,
        "logloss_over_zero": logloss_value,
        "auc_over_zero": auc,
        "actual_over_zero_rate": actual_rate,
        "prob_over_zero_mean": prob_mean,
        "model_pick_accuracy": model_pick_accuracy,
        "mean_abs_model_edge": float(pd.to_numeric(group.get("abs_model_edge", pd.Series(np.nan, index=group.index)), errors="coerce").mean()),
        "mean_abs_model_error": float(pd.to_numeric(group.get("abs_model_error", pd.Series(np.nan, index=group.index)), errors="coerce").mean()),
        "mean_abs_line_error": float(pd.to_numeric(group.get("abs_line_error", pd.Series(np.nan, index=group.index)), errors="coerce").mean()),
        "generated_at": generated_at,
    }


def _summaries_for_frame(frame: pd.DataFrame, *, evaluation_scope: str, generated_at: str) -> list[dict[str, object]]:
    if frame.empty:
        return []
    rows: list[dict[str, object]] = [
        _summary_row(
            frame,
            evaluation_scope=evaluation_scope,
            scope="overall",
            market="all",
            confidence_tier="all",
            generated_at=generated_at,
        )
    ]
    for tier, group in frame.groupby("confidence_tier", dropna=False, sort=False):
        rows.append(
            _summary_row(
                group,
                evaluation_scope=evaluation_scope,
                scope="confidence_tier",
                market="all",
                confidence_tier=str(tier),
                generated_at=generated_at,
            )
        )
    for market, group in frame.groupby("market", dropna=False, sort=True):
        rows.append(
            _summary_row(
                group,
                evaluation_scope=evaluation_scope,
                scope="market",
                market=str(market),
                confidence_tier="all",
                generated_at=generated_at,
            )
        )
        for tier, tier_group in group.groupby("confidence_tier", dropna=False, sort=False):
            rows.append(
                _summary_row(
                    tier_group,
                    evaluation_scope=evaluation_scope,
                    scope="market_confidence_tier",
                    market=str(market),
                    confidence_tier=str(tier),
                    generated_at=generated_at,
                )
            )
    return rows


def build_confidence_tier_metrics(
    *,
    model_predictions: pd.DataFrame,
    feature_frames: Sequence[pd.DataFrame],
    line_eval: pd.DataFrame | None = None,
    generated_at: str | None = None,
) -> pd.DataFrame:
    generated_at = generated_at or datetime.now(timezone.utc).isoformat()
    sample_lookup = _feature_sample_lookup(feature_frames)
    rows: list[dict[str, object]] = []
    model = attach_confidence_tiers(model_predictions, sample_lookup)
    rows.extend(_summaries_for_frame(model, evaluation_scope="model_predictions", generated_at=generated_at))
    if line_eval is not None and not line_eval.empty:
        lines = attach_confidence_tiers(line_eval, sample_lookup)
        rows.extend(_summaries_for_frame(lines, evaluation_scope="historical_lines", generated_at=generated_at))
    if not rows:
        return pd.DataFrame(columns=METRIC_COLUMNS)
    out = pd.DataFrame(rows, columns=METRIC_COLUMNS)
    tier_rank = out["confidence_tier"].map(CONFIDENCE_TIER_ORDER).fillna(99)
    out = out.assign(_tier_rank=tier_rank)
    out = out.sort_values(["evaluation_scope", "scope", "market", "_tier_rank"], kind="stable")
    return out.drop(columns=["_tier_rank"]).reset_index(drop=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build player prop confidence-tier calibration summaries.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-predictions", type=Path, default=DEFAULT_MODEL_PREDICTIONS_PATH)
    parser.add_argument("--offense-features", type=Path, default=DEFAULT_OFFENSE_OUTPUT_PATH)
    parser.add_argument("--defense-features", type=Path, default=DEFAULT_DEFENSE_OUTPUT_PATH)
    parser.add_argument("--novelty-features", type=Path, default=DEFAULT_NOVELTY_OUTPUT_PATH)
    parser.add_argument("--line-eval", type=Path, default=DEFAULT_LINE_EVAL_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    model_predictions = _read_table(args.model_predictions)
    feature_frames = [_read_table(args.offense_features), _read_table(args.defense_features), _read_table(args.novelty_features)]
    line_eval = _read_table(args.line_eval)
    metrics = build_confidence_tier_metrics(
        model_predictions=model_predictions,
        feature_frames=feature_frames,
        line_eval=line_eval,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output, index=False)
    if args.debug:
        print("[player_props.confidence_tiers] metrics:")
        print(metrics.to_string(index=False))
    print(f"[player_props.confidence_tiers] wrote metrics rows={len(metrics)} -> {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
