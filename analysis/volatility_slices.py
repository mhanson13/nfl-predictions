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
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)


import matplotlib.pyplot as plt
import pyarrow.parquet as pq


ANALYSIS_DIR = Path("analysis")
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

MERGED_PRED_PATH = Path("predictions/evaluation/merged_predictions_actuals.csv")
FEATURES_PATH = Path("data/processed/matchup_features.parquet")


def _load_predictions() -> pd.DataFrame:
    if not MERGED_PRED_PATH.exists():
        raise FileNotFoundError(
            f"{MERGED_PRED_PATH} not found. Run `python -m src.evaluation.evaluate_predictions` first."
        )
    df = pd.read_csv(MERGED_PRED_PATH)
    needed = {"game_id", "season", "home_win_prob", "actual_home_win", "home_margin", "pred_home_margin"}
    missing = needed.difference(df.columns)
    if missing:
        raise ValueError(f"merged_predictions_actuals is missing expected columns: {sorted(missing)}")
    return df


def _load_features(columns: Iterable[str]) -> pd.DataFrame:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"{FEATURES_PATH} not found. Run `python -m src.features.build_features` first."
        )
    requested = list(columns)
    available_cols = requested
    missing_cols: list[str] = []
    try:
        schema_names = set(pq.ParquetFile(FEATURES_PATH).schema_arrow.names)
        available_cols = [c for c in requested if c in schema_names]
        missing_cols = [c for c in requested if c not in schema_names]
        if not available_cols:
            available_cols = [c for c in ("game_id",) if c in schema_names]
    except Exception:
        schema_names = None
    try:
        feat = pd.read_parquet(FEATURES_PATH, columns=available_cols if available_cols else None)
    except Exception:
        feat = pd.read_parquet(FEATURES_PATH, columns=available_cols if available_cols else None, engine="fastparquet")
    for col in missing_cols:
        feat[col] = np.nan
    if "season" in feat.columns and "week" in feat.columns:
        feat = feat.sort_values(["season", "week", "game_id"])
    if feat.duplicated("game_id").any():
        dupes = feat[feat.duplicated("game_id", keep=False)]["game_id"].unique()
        print(f"[volatility] warning: dropping duplicate game_ids from matchup_features (examples: {dupes[:5]!r})")
        feat = feat.drop_duplicates("game_id", keep="last")
    return feat


def _coalesce(df: pd.DataFrame, cols: Iterable[str], default=np.nan) -> pd.Series:
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.Series(default, index=df.index)
    arr = df[cols].copy()
    out = arr.bfill(axis=1).iloc[:, 0]
    out = out.fillna(default)
    return out


def build_dataset() -> pd.DataFrame:
    base = _load_predictions()
    feature_cols = [
        "game_id",
        "home_team",
        "away_team",
        "weather_wind_mph",
        "weather_wind_mph_wx",
        "weather_temp_f",
        "weather_temp_f_wx",
        "weather_temp_kickoff_f",
        "weather_temp_kickoff_f_wx",
        "wind",
        "temp",
        "inj_qb_questionable_home",
        "inj_qb_questionable_away",
        "inj_qb_doubtful_home",
        "inj_qb_doubtful_away",
        "inj_qb_out_home",
        "inj_qb_out_away",
        "inj_qb_reserve_home",
        "inj_qb_reserve_away",
        "inj_practice_limited_home",
        "inj_practice_limited_away",
        "inj_practice_limited_rolling3_home",
        "inj_practice_limited_rolling3_away",
        "inj_qb_questionable_rolling3_home",
        "inj_qb_questionable_rolling3_away",
        "inj_qb_doubtful_rolling3_home",
        "inj_qb_doubtful_rolling3_away",
        "inj_qb_reserve_rolling3_home",
        "inj_qb_reserve_rolling3_away",
        "home_rest",
        "away_rest",
        "sched_rest_days_home",
        "sched_rest_days_away",
        "sched_back_to_back_travel_home",
        "sched_back_to_back_travel_away",
        "roof_is_dome",
        "home_indoor",
        "away_indoor",
        "away_travel_distance_km",
        "away_travel_distance_miles",
        "home_travel_distance_km",
        "timezone_diff_hours",
        "timezone_diff_hours_abs",
        "away_travel_east",
        "away_travel_west",
        "travel_km_short_rest",
        "travel_km_back_to_back",
        "travel_km_per_rest_day",
        "timezone_diff_short_rest",
    ]
    features = _load_features(feature_cols)
    available_games = set(features["game_id"])
    base = base[base["game_id"].isin(available_games)].copy()
    merged = base.merge(features, on="game_id", how="left", validate="one_to_one")

    if merged["home_rest"].isna().any():
        merged = merged[merged["home_rest"].notna()].copy()

    merged["wind_mph"] = _coalesce(
        merged,
        ["weather_wind_mph", "weather_wind_mph_wx", "wind"],
        default=np.nan,
    )
    merged["wind_mph"] = pd.to_numeric(merged["wind_mph"], errors="coerce")
    merged["temp_f"] = _coalesce(
        merged,
        [
            "weather_temp_kickoff_f",
            "weather_temp_kickoff_f_wx",
            "weather_temp_f",
            "weather_temp_f_wx",
            "temp",
        ],
        default=np.nan,
    )
    merged["temp_f"] = pd.to_numeric(merged["temp_f"], errors="coerce")

    qb_current_cols = [
        "inj_qb_questionable_home",
        "inj_qb_questionable_away",
        "inj_qb_doubtful_home",
        "inj_qb_doubtful_away",
        "inj_qb_reserve_home",
        "inj_qb_reserve_away",
        "inj_qb_out_home",
        "inj_qb_out_away",
    ]
    qb_roll_cols = [
        "inj_qb_questionable_rolling3_home",
        "inj_qb_questionable_rolling3_away",
        "inj_qb_doubtful_rolling3_home",
        "inj_qb_doubtful_rolling3_away",
        "inj_qb_reserve_rolling3_home",
        "inj_qb_reserve_rolling3_away",
        "inj_practice_limited_rolling3_home",
        "inj_practice_limited_rolling3_away",
    ]
    qb_df = merged[qb_current_cols + qb_roll_cols].fillna(0)
    merged["qb_uncertain"] = (qb_df[qb_current_cols] > 0).any(axis=1) | (qb_df[qb_roll_cols] > 0).any(axis=1)
    merged["qb_uncertainty_score"] = qb_df[qb_current_cols].sum(axis=1) + qb_df[qb_roll_cols].sum(axis=1)

    rest_df = merged[["home_rest", "away_rest"]].astype(float)
    merged["short_rest"] = rest_df.min(axis=1) <= 6

    travel_cols = ["sched_back_to_back_travel_home", "sched_back_to_back_travel_away"]
    travel_df = merged[travel_cols].fillna(0)
    merged["back_to_back_travel"] = (travel_df > 0).any(axis=1)

    merged["indoor_game"] = merged[["roof_is_dome", "home_indoor", "away_indoor"]].fillna(0).astype(bool).any(axis=1)
    merged["outdoor_game"] = ~merged["indoor_game"]

    return merged


MetricResult = Dict[str, float]


def compute_slice_metrics(df: pd.DataFrame) -> MetricResult:
    y_true = df["actual_home_win"].astype(int)
    y_prob = df["home_win_prob"].astype(float)
    y_pred = (y_prob >= 0.5).astype(int)
    margin_true = df["home_margin"].astype(float)
    margin_pred = df["pred_home_margin"].astype(float)

    y_prob = np.clip(y_prob, 1e-6, 1 - 1e-6)

    metrics: MetricResult = {
        "n_games": int(len(df)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_prob)) if len(df["actual_home_win"].unique()) > 1 else np.nan,
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob)),
        "mae": float(mean_absolute_error(margin_true, margin_pred)),
        "rmse": float(np.sqrt(mean_squared_error(margin_true, margin_pred))),
    }
    return metrics


def build_slices(df: pd.DataFrame) -> Dict[str, pd.Series]:
    slices: Dict[str, pd.Series] = {}
    slices["high_wind"] = df["wind_mph"] >= 15
    slices["extreme_temp"] = df["temp_f"].notna() & ((df["temp_f"] <= 25) | (df["temp_f"] >= 90))
    slices["qb_uncertainty"] = df["qb_uncertain"]
    slices["short_rest"] = df["short_rest"]
    slices["back_to_back_travel"] = df["back_to_back_travel"]
    slices["indoor"] = df["indoor_game"]
    slices["outdoor"] = df["outdoor_game"]
    if "timezone_diff_hours_abs" in df.columns:
        slices["long_travel_timezone"] = df["timezone_diff_hours_abs"].notna() & (df["timezone_diff_hours_abs"] >= 2)
    if "away_travel_distance_miles" in df.columns:
        slices["long_travel_distance"] = df["away_travel_distance_miles"].notna() & (df["away_travel_distance_miles"] >= 1500)
    return slices


def run_analysis(df: pd.DataFrame, min_games: int = 30) -> Tuple[pd.DataFrame, Dict[str, MetricResult]]:
    slices = build_slices(df)
    results: Dict[str, MetricResult] = {}
    records = []

    for name, mask in slices.items():
        subset = df[mask.fillna(False)]
        if len(subset) < min_games:
            continue
        try:
            metrics = compute_slice_metrics(subset)
        except ValueError as exc:  # likely all y_true identical
            metrics = {
                "n_games": int(len(subset)),
                "accuracy": np.nan,
                "auc": np.nan,
                "brier": np.nan,
                "log_loss": np.nan,
                "mae": np.nan,
                "rmse": np.nan,
                "error": str(exc),
            }
        metrics["slice"] = name
        metrics["coverage_pct"] = float(len(subset) / len(df))
        results[name] = metrics
        records.append(metrics)

        # Also add complement slice where meaningful
        complement = df[~mask.fillna(False)]
        if len(complement) >= min_games:
            comp_name = f"not_{name}"
            try:
                comp_metrics = compute_slice_metrics(complement)
            except ValueError as exc:
                comp_metrics = {
                    "n_games": int(len(complement)),
                    "accuracy": np.nan,
                    "auc": np.nan,
                    "brier": np.nan,
                    "log_loss": np.nan,
                    "mae": np.nan,
                    "rmse": np.nan,
                    "error": str(exc),
                }
            comp_metrics["slice"] = comp_name
            comp_metrics["coverage_pct"] = float(len(complement) / len(df))
            results[comp_name] = comp_metrics
            records.append(comp_metrics)

    summary_df = pd.DataFrame.from_records(records).sort_values("slice")
    return summary_df, results


def _compute_error_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    eps = 1e-6
    probs = np.clip(out["home_win_prob"], eps, 1 - eps)
    out["abs_margin_error"] = np.abs(out["home_margin"] - out["pred_home_margin"])
    out["log_loss_per_game"] = -(out["actual_home_win"] * np.log(probs) + (1 - out["actual_home_win"]) * np.log(1 - probs))
    return out


def _plot_metric_trends(
    df: pd.DataFrame,
    feature: str,
    bucket_labels: Iterable[str],
    mae_values: Iterable[float],
    logloss_values: Iterable[float],
    out_path: Path,
    feature_label: str,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(bucket_labels, mae_values, marker="o")
    axes[0].set_ylabel("Mean MAE")
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)
    axes[1].plot(bucket_labels, logloss_values, marker="o", color="#d95f02")
    axes[1].set_ylabel("Mean LogLoss")
    axes[1].set_xlabel(feature_label)
    axes[1].grid(axis="y", linestyle="--", alpha=0.3)
    fig.suptitle(f"Error vs {feature_label}")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def create_diagnostic_plots(df: pd.DataFrame) -> None:
    enriched = _compute_error_columns(df)

    # QB uncertainty buckets
    qb_df = enriched[enriched["qb_uncertainty_score"].notna()].copy()
    if not qb_df.empty:
        qb_df["qb_bucket"] = qb_df["qb_uncertainty_score"].clip(upper=6).astype(int)
        qb_group = qb_df.groupby("qb_bucket").agg(
            mae=("abs_margin_error", "mean"),
            logloss=("log_loss_per_game", "mean"),
            n_games=("qb_bucket", "size"),
        )
        qb_group = qb_group[qb_group["n_games"] >= 10]
        if not qb_group.empty:
            out_path = ANALYSIS_DIR / "volatility_qb_error.png"
            _plot_metric_trends(
                qb_group,
                feature="qb_bucket",
                bucket_labels=[str(i) for i in qb_group.index],
                mae_values=qb_group["mae"],
                logloss_values=qb_group["logloss"],
                out_path=out_path,
                feature_label="QB uncertainty score",
            )
            print(f"Saved QB uncertainty diagnostic plot to {out_path.resolve()}")
        else:
            print("[volatility] Skipped QB uncertainty plot (insufficient coverage)")
    else:
        print("[volatility] Skipped QB uncertainty plot (no data)")

    # Wind buckets (only outdoor games with wind info)
    wind_df = enriched[enriched["wind_mph"].notna()].copy()
    if not wind_df.empty:
        bins = [0, 5, 10, 15, 20, 40]
        wind_df["wind_bucket"] = pd.cut(wind_df["wind_mph"], bins=bins, right=False)
        wind_group = wind_df.groupby("wind_bucket").agg(
            mae=("abs_margin_error", "mean"),
            logloss=("log_loss_per_game", "mean"),
            n_games=("wind_bucket", "size"),
        )
        wind_group = wind_group[wind_group["n_games"] >= 5]
        if not wind_group.empty:
            labels = [f"[{int(b.left)}, {int(b.right)})" for b in wind_group.index]
            out_path = ANALYSIS_DIR / "volatility_wind_error.png"
            _plot_metric_trends(
                wind_group,
                feature="wind_bucket",
                bucket_labels=labels,
                mae_values=wind_group["mae"],
                logloss_values=wind_group["logloss"],
                out_path=out_path,
                feature_label="Wind speed (mph)",
            )
            print(f"Saved wind diagnostic plot to {out_path.resolve()}")
        else:
            print("[volatility] Skipped wind plot (insufficient data per bucket)")
    else:
        print("[volatility] Skipped wind plot (no wind observations)")

    travel_df = enriched[enriched.get("away_travel_distance_km").notna()].copy()
    if not travel_df.empty:
        bins = [0, 500, 1000, 1500, 2000, 4000, 6000]
        travel_df["travel_bucket"] = pd.cut(travel_df["away_travel_distance_km"], bins=bins, right=False)
        travel_group = travel_df.groupby("travel_bucket").agg(
            mae=("abs_margin_error", "mean"),
            logloss=("log_loss_per_game", "mean"),
            n_games=("travel_bucket", "size"),
        )
        travel_group = travel_group[travel_group["n_games"] >= 10]
        if not travel_group.empty:
            labels = [f"[{int(b.left)}, {int(b.right)})" for b in travel_group.index]
            out_path = ANALYSIS_DIR / "volatility_travel_error.png"
            _plot_metric_trends(
                travel_group,
                feature="travel_bucket",
                bucket_labels=labels,
                mae_values=travel_group["mae"],
                logloss_values=travel_group["logloss"],
                out_path=out_path,
                feature_label="Away travel distance (km)",
            )
            print(f"Saved travel diagnostic plot to {out_path.resolve()}")
        else:
            print("[volatility] Skipped travel plot (insufficient data per bucket)")
    else:
        print("[volatility] Skipped travel plot (no travel data)")

    tz_df = enriched[enriched.get("timezone_diff_hours_abs").notna()].copy()
    if not tz_df.empty:
        tz_df["tz_bucket"] = pd.cut(tz_df["timezone_diff_hours_abs"], bins=[0, 1, 2, 3, 5], right=False)
        tz_group = tz_df.groupby("tz_bucket").agg(
            mae=("abs_margin_error", "mean"),
            logloss=("log_loss_per_game", "mean"),
            n_games=("tz_bucket", "size"),
        )
        tz_group = tz_group[tz_group["n_games"] >= 10]
        if not tz_group.empty:
            labels = [f"[{b.left:.0f}, {b.right:.0f})" for b in tz_group.index]
            out_path = ANALYSIS_DIR / "volatility_timezone_error.png"
            _plot_metric_trends(
                tz_group,
                feature="tz_bucket",
                bucket_labels=labels,
                mae_values=tz_group["mae"],
                logloss_values=tz_group["logloss"],
                out_path=out_path,
                feature_label="Away timezone delta (hours)",
            )
            print(f"Saved timezone diagnostic plot to {out_path.resolve()}")
        else:
            print("[volatility] Skipped timezone plot (insufficient data per bucket)")
    else:
        print("[volatility] Skipped timezone plot (no timezone data)")


def main(args: argparse.Namespace) -> None:
    df = build_dataset()
    if args.start_season is not None:
        df = df[df["season"] >= args.start_season]
    if args.end_season is not None:
        df = df[df["season"] <= args.end_season]
    if df.empty:
        raise ValueError("No games available after applying season filters.")

    summary_df, _ = run_analysis(df, min_games=args.min_games)
    out_path = ANALYSIS_DIR / "volatility_slice_metrics.csv"
    summary_df.to_csv(out_path, index=False)

    print("=== Volatility slice metrics ===")
    numeric_cols = ["n_games", "coverage_pct", "accuracy", "auc", "brier", "log_loss", "mae", "rmse"]
    print(summary_df[numeric_cols + ["slice"]].set_index("slice"))
    print(f"\nWrote slice metrics to {out_path.resolve()}")

    create_diagnostic_plots(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose model error across volatility-driven slices.")
    parser.add_argument("--start-season", type=int, default=None, help="Optional lower bound for seasons.")
    parser.add_argument("--end-season", type=int, default=None, help="Optional upper bound for seasons.")
    parser.add_argument("--min-games", type=int, default=30, help="Minimum sample size required to report a slice.")
    args = parser.parse_args()
    main(args)
