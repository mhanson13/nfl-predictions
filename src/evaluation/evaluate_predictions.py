from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


PREDICTION_PATTERN = re.compile(r"^w(\d+)_predictions(?:_full|_history.*)?\.csv$", re.IGNORECASE)
logger = logging.getLogger("evaluation")


@dataclass(frozen=True)
class PredictionFrame:
    path: Path
    week: int
    df: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate weekly NFL predictions against historical outcomes."
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=Path("predictions"),
        help="Directory containing weekly prediction CSV exports.",
    )
    parser.add_argument(
        "--features-path",
        type=Path,
        default=Path("data/processed/matchup_features.parquet"),
        help="Parquet file with matchup features (including final scores).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("predictions/evaluation"),
        help="Directory to store evaluation artifacts.",
    )
    parser.add_argument(
        "--min-games",
        type=int,
        default=4,
        help="Minimum completed games required in a week to report metrics.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging.",
    )
    parser.add_argument(
        "--start-season",
        type=int,
        default=None,
        help="If provided, only evaluate seasons >= this value.",
    )
    parser.add_argument(
        "--end-season",
        type=int,
        default=None,
        help="If provided, only evaluate seasons <= this value.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip generating Matplotlib plots (useful on headless systems).",
    )
    return parser.parse_args()


def find_prediction_files(pred_dir: Path) -> list[Path]:
    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory {pred_dir} does not exist.")
    paths = sorted(p for p in pred_dir.rglob("*.csv") if p.is_file())
    matches: list[Path] = []
    for path in paths:
        if PREDICTION_PATTERN.match(path.name):
            matches.append(path)
    return matches


def load_prediction_frames(paths: Iterable[Path]) -> list[PredictionFrame]:
    frames: list[PredictionFrame] = []
    for path in paths:
        match = PREDICTION_PATTERN.match(path.name)
        if not match:
            continue
        week = int(match.group(1))
        df = pd.read_csv(path)

        if "week" not in df.columns:
            df["week"] = week
        else:
            df["week"] = pd.to_numeric(df["week"], errors="coerce").fillna(week).astype(int)
        if "season" in df.columns:
            df["season"] = pd.to_numeric(df["season"], errors="coerce", downcast="integer")
        else:
            df["season"] = pd.NA

        metric_cols = [
            "home_win_prob",
            "pred_home_margin",
            "home_confidence_pct",
        ]
        for col in metric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        frames.append(PredictionFrame(path=path, week=week, df=df))
    return frames


def load_actuals(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Processed features file {path} not found.")
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
    try:
        df = pd.read_parquet(path, columns=cols)
    except Exception as exc:
        logger.warning("Primary parquet read failed (%s); retrying with fastparquet.", exc)
        df = pd.read_parquet(path, columns=cols, engine="fastparquet")
    df = df.drop_duplicates(subset=["game_id"], keep="last")
    df = df[df["home_margin"].notna()]
    df["actual_home_win"] = np.where(
        df["home_margin"] > 0,
        1,
        np.where(df["home_margin"] < 0, 0, np.nan),
    )
    return df


def unify_predictions(frames: list[PredictionFrame]) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for frame in frames:
        df = frame.df
        df = df.copy()
        df["prediction_source"] = frame.path.name
        mod_time = datetime.fromtimestamp(frame.path.stat().st_mtime, tz=timezone.utc)
        df["prediction_made_at"] = mod_time
        needed = {"game_id", "home_team", "away_team"}
        missing = needed - set(df.columns)
        if missing:
            logger.warning("Skipping %s missing columns %s", frame.path, sorted(missing))
            continue
        required_metrics = {
            "season",
            "week",
            "home_win_prob",
            "pred_home_margin",
        }
        missing_metrics = required_metrics - set(df.columns)
        if missing_metrics:
            logger.warning(
                "Skipping %s missing probability columns %s",
                frame.path,
                sorted(missing_metrics),
            )
            continue
        df = df[
            [
                "season",
                "week",
                "game_id",
                "home_team",
                "away_team",
                "home_win_prob",
                "pred_home_margin",
                "prediction_source",
                "prediction_made_at",
            ]
        ]
        records.append(df)
    if not records:
        return pd.DataFrame(
            columns=[
                "season",
                "week",
                "game_id",
                "home_team",
                "away_team",
                "home_win_prob",
                "pred_home_margin",
                "prediction_source",
                "prediction_made_at",
            ]
        )
    combined = pd.concat(records, ignore_index=True)
    combined["season"] = combined["season"].ffill()
    combined["season"] = pd.to_numeric(combined["season"], errors="coerce", downcast="integer")
    combined = combined.sort_values("prediction_made_at")
    combined = combined.drop_duplicates(subset=["season", "week", "game_id"], keep="last")
    return combined


def compute_metrics(preds: pd.DataFrame, actuals: pd.DataFrame, min_games: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = preds.merge(actuals, on=["season", "week", "game_id"], suffixes=("", "_actual"))
    merged = merged[merged["home_margin"].notna()]
    if merged.empty:
        return merged.assign(), pd.DataFrame()

    merged["home_win_prob"] = merged["home_win_prob"].clip(lower=0.0, upper=1.0)
    merged["pred_home_margin"] = pd.to_numeric(merged["pred_home_margin"], errors="coerce")
    merged["predicted_home_win"] = (merged["home_win_prob"] >= 0.5).astype(int)
    merged["actual_home_win"] = np.where(
        merged["home_margin"] > 0,
        1,
        np.where(merged["home_margin"] < 0, 0, np.nan),
    )
    merged = merged[merged["actual_home_win"].notna()]
    merged["prob_error_sq"] = (merged["home_win_prob"] - merged["actual_home_win"]) ** 2
    merged["margin_error"] = merged["pred_home_margin"] - merged["home_margin"]
    merged["abs_margin_error"] = merged["margin_error"].abs()
    merged["sq_margin_error"] = merged["margin_error"] ** 2
    eps = 1e-9
    merged["log_loss_component"] = (
        merged["actual_home_win"] * np.log(np.clip(merged["home_win_prob"], eps, 1 - eps))
        + (1 - merged["actual_home_win"]) * np.log(np.clip(1 - merged["home_win_prob"], eps, 1 - eps))
    )
    merged["misclassified"] = merged["predicted_home_win"] != merged["actual_home_win"]

    weekly = (
        merged.groupby(["season", "week"])
        .agg(
            n_games=("game_id", "count"),
            accuracy=("misclassified", lambda x: 1.0 - x.mean() if len(x) else np.nan),
            brier=("prob_error_sq", "mean"),
            log_loss=("log_loss_component", lambda x: -x.mean() if len(x) else np.nan),
            mae_margin=("abs_margin_error", "mean"),
            rmse_margin=("sq_margin_error", lambda x: math.sqrt(x.mean()) if len(x) else np.nan),
        )
        .reset_index()
    )
    weekly = weekly[weekly["n_games"] >= max(min_games, 1)]

    auc_records: list[dict[str, float]] = []
    for (season, week), group in merged.groupby(["season", "week"], sort=False):
        if group["actual_home_win"].nunique() == 2:
            auc_value = float(roc_auc_score(group["actual_home_win"], group["home_win_prob"]))
        else:
            auc_value = float("nan")
        auc_records.append({"season": season, "week": week, "auc": auc_value})
    auc_df = pd.DataFrame(auc_records)
    if not auc_df.empty:
        weekly = weekly.merge(auc_df, on=["season", "week"], how="left")
    else:
        weekly["auc"] = np.nan

    weekly["week_order"] = weekly["season"] * 100 + weekly["week"]
    weekly = weekly.sort_values("week_order")
    return merged, weekly

def compute_overall_metrics(merged: pd.DataFrame) -> dict[str, float]:
    if merged.empty:
        return {}
    n_games = int(len(merged))
    accuracy = float(1.0 - merged["misclassified"].mean()) if n_games else float("nan")
    brier = float(merged["prob_error_sq"].mean()) if n_games else float("nan")
    log_loss_val = float(-merged["log_loss_component"].mean()) if n_games else float("nan")
    mae = float(merged["abs_margin_error"].mean()) if n_games else float("nan")
    rmse = float(math.sqrt(merged["sq_margin_error"].mean())) if n_games else float("nan")
    auc = float(roc_auc_score(merged["actual_home_win"], merged["home_win_prob"])) if merged["actual_home_win"].nunique() == 2 else float("nan")
    return {
        "n_games": n_games,
        "accuracy": accuracy,
        "brier": brier,
        "log_loss": log_loss_val,
        "auc": auc,
        "mae_margin": mae,
        "rmse_margin": rmse,
    }



def summarize_underperformance(merged: pd.DataFrame, output_dir: Path) -> None:
    if merged.empty:
        return
    underperf = merged.copy()
    underperf["confidence"] = (merged["home_win_prob"] - 0.5).abs()
    largest_margin = underperf.sort_values("abs_margin_error", ascending=False).head(10)
    largest_margin.to_csv(output_dir / "largest_margin_errors.csv", index=False)

    high_confidence_misses = underperf[
        (underperf["misclassified"]) & (underperf["confidence"] >= 0.2)
    ].sort_values("confidence", ascending=False)
    high_confidence_misses.head(10).to_csv(
        output_dir / "high_confidence_misclassifications.csv", index=False
    )

    prob_bins = pd.cut(underperf["home_win_prob"], bins=np.linspace(0, 1, 11))
    calib = (
        underperf.groupby(prob_bins, observed=False)
        .agg(
            n_games=("game_id", "count"),
            mean_prob=("home_win_prob", "mean"),
            actual_win_rate=("actual_home_win", "mean"),
            mean_margin_error=("margin_error", "mean"),
        )
        .reset_index()
    )
    calib.to_csv(output_dir / "calibration_by_prob_bin.csv", index=False)

    team_cols = ["home_team", "away_team"]
    team_frames: list[pd.DataFrame] = []
    for col in team_cols:
        tmp = underperf[[col, "abs_margin_error", "misclassified"]].copy()
        tmp = tmp.rename(columns={col: "team"})
        tmp["role"] = "home" if col == "home_team" else "away"
        team_frames.append(tmp)
    team_eval = pd.concat(team_frames, ignore_index=True)
    team_summary = (
        team_eval.groupby("team")
        .agg(
            n_games=("role", "count"),
            mean_abs_margin_error=("abs_margin_error", "mean"),
            misclassified_rate=("misclassified", "mean"),
        )
        .reset_index()
        .sort_values("mean_abs_margin_error", ascending=False)
    )
    team_summary.to_csv(output_dir / "team_error_summary.csv", index=False)


def plot_metrics(weekly: pd.DataFrame, output_dir: Path) -> bool:
    if weekly.empty:
        return False
    max_points = 180
    if len(weekly) > max_points:
        logger.warning(
            "Skipping plot generation (%d weekly rows exceeds threshold %d).",
            len(weekly),
            max_points,
        )
        return False
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = weekly.apply(lambda r: f"{int(r['season'])}-W{int(r['week']):02d}", axis=1)
    label_values = labels.to_numpy()
    x = np.arange(len(label_values))
    tick_step = max(1, len(label_values) // 40)
    tick_idx = np.arange(0, len(label_values), tick_step)
    if tick_idx.size == 0 or tick_idx[-1] != len(label_values) - 1:
        tick_idx = np.append(tick_idx, len(label_values) - 1)
    tick_idx = np.unique(tick_idx)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, weekly["accuracy"], marker="o", label="Accuracy")
    ax.plot(x, weekly["brier"], marker="o", label="Brier (lower better)")
    ax.plot(x, weekly["log_loss"], marker="o", label="Log loss (lower better)")
    ax.set_title("Classification Metrics by Week")
    ax.set_xlabel("Week")
    ax.set_ylabel("Metric value")
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(label_values[tick_idx], rotation=45, ha="right")
    fig.subplots_adjust(bottom=0.28)
    fig.savefig(output_dir / "classification_metrics.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, weekly["mae_margin"], marker="o", label="MAE (pts)")
    ax.plot(x, weekly["rmse_margin"], marker="o", label="RMSE (pts)")
    ax.set_title("Margin Regression Metrics by Week")
    ax.set_xlabel("Week")
    ax.set_ylabel("Points")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(label_values[tick_idx], rotation=45, ha="right")
    fig.subplots_adjust(bottom=0.28)
    fig.savefig(output_dir / "margin_metrics.png", dpi=150)
    plt.close(fig)
    return True


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.WARNING,
        format="[evaluation] %(message)s",
    )
    # Keep Matplotlib's verbose font debugging quiet even in debug mode.
    logging.getLogger("matplotlib").setLevel(logging.INFO if args.debug else logging.WARNING)
    logging.getLogger("matplotlib.font_manager").setLevel(
        logging.INFO if args.debug else logging.WARNING
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pred_paths = find_prediction_files(args.pred_dir)
    logger.debug("Found %d prediction files under %s", len(pred_paths), args.pred_dir)
    if not pred_paths:
        raise FileNotFoundError(f"No weekly prediction files found in {args.pred_dir}")

    frames = load_prediction_frames(pred_paths)
    logger.debug("Loaded %d prediction frames", len(frames))
    preds = unify_predictions(frames)
    logger.debug("Unified predictions shape: %s", preds.shape)
    if 'season' in preds.columns:
        preds['season'] = pd.to_numeric(preds['season'], errors='coerce')
        if args.start_season is not None:
            preds = preds[preds['season'] >= args.start_season]
        if args.end_season is not None:
            preds = preds[preds['season'] <= args.end_season]
    logger.debug('Filtered predictions shape: %s', preds.shape)
    if preds.empty:
        raise RuntimeError("Failed to assemble prediction data for evaluation.")

    actuals = load_actuals(args.features_path)
    logger.debug("Loaded actuals shape: %s", actuals.shape)
    if 'season' in actuals.columns:
        actuals['season'] = pd.to_numeric(actuals['season'], errors='coerce')
        if args.start_season is not None:
            actuals = actuals[actuals['season'] >= args.start_season]
        if args.end_season is not None:
            actuals = actuals[actuals['season'] <= args.end_season]
    logger.debug('Filtered actuals shape: %s', actuals.shape)
    merged, weekly = compute_metrics(preds, actuals, args.min_games)
    logger.debug("Merged rows: %s, Weekly rows: %s", merged.shape[0], weekly.shape[0])

    merged.to_csv(args.output_dir / "merged_predictions_actuals.csv", index=False)
    weekly.to_csv(args.output_dir / "weekly_metrics.csv", index=False)
    plotted = False
    if args.skip_plots:
        logger.info("Skipping plot generation (--skip-plots).")
    else:
        plotted = plot_metrics(weekly, args.output_dir)
    summarize_underperformance(merged, args.output_dir)

    overall = compute_overall_metrics(merged)
    overall_path: Path | None = None
    if overall:
        overall_path = args.output_dir / "overall_metrics.csv"
        new_row = overall.copy()
        new_row["run_timestamp"] = datetime.utcnow().isoformat() + "Z"
        if overall_path.exists():
            existing = pd.read_csv(overall_path)
            if "run_timestamp" not in existing.columns:
                existing["run_timestamp"] = pd.NA
            combined = pd.concat([existing, pd.DataFrame([new_row])], ignore_index=True)
            combined.to_csv(overall_path, index=False)
        else:
            pd.DataFrame([new_row]).to_csv(overall_path, index=False)

    if weekly.empty:
        print("No completed weeks met the minimum game threshold; metrics not generated.")
    else:
        print("Saved weekly metrics to", args.output_dir / "weekly_metrics.csv")
        if args.skip_plots or not plotted:
            print("Skipped plot image generation; see logs for details.")
        else:
            print("Saved plots to", args.output_dir)

    if overall:
        auc_val = overall["auc"]
        auc_str = f"{auc_val:.3f}" if not math.isnan(auc_val) else "nan"
        summary_line = (
            f"Overall: n={overall['n_games']}  ACC={overall['accuracy']:.3f}  "
            f"AUC={auc_str}  Brier={overall['brier']:.3f}  LogLoss={overall['log_loss']:.3f}  "
            f"MAE={overall['mae_margin']:.3f}  RMSE={overall['rmse_margin']:.3f}"
        )
        print(summary_line)
        if overall_path:
            print("Saved overall metrics to", overall_path)
    else:
        print("Overall metrics unavailable (no completed games).")


if __name__ == "__main__":
    main()
