from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.utils.io import MODELS_DIR


def _load_history(paths: Iterable[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        try:
            df = pd.read_csv(path)
            df["_source_path"] = str(path)
            frames.append(df)
        except Exception as ex:
            print(f"[calibrate] skip {path}: {ex}")
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    return combined


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate win probability model using historical predictions.")
    parser.add_argument("--history-dir", type=Path, default=Path("predictions/history"), help="Directory with prediction history CSVs")
    parser.add_argument("--seasons", type=int, nargs="*", help="Explicit seasons to include")
    parser.add_argument("--season-window", type=int, default=2, help="If seasons not provided, use this many most recent seasons")
    parser.add_argument("--min-games", type=int, default=250, help="Minimum game count required for calibration")
    parser.add_argument("--output", type=Path, default=MODELS_DIR / "winprob_calibrator.pkl", help="Destination for calibrator artifact (deprecated; use --save-calibrator)")
    parser.add_argument("--save-calibrator", type=Path, default=None, help="Path to persist the calibrator artifact")
    parser.add_argument("--volatility-dataset", type=Path, default=Path("analysis/volatility_classifier_dataset.csv"), help="Path to volatility probabilities dataset")
    parser.add_argument("--volatility-threshold", type=float, default=0.55, help="Only adjust predictions when volatility_prob >= threshold")
    parser.add_argument("--volatility-strength", type=float, default=0.35, help="Strength of shrinkage toward 0.5 for volatile games (0-1)")
    parser.add_argument("--disable-volatility", action="store_true", help="Skip volatility-based shrinkage prior to calibration")
    parser.add_argument("--apply-isotonic", dest="apply_isotonic", action="store_true", help="Apply isotonic regression calibrator after volatility shrinkage")
    parser.add_argument("--skip-isotonic", dest="apply_isotonic", action="store_false", help="Skip isotonic calibration step")
    parser.set_defaults(apply_isotonic=True)
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    logging.getLogger("urllib3").setLevel(logging.DEBUG if args.debug else logging.INFO)

    if not args.history_dir.exists():
        raise FileNotFoundError(f"History directory {args.history_dir} not found")

    csv_paths = sorted(args.history_dir.glob("*.csv"))
    history = _load_history(csv_paths)
    if history.empty:
        raise RuntimeError("No prediction history found")

    if "home_win_prob" not in history.columns:
        raise RuntimeError("History files must include home_win_prob column")

    if "home_margin" not in history.columns and "actual_home_win" not in history.columns:
        raise RuntimeError("History files must include home_margin or actual_home_win column")

    # Compute season column if missing (try to infer from filename)
    if "season" not in history.columns or history["season"].isna().all():
        inferred = []
        for src in history["_source_path"]:
            stem = Path(src).stem
            season_digits = "".join([c for c in stem if c.isdigit()])
            inferred.append(int(season_digits[:4]) if len(season_digits) >= 4 else np.nan)
        history["season"] = inferred

    history["season"] = pd.to_numeric(history["season"], errors="coerce")
    history = history.dropna(subset=["season"])
    history["season"] = history["season"].astype(int)

    if args.seasons:
        seasons = sorted(set(args.seasons))
    else:
        seasons = sorted(history["season"].unique())[-args.season_window:]
    history = history[history["season"].isin(seasons)].copy()

    if "actual_home_win" in history.columns:
        actual = pd.to_numeric(history["actual_home_win"], errors="coerce")
    else:
        margin = pd.to_numeric(history["home_margin"], errors="coerce")
        actual = pd.Series(np.where(margin > 0, 1, np.where(margin < 0, 0, np.nan)), index=history.index, name="actual")
    preds = pd.to_numeric(history["home_win_prob"], errors="coerce")

    actual = pd.Series(actual, index=history.index, name="actual")
    preds = pd.Series(preds, index=history.index, name="pred")

    mask = preds.notna() & actual.notna()
    preds = preds[mask]
    actual = actual[mask]

    original_preds = preds.copy()
    volatility_applied = False
    volatility_strength = float(args.volatility_strength)
    volatility_threshold = float(args.volatility_threshold)
    volatility_meta: dict[str, float] = {}

    if not args.disable_volatility and args.volatility_dataset.exists():
        try:
            vol_df = pd.read_csv(args.volatility_dataset, usecols=["game_id", "volatility_prob"])
            vol_map = vol_df.set_index("game_id")["volatility_prob"]
            history["volatility_prob"] = history["game_id"].map(vol_map)
            vol_series = pd.Series(history["volatility_prob"], index=history.index).astype(float)
            vol_series = vol_series.fillna(0.0)
            vol_series = vol_series.loc[mask]
            clip_strength = np.clip(volatility_strength, 0.0, 1.0)
            adjust_mask = vol_series >= volatility_threshold
            if adjust_mask.any() and clip_strength > 0:
                shrink = 1.0 - clip_strength * vol_series
                shrink = np.where(adjust_mask, shrink, 1.0)
                preds = 0.5 + (preds - 0.5) * shrink
                volatility_applied = True
                volatility_meta = {
                    "volatility_threshold": float(volatility_threshold),
                    "volatility_strength": float(clip_strength),
                    "volatility_coverage": float(adjust_mask.mean()),
                    "volatility_mean_prob": float(vol_series.mean()),
                }
                adj_brier = brier_score_loss(actual, preds)
                adj_auc = roc_auc_score(actual, preds)
                print(
                    "[calibrate] volatility adjustment applied: "
                    f"coverage={adjust_mask.mean():.2%} strength={clip_strength:.2f} "
                    f"Brier(after adjust)={adj_brier:.4f} AUC(after adjust)={adj_auc:.3f}"
                )
        except Exception as exc:  # pragma: no cover - diagnostic only
            print(f"[calibrate] volatility adjustment skipped: {exc}")

    if len(preds) < args.min_games:
        raise RuntimeError(f"Not enough samples for calibration: {len(preds)} < {args.min_games}")

    before_brier = brier_score_loss(actual, preds)
    before_auc = roc_auc_score(actual, preds)
    print(f"[calibrate] samples={len(preds)} seasons={seasons}  baseline Brier={before_brier:.4f} AUC={before_auc:.3f}")

    calibrated = preds.copy()
    calibrator: IsotonicRegression | None = None
    after_brier = before_brier
    after_auc = before_auc
    if args.apply_isotonic:
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(preds, actual)
        calibrated = calibrator.predict(preds)
        after_brier = brier_score_loss(actual, calibrated)
        after_auc = roc_auc_score(actual, calibrated)
        print(f"[calibrate] isotonic Brier={after_brier:.4f} AUC={after_auc:.3f}")
    else:
        print("[calibrate] isotonic calibration skipped (apply_isotonic=False)")

    artifact = {
        "calibrator": calibrator,
        "method": "isotonic" if args.apply_isotonic else "none",
        "sample_count": int(len(preds)),
        "seasons": seasons,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "baseline_brier": float(before_brier),
        "calibrated_brier": float(after_brier),
        "baseline_auc": float(before_auc),
        "calibrated_auc": float(after_auc),
        "volatility_used": volatility_applied,
    }
    artifact["volatility_metadata"] = volatility_meta

    target_path = args.save_calibrator or args.output
    target_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, target_path)
    print(f"[calibrate] saved calibrator to {target_path}")
    # Maintain backward-compatible copy if using new filename
    default_path = MODELS_DIR / "winprob_calibrator.pkl"
    if target_path.resolve() != default_path.resolve():
        joblib.dump(artifact, default_path)
        print(f"[calibrate] also wrote calibrator to legacy path {default_path}")

    # Save metrics snapshot
    metrics_rows = [
        {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "stage": "baseline",
            "samples": int(len(preds)),
            "brier": float(before_brier),
            "auc": float(before_auc),
            "volatility_threshold": float(volatility_threshold),
            "volatility_strength": float(volatility_strength),
            "volatility_coverage": volatility_meta.get("volatility_coverage", 0.0),
        },
        {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "stage": "calibrated" if args.apply_isotonic else "shrink_only",
            "samples": int(len(preds)),
            "brier": float(after_brier),
            "auc": float(after_auc),
            "volatility_threshold": float(volatility_threshold),
            "volatility_strength": float(volatility_strength),
            "volatility_coverage": volatility_meta.get("volatility_coverage", 0.0),
        },
    ]
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = Path("predictions/evaluation/overall_metrics.csv")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    if metrics_path.exists():
        try:
            existing = pd.read_csv(metrics_path)
            metrics_df = pd.concat([existing, metrics_df], ignore_index=True)
        except Exception:
            pass
    metrics_df.to_csv(metrics_path, index=False)
    print(f"[calibrate] wrote metrics snapshot -> {metrics_path}")

    # Reliability plot (calibrated predictions if available)
    try:
        bins = np.linspace(0.0, 1.0, 11)
        bin_ids = np.digitize(calibrated, bins) - 1
        grouped = pd.DataFrame({"pred": calibrated, "actual": actual}).groupby(bin_ids)
        bin_centers = np.clip((bins[:-1] + bins[1:]) / 2, 0.0, 1.0)
        mean_pred = grouped["pred"].mean().reindex(range(len(bin_centers))).fillna(np.nan)
        mean_actual = grouped["actual"].mean().reindex(range(len(bin_centers))).fillna(np.nan)

        plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], linestyle="--", color="#aaaaaa", label="Perfect calibration")
        plt.plot(bin_centers, mean_actual, marker="o", label="Model")
        plt.xlabel("Predicted win probability")
        plt.ylabel("Observed win rate")
        plt.title("Win Probability Reliability Curve")
        plt.legend()
        plt.grid(alpha=0.2)
        curve_path = Path("analysis/reliability_curve.png")
        curve_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(curve_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[calibrate] saved reliability plot -> {curve_path}")
    except Exception as exc:
        print(f"[calibrate] reliability plot skipped: {exc}")


if __name__ == "__main__":
    main()
