from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Iterable

import joblib
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
    parser.add_argument("--output", type=Path, default=MODELS_DIR / "winprob_calibrator.pkl", help="Destination for calibrator artifact")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

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

    if len(preds) < args.min_games:
        raise RuntimeError(f"Not enough samples for calibration: {len(preds)} < {args.min_games}")

    before_brier = brier_score_loss(actual, preds)
    before_auc = roc_auc_score(actual, preds)
    print(f"[calibrate] samples={len(preds)} seasons={seasons}  baseline Brier={before_brier:.4f} AUC={before_auc:.3f}")

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(preds, actual)
    calibrated = calibrator.predict(preds)
    after_brier = brier_score_loss(actual, calibrated)
    after_auc = roc_auc_score(actual, calibrated)
    print(f"[calibrate] isotonic Brier={after_brier:.4f} AUC={after_auc:.3f}")

    artifact = {
        "calibrator": calibrator,
        "method": "isotonic",
        "sample_count": int(len(preds)),
        "seasons": seasons,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "baseline_brier": float(before_brier),
        "calibrated_brier": float(after_brier),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output)
    print(f"[calibrate] saved calibrator to {args.output}")


if __name__ == "__main__":
    main()
