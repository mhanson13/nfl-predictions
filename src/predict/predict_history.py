from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Optional

import joblib
import numpy as np
import pandas as pd

from src.utils.io import PROC_DIR, MODELS_DIR
from src.utils.logging import configure as configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate retrospective predictions for completed games using trained models."
    )
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        required=True,
        help="Season years to score.",
    )
    parser.add_argument(
        "--weeks",
        type=int,
        nargs="+",
        help="Optional week numbers to limit scoring (regular-season week numbering).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("predictions/history"),
        help="Where to write history prediction CSVs.",
    )
    parser.add_argument(
        "--features-path",
        type=Path,
        default=PROC_DIR / "matchup_features.parquet",
        help="Parquet file containing matchup features (including historical games).",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=MODELS_DIR,
        help="Directory containing trained model pickles.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing prediction files instead of skipping.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable verbose logging.")
    return parser.parse_args()


def _load_model_bundle(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Expected model bundle at {path}")
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "model" not in bundle or "features" not in bundle:
        raise ValueError(f"Unexpected model bundle format for {path}")
    return bundle


def _prepare_matrix(df: pd.DataFrame, columns: Iterable[str], ranges: Optional[dict] = None) -> pd.DataFrame:
    X = df.reindex(columns=list(columns), fill_value=0.0)
    X = X.astype(float)
    if ranges:
        for col, bounds in ranges.items():
            if col in X.columns and isinstance(bounds, (tuple, list)) and len(bounds) == 2:
                lo, hi = bounds
                X[col] = X[col].clip(lo, hi)
    return X


def _format_margin(pred: float) -> str:
    if pd.isna(pred):
        return ""
    if pred >= 0:
        return f"Home by {pred:.1f} pts"
    return f"Away by {abs(pred):.1f} pts"


def main() -> None:
    args = parse_args()
    configure_logging(args.debug)
    logger = logging.getLogger("predict_history")
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)

    if not args.features_path.exists():
        raise FileNotFoundError(f"Features parquet not found at {args.features_path}")
    df = pd.read_parquet(args.features_path)
    df = df[df["season"].isin(args.seasons)].copy()
    if args.weeks:
        df = df[df["week"].isin(args.weeks)]

    if df.empty:
        logger.warning("No games found for requested seasons/weeks.")
        return

    # Only evaluate games with completed scores (avoid future games)
    df = df[df["home_margin"].notna()].copy()
    if df.empty:
        logger.warning("No completed games available for selected filters.")
        return

    win_bundle = _load_model_bundle(args.models_dir / "winprob_gb.pkl")
    spread_bundle = _load_model_bundle(args.models_dir / "spread_gb.pkl")

    win_model = win_bundle["model"]
    win_features = list(win_bundle["features"])
    win_ranges = win_bundle.get("feature_ranges") or {}

    spread_model = spread_bundle["model"]
    spread_features = list(spread_bundle["features"])
    spread_imputer = spread_bundle.get("imputer")

    X_win = _prepare_matrix(df, win_features, win_ranges)
    proba = win_model.predict_proba(X_win)[:, 1]

    X_spread_raw = _prepare_matrix(df, spread_features, None)
    if spread_imputer is not None:
        X_spread = spread_imputer.transform(X_spread_raw)
    else:
        X_spread = X_spread_raw.values
    margin_pred = spread_model.predict(X_spread)

    out_cols = [
        "season",
        "week",
        "game_id",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "home_margin",
    ]
    base = df[out_cols].copy()
    base["home_win_prob"] = proba
    base["pred_home_margin"] = margin_pred
    base["pred_home_margin_expl"] = base["pred_home_margin"].apply(_format_margin)
    base["home_pick"] = np.where(base["home_win_prob"] >= 0.5, base["home_team"], base["away_team"])
    base["home_confidence_pct"] = (base["home_win_prob"] * 100).round(1)
    base["prediction_source"] = "history"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for (season, week), group in base.groupby(["season", "week"], sort=True):
        filename = args.output_dir / f"w{int(week):02d}_predictions_history_{season}.csv"
        if filename.exists() and not args.overwrite:
            logger.debug("Skipping existing history file %s (use --overwrite to replace).", filename)
            continue
        group = group.sort_values("game_id")
        group.to_csv(filename, index=False)
        written += 1
        logger.debug("Wrote %s rows to %s", len(group), filename)

    logger.info("History predictions written for %s week files.", written)


if __name__ == "__main__":
    main()

