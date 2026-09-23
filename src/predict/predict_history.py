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

"""Backfill predictions for historical games using archived models and features."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Optional

import joblib
import numpy as np
import pandas as pd
from dateutil import tz
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.io import PROC_DIR, MODELS_DIR
from src.utils.logging_config import setup_logging
from src.models import train as train_module
from src.models.feature_columns import make_feature_diffs, select_feature_columns
from src.predict.utils import apply_probability_caps, moneyline_to_prob
from src.predict.volatility import (
    apply_shrinkage as apply_volatility_shrinkage,
    load_volatility_artifact,
    score_volatility,
    DEFAULT_VOLATILITY_ARTIFACT,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the retrospective prediction job."""
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
        "--walk-forward",

        action="store_true",

        help="Train by prior seasons and score each requested season out of sample.",

    )

    parser.add_argument("--train-start-year", type=int, default=None, help="Earliest season available to walk-forward fold training.")
    parser.add_argument("--min-train-games", type=int, default=200, help="Minimum completed prior games required for a walk-forward fold.")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU-accelerated XGBoost models for walk-forward fold training.")
    parser.add_argument("--winprob-param-config", type=Path, default=None, help="Optional tuned win-probability parameter config for walk-forward folds.")
    parser.add_argument("--spread-param-config", type=Path, default=None, help="Optional tuned spread parameter config for walk-forward folds.")
    parser.add_argument("--skip-logit", action="store_true", help="Skip logistic companion model in walk-forward history scoring.")

    parser.add_argument(

        "--overwrite",
        action="store_true",
        help="Overwrite existing prediction files instead of skipping.",
    )
    parser.add_argument(
        "--volatility-artifact",
        type=Path,
        default=DEFAULT_VOLATILITY_ARTIFACT,
        help="Path to trained volatility classifier artifact.",
    )
    parser.add_argument(
        "--volatility-threshold",
        type=float,
        default=None,
        help="Override volatility probability threshold (defaults to artifact value).",
    )
    parser.add_argument(
        "--volatility-strength",
        type=float,
        default=0.35,
        help="Shrinkage strength (0-1) towards 0.5 for high-volatility win probabilities.",
    )
    parser.add_argument(
        "--volatility-margin-strength",
        type=float,
        default=0.45,
        help="Shrinkage strength (0-1) towards 0 margin for high-volatility spreads.",
    )
    parser.add_argument(
        "--disable-volatility",
        action="store_true",
        help="Skip volatility-based adjustments even if an artifact is available.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable verbose logging.")
    return parser.parse_args()


def _load_model_bundle(path: Path) -> dict:
    """Load the trained model plus metadata from disk (raising if missing)."""
    if not path.exists():
        raise FileNotFoundError(f"Expected model bundle at {path}")
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "model" not in bundle or "features" not in bundle:
        raise ValueError(f"Unexpected model bundle format for {path}")
    return bundle


def _prepare_matrix(df: pd.DataFrame, columns: Iterable[str], ranges: Optional[dict] = None) -> pd.DataFrame:
    """Select and type-cast the exact columns expected by the trained model."""
    X = df.reindex(columns=list(columns), fill_value=0.0)
    X = X.astype(float)
    if ranges:
        for col, bounds in ranges.items():
            if col in X.columns and isinstance(bounds, (tuple, list)) and len(bounds) == 2:
                lo, hi = bounds
                X[col] = X[col].clip(lo, hi)
    return X


def _format_margin(pred: float) -> str:
    """Pretty-print spread predictions with a leading sign."""
    if pd.isna(pred):
        return ""
    if pred >= 0:
        return f"Home by {pred:.1f} pts"
    return f"Away by {abs(pred):.1f} pts"


MOUNTAIN_TZ = tz.gettz("America/Denver")


def _format_mountain_time(series: pd.Series) -> pd.Series:
    """Convert UTC timestamps to mountain time strings for reporting."""
    """Return Mountain Time formatted strings for kickoff timestamps."""
    if series is None or len(series) == 0:
        return pd.Series([], dtype="object")
    kickoff_dt = pd.to_datetime(series, utc=True, errors="coerce")
    if kickoff_dt.isna().all():
        return pd.Series([""] * len(series), index=series.index, dtype="object")
    if MOUNTAIN_TZ is None:
        formatted = kickoff_dt.dt.strftime("%Y-%m-%d %H:%M UTC")
    else:
        kickoff_mt = kickoff_dt.dt.tz_convert(MOUNTAIN_TZ)
        formatted = kickoff_mt.dt.strftime("%Y-%m-%d %H:%M %Z")
    return formatted.fillna("")


def _ensure_home_win(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "home_win" not in out.columns and "home_margin" in out.columns:
        margin = pd.to_numeric(out["home_margin"], errors="coerce")
        out["home_win"] = np.where(margin > 0, 1, np.where(margin < 0, 0, np.nan))
    return out


def _walk_forward_split_frames(
    df: pd.DataFrame,
    *,
    score_season: int,
    train_start_year: int | None,
    weeks: list[int] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    season_values = pd.to_numeric(df["season"], errors="coerce")
    train_mask = season_values < int(score_season)
    if train_start_year is not None:
        train_mask &= season_values >= int(train_start_year)
    score_mask = season_values == int(score_season)
    if weeks:
        score_mask &= pd.to_numeric(df["week"], errors="coerce").isin([int(w) for w in weeks])
    train_df = df.loc[train_mask].copy()
    score_df = df.loc[score_mask].copy()
    train_df = train_df[train_df["home_margin"].notna()].copy()
    if "home_win" in train_df.columns:
        train_df = train_df[train_df["home_win"].notna()].copy()
    score_df = score_df[score_df["home_margin"].notna()].copy()
    return train_df, score_df


def _load_param_config(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    params, _ = train_module.load_param_config(str(path))
    return params


def _xgb_available() -> bool:
    return bool(getattr(train_module, "XGB_AVAILABLE", False))


def _fit_walk_forward_win_model(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    param_config: dict[str, object] | None,
    use_gpu: bool,
    skip_logit: bool,
) -> dict[str, object]:
    if use_gpu and not _xgb_available():
        raise RuntimeError("GPU walk-forward training requested but XGBoost is not installed.")
    use_cols = [c for c in feature_cols if c in train_df.columns]
    X = _prepare_matrix(train_df, use_cols)
    y = train_df["home_win"].astype(int).to_numpy()
    model_choice = str(param_config.get("model_type", "")).lower() if param_config else ""
    use_xgb = _xgb_available() and model_choice not in {"hist", "hgb", "gb_hist"}
    if use_xgb:
        xgb_params: dict[str, object] = {
            "n_estimators": 800,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "reg_alpha": 0.0,
            "random_state": 42,
            "eval_metric": "logloss",
            "verbosity": 0,
            "use_label_encoder": False,
        }
        if param_config:
            xgb_params.update(train_module._apply_prefixed_params(param_config, "xgb_"))
        xgb_params.update(train_module.get_xgb_tree_config(use_gpu))
        model = train_module.XGBClassifier(**xgb_params)
        train_module._fit_xgb_with_fallback(model, X.values, y, use_gpu, "[predict_history][walk_forward][win_prob]")
    else:
        hgb_params: dict[str, object] = {
            "random_state": 42,
            "max_leaf_nodes": 31,
            "learning_rate": 0.05,
            "l2_regularization": 0.1,
            "min_samples_leaf": 20,
            "max_bins": 255,
        }
        if param_config:
            hgb_params.update(train_module._apply_prefixed_params(param_config, "hgb_"))
        model = HistGradientBoostingClassifier(**hgb_params)
        model.fit(X.values, y)

    logistic_model = None
    logistic_features: list[str] = []
    if not skip_logit:
        logistic_candidates = [
            "spread_line",
            "home_moneyline",
            "away_moneyline",
            "total_line",
            "inj_listed_total_diff",
            "inj_out_diff",
            "roster_injured_pct_diff",
            "team_news_injury_kw7_diff",
            "news_injury_kw7_diff",
        ]
        logistic_features = [c for c in logistic_candidates if c in train_df.columns]
        if logistic_features:
            logistic_model = Pipeline(
                steps=[
                    ("imputer", train_module._make_median_imputer()),
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
                ]
            )
            logistic_model.fit(train_df[logistic_features].astype(float).values, y)

    return {
        "model": model,
        "features": use_cols,
        "logistic_model": logistic_model,
        "logistic_features": logistic_features,
        "ensemble_weight": 1.0,
    }


def _fit_walk_forward_spread_model(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    param_config: dict[str, object] | None,
    use_gpu: bool,
) -> dict[str, object]:
    if use_gpu and not _xgb_available():
        raise RuntimeError("GPU walk-forward training requested but XGBoost is not installed.")
    use_cols = [c for c in feature_cols if c in train_df.columns]
    X_raw = _prepare_matrix(train_df, use_cols)
    y = train_df["home_margin"].astype(float).to_numpy()
    imputer = train_module._make_median_imputer()
    X = imputer.fit_transform(X_raw.values)
    model_choice = str(param_config.get("model_type", "")).lower() if param_config else ""
    use_xgb = _xgb_available() and model_choice not in {"hist", "hgb", "gb_hist", "gb", "gbr", "gbdt", "huber"}
    if use_xgb:
        xgb_params: dict[str, object] = {
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "random_state": 42,
            "objective": "reg:squarederror",
            "verbosity": 0,
        }
        if param_config:
            xgb_params.update(train_module._apply_prefixed_params(param_config, "xgb_"))
        xgb_params.update(train_module.get_xgb_tree_config(use_gpu))
        model = train_module.XGBRegressor(**xgb_params)
        train_module._fit_xgb_with_fallback(model, X, y, use_gpu, "[predict_history][walk_forward][spread]")
    else:
        gbr_params: dict[str, object] = {
            "loss": "huber",
            "alpha": 0.9,
            "learning_rate": 0.05,
            "n_estimators": 600,
            "max_depth": 4,
            "min_samples_leaf": 20,
            "max_features": "sqrt",
            "subsample": 0.8,
            "random_state": 42,
        }
        if param_config:
            gbr_params.update(train_module._apply_prefixed_params(param_config, "gbr_"))
        model = GradientBoostingRegressor(**gbr_params)
        model.fit(X, y)
    train_pred = model.predict(X)
    return {
        "model": model,
        "features": use_cols,
        "imputer": imputer,
        "bias_correction": float((train_pred - y).mean()),
    }


def _predict_walk_forward_win(bundle: dict[str, object], score_df: pd.DataFrame) -> np.ndarray:
    features = list(bundle["features"])
    model = bundle["model"]
    X = _prepare_matrix(score_df, features)
    proba = model.predict_proba(X.values)[:, 1]
    logistic_model = bundle.get("logistic_model")
    logistic_features = list(bundle.get("logistic_features") or [])
    if logistic_model is not None and logistic_features:
        log_proba = logistic_model.predict_proba(score_df[logistic_features].astype(float).values)[:, 1]
        weight = float(bundle.get("ensemble_weight", 1.0))
        proba = weight * proba + (1.0 - weight) * log_proba
    return np.asarray(proba, dtype=float)


def _predict_walk_forward_spread(bundle: dict[str, object], score_df: pd.DataFrame) -> np.ndarray:
    features = list(bundle["features"])
    model = bundle["model"]
    imputer = bundle.get("imputer")
    X_raw = _prepare_matrix(score_df, features)
    X = imputer.transform(X_raw.values) if imputer is not None else X_raw.values
    return np.asarray(model.predict(X), dtype=float) - float(bundle.get("bias_correction", 0.0))


def _build_walk_forward_history(
    df: pd.DataFrame,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> pd.DataFrame:
    df = _ensure_home_win(df)
    feat_df = make_feature_diffs(df)
    feat_df = _ensure_home_win(feat_df)
    feat_df = feat_df.loc[:, ~pd.Index(feat_df.columns).duplicated()].copy()
    feature_cols = [c for c in select_feature_columns(feat_df) if c in feat_df.columns]
    if not feature_cols:
        raise ValueError("No usable features for walk-forward history scoring.")
    win_params = _load_param_config(args.winprob_param_config)
    spread_params = _load_param_config(args.spread_param_config)
    frames: list[pd.DataFrame] = []
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
    for season in sorted(set(int(s) for s in args.seasons)):
        train_df, score_df = _walk_forward_split_frames(
            feat_df,
            score_season=season,
            train_start_year=args.train_start_year,
            weeks=args.weeks,
        )
        if score_df.empty:
            logger.warning("No completed games to score for walk-forward season %s.", season)
            continue
        if len(train_df) < int(args.min_train_games) or train_df["home_win"].nunique(dropna=True) < 2:
            logger.warning(
                "Skipping walk-forward season %s: train_rows=%s, classes=%s.",
                season,
                len(train_df),
                train_df["home_win"].nunique(dropna=True),
            )
            continue
        win_bundle = _fit_walk_forward_win_model(
            train_df,
            feature_cols,
            param_config=win_params,
            use_gpu=bool(args.use_gpu),
            skip_logit=bool(args.skip_logit),
        )
        spread_bundle = _fit_walk_forward_spread_model(
            train_df,
            feature_cols,
            param_config=spread_params,
            use_gpu=bool(args.use_gpu),
        )
        raw_proba = _predict_walk_forward_win(win_bundle, score_df)
        market_caps = None
        if "sched_implied_prob_home" in score_df.columns:
            market_caps = pd.to_numeric(score_df["sched_implied_prob_home"], errors="coerce")
        elif "home_moneyline" in score_df.columns:
            market_caps = moneyline_to_prob(score_df["home_moneyline"])
        capped_proba = apply_probability_caps(raw_proba, market_caps)
        margin_pred = _predict_walk_forward_spread(spread_bundle, score_df)
        base = score_df[out_cols].copy()
        base["home_win_prob_model_raw"] = raw_proba
        base["home_win_prob_raw"] = raw_proba
        base["home_win_prob_capped"] = capped_proba
        base["home_win_prob"] = capped_proba
        base["pred_home_margin"] = margin_pred
        base["pred_home_margin_raw"] = margin_pred
        base["prediction_source"] = "walk_forward_history"
        base["walk_forward_train_start"] = int(train_df["season"].min())
        base["walk_forward_train_end"] = int(train_df["season"].max())
        base["walk_forward_train_rows"] = int(len(train_df))
        if "kickoff" in score_df.columns and score_df["kickoff"].notna().any():
            base["kickoff"] = score_df["kickoff"]
            base["kickoff_mt"] = _format_mountain_time(base["kickoff"])
        else:
            base["kickoff_mt"] = ""
        if not args.disable_volatility:
            artifact_path = Path(args.volatility_artifact)
            if artifact_path.exists():
                try:
                    artifact = load_volatility_artifact(artifact_path)
                    threshold = args.volatility_threshold if args.volatility_threshold is not None else artifact.threshold
                    vol_prob, _ = score_volatility(score_df, artifact)
                    win_series = pd.Series(base["home_win_prob"], index=base.index, dtype=float)
                    margin_series = pd.Series(base["pred_home_margin"], index=base.index, dtype=float)
                    adj_prob, adj_margin, labels = apply_volatility_shrinkage(
                        win_series,
                        margin_series,
                        vol_prob.reindex(base.index),
                        threshold=float(threshold),
                        prob_strength=float(args.volatility_strength),
                        margin_strength=float(args.volatility_margin_strength),
                    )
                    base["home_win_prob"] = adj_prob
                    base["pred_home_margin"] = adj_margin
                    base["volatility_prob"] = vol_prob.reindex(base.index)
                    base["volatility_label"] = labels.reindex(base.index)
                except Exception as exc:
                    logger.warning("Walk-forward volatility adjustment skipped for %s: %s", season, exc)
        base["pred_home_margin_expl"] = base["pred_home_margin"].apply(_format_margin)
        base["home_pick"] = np.where(base["home_win_prob"] >= 0.5, base["home_team"], base["away_team"])
        base["home_confidence_pct"] = (base["home_win_prob"] * 100).round(1)
        frames.append(base)
        logger.info(
            "Walk-forward season %s scored rows=%s using train seasons %s-%s.",
            season,
            len(base),
            int(train_df["season"].min()),
            int(train_df["season"].max()),
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _write_history_files(base: pd.DataFrame, output_dir: Path, overwrite: bool, logger: logging.Logger) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for (season, week), group in base.groupby(["season", "week"], sort=True):
        filename = output_dir / f"w{int(week):02d}_predictions_history_{season}.csv"
        if filename.exists() and not overwrite:
            logger.debug("Skipping existing history file %s (use --overwrite to replace).", filename)
            continue
        group = group.sort_values("game_id")
        group.to_csv(filename, index=False)
        written += 1
        logger.debug("Wrote %s rows to %s", len(group), filename)
    return written


def _clear_history_files(output_dir: Path, seasons: Iterable[int], logger: logging.Logger) -> int:
    if not output_dir.exists():
        return 0
    season_suffixes = {f"_{int(season)}.csv" for season in seasons}
    removed = 0
    for path in output_dir.glob("w*_predictions_history_*.csv"):
        if any(path.name.endswith(suffix) for suffix in season_suffixes):
            path.unlink()
            removed += 1
            logger.debug("Removed stale history file %s", path)
    return removed


def main() -> None:
    """Entry point for generating historical predictions over completed games."""
    args = parse_args()
    setup_logging("nfl_predictions", level="DEBUG" if args.debug else "INFO")
    logger = logging.getLogger("predict_history")
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)

    if not args.features_path.exists():
        raise FileNotFoundError(f"Features parquet not found at {args.features_path}")
    df = pd.read_parquet(args.features_path)
    if args.walk_forward:
        base = _build_walk_forward_history(df, args, logger)
        if base.empty:
            logger.warning("No walk-forward history predictions were generated.")
            return
        if args.overwrite:
            removed = _clear_history_files(args.output_dir, args.seasons, logger)
            if removed:
                logger.info("Removed %s stale walk-forward history files before writing.", removed)
        written = _write_history_files(base, args.output_dir, args.overwrite, logger)
        logger.info("Walk-forward history predictions written for %s week files.", written)
        return

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
    spread_bias = float(spread_bundle.get("bias_correction", 0.0))

    X_win = _prepare_matrix(df, win_features, win_ranges)
    proba_raw = win_model.predict_proba(X_win)[:, 1]
    market_caps = None
    if "sched_implied_prob_home" in df.columns:
        market_caps = pd.to_numeric(df["sched_implied_prob_home"], errors="coerce")
    elif "home_moneyline" in df.columns:
        market_caps = moneyline_to_prob(df["home_moneyline"])
    proba = apply_probability_caps(proba_raw, market_caps)

    X_spread_raw = _prepare_matrix(df, spread_features, None)
    if spread_imputer is not None:
        X_spread = spread_imputer.transform(X_spread_raw)
    else:
        X_spread = X_spread_raw.values
    margin_pred = spread_model.predict(X_spread)
    if spread_bias:
        margin_pred = margin_pred - spread_bias

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
    base["home_win_prob_model_raw"] = proba_raw
    base["home_win_prob_raw"] = proba_raw
    base["home_win_prob_capped"] = proba
    base["home_win_prob"] = proba
    base["pred_home_margin"] = margin_pred
    base["prediction_source"] = "history"
    if "kickoff" in df.columns and df["kickoff"].notna().any():
        base["kickoff"] = df["kickoff"]
        base["kickoff_mt"] = _format_mountain_time(base["kickoff"])
    else:
        base["kickoff_mt"] = ""

    if not args.disable_volatility:
        artifact_path = Path(args.volatility_artifact)
        if artifact_path.exists():
            try:
                artifact = load_volatility_artifact(artifact_path)
                threshold = args.volatility_threshold if args.volatility_threshold is not None else artifact.threshold
                vol_prob, _ = score_volatility(df, artifact)
                win_series = pd.Series(base["home_win_prob"], index=base.index, dtype=float)
                margin_series = pd.Series(base["pred_home_margin"], index=base.index, dtype=float) if "pred_home_margin" in base else None
                adj_prob, adj_margin, labels = apply_volatility_shrinkage(
                    win_series,
                    margin_series,
                    vol_prob.reindex(base.index),
                    threshold=float(threshold),
                    prob_strength=float(args.volatility_strength),
                    margin_strength=float(args.volatility_margin_strength),
                )
                base["home_win_prob"] = adj_prob
                if adj_margin is not None:
                    base["pred_home_margin_raw"] = base["pred_home_margin"]
                    base["pred_home_margin"] = adj_margin
                base["volatility_prob"] = vol_prob.reindex(base.index)
                base["volatility_label"] = labels.reindex(base.index)
                coverage = float((base["volatility_label"] == 1).mean())
                logger.debug(
                    "Applied volatility shrinkage: threshold=%.2f strength=%.2f margin_strength=%.2f coverage=%.2f%%",
                    threshold,
                    args.volatility_strength,
                    args.volatility_margin_strength,
                    coverage * 100,
                )
            except Exception as exc:
                logger.warning("Volatility adjustment skipped: %s", exc)
        else:
            logger.debug("Volatility artifact %s not found; skipping adjustments.", artifact_path)

    base["pred_home_margin_expl"] = base["pred_home_margin"].apply(_format_margin)
    base["home_pick"] = np.where(base["home_win_prob"] >= 0.5, base["home_team"], base["away_team"])
    base["home_confidence_pct"] = (base["home_win_prob"] * 100).round(1)

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

