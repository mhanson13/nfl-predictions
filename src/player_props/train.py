from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Sequence
import warnings

# Set before sklearn/joblib imports to avoid noisy Windows physical-core probes.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, (os.cpu_count() or 2) - 1)))
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"joblib\.externals\.loky\.backend\.context",
)

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.player_props.evaluate_baselines import (
    BASELINE_METRICS_NAME,
    BASELINE_PREDICTIONS_NAME,
    _binary_metrics,
)
from src.player_props.features import (
    DEFAULT_DEFENSE_OUTPUT_PATH,
    DEFAULT_OFFENSE_OUTPUT_PATH,
    DEFENSE_MARKETS,
    OFFENSE_MARKETS,
)
from src.utils.io import MODELS_DIR, read_df
from src.utils.week_filter import filter_before_week


DEFAULT_OUTPUT_DIR = Path("predictions/evaluation/player_props")
DEFAULT_MODELS_DIR = MODELS_DIR / "player_props"
MODEL_PREDICTIONS_NAME = "model_predictions.csv"
MODEL_METRICS_NAME = "model_metrics.csv"
MODEL_VS_BASELINE_NAME = "model_vs_baseline.csv"

NON_FEATURE_COLUMNS = {
    "season",
    "week",
    "game_id",
    "team",
    "opponent",
    "player_id",
    "player_name",
    "position",
    "position_group",
    "market",
    "market_family",
    "actual_value",
    "actual_over_zero",
    "played_flag",
    "active_flag",
    "value_type",
    "source",
}


@dataclass(frozen=True)
class MarketTrainingConfig:
    market: str
    family: str
    target_type: str


MARKET_CONFIGS: tuple[MarketTrainingConfig, ...] = (
    MarketTrainingConfig("qb_passing_yards", "offense", "regression"),
    MarketTrainingConfig("rb_rushing_yards", "offense", "regression"),
    MarketTrainingConfig("wrte_receiving_yards", "offense", "regression"),
    MarketTrainingConfig("def_sacks", "defense", "count_event"),
)


def _make_median_imputer() -> SimpleImputer:
    try:
        return SimpleImputer(strategy="median", keep_empty_features=True)
    except TypeError:  # pragma: no cover - older scikit-learn
        return SimpleImputer(strategy="median")


def _regressor(model_type: str, *, random_state: int) -> Pipeline:
    if model_type == "ridge":
        return Pipeline(
            [
                ("imputer", _make_median_imputer()),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        )
    return Pipeline(
        [
            ("imputer", _make_median_imputer()),
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=0.05,
                    max_iter=200,
                    l2_regularization=0.05,
                    random_state=random_state,
                ),
            ),
        ]
    )


def _classifier(model_type: str, *, random_state: int) -> Pipeline:
    if model_type == "ridge":
        return Pipeline(
            [
                ("imputer", _make_median_imputer()),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=random_state)),
            ]
        )
    return Pipeline(
        [
            ("imputer", _make_median_imputer()),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_iter=150,
                    l2_regularization=0.05,
                    random_state=random_state,
                ),
            ),
        ]
    )


def _load_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return read_df(path)


def _prepare_features(
    frame: pd.DataFrame,
    *,
    start_season: int | None = None,
    end_season: int | None = None,
    exclude_from_season: int | None = None,
    exclude_from_week: int | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    out["season"] = pd.to_numeric(out["season"], errors="coerce")
    out["week"] = pd.to_numeric(out["week"], errors="coerce")
    out["actual_value"] = pd.to_numeric(out["actual_value"], errors="coerce")
    out = out[out["season"].notna() & out["week"].notna() & out["actual_value"].notna()]
    out["season"] = out["season"].astype(int)
    out["week"] = out["week"].astype(int)
    if "actual_over_zero" not in out.columns:
        out["actual_over_zero"] = out["actual_value"] > 0
    else:
        out["actual_over_zero"] = out["actual_over_zero"].astype(bool)
    if start_season is not None:
        out = out[out["season"] >= int(start_season)]
    if end_season is not None:
        out = out[out["season"] <= int(end_season)]
    out = filter_before_week(out, exclude_from_season, exclude_from_week)
    return out.reset_index(drop=True)


def select_feature_columns(frame: pd.DataFrame) -> list[str]:
    feature_cols: list[str] = []
    for col in frame.columns:
        if col in NON_FEATURE_COLUMNS:
            continue
        series = frame[col]
        if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
            feature_cols.append(col)
    return feature_cols


def _metric_row(
    *,
    scope: str,
    season: int | None,
    market: str,
    family: str,
    model_type: str,
    generated_at: str,
    n_train: int,
    n_scored: int,
    actual: pd.Series,
    projection: pd.Series,
    prob_over_zero: pd.Series | None = None,
) -> dict[str, object]:
    if n_scored:
        errors = projection.astype(float) - actual.astype(float)
        mae = float(mean_absolute_error(actual, projection))
        rmse = float(mean_squared_error(actual, projection) ** 0.5)
        median_abs_error = float(errors.abs().median())
        mean_error = float(errors.mean())
        actual_mean = float(actual.mean())
        projection_mean = float(projection.mean())
        corr = (
            float(actual.corr(projection))
            if len(actual) > 1 and actual.nunique(dropna=True) > 1 and projection.nunique(dropna=True) > 1
            else float("nan")
        )
    else:
        mae = rmse = median_abs_error = mean_error = actual_mean = projection_mean = corr = float("nan")

    brier = logloss_value = auc = prob_mean = float("nan")
    if prob_over_zero is not None and n_scored:
        binary_frame = pd.DataFrame(
            {
                "actual_over_zero": actual.astype(float) > 0,
                "prob_over_zero": prob_over_zero,
            }
        )
        brier, logloss_value, auc, prob_mean = _binary_metrics(binary_frame)

    return {
        "scope": scope,
        "season": season if season is not None else pd.NA,
        "market": market,
        "market_family": family,
        "model_type": model_type,
        "generated_at": generated_at,
        "n_train": n_train,
        "n_scored": n_scored,
        "mae": mae,
        "rmse": rmse,
        "median_abs_error": median_abs_error,
        "mean_error": mean_error,
        "actual_mean": actual_mean,
        "projection_mean": projection_mean,
        "corr": corr,
        "brier_over_zero": brier,
        "logloss_over_zero": logloss_value,
        "auc_over_zero": auc,
        "prob_over_zero_mean": prob_mean,
    }


def _predict_probability(
    classifier: Pipeline | None,
    reg_projection: np.ndarray,
    test_x: pd.DataFrame,
) -> np.ndarray:
    if classifier is not None:
        return classifier.predict_proba(test_x)[:, 1]
    return 1.0 - np.exp(-np.clip(reg_projection, 0, None))


def _train_market_walk_forward(
    frame: pd.DataFrame,
    *,
    config: MarketTrainingConfig,
    feature_cols: list[str],
    model_type: str,
    min_train_rows: int,
    random_state: int,
    generated_at: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object] | None]:
    market_df = frame[frame["market"].eq(config.market)].copy()
    market_df = market_df.sort_values(["season", "week", "game_id", "team", "player_id"], kind="stable")
    if market_df.empty or not feature_cols:
        return pd.DataFrame(), pd.DataFrame(), None

    prediction_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    seasons = sorted(market_df["season"].dropna().astype(int).unique())

    for season in seasons:
        train = market_df[market_df["season"] < season]
        test = market_df[market_df["season"] == season]
        if len(train) < min_train_rows or test.empty:
            continue

        reg = _regressor(model_type, random_state=random_state)
        reg.fit(train[feature_cols], train["actual_value"].astype(float))
        projection = np.clip(reg.predict(test[feature_cols]), 0, None)

        classifier: Pipeline | None = None
        prob_over_zero = np.full(len(test), np.nan)
        if config.target_type == "count_event":
            y_binary = train["actual_over_zero"].astype(bool).astype(int)
            if y_binary.nunique() == 2:
                classifier = _classifier(model_type, random_state=random_state)
                classifier.fit(train[feature_cols], y_binary)
            prob_over_zero = _predict_probability(classifier, projection, test[feature_cols])

        pred = test[
            [
                "season",
                "week",
                "game_id",
                "team",
                "opponent",
                "player_id",
                "player_name",
                "position",
                "position_group",
                "market",
                "market_family",
                "actual_value",
                "actual_over_zero",
            ]
        ].copy()
        pred["model_projection"] = projection
        pred["prob_over_zero"] = prob_over_zero
        pred["model_type"] = model_type
        pred["evaluation_method"] = "season_walk_forward"
        pred["train_rows"] = len(train)
        pred["feature_count"] = len(feature_cols)
        prediction_rows.append(pred)

        metric_rows.append(
            _metric_row(
                scope="season",
                season=int(season),
                market=config.market,
                family=config.family,
                model_type=model_type,
                generated_at=generated_at,
                n_train=len(train),
                n_scored=len(test),
                actual=test["actual_value"].astype(float),
                projection=pd.Series(projection, index=test.index),
                prob_over_zero=pd.Series(prob_over_zero, index=test.index)
                if config.target_type == "count_event"
                else None,
            )
        )

    if not prediction_rows:
        return pd.DataFrame(), pd.DataFrame(), None

    predictions = pd.concat(prediction_rows, ignore_index=True)
    metric_rows.append(
        _metric_row(
            scope="overall",
            season=None,
            market=config.market,
            family=config.family,
            model_type=model_type,
            generated_at=generated_at,
            n_train=int(predictions["train_rows"].max()),
            n_scored=len(predictions),
            actual=predictions["actual_value"].astype(float),
            projection=predictions["model_projection"].astype(float),
            prob_over_zero=predictions["prob_over_zero"] if config.target_type == "count_event" else None,
        )
    )
    metrics = pd.DataFrame(metric_rows)
    final_model = train_final_market_model(
        market_df,
        config=config,
        feature_cols=feature_cols,
        model_type=model_type,
        min_train_rows=min_train_rows,
        random_state=random_state,
    )
    return predictions, metrics, final_model


def train_final_market_model(
    frame: pd.DataFrame,
    *,
    config: MarketTrainingConfig,
    feature_cols: list[str],
    model_type: str,
    min_train_rows: int,
    random_state: int,
) -> dict[str, object] | None:
    if len(frame) < min_train_rows or not feature_cols:
        return None
    reg = _regressor(model_type, random_state=random_state)
    reg.fit(frame[feature_cols], frame["actual_value"].astype(float))

    classifier: Pipeline | None = None
    if config.target_type == "count_event":
        y_binary = frame["actual_over_zero"].astype(bool).astype(int)
        if y_binary.nunique() == 2:
            classifier = _classifier(model_type, random_state=random_state)
            classifier.fit(frame[feature_cols], y_binary)

    return {
        "market": config.market,
        "market_family": config.family,
        "target_type": config.target_type,
        "model_type": model_type,
        "feature_cols": feature_cols,
        "regressor": reg,
        "classifier": classifier,
    }


def _save_model_bundle(
    bundle: dict[str, object],
    *,
    market_frame: pd.DataFrame,
    model_dir: Path,
    metrics: pd.DataFrame,
    generated_at: str,
) -> None:
    market = str(bundle["market"])
    family = str(bundle["market_family"])
    target_dir = model_dir / family
    target_dir.mkdir(parents=True, exist_ok=True)
    model_path = target_dir / f"{market}_model.pkl"
    metadata_path = target_dir / f"{market}_metadata.json"
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, (os.cpu_count() or 2) - 1)))
    import joblib

    joblib.dump(bundle, model_path)

    overall = metrics[(metrics["scope"].eq("overall")) & (metrics["market"].eq(market))]
    metadata = {
        "market": market,
        "market_family": family,
        "target_type": bundle["target_type"],
        "model_type": bundle["model_type"],
        "generated_at": generated_at,
        "model_path": model_path.as_posix(),
        "feature_count": len(bundle["feature_cols"]),
        "train_rows": int(len(market_frame)),
        "train_seasons": [int(season) for season in sorted(market_frame["season"].dropna().unique())],
        "walk_forward_metrics": overall.iloc[0].dropna().to_dict() if not overall.empty else {},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _same_sample_model_vs_baseline(
    predictions: pd.DataFrame,
    baseline_predictions_path: Path,
    *,
    generated_at: str,
) -> pd.DataFrame:
    if predictions.empty or not baseline_predictions_path.exists():
        return pd.DataFrame()
    baseline = pd.read_csv(baseline_predictions_path)
    if baseline.empty or "baseline_projection" not in baseline.columns:
        return pd.DataFrame()
    baseline = baseline[baseline.get("label_available", False).astype(bool)].copy()
    keys = ["season", "week", "game_id", "team", "player_id", "market"]
    missing = [col for col in keys if col not in baseline.columns or col not in predictions.columns]
    if missing:
        return pd.DataFrame()

    model_cols = keys + ["model_projection", "prob_over_zero"]
    merged = baseline.merge(predictions[model_cols], on=keys, how="inner")
    if merged.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for market, group in merged.groupby("market", sort=True):
        actual = pd.to_numeric(group["actual_value"], errors="coerce")
        baseline_projection = pd.to_numeric(group["baseline_projection"], errors="coerce")
        model_projection = pd.to_numeric(group["model_projection"], errors="coerce")
        valid = actual.notna() & baseline_projection.notna() & model_projection.notna()
        if not valid.any():
            continue
        actual = actual[valid]
        baseline_projection = baseline_projection[valid]
        model_projection = model_projection[valid]
        baseline_mae = float(mean_absolute_error(actual, baseline_projection))
        model_mae = float(mean_absolute_error(actual, model_projection))
        baseline_rmse = float(mean_squared_error(actual, baseline_projection) ** 0.5)
        model_rmse = float(mean_squared_error(actual, model_projection) ** 0.5)
        rows.append(
            {
                "comparison_scope": "same_baseline_rows",
                "market": market,
                "generated_at": generated_at,
                "n_scored": int(len(actual)),
                "baseline_mae": baseline_mae,
                "model_mae": model_mae,
                "mae_delta_model_minus_baseline": model_mae - baseline_mae,
                "baseline_rmse": baseline_rmse,
                "model_rmse": model_rmse,
                "rmse_delta_model_minus_baseline": model_rmse - baseline_rmse,
            }
        )
    return pd.DataFrame(rows)


def _overall_model_vs_baseline(
    model_metrics: pd.DataFrame,
    baseline_metrics_path: Path,
    *,
    generated_at: str,
) -> pd.DataFrame:
    if model_metrics.empty or not baseline_metrics_path.exists():
        return pd.DataFrame()
    baseline = pd.read_csv(baseline_metrics_path)
    baseline = baseline[baseline["scope"].eq("overall")].copy() if "scope" in baseline.columns else pd.DataFrame()
    model = model_metrics[model_metrics["scope"].eq("overall")].copy()
    if baseline.empty or model.empty:
        return pd.DataFrame()
    merged = model.merge(
        baseline,
        on="market",
        how="inner",
        suffixes=("_model", "_baseline"),
    )
    if merged.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "comparison_scope": "native_eval_samples",
            "market": merged["market"],
            "generated_at": generated_at,
            "n_model": merged["n_scored_model"],
            "n_baseline": merged["n_scored_baseline"],
            "baseline_mae": merged["mae_baseline"],
            "model_mae": merged["mae_model"],
            "mae_delta_model_minus_baseline": merged["mae_model"] - merged["mae_baseline"],
            "baseline_rmse": merged["rmse_baseline"],
            "model_rmse": merged["rmse_model"],
            "rmse_delta_model_minus_baseline": merged["rmse_model"] - merged["rmse_baseline"],
        }
    )


def train_and_evaluate(
    *,
    offense_features: pd.DataFrame,
    defense_features: pd.DataFrame,
    markets: Sequence[str] | None = None,
    model_type: str = "hgb",
    min_train_rows: int = 100,
    random_state: int = 42,
    generated_at: str | None = None,
    start_season: int | None = None,
    end_season: int | None = None,
    exclude_from_season: int | None = None,
    exclude_from_week: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, object]]]:
    generated_at = generated_at or datetime.now(timezone.utc).isoformat()
    requested = {market.lower() for market in markets} if markets else {cfg.market for cfg in MARKET_CONFIGS}
    offense = _prepare_features(
        offense_features,
        start_season=start_season,
        end_season=end_season,
        exclude_from_season=exclude_from_season,
        exclude_from_week=exclude_from_week,
    )
    defense = _prepare_features(
        defense_features,
        start_season=start_season,
        end_season=end_season,
        exclude_from_season=exclude_from_season,
        exclude_from_week=exclude_from_week,
    )

    predictions: list[pd.DataFrame] = []
    metrics: list[pd.DataFrame] = []
    bundles: dict[str, dict[str, object]] = {}

    for config in MARKET_CONFIGS:
        if config.market not in requested:
            continue
        frame = offense if config.family == "offense" else defense
        frame = frame[frame["market"].eq(config.market)].copy()
        if frame.empty:
            continue
        feature_cols = select_feature_columns(frame)
        market_predictions, market_metrics, bundle = _train_market_walk_forward(
            frame,
            config=config,
            feature_cols=feature_cols,
            model_type=model_type,
            min_train_rows=min_train_rows,
            random_state=random_state,
            generated_at=generated_at,
        )
        if not market_predictions.empty:
            predictions.append(market_predictions)
        if not market_metrics.empty:
            metrics.append(market_metrics)
        if bundle is not None:
            bundles[config.market] = bundle

    predictions_df = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    metrics_df = pd.concat(metrics, ignore_index=True) if metrics else pd.DataFrame()
    return predictions_df, metrics_df, bundles


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and walk-forward evaluate first player prop models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--offense-features", type=Path, default=DEFAULT_OFFENSE_OUTPUT_PATH)
    parser.add_argument("--defense-features", type=Path, default=DEFAULT_DEFENSE_OUTPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--baseline-metrics", type=Path, default=DEFAULT_OUTPUT_DIR / BASELINE_METRICS_NAME)
    parser.add_argument("--baseline-predictions", type=Path, default=DEFAULT_OUTPUT_DIR / BASELINE_PREDICTIONS_NAME)
    parser.add_argument("--markets", nargs="+", default=None, choices=[cfg.market for cfg in MARKET_CONFIGS])
    parser.add_argument("--model-type", choices=["hgb", "ridge"], default="hgb")
    parser.add_argument("--min-train-rows", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--start-season", type=int, default=None)
    parser.add_argument("--end-season", type=int, default=None)
    parser.add_argument("--exclude-from-season", type=int, default=None)
    parser.add_argument("--exclude-from-week", type=int, default=None)
    parser.add_argument("--skip-save-models", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    generated_at = datetime.now(timezone.utc).isoformat()
    offense = _load_features(args.offense_features)
    defense = _load_features(args.defense_features)
    predictions, metrics, bundles = train_and_evaluate(
        offense_features=offense,
        defense_features=defense,
        markets=args.markets,
        model_type=args.model_type,
        min_train_rows=args.min_train_rows,
        random_state=args.random_state,
        generated_at=generated_at,
        start_season=args.start_season,
        end_season=args.end_season,
        exclude_from_season=args.exclude_from_season,
        exclude_from_week=args.exclude_from_week,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / MODEL_PREDICTIONS_NAME
    metrics_path = args.output_dir / MODEL_METRICS_NAME
    comparison_path = args.output_dir / MODEL_VS_BASELINE_NAME
    predictions.to_csv(predictions_path, index=False)
    metrics.to_csv(metrics_path, index=False)

    comparisons = [
        _overall_model_vs_baseline(metrics, args.baseline_metrics, generated_at=generated_at),
        _same_sample_model_vs_baseline(predictions, args.baseline_predictions, generated_at=generated_at),
    ]
    comparison = pd.concat([df for df in comparisons if not df.empty], ignore_index=True) if comparisons else pd.DataFrame()
    comparison.to_csv(comparison_path, index=False)

    if not args.skip_save_models:
        for market, bundle in bundles.items():
            family = str(bundle["market_family"])
            feature_frame = offense if family == "offense" else defense
            market_frame = feature_frame[feature_frame["market"].eq(market)]
            market_frame = _prepare_features(
                market_frame,
                start_season=args.start_season,
                end_season=args.end_season,
                exclude_from_season=args.exclude_from_season,
                exclude_from_week=args.exclude_from_week,
            )
            _save_model_bundle(
                bundle,
                market_frame=market_frame,
                model_dir=args.models_dir,
                metrics=metrics,
                generated_at=generated_at,
            )

    if args.debug and not metrics.empty:
        summary = metrics[metrics["scope"].eq("overall")][
            ["market", "n_scored", "mae", "rmse", "brier_over_zero", "auc_over_zero"]
        ]
        print("[player_props.train] overall model metrics:")
        print(summary.to_string(index=False))
    if args.debug and not comparison.empty:
        print("[player_props.train] model vs baseline:")
        print(comparison.to_string(index=False))

    print(f"[player_props.train] wrote {len(predictions)} prediction rows -> {predictions_path.resolve()}")
    print(f"[player_props.train] wrote {len(metrics)} metric rows -> {metrics_path.resolve()}")
    print(f"[player_props.train] wrote {len(comparison)} comparison rows -> {comparison_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
