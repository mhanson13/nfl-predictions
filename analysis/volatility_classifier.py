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
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datetime import datetime, timezone
import joblib

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

try:
    from xgboost import XGBClassifier

    XGB_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    XGB_AVAILABLE = False

from analysis.volatility_slices import build_dataset, _compute_error_columns
from src.features.volatility import build_volatility_feature_matrix, VolatilityFeatureFrame
from src.utils.week_filter import filter_before_week


ANALYSIS_DIR = Path("analysis")
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_ARTIFACT_PATH = ANALYSIS_DIR / "volatility_classifier_model.pkl"
CALIBRATION_PLOT_PATH = ANALYSIS_DIR / "volatility_calibration.png"
NEW_VOLATILITY_FEATURES = {
    "weather_temp_deviation",
    "wind_temp_interaction",
    "weather_temp_delta",
    "wind_mph_delta",
    "qb_uncertainty_trend",
    "travel_timezone_product",
    "travel_short_rest_flag",
    "travel_rest_pressure",
    "model_confidence_abs",
    "model_uncertainty",
    "model_logit_abs",
    "pred_margin_abs",
    "pred_margin_confidence",
    "model_margin_disagreement",
}
MIN_PRED_POSITIVE_RATE = 0.02
MAX_PRED_POSITIVE_RATE = 0.85


def save_calibration_plot(probs: np.ndarray, targets: np.ndarray, threshold: float, path: Path) -> None:
    if len(probs) != len(targets) or len(probs) == 0:
        raise ValueError("Cannot plot calibration with empty or mismatched arrays.")
    n_bins = min(15, max(5, int(len(probs) / 100)))
    prob_true, prob_pred = calibration_curve(targets, probs, n_bins=n_bins, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(prob_pred, prob_true, marker="o", label="Observed")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", label="Perfect")
    ax.axvline(threshold, color="#d95f02", linestyle=":", label=f"Threshold {threshold:.2f}")
    ax.set_xlabel("Predicted volatility probability")
    ax.set_ylabel("Observed volatility rate")
    ax.set_title("Volatility Classifier Calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_model_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    enriched_errors = _compute_error_columns(df)
    vol_frame: VolatilityFeatureFrame = build_volatility_feature_matrix(enriched_errors)
    feature_df = vol_frame.features.copy()
    feature_df["rest_diff"] = feature_df["home_rest"] - feature_df["away_rest"]
    feature_df["rest_days_diff"] = feature_df["sched_rest_days_home"] - feature_df["sched_rest_days_away"]
    return feature_df, vol_frame.enriched


def build_labels(
    df: pd.DataFrame,
    percentile: float,
    use_logloss: bool,
    use_margin: bool,
) -> tuple[pd.Series, pd.Series]:
    percentile = float(np.clip(percentile, 0.0, 1.0))
    metric_ranks: list[pd.Series] = []

    if use_margin and "abs_margin_error" in df.columns:
        margin_error = pd.to_numeric(df["abs_margin_error"], errors="coerce")
        if margin_error.notna().any():
            metric_ranks.append(margin_error.rank(pct=True, method="average"))

    if use_logloss and "log_loss_per_game" in df.columns:
        logloss = pd.to_numeric(df["log_loss_per_game"], errors="coerce")
        if logloss.notna().any():
            metric_ranks.append(logloss.rank(pct=True, method="average"))

    if not metric_ranks:
        if "prob_error_sq" not in df.columns:
            raise ValueError("No usable error columns available for volatility labels.")
        prob_error = pd.to_numeric(df["prob_error_sq"], errors="coerce")
        if not prob_error.notna().any():
            raise ValueError("No finite error values available for volatility labels.")
        metric_ranks.append(prob_error.rank(pct=True, method="average"))

    error_score = pd.concat(metric_ranks, axis=1).mean(axis=1).fillna(0.0)
    labels = (error_score >= percentile).astype(int)
    stable_mask = (labels == 0).astype(int)
    return labels, stable_mask


def _candidate_thresholds(y_prob: np.ndarray, extras: tuple[float, ...] = ()) -> np.ndarray:
    probs = np.asarray(y_prob, dtype=float)
    probs = probs[np.isfinite(probs)]
    if probs.size == 0:
        return np.array([0.5], dtype=float)

    quantile_thresholds = np.quantile(probs, np.linspace(0.02, 0.98, 49))
    thresholds = np.concatenate([quantile_thresholds, np.asarray(extras or (0.5,), dtype=float)])
    thresholds = thresholds[np.isfinite(thresholds)]
    thresholds = np.clip(thresholds, 0.0, 1.0)
    return np.unique(np.round(thresholds, 6))


def _threshold_record(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    auc_value: float,
) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / max(tn + fp, 1)
    balanced_accuracy = 0.5 * (recall + specificity)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy),
        "f1": float(f1),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc": float(auc_value),
        "pred_positive_rate": float(y_pred.mean()),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def build_threshold_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    decision_threshold: Optional[float] = None,
) -> list[dict[str, float]]:
    auc_value = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    extras = (float(decision_threshold),) if decision_threshold is not None else (0.5,)
    thresholds = _candidate_thresholds(y_prob, extras=extras)
    return [_threshold_record(y_true, y_prob, float(threshold), auc_value) for threshold in thresholds]


def select_threshold_from_sweep(
    sweep_records: list[dict[str, float]],
    *,
    threshold_metric: str,
    decision_threshold: Optional[float] = None,
    min_pred_positive_rate: float = MIN_PRED_POSITIVE_RATE,
    max_pred_positive_rate: float = MAX_PRED_POSITIVE_RATE,
) -> dict[str, float]:
    if not sweep_records:
        raise ValueError("Cannot select threshold from an empty sweep.")
    if decision_threshold is not None:
        target = float(np.clip(decision_threshold, 0.0, 1.0))
        return min(sweep_records, key=lambda r: abs(r["threshold"] - target))

    candidates = [
        r
        for r in sweep_records
        if min_pred_positive_rate <= r["pred_positive_rate"] <= max_pred_positive_rate
    ]
    if not candidates:
        candidates = sweep_records

    strategy = (threshold_metric or "balanced").lower()
    if strategy == "f1":
        return max(candidates, key=lambda r: (r["f1"], r["balanced_accuracy"], r["precision"], r["recall"]))
    return max(candidates, key=lambda r: (r["balanced_accuracy"], r["f1"], r["precision"], r["recall"]))


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: str,
    calibrate: bool,
    random_state: int = 42,
) -> object:
    base_model: object
    name = model_name.lower()
    if name == "logreg":
        from sklearn.linear_model import LogisticRegression

        base_model = Pipeline(
            steps=[
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        penalty="l2",
                        solver="liblinear",
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    elif name == "rf":
        from sklearn.ensemble import RandomForestClassifier

        base_model = RandomForestClassifier(
            n_estimators=400,
            max_depth=6,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=random_state,
        )
    else:  # default xgb
        if not XGB_AVAILABLE:
            from sklearn.ensemble import RandomForestClassifier

            base_model = RandomForestClassifier(
                n_estimators=400,
                max_depth=6,
                min_samples_leaf=20,
                class_weight="balanced",
                random_state=random_state,
            )
        else:
            pos_count = float(np.sum(y_train))
            neg_count = float(len(y_train) - pos_count)
            scale_pos_weight = (neg_count / pos_count) if pos_count > 0 else 1.0
            base_model = XGBClassifier(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.9,
                colsample_bytree=0.9,
                min_child_weight=1.0,
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
                random_state=random_state,
            )

    if calibrate:
        model = CalibratedClassifierCV(base_model, method="isotonic", cv=5)
    else:
        model = base_model

    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    decision_threshold: Optional[float] = None,
    threshold_metric: str = 'balanced',
) -> tuple[Dict[str, float], np.ndarray, np.ndarray, float]:
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        y_prob = 1 / (1 + np.exp(-scores))
    else:
        y_prob = model.predict(X_test)

    if decision_threshold is not None:
        best_threshold = float(np.clip(decision_threshold, 0.0, 1.0))
    else:
        strategy = (threshold_metric or "balanced").lower()
        if strategy == "balanced":
            fpr, tpr, thresholds = roc_curve(y_test, y_prob)
            if thresholds.size == 0:
                best_threshold = 0.5
            else:
                balanced = tpr + (1 - fpr)
                best_idx = int(np.nanargmax(balanced))
                best_threshold = float(thresholds[best_idx])
        elif strategy == "f1":
            precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
            if thresholds.size == 0:
                best_threshold = 0.5
            else:
                f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-9, None)
                best_idx = int(np.nanargmax(f1))
                best_threshold = float(thresholds[best_idx])
        else:
            best_threshold = 0.5
        if not np.isfinite(best_threshold):
            best_threshold = 0.5
        best_threshold = float(np.clip(best_threshold, 0.0, 1.0))

    sweep_records = build_threshold_sweep(
        y_test,
        y_prob,
        decision_threshold=decision_threshold,
    )
    selected = select_threshold_from_sweep(
        sweep_records,
        threshold_metric=threshold_metric,
        decision_threshold=decision_threshold,
    )
    best_threshold = float(selected["threshold"])
    y_pred = (y_prob >= best_threshold).astype(int)

    metrics = {
        "auc": roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else float("nan"),
        "average_precision": average_precision_score(y_test, y_prob),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),

        "specificity": selected["specificity"],

        "balanced_accuracy": selected["balanced_accuracy"],

        "f1": selected["f1"],
        "positive_rate": float(y_test.mean()),
        "pred_positive_rate": float(y_pred.mean()),
        "threshold": best_threshold,
    }
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    metrics.update({"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)})
    return metrics, y_prob, y_pred, best_threshold


def get_feature_importance(model, feature_names: List[str]) -> pd.DataFrame:
    estimator = model
    if hasattr(model, "base_estimator_"):
        estimator = model.base_estimator_
    elif hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
        estimator = model.calibrated_classifiers_[0].estimator
    if isinstance(estimator, Pipeline):
        estimator = estimator.named_steps.get("clf", estimator.steps[-1][1])

    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importances = np.abs(estimator.coef_).ravel()
    elif XGB_AVAILABLE and hasattr(estimator, "get_booster"):
        booster = estimator.get_booster()
        gain_dict = booster.get_score(importance_type="gain")
        importances = np.array([gain_dict.get(f"f{i}", 0.0) for i in range(len(feature_names))])
    else:
        importances = np.zeros(len(feature_names))

    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    return df


def main(args: argparse.Namespace) -> None:
    df = build_dataset()
    if args.start_season is not None:
        df = df[df["season"] >= args.start_season]
    if args.end_season is not None:
        df = df[df["season"] <= args.end_season]

    before_cutoff = len(df)
    df = filter_before_week(df, args.exclude_from_season, args.exclude_from_week)
    if args.exclude_from_season is not None and args.exclude_from_week is not None and len(df) != before_cutoff:
        print(
            f"[volatility] excluded {before_cutoff - len(df)} rows at/after "
            f"{args.exclude_from_season} Week {args.exclude_from_week}"
        )
    if df.empty:
        raise ValueError("No games remaining after applying filters.")

    feature_df, enriched = build_model_features(df)
    labels, stable_flags = build_labels(
        enriched,
        percentile=args.percentile,
        use_logloss=not args.disable_logloss,
        use_margin=not args.disable_margin,
    )
    enriched = enriched.assign(high_error=labels, stable_label=stable_flags)

    if labels.sum() < 30:
        raise ValueError("Not enough high-error examples to train classifier (need >= 30).")

    if args.disable_season_split:
        X_train, X_test, y_train, y_test = train_test_split(
            feature_df.values,
            labels.values,
            test_size=0.25,
            stratify=labels.values,
            random_state=args.random_state,
        )
    elif args.test_start_season is not None:
        train_mask = enriched["season"] <= args.train_end_season
        test_mask = enriched["season"] >= args.test_start_season
        if not train_mask.any() or not test_mask.any():
            raise ValueError("Train/test season split is empty. Adjust train_end/test_start seasons.")
        X_train = feature_df.loc[train_mask].values
        y_train = labels.loc[train_mask].values
        X_test = feature_df.loc[test_mask].values
        y_test = labels.loc[test_mask].values

    model = train_classifier(
        X_train,
        y_train,
        model_name=args.model,
        calibrate=args.calibrate,
        random_state=args.random_state,
    )
    metrics, y_prob_test, y_pred_test, threshold = evaluate_model(
        model,
        X_test,
        y_test,
        decision_threshold=args.decision_threshold,
        threshold_metric=args.threshold_metric,
    )
    importances = get_feature_importance(model, list(feature_df.columns))
    highlighted = importances[importances["feature"].isin(NEW_VOLATILITY_FEATURES)].head(5)
    if not highlighted.empty:
        print("[volatility] new feature contributions:")
        for row in highlighted.itertuples(index=False):
            print(f"  {row.feature}: importance={row.importance:.4f}")

    sweep_records = build_threshold_sweep(
        y_test,
        y_prob_test,
        decision_threshold=args.decision_threshold,
    )
    chosen = select_threshold_from_sweep(
        sweep_records,
        threshold_metric=args.threshold_metric,
        decision_threshold=args.decision_threshold,
    )
    selected_threshold = float(np.clip(chosen["threshold"], 0.0, 1.0))
    final_pred = (y_prob_test >= selected_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, final_pred, labels=[0, 1]).ravel()
    final_metrics = {
        "auc": float(chosen["auc"]),
        "average_precision": average_precision_score(y_test, y_prob_test),
        "accuracy": float(chosen["accuracy"]),
        "precision": float(chosen["precision"]),
        "recall": float(chosen["recall"]),
        "specificity": float(chosen["specificity"]),
        "balanced_accuracy": float(chosen["balanced_accuracy"]),
        "f1": float(chosen["f1"]),
        "positive_rate": float(y_test.mean()),
        "pred_positive_rate": float(chosen["pred_positive_rate"]),
        "threshold": selected_threshold,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }
    print("[volatility] top threshold candidates:")
    ranked_sweep = sorted(
        sweep_records,
        key=lambda r: (r.get("balanced_accuracy", 0.0), r.get("f1", 0.0)),
        reverse=True,
    )
    for record in ranked_sweep[:8]:
        print(
            f"  thr={record['threshold']:.3f} precision={record['precision']:.3f} "
            f"recall={record['recall']:.3f} bal_acc={record['balanced_accuracy']:.3f} "
            f"auc={record['auc']:.3f} pred%={record['pred_positive_rate']:.3f}"
        )
    print("[volatility] recommended threshold "
          f"{selected_threshold:.2f} -> precision={final_metrics['precision']:.3f} "
          f"recall={final_metrics['recall']:.3f} accuracy={final_metrics['accuracy']:.3f}")
    print(
        f"[volatility] confusion matrix (thr={selected_threshold:.2f}): "
        f"TP={tp} FP={fp} TN={tn} FN={fn}"
    )

    # Score entire dataset for downstream usage
    if hasattr(model, "predict_proba"):
        all_probs = model.predict_proba(feature_df.values)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(feature_df.values)
        all_probs = 1.0 / (1.0 + np.exp(-scores))
    else:
        all_probs = model.predict(feature_df.values)
    all_labels = (all_probs >= selected_threshold).astype(int)

    model_label = args.model + ("+calibrated" if args.calibrate else "")
    stable_rate = float(stable_flags.mean())
    volatility_rate = float(labels.mean())

    metrics_out = {
        "settings": {
            "percentile": args.percentile,
            "start_season": args.start_season,
            "end_season": args.end_season,
            "exclude_from_season": args.exclude_from_season,
            "exclude_from_week": args.exclude_from_week,
            "train_end_season": args.train_end_season,
            "test_start_season": args.test_start_season,
            "disable_season_split": args.disable_season_split,
            "disable_logloss": args.disable_logloss,
            "disable_margin": args.disable_margin,
            "model": model_label,
            "xgb_available": XGB_AVAILABLE,
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "decision_threshold": selected_threshold,
            "threshold_metric": args.threshold_metric,
            "threshold_sweep": sweep_records,
            "stable_rate": stable_rate,
            "volatile_rate": volatility_rate,
        },
        "metrics": final_metrics,
    }

    metrics_path = ANALYSIS_DIR / "volatility_classifier_metrics.json"
    importances_path = ANALYSIS_DIR / "volatility_classifier_importance.csv"
    dataset_path = ANALYSIS_DIR / "volatility_classifier_dataset.csv"

    metrics_path.write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")
    importances.to_csv(importances_path, index=False)
    dataset = enriched.assign(
        **feature_df,
        volatility_prob=all_probs,
        volatility_label=all_labels,
        volatility_threshold=selected_threshold,
    )
    dataset.to_csv(dataset_path, index=False)

    artifact = {
        "model": model,
        "feature_columns": list(feature_df.columns),
        "threshold": float(selected_threshold),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metrics": final_metrics,
        "settings": metrics_out["settings"],
    }
    joblib.dump(artifact, MODEL_ARTIFACT_PATH)

    try:
        save_calibration_plot(np.asarray(all_probs, dtype=float), labels.values.astype(int), float(selected_threshold), CALIBRATION_PLOT_PATH)
        calibration_msg = f"Calibration plot saved to {CALIBRATION_PLOT_PATH.resolve()}"
    except Exception as exc:
        calibration_msg = f"Calibration plot skipped: {exc}"

    print("=== Volatility classifier metrics ===")
    for key, value in final_metrics.items():
        if isinstance(value, float):
            print(f"{key:>16}: {value:.4f}")
        else:
            print(f"{key:>16}: {value}")
    print(f"\nMetrics saved to {metrics_path.resolve()}")
    print(f"Feature importance saved to {importances_path.resolve()}")
    print(f"Labeled dataset saved to {dataset_path.resolve()}")
    print(f"Model artifact saved to {MODEL_ARTIFACT_PATH.resolve()}")
    print(calibration_msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a classifier to flag high-error (volatile) games.")
    parser.add_argument("--start-season", type=int, default=2016, help="Lower bound season for dataset.")
    parser.add_argument("--end-season", type=int, default=None, help="Upper bound season for dataset.")
    parser.add_argument("--exclude-from-season", type=int, default=None, help="Exclude this season/week and later rows from volatility training.")
    parser.add_argument("--exclude-from-week", type=int, default=None, help="Exclude this season/week and later rows from volatility training.")
    parser.add_argument(
        "--train-end-season",
        type=int,
        default=2023,
        help="Last season used for training (inclusive).",
    )
    parser.add_argument(
        "--test-start-season",
        type=int,
        default=2024,
        help="First season used for testing (inclusive). Set to None for random split.",
    )
    parser.add_argument(
        "--disable-season-split",
        action="store_true",
        help="Use random train/test split instead of season-based split.",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=0.75,
        help="Percentile for labeling high-error games (0.5 - 0.99).",
    )
    parser.add_argument(
        "--model",
        choices=["xgb", "logreg", "rf"],
        default="xgb",
        help="Base classifier to use.",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Apply isotonic calibration to the chosen classifier.",
    )
    parser.add_argument(
        "--disable-logloss",
        action="store_true",
        help="Exclude log-loss errors when labeling high-error games.",
    )
    parser.add_argument(
        "--disable-margin",
        action="store_true",
        help="Exclude margin errors when labeling high-error games.",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for splits/models.")
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=None,
        help="Optional fixed probability threshold for labeling volatile games.",
    )
    parser.add_argument(
        "--threshold-metric",
        choices=["balanced", "f1"],
        default="balanced",
        help="Strategy for auto-selecting the probability threshold when --decision-threshold is not supplied.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging.")
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s:%(name)s:%(message)s")
        logging.getLogger("urllib3").setLevel(logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)

    main(args)
