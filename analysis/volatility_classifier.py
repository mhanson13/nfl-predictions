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
from typing import Dict, List, Tuple

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
}


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
    indoor = df.get("indoor_game", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    wind = pd.to_numeric(df.get("wind_mph"), errors="coerce").fillna(999.0)
    wind_ok = wind < 10.0
    qb_uncertain = df.get("qb_uncertain", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    qb_score = pd.to_numeric(df.get("qb_uncertainty_score"), errors="coerce").fillna(0.0)
    qb_recent = pd.to_numeric(df.get("qb_uncertainty_recent"), errors="coerce").fillna(0.0)
    qb_ok = (~qb_uncertain) & (qb_score <= 0.0) & (qb_recent <= 0.0)
    home_rest = pd.to_numeric(df.get("home_rest"), errors="coerce").fillna(0.0)
    away_rest = pd.to_numeric(df.get("away_rest"), errors="coerce").fillna(0.0)
    rest_ok = (home_rest >= 7.0) & (away_rest >= 7.0)
    short_rest = df.get("short_rest", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    back_to_back = df.get("back_to_back_travel", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    travel_flag = pd.to_numeric(df.get("travel_short_rest_flag"), errors="coerce").fillna(0.0)
    travel_ok = (~short_rest) & (~back_to_back) & (travel_flag <= 0.0)

    stable_mask = indoor & wind_ok & qb_ok & rest_ok & travel_ok
    labels = (~stable_mask).astype(int)
    return labels, stable_mask.astype(int)


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

    y_pred = (y_prob >= best_threshold).astype(int)

    metrics = {
        "auc": roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else float("nan"),
        "average_precision": average_precision_score(y_test, y_prob),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "positive_rate": float(y_test.mean()),
        "pred_positive_rate": float(y_pred.mean()),
        "threshold": best_threshold,
    }
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
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

    sweep_thresholds = np.round(np.arange(0.50, 0.91, 0.05), 2)
    sweep_records: list[dict[str, float]] = []
    auc_value = metrics.get("auc", float("nan"))
    print("[volatility] threshold sweep (0.50 -> 0.90):")
    for thr in sweep_thresholds:
        preds_thr = (y_prob_test >= thr).astype(int)
        prec = precision_score(y_test, preds_thr, zero_division=0)
        rec = recall_score(y_test, preds_thr, zero_division=0)
        acc = accuracy_score(y_test, preds_thr)
        pred_rate = float(preds_thr.mean())
        record = {
            "threshold": float(thr),
            "precision": float(prec),
            "recall": float(rec),
            "accuracy": float(acc),
            "auc": float(auc_value),
            "pred_positive_rate": pred_rate,
        }
        sweep_records.append(record)
        print(
            f"  thr={thr:.2f} precision={prec:.3f} recall={rec:.3f} "
            f"accuracy={acc:.3f} auc={auc_value:.3f} pred%={pred_rate:.3f}"
        )

    candidates = [r for r in sweep_records if r["precision"] >= 0.7]
    if candidates:
        chosen = min(candidates, key=lambda r: abs(r["recall"] - 0.8))
    else:
        chosen = max(sweep_records, key=lambda r: (r["precision"], r["recall"]))
    selected_threshold = float(np.clip(chosen["threshold"], 0.0, 1.0))
    final_pred = (y_prob_test >= selected_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, final_pred).ravel()
    final_metrics = {
        "auc": float(auc_value),
        "average_precision": average_precision_score(y_test, y_prob_test),
        "accuracy": float(accuracy_score(y_test, final_pred)),
        "precision": float(precision_score(y_test, final_pred, zero_division=0)),
        "recall": float(recall_score(y_test, final_pred, zero_division=0)),
        "positive_rate": float(y_test.mean()),
        "pred_positive_rate": float(final_pred.mean()),
        "threshold": selected_threshold,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }
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

    main(args)
