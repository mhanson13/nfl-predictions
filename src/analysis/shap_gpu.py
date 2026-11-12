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

"""GPU-accelerated SHAP analysis for the win-probability XGBoost model."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

plt.switch_backend("Agg")

from src.models.feature_columns import make_feature_diffs, select_feature_columns

MATCHUP_PATH = Path("data/processed/matchup_features.parquet")
MODEL_PATH = Path("models/xgb_winprob.json")
MODEL_BUNDLE_PATH = Path("models/winprob_gb.pkl")
OUTPUT_DIR = Path("analysis/shap")
SAMPLE_SIZE = 2000
MAX_PLOT_FEATURES = 20


def _prepare_feature_matrix(
    df: pd.DataFrame, *, feature_order: Optional[List[str]] = None
) -> tuple[pd.DataFrame, List[str]]:
    """Produce the exact feature matrix expected by the trained model."""
    feat_df = make_feature_diffs(df)
    feat_cols = select_feature_columns(feat_df)
    if not feat_cols:
        raise ValueError("No usable feature columns detected for SHAP computation.")
    dedup_mask = ~pd.Index(feat_df.columns).duplicated()
    feat_df = feat_df.loc[:, dedup_mask]
    missing = [c for c in feat_cols if c not in feat_df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing[:10]}")
    X = feat_df[feat_cols].astype(float)
    if feature_order:
        missing_order = [col for col in feature_order if col not in X.columns]
        if missing_order:
            raise ValueError(f"Feature columns required by the model are missing: {missing_order[:10]}")
        X = X.loc[:, feature_order]
        feat_cols = feature_order
    return X, feat_cols


def _load_model() -> Tuple[xgb.Booster, Optional[List[str]]]:
    """Load the booster (from JSON or joblib bundle) plus its feature order."""
    feature_list: Optional[List[str]] = None
    bundle = None
    if MODEL_BUNDLE_PATH.exists():
        bundle = joblib.load(MODEL_BUNDLE_PATH)
        bundle_features = bundle.get("features")
        if isinstance(bundle_features, list):
            feature_list = [str(col) for col in bundle_features]
    if MODEL_PATH.exists():
        booster = xgb.Booster()
        booster.load_model(MODEL_PATH)
        return booster, feature_list
    if bundle is not None:
        model = bundle.get("model")
        if model is None:
            raise ValueError(f"No 'model' entry inside {MODEL_BUNDLE_PATH}")
        if hasattr(model, "get_booster"):
            booster = model.get_booster()
            if booster is None:
                raise ValueError("XGBoost booster unavailable inside winprob_gb.pkl (model not XGB-based).")
            return booster, feature_list or getattr(booster, "feature_names", None)
        raise ValueError(
            "Stored win probability model does not expose get_booster(); "
            "ensure XGBoost is used or export models/xgb_winprob.json manually."
        )
    raise FileNotFoundError(f"Model not found at {MODEL_PATH} or {MODEL_BUNDLE_PATH}")


def _native_tree_shap_values(
    booster: xgb.Booster,
    sample: pd.DataFrame,
) -> tuple[np.ndarray, float | np.ndarray]:
    """Compute TreeSHAP contributions with XGBoost's native predictor."""
    dmatrix = xgb.DMatrix(sample, feature_names=list(sample.columns))
    contribs = np.asarray(
        booster.predict(dmatrix, pred_contribs=True, validate_features=True)
    )
    feature_count = sample.shape[1]
    if contribs.ndim == 3:
        if contribs.shape[1] == feature_count + 1:
            contribs = contribs[:, :, 0]
        elif contribs.shape[2] == feature_count + 1:
            contribs = contribs[:, 0, :]
    if contribs.ndim != 2 or contribs.shape[1] != feature_count + 1:
        raise ValueError(
            "Unexpected XGBoost SHAP contribution shape "
            f"{contribs.shape}; expected (*, {feature_count + 1})."
        )
    shap_array = contribs[:, :-1]
    bias = contribs[:, -1]
    if bias.size and np.allclose(bias, bias[0], equal_nan=True):
        base_value: float | np.ndarray = float(bias[0])
    else:
        base_value = bias
    return shap_array, base_value


def _top_shap_importance(shap_array: np.ndarray, feature_names: list[str]) -> pd.Series:
    """Return mean absolute SHAP importance, largest first."""
    if shap_array.ndim != 2:
        raise ValueError(f"Expected 2D SHAP array, got shape {shap_array.shape}.")
    if shap_array.shape[1] != len(feature_names):
        raise ValueError(
            "SHAP feature width does not match feature names: "
            f"{shap_array.shape[1]} vs {len(feature_names)}."
        )
    return pd.Series(np.abs(shap_array).mean(axis=0), index=feature_names).sort_values(ascending=False)


def _plot_shap_bar(shap_array: np.ndarray, sample: pd.DataFrame, output_path: Path) -> None:
    """Write a compact mean-absolute-SHAP bar chart without importing shap."""
    importance = _top_shap_importance(shap_array, list(sample.columns)).head(MAX_PLOT_FEATURES)
    display = importance.iloc[::-1]
    height = max(5.0, 0.34 * len(display) + 1.2)
    fig, ax = plt.subplots(figsize=(10, height))
    ax.barh(display.index, display.values, color="#4c78a8")
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title("Top win-probability feature impacts")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_shap_summary(shap_array: np.ndarray, sample: pd.DataFrame, output_path: Path) -> None:
    """Write a beeswarm-style SHAP summary using only matplotlib/numpy."""
    importance = _top_shap_importance(shap_array, list(sample.columns)).head(MAX_PLOT_FEATURES)
    top_features = list(importance.index)[::-1]
    col_positions = {col: idx for idx, col in enumerate(sample.columns)}
    rng = np.random.default_rng(42)
    height = max(6.0, 0.36 * len(top_features) + 1.4)
    fig, ax = plt.subplots(figsize=(10, height))

    for y_pos, feature in enumerate(top_features):
        shap_values = shap_array[:, col_positions[feature]].astype(float)
        feature_values = pd.to_numeric(sample[feature], errors="coerce").to_numpy(dtype=float)
        valid_values = feature_values[np.isfinite(feature_values)]
        if valid_values.size:
            low, high = np.nanpercentile(valid_values, [5, 95])
            denom = high - low
            if denom <= 0:
                colors = np.full_like(shap_values, 0.5, dtype=float)
            else:
                colors = np.clip((feature_values - low) / denom, 0.0, 1.0)
                colors = np.where(np.isfinite(colors), colors, 0.5)
        else:
            colors = np.full_like(shap_values, 0.5, dtype=float)
        jitter = rng.normal(0.0, 0.08, size=len(shap_values))
        scatter = ax.scatter(
            shap_values,
            np.full_like(shap_values, y_pos, dtype=float) + jitter,
            c=colors,
            cmap="coolwarm",
            s=10,
            alpha=0.65,
            linewidths=0,
        )

    ax.axvline(0.0, color="#555555", linewidth=1.0, alpha=0.7)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features)
    ax.set_xlabel("SHAP value")
    ax.set_title("Win-probability SHAP summary")
    ax.grid(axis="x", alpha=0.25)
    cbar = fig.colorbar(scatter, ax=ax, pad=0.01)
    cbar.set_label("feature value")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Entrypoint for generating SHAP values/plots used by diagnostics."""
    if not MATCHUP_PATH.exists():
        print(f"[shap_gpu] Feature file missing: {MATCHUP_PATH}")
        return
    try:
        df = pd.read_parquet(MATCHUP_PATH)
    except Exception as exc:
        print(f"[shap_gpu] Failed to read {MATCHUP_PATH}: {exc}")
        return
    try:
        booster, ordered_features = _load_model()
    except Exception as exc:
        print(f"[shap_gpu] Unable to load model: {exc}")
        return
    try:
        X, feat_cols = _prepare_feature_matrix(df, feature_order=ordered_features)
    except ValueError as exc:
        print(f"[shap_gpu] {exc}")
        return
    if X.empty:
        print("[shap_gpu] No rows available for SHAP computation.")
        return
    sample = X.sample(n=min(len(X), SAMPLE_SIZE), random_state=42)
    print(f"[shap_gpu] Computing SHAP for {len(sample)} rows and {len(feat_cols)} features.")

    shap_array, base_value = _native_tree_shap_values(booster, sample)
    shap_dir = OUTPUT_DIR
    shap_dir.mkdir(parents=True, exist_ok=True)
    np.save(shap_dir / "shap_values_winprob.npy", shap_array)
    sample.to_parquet(shap_dir / "shap_sample_features.parquet")
    with open(shap_dir / "shap_base_value_winprob.txt", "w", encoding="utf-8") as fh:
        fh.write(str(base_value))
    _plot_shap_summary(shap_array, sample, shap_dir / "summary_winprob.png")
    _plot_shap_bar(shap_array, sample, shap_dir / "summary_bar_winprob.png")
    print(f"[shap_gpu] SHAP artifacts saved to {shap_dir}")


if __name__ == "__main__":
    main()
