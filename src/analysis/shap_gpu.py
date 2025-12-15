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
import shap
import xgboost as xgb

plt.switch_backend("Agg")

from src.models.feature_columns import make_feature_diffs, select_feature_columns

MATCHUP_PATH = Path("data/processed/matchup_features.parquet")
MODEL_PATH = Path("models/xgb_winprob.json")
MODEL_BUNDLE_PATH = Path("models/winprob_gb.pkl")
OUTPUT_DIR = Path("analysis/shap")
SAMPLE_SIZE = 2000


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
    dmatrix = xgb.DMatrix(sample, feature_names=list(sample.columns))
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(dmatrix)
    base_value = explainer.expected_value
    if isinstance(shap_values, list):
        shap_array = shap_values[0]
    else:
        shap_array = shap_values
    shap_dir = OUTPUT_DIR
    shap_dir.mkdir(parents=True, exist_ok=True)
    np.save(shap_dir / "shap_values_winprob.npy", shap_array)
    sample.to_parquet(shap_dir / "shap_sample_features.parquet")
    with open(shap_dir / "shap_base_value_winprob.txt", "w", encoding="utf-8") as fh:
        fh.write(str(base_value))
    plt.figure()
    shap.summary_plot(shap_array, sample, show=False)
    plt.tight_layout()
    plt.savefig(shap_dir / "summary_winprob.png", bbox_inches="tight")
    plt.close()
    plt.figure()
    shap.summary_plot(shap_array, sample, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(shap_dir / "summary_bar_winprob.png", bbox_inches="tight")
    plt.close()
    print(f"[shap_gpu] SHAP artifacts saved to {shap_dir}")


if __name__ == "__main__":
    main()
