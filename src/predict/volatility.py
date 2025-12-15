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

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import joblib
import numpy as np
import pandas as pd

from src.features.volatility import build_volatility_feature_matrix

DEFAULT_VOLATILITY_ARTIFACT = Path("analysis/volatility_classifier_model.pkl")


@dataclass
class VolatilityArtifact:
    model: Any
    feature_columns: list[str]
    threshold: float
    created_at: Optional[str] = None
    settings: Optional[dict[str, Any]] = None
    metrics: Optional[dict[str, Any]] = None


def _ensure_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        for col in missing:
            frame[col] = 0.0
    return frame.reindex(columns=list(columns), copy=False)


def load_volatility_artifact(path: Path | str = DEFAULT_VOLATILITY_ARTIFACT) -> VolatilityArtifact:
    artifact_path = Path(path)
    data = joblib.load(artifact_path)
    if isinstance(data, VolatilityArtifact):
        return data
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected volatility artifact format at {artifact_path}")
    model = data.get("model", data)
    feature_columns = data.get("feature_columns")
    threshold = data.get("threshold")
    if feature_columns is None or threshold is None:
        raise ValueError(f"Volatility artifact missing metadata (feature_columns/threshold) at {artifact_path}")
    created_at = data.get("created_at")
    settings = data.get("settings")
    metrics = data.get("metrics")
    return VolatilityArtifact(
        model=model,
        feature_columns=list(feature_columns),
        threshold=float(threshold),
        created_at=created_at,
        settings=settings,
        metrics=metrics,
    )


def score_volatility(
    df: pd.DataFrame,
    artifact: VolatilityArtifact,
) -> tuple[pd.Series, pd.DataFrame]:
    frame = build_volatility_feature_matrix(df)
    features = frame.features.copy()
    features = _ensure_columns(features, artifact.feature_columns)
    matrix = features.astype(float).values
    model = artifact.model
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(matrix)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(matrix)
        prob = 1.0 / (1.0 + np.exp(-scores))
    else:
        prob = np.asarray(model.predict(matrix), dtype=float)
    return pd.Series(prob, index=df.index, name="volatility_prob"), frame.enriched


def apply_shrinkage(
    base_probs: pd.Series,
    base_margin: Optional[pd.Series],
    volatility_prob: pd.Series,
    *,
    threshold: float,
    prob_strength: float,
    margin_strength: float = 0.0,
) -> tuple[pd.Series, Optional[pd.Series], pd.Series]:
    probs = base_probs.copy()
    margin = base_margin.copy() if base_margin is not None else None
    vol = volatility_prob.fillna(0.0).astype(float)
    clip_prob = float(np.clip(prob_strength, 0.0, 1.0))
    clip_margin = float(np.clip(margin_strength, 0.0, 1.0))
    mask = vol >= threshold
    if clip_prob > 0:
        shrink = 1.0 - clip_prob * vol
        probs.loc[mask] = 0.5 + (probs.loc[mask] - 0.5) * shrink.loc[mask]
    if margin is not None and clip_margin > 0:
        shrink_margin = 1.0 - clip_margin * vol
        margin.loc[mask] = margin.loc[mask] * shrink_margin.loc[mask]
    labels = (vol >= threshold).astype(int)
    return probs, margin, labels
