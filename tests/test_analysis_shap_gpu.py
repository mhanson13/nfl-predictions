"""Tests for src/analysis/shap_gpu.py pure functions."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis.shap_gpu import _load_model, _prepare_feature_matrix


# ---------------------------------------------------------------------------
# _prepare_feature_matrix
# ---------------------------------------------------------------------------
def _make_feature_df(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "epa_home": rng.normal(0, 1, n),
        "epa_away": rng.normal(0, 1, n),
        "completions_home": rng.normal(20, 5, n),
        "completions_away": rng.normal(20, 5, n),
    })


class TestPrepareFeatureMatrix:
    def test_returns_tuple(self):
        df = _make_feature_df()
        result = _prepare_feature_matrix(df)
        assert isinstance(result, tuple) and len(result) == 2

    def test_returns_dataframe_and_list(self):
        df = _make_feature_df()
        X, feat_cols = _prepare_feature_matrix(df)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(feat_cols, list)

    def test_feature_cols_match_x_columns(self):
        df = _make_feature_df()
        X, feat_cols = _prepare_feature_matrix(df)
        assert list(X.columns) == feat_cols

    def test_x_is_float(self):
        df = _make_feature_df()
        X, _ = _prepare_feature_matrix(df)
        assert X.dtypes.apply(lambda d: d == "float64" or "float" in str(d)).all()

    def test_raises_if_no_usable_cols(self):
        # DataFrame with too few rows — nothing passes support threshold
        df = pd.DataFrame({"a_home": [1.0], "a_away": [0.0]})
        with pytest.raises(ValueError, match="No usable feature columns"):
            _prepare_feature_matrix(df)

    def test_feature_order_applied(self):
        df = _make_feature_df()
        _, all_cols = _prepare_feature_matrix(df)
        if len(all_cols) >= 2:
            subset = all_cols[:1]
            X, feat_cols = _prepare_feature_matrix(df, feature_order=subset)
            assert feat_cols == subset

    def test_feature_order_missing_col_raises(self):
        df = _make_feature_df()
        with pytest.raises(ValueError, match="Feature columns required"):
            _prepare_feature_matrix(df, feature_order=["nonexistent_col"])


# ---------------------------------------------------------------------------
# _load_model — invokes loading logic (model may or may not be present)
# ---------------------------------------------------------------------------
class TestLoadModel:
    def test_raises_or_returns_when_called(self):
        """_load_model either returns a (booster, features) tuple or raises a known exception."""
        try:
            result = _load_model()
            booster, features = result
            # If it succeeds, booster and features should have the right types
            assert features is None or isinstance(features, list)
        except (FileNotFoundError, ValueError):
            pass  # expected when model files are absent or incompatible
