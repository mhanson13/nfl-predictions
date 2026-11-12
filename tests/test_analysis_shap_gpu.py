"""Tests for src/analysis/shap_gpu.py pure functions."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis.shap_gpu import (
    _load_model,
    _native_tree_shap_values,
    _plot_shap_bar,
    _plot_shap_summary,
    _prepare_feature_matrix,
)


# ---------------------------------------------------------------------------
# _prepare_feature_matrix
# ---------------------------------------------------------------------------
def _make_feature_df(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "pass_epa_rolling_home": rng.normal(0, 1, n),
        "pass_epa_rolling_away": rng.normal(0, 1, n),
        "rest_days_home": rng.normal(7, 1, n),
        "rest_days_away": rng.normal(7, 1, n),
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
# _native_tree_shap_values
# ---------------------------------------------------------------------------
class _FakeBooster:
    def __init__(self, contribs: np.ndarray):
        self.contribs = contribs

    def predict(self, dmatrix, pred_contribs=False, validate_features=True):
        assert pred_contribs is True
        assert validate_features is True
        return self.contribs


class TestNativeTreeShapValues:
    def test_splits_feature_contributions_from_bias_column(self):
        sample = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        booster = _FakeBooster(
            np.array(
                [
                    [0.1, 0.2, 0.7],
                    [0.3, 0.4, 0.7],
                ]
            )
        )

        shap_array, base_value = _native_tree_shap_values(booster, sample)

        assert shap_array.tolist() == [[0.1, 0.2], [0.3, 0.4]]
        assert base_value == pytest.approx(0.7)

    def test_rejects_unexpected_contribution_shape(self):
        sample = pd.DataFrame({"a": [1.0], "b": [2.0]})
        booster = _FakeBooster(np.array([[0.1, 0.2]]))

        with pytest.raises(ValueError, match="Unexpected XGBoost SHAP contribution shape"):
            _native_tree_shap_values(booster, sample)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
class TestPlotHelpers:
    def test_local_plot_helpers_write_pngs(self, tmp_path: Path):
        sample = pd.DataFrame(
            {
                "a": np.linspace(0.0, 1.0, 12),
                "b": np.linspace(1.0, 0.0, 12),
                "c": np.ones(12),
            }
        )
        shap_array = np.column_stack(
            [
                np.linspace(-0.5, 0.5, 12),
                np.linspace(0.2, -0.2, 12),
                np.zeros(12),
            ]
        )
        summary_path = tmp_path / "summary.png"
        bar_path = tmp_path / "bar.png"

        _plot_shap_summary(shap_array, sample, summary_path)
        _plot_shap_bar(shap_array, sample, bar_path)

        assert summary_path.stat().st_size > 0
        assert bar_path.stat().st_size > 0


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
