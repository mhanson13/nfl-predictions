"""Tests for src/models/feature_columns.py."""
import numpy as np
import pandas as pd
import pytest

from src.models.feature_columns import (
    LEAKAGE_PATTERNS,
    make_feature_diffs,
    select_feature_columns,
)


# ---------------------------------------------------------------------------
# make_feature_diffs
# ---------------------------------------------------------------------------

class TestMakeFeatureDiffs:

    def _basic_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "epa_home": [1.0, 2.0, 3.0],
            "epa_away": [0.5, 1.5, 2.5],
            "completions_home": [20.0, 22.0, 18.0],
            "completions_away": [18.0, 20.0, 16.0],
        })

    def test_returns_dataframe(self):
        df = self._basic_df()
        result = make_feature_diffs(df)
        assert isinstance(result, pd.DataFrame)

    def test_diff_column_created(self):
        df = self._basic_df()
        result = make_feature_diffs(df)
        assert "epa_diff" in result.columns
        assert "completions_diff" in result.columns

    def test_diff_values_correct(self):
        df = self._basic_df()
        result = make_feature_diffs(df)
        expected = pd.Series([0.5, 0.5, 0.5], name="epa_diff")
        pd.testing.assert_series_equal(result["epa_diff"].reset_index(drop=True), expected, check_names=False)

    def test_original_columns_preserved(self):
        df = self._basic_df()
        result = make_feature_diffs(df)
        for col in df.columns:
            assert col in result.columns

    def test_leakage_columns_excluded(self):
        """Columns matching LEAKAGE_PATTERNS must not produce a _diff."""
        df = pd.DataFrame({
            "score_home": [24.0, 17.0],
            "score_away": [17.0, 24.0],
            "margin_home": [7.0, -7.0],
            "margin_away": [0.0, 0.0],
        })
        result = make_feature_diffs(df)
        assert "score_diff" not in result.columns
        assert "margin_diff" not in result.columns

    def test_non_numeric_column_skipped(self):
        df = pd.DataFrame({
            "team_home": ["KC", "BUF"],
            "team_away": ["BUF", "KC"],
            "epa_home": [1.0, 2.0],
            "epa_away": [0.5, 1.5],
        })
        result = make_feature_diffs(df)
        assert "team_diff" not in result.columns
        assert "epa_diff" in result.columns

    def test_no_matching_away_col_skipped(self):
        df = pd.DataFrame({
            "epa_home": [1.0, 2.0],
            "other": [5.0, 6.0],
        })
        result = make_feature_diffs(df)
        assert "epa_diff" not in result.columns

    def test_empty_dataframe_returns_empty(self):
        result = make_feature_diffs(pd.DataFrame())
        assert result.empty

    def test_does_not_mutate_input(self):
        df = self._basic_df()
        original_cols = list(df.columns)
        make_feature_diffs(df)
        assert list(df.columns) == original_cols


# ---------------------------------------------------------------------------
# select_feature_columns
# ---------------------------------------------------------------------------

def _large_df(n: int = 100) -> pd.DataFrame:
    """Build a DataFrame with enough rows to pass the support threshold (>= 50)."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "epa_diff": rng.normal(0, 1, n),
        "completions_diff": rng.normal(0, 2, n),
        "sched_spread_close_diff": rng.normal(0, 4, n),
        "inj_out_diff": rng.normal(0, 1, n),
        "pass_epa_per_db_rolling3_diff": rng.normal(0, 0.2, n),
        "drive_epa_mean_rolling3_diff": rng.normal(0, 0.4, n),
        "team_news_count7_post_diff": rng.normal(0, 1, n),
        "weather_temperature": rng.normal(55, 15, n),
        "roof_is_dome": rng.integers(0, 2, n).astype(float),
        "wx_wind_diff": rng.normal(0, 3, n),
        "win_diff": rng.normal(0, 1, n),   # leakage
        "score_diff": rng.normal(0, 1, n), # leakage
        "team_home": ["KC"] * n,           # non-numeric
    })


class TestSelectFeatureColumns:

    def test_returns_list(self):
        df = _large_df()
        result = select_feature_columns(df)
        assert isinstance(result, list)

    def test_diff_columns_included(self):
        df = _large_df()
        result = select_feature_columns(df)
        assert "sched_spread_close_diff" in result
        assert "inj_out_diff" in result
        assert "pass_epa_per_db_rolling3_diff" in result
        assert "drive_epa_mean_rolling3_diff" in result

    def test_raw_performance_columns_excluded(self):
        df = _large_df()
        result = select_feature_columns(df)
        assert "epa_diff" not in result
        assert "completions_diff" not in result
        assert "team_news_count7_post_diff" not in result

    def test_weather_columns_included(self):
        df = _large_df()
        result = select_feature_columns(df)
        assert "weather_temperature" in result
        assert "roof_is_dome" in result

    def test_wx_columns_included(self):
        df = _large_df()
        result = select_feature_columns(df)
        assert "wx_wind_diff" in result

    def test_leakage_columns_excluded(self):
        df = _large_df()
        result = select_feature_columns(df)
        assert "win_diff" not in result
        assert "score_diff" not in result

    def test_non_numeric_excluded(self):
        df = _large_df()
        result = select_feature_columns(df)
        assert "team_home" not in result

    def test_no_duplicates(self):
        df = _large_df()
        result = select_feature_columns(df)
        assert len(result) == len(set(result))

    def test_small_df_returns_empty_or_subset(self):
        """With < 50 rows no diff column should pass the support check."""
        rng = np.random.default_rng(7)
        df = pd.DataFrame({"sched_spread_close_diff": rng.normal(0, 1, 10)})
        result = select_feature_columns(df)
        # Column with fewer than 50 non-NA values should NOT be selected.
        assert "sched_spread_close_diff" not in result

    def test_zero_variance_excluded(self):
        """Constant columns must not be selected."""
        rng = np.random.default_rng(7)
        n = 100
        df = pd.DataFrame({
            "sched_spread_close_diff": [1.0] * n,   # zero variance
            "inj_out_diff": rng.normal(0, 2, n),
        })
        result = select_feature_columns(df)
        assert "sched_spread_close_diff" not in result

    def test_duplicate_columns_handled(self):
        """Duplicate column names must not raise."""
        rng = np.random.default_rng(7)
        n = 100
        df = pd.DataFrame(np.random.randn(n, 2), columns=["sched_spread_close_diff", "sched_spread_close_diff"])
        result = select_feature_columns(df)
        assert isinstance(result, list)

    def test_all_nan_column_excluded(self):
        n = 100
        df = pd.DataFrame({"sched_spread_close_diff": [float("nan")] * n})
        result = select_feature_columns(df)
        assert "sched_spread_close_diff" not in result

    def test_leakage_patterns_constant(self):
        assert "win" in LEAKAGE_PATTERNS
        assert "score" in LEAKAGE_PATTERNS
