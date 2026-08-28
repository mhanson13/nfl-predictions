"""Tests for src/utils/schemas.py — pandera-based validation functions."""
from __future__ import annotations

import pandas as pd
import pytest

from src.utils.schemas import (
    check_completeness,
    check_data_freshness,
    detect_outliers,
    validate_dataframe,
    validate_features,
    validate_injuries,
    validate_predictions,
    validate_schedule,
    validate_team_stats,
    validate_weather,
)


# ---------------------------------------------------------------------------
# validate_dataframe
# ---------------------------------------------------------------------------
class TestValidateDataframe:
    def test_dict_schema_passes_when_all_cols_present(self):
        df = pd.DataFrame({"season": [2023], "week": [1]})
        is_valid, err = validate_dataframe(df, {"season": int, "week": int}, "test")
        assert is_valid
        assert err is None

    def test_dict_schema_fails_when_col_missing(self):
        df = pd.DataFrame({"season": [2023]})
        is_valid, err = validate_dataframe(df, {"season": int, "week": int}, "test")
        assert not is_valid
        assert err is not None

    def test_raise_on_error_raises_valueerror(self):
        df = pd.DataFrame({"season": [2023]})
        with pytest.raises(ValueError):
            validate_dataframe(df, {"week": int}, raise_on_error=True)


# ---------------------------------------------------------------------------
# validate_schedule
# ---------------------------------------------------------------------------
class TestValidateSchedule:
    def _valid_df(self, n: int = 3) -> pd.DataFrame:
        return pd.DataFrame({
            "season": [2023] * n,
            "week": [1] * n,
            "game_id": [f"2023_01_KC_BUF_{i}" for i in range(n)],
            "home_team": ["KC"] * n,
            "away_team": ["BUF"] * n,
            "home_score": [24.0] * n,
            "away_score": [17.0] * n,
            "gameday": pd.to_datetime(["2023-09-10"] * n).tz_localize("UTC"),
        })

    def test_valid_df_passes(self):
        is_valid, err = validate_schedule(self._valid_df())
        assert is_valid

    def test_invalid_df_fails_gracefully(self):
        df = pd.DataFrame({"other": [1]})
        is_valid, err = validate_schedule(df)
        assert not is_valid


# ---------------------------------------------------------------------------
# validate_team_stats
# ---------------------------------------------------------------------------
class TestValidateTeamStats:
    def test_valid_df_passes(self):
        df = pd.DataFrame({"season": [2023], "abbr": ["KC"]})
        is_valid, err = validate_team_stats(df)
        assert is_valid

    def test_missing_col_fails(self):
        df = pd.DataFrame({"season": [2023]})
        is_valid, err = validate_team_stats(df)
        assert not is_valid


# ---------------------------------------------------------------------------
# validate_weather
# ---------------------------------------------------------------------------
class TestValidateWeather:
    def test_valid_df_passes(self):
        df = pd.DataFrame({
            "game_id": ["2023_01_KC_BUF"],
            "season": [2023],
            "week": [1],
            "weather_temp_f": [65.0],
            "weather_wind_mph": [5.0],
            "weather_humidity_pct": [50.0],
        })
        is_valid, err = validate_weather(df)
        assert is_valid


# ---------------------------------------------------------------------------
# validate_injuries / validate_features / validate_predictions
# ---------------------------------------------------------------------------
class TestValidateInjuries:
    def test_valid_df_passes(self):
        df = pd.DataFrame({
            "season": [2023],
            "week": [1],
            "team": ["KC"],
            "player_id": ["p001"],
            "status": ["Questionable"],
        })
        is_valid, err = validate_injuries(df)
        assert is_valid


class TestValidateFeatures:
    def test_valid_df_passes(self):
        df = pd.DataFrame({
            "season": [2023],
            "week": [1],
            "game_id": ["2023_01_KC_BUF"],
            "home_team": ["KC"],
            "away_team": ["BUF"],
            "home_score": [24.0],
            "away_score": [17.0],
            "home_margin": [7.0],
            "home_win": [1.0],
        })
        is_valid, err = validate_features(df)
        assert is_valid


class TestValidatePredictions:
    def test_valid_df_passes(self):
        df = pd.DataFrame({
            "game_id": ["2023_01_KC_BUF"],
            "season": [2023],
            "week": [1],
            "home_team": ["KC"],
            "away_team": ["BUF"],
            "predicted_home_win_prob": [0.65],
            "predicted_spread": [-3.0],
            "confidence": ["high"],
        })
        is_valid, err = validate_predictions(df)
        assert is_valid


# ---------------------------------------------------------------------------
# check_data_freshness
# ---------------------------------------------------------------------------
class TestCheckDataFreshness:
    def test_fresh_data_returns_true(self):
        df = pd.DataFrame({"gameday": [pd.Timestamp.now(tz="UTC")]})
        assert check_data_freshness(df, "gameday", max_age_days=1)

    def test_stale_data_returns_false(self):
        df = pd.DataFrame({"gameday": [pd.Timestamp("2020-01-01", tz="UTC")]})
        assert not check_data_freshness(df, "gameday", max_age_days=30)

    def test_missing_col_returns_false(self):
        df = pd.DataFrame({"other": [1]})
        assert not check_data_freshness(df, "gameday")


# ---------------------------------------------------------------------------
# check_completeness
# ---------------------------------------------------------------------------
class TestCheckCompleteness:
    def test_all_present(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = check_completeness(df, ["a", "b"])
        assert result["a"]["passes"]
        assert result["b"]["passes"]

    def test_missing_col(self):
        df = pd.DataFrame({"a": [1, 2]})
        result = check_completeness(df, ["a", "b"])
        assert not result["b"]["present"]

    def test_low_fill_rate_fails(self):
        df = pd.DataFrame({"a": [1, None, None, None, None]})
        result = check_completeness(df, ["a"], min_fill_rate=0.9)
        assert not result["a"]["passes"]


# ---------------------------------------------------------------------------
# detect_outliers
# ---------------------------------------------------------------------------
class TestDetectOutliers:
    def test_returns_series(self):
        df = pd.DataFrame({"val": [1, 2, 3, 100]})
        result = detect_outliers(df, "val", n_std=2.0)
        assert isinstance(result, pd.Series)

    def test_outlier_detected(self):
        df = pd.DataFrame({"val": [1, 2, 3, 4, 5, 100]})
        result = detect_outliers(df, "val", n_std=2.0)
        assert result.iloc[-1]  # 100 should be an outlier

    def test_missing_col_returns_false_series(self):
        df = pd.DataFrame({"other": [1, 2]})
        result = detect_outliers(df, "missing_col")
        assert not result.any()
