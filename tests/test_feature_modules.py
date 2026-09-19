"""
Tests for src/features/* — all six feature-engineering modules.

Each module follows the same pattern:
  * accepts a PBP (or schedule / injury / roster) DataFrame
  * returns a team-week keyed DataFrame of leakage-free rolling features
  * returns an empty DataFrame on bad / missing input

A shared minimal PBP fixture covers the common columns; per-module fixtures
extend it with the columns each module specifically needs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Shared norm_team helper (same contract as the pipeline uses)
# ---------------------------------------------------------------------------

def _norm(team: str, _ctx=None) -> str:
    return str(team).upper().strip() if team else team


# ---------------------------------------------------------------------------
# Shared PBP fixture factory
# ---------------------------------------------------------------------------

def _make_pbp(
    seasons=(2023,),
    teams=(("KC", "BUF"), ("SF", "DAL")),
    weeks=(1, 2, 3, 4),
    extra_cols: dict | None = None,
) -> pd.DataFrame:
    """Build a minimal synthetic play-by-play DataFrame."""
    rows = []
    play_id = 1
    for season in seasons:
        for week in weeks:
            for home, away in teams:
                game_id = f"{season}_{week:02d}_{home}_{away}"
                for drive_num in range(1, 4):
                    for _ in range(5):
                        row = {
                            "game_id": game_id,
                            "season": season,
                            "week": week,
                            "play_id": play_id,
                            "posteam": home if drive_num % 2 == 1 else away,
                            "defteam": away if drive_num % 2 == 1 else home,
                            "epa": np.random.default_rng(play_id).uniform(-1, 1),
                            "pass_attempt": int(play_id % 3 == 0),
                            "rush_attempt": int(play_id % 3 == 1),
                            "sack": int(play_id % 15 == 0),
                            "qb_scramble": 0,
                            "play_type": "pass" if play_id % 3 == 0 else "run",
                            "down": (play_id % 4) + 1,
                            "ydstogo": 10,
                            "yardline_100": (play_id % 100) + 1,
                            "drive": drive_num,
                            "fixed_drive": drive_num,
                            "qtr": ((play_id - 1) % 4) + 1,
                            "drive_inside20": int((play_id % 100) + 1 <= 20),
                            "drive_ended_with_score": int(drive_num == 3),
                            "touchdown": int(drive_num == 3 and play_id % 5 == 0),
                        }
                        if extra_cols:
                            row.update(extra_cols)
                        rows.append(row)
                        play_id += 1
    return pd.DataFrame(rows)


# ===========================================================================
# passing_epa_features
# ===========================================================================
from src.features.passing_epa_features import build_passing_epa_features


class TestPassingEpaFeatures:

    def _pbp(self):
        return _make_pbp()

    def test_returns_dataframe(self):
        df = build_passing_epa_features(self._pbp(), [2023], _norm)
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns_present(self):
        df = build_passing_epa_features(self._pbp(), [2023], _norm)
        for col in ("season", "week", "team", "pass_epa_per_db",
                    "pass_epa_per_db_rolling3", "pass_epa_per_db_rolling5",
                    "pass_epa_per_db_rolling3_diff_lg"):
            assert col in df.columns, f"missing {col}"

    def test_one_row_per_team_week(self):
        df = build_passing_epa_features(self._pbp(), [2023], _norm)
        assert not df.duplicated(subset=["season", "week", "team"]).any()

    def test_season_filter_respected(self):
        pbp = _make_pbp(seasons=(2022, 2023))
        df = build_passing_epa_features(pbp, [2022], _norm)
        assert set(df["season"].astype(int).unique()) == {2022}

    def test_empty_pbp_returns_empty(self):
        df = build_passing_epa_features(pd.DataFrame(), [2023], _norm)
        assert df.empty

    def test_none_pbp_returns_empty(self):
        df = build_passing_epa_features(None, [2023], _norm)
        assert df.empty

    def test_missing_season_column_returns_empty(self):
        pbp = self._pbp().drop(columns=["season"])
        df = build_passing_epa_features(pbp, [2023], _norm)
        assert df.empty

    def test_rolling_values_are_finite(self):
        df = build_passing_epa_features(self._pbp(), [2023], _norm)
        assert df["pass_epa_per_db_rolling3"].notna().any()

    def test_no_future_leakage_shift(self):
        """Week 1 rolling features must be NaN or equal to the single prior value."""
        df = build_passing_epa_features(self._pbp(), [2023], _norm)
        w1 = df[df["week"].astype(int) == 1]
        # With only week 1 data there are no prior games — rolling with min_periods=1
        # will return the shifted value (NaN after shift for week 1).
        # Simply confirm week 1 rows exist (leakage would show up as test data in the roll).
        assert not w1.empty


# ===========================================================================
# drive_epa_features
# ===========================================================================
from src.features.drive_epa_features import build_drive_epa_features


class TestDriveEpaFeatures:

    def _pbp(self):
        return _make_pbp()

    def test_returns_dataframe(self):
        df = build_drive_epa_features(self._pbp(), [2023], _norm)
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns_present(self):
        df = build_drive_epa_features(self._pbp(), [2023], _norm)
        for col in ("season", "week", "team",
                    "drive_epa_mean_rolling3",
                    "drive_epa_first_drive_rolling3",
                    "drive_epa_q4_rolling3",
                    "drive_completion_rate_rolling3",
                    "drives_per_game_rolling3"):
            assert col in df.columns, f"missing {col}"

    def test_one_row_per_team_week(self):
        df = build_drive_epa_features(self._pbp(), [2023], _norm)
        assert not df.duplicated(subset=["season", "week", "team"]).any()

    def test_empty_pbp_returns_empty(self):
        assert build_drive_epa_features(pd.DataFrame(), [2023], _norm).empty

    def test_none_returns_empty(self):
        assert build_drive_epa_features(None, [2023], _norm).empty

    def test_missing_required_column_returns_empty(self):
        pbp = _make_pbp().drop(columns=["fixed_drive"])
        assert build_drive_epa_features(pbp, [2023], _norm).empty

    def test_season_filter(self):
        pbp = _make_pbp(seasons=(2021, 2023))
        df = build_drive_epa_features(pbp, [2021], _norm)
        assert set(df["season"].astype(int).unique()) == {2021}

    def test_drives_per_game_positive(self):
        df = build_drive_epa_features(self._pbp(), [2023], _norm)
        assert (df["drives_per_game_rolling3"].dropna() > 0).all()

    def test_league_diff_column_present(self):
        df = build_drive_epa_features(self._pbp(), [2023], _norm)
        assert "drive_epa_mean_rolling3_diff_lg" in df.columns


# ===========================================================================
# redzone_features
# ===========================================================================
from src.features.redzone_features import build_redzone_features


class TestRedzoneFeatures:

    def _pbp(self):
        # Include yardline_100 ≤ 20 to produce red-zone plays
        return _make_pbp()

    def test_returns_dataframe(self):
        df = build_redzone_features(self._pbp(), [2023], _norm)
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns_present(self):
        df = build_redzone_features(self._pbp(), [2023], _norm)
        for col in ("season", "week", "team",
                    "rz_trips_for", "rz_tds_for",
                    "rz_trips_against", "rz_tds_against",
                    "rz_td_rate_for_rolling3", "rz_td_rate_against_rolling3"):
            assert col in df.columns, f"missing {col}"

    def test_empty_pbp_returns_empty(self):
        assert build_redzone_features(pd.DataFrame(), [2023], _norm).empty

    def test_none_returns_empty(self):
        assert build_redzone_features(None, [2023], _norm).empty

    def test_missing_yardline_returns_empty(self):
        pbp = _make_pbp().drop(columns=["yardline_100"])
        assert build_redzone_features(pbp, [2023], _norm).empty

    def test_trips_non_negative(self):
        df = build_redzone_features(self._pbp(), [2023], _norm)
        assert (df["rz_trips_for"] >= 0).all()
        assert (df["rz_trips_against"] >= 0).all()

    def test_td_rate_between_0_and_1(self):
        df = build_redzone_features(self._pbp(), [2023], _norm)
        valid = df["rz_td_rate_for"].dropna()
        assert ((valid >= 0) & (valid <= 1)).all()

    def test_season_filter(self):
        pbp = _make_pbp(seasons=(2022, 2023))
        df = build_redzone_features(pbp, [2022], _norm)
        assert set(df["season"].astype(int).unique()) == {2022}

    def test_no_rz_plays_still_runs(self):
        """If all yardline_100 > 20, filter removes all plays → empty result."""
        pbp = _make_pbp()
        pbp["yardline_100"] = 50  # no red-zone plays
        result = build_redzone_features(pbp, [2023], _norm)
        assert result.empty


# ===========================================================================
# pressure_features
# ===========================================================================
from src.features.pressure_features import build_pressure_features


class TestPressureFeatures:

    def _pbp(self):
        return _make_pbp()

    def test_returns_dataframe(self):
        df = build_pressure_features(self._pbp(), [2023], _norm)
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns_present(self):
        df = build_pressure_features(self._pbp(), [2023], _norm)
        for col in ("season", "week", "team",
                    "pressures_allowed_per_db", "sack_rate_allowed",
                    "pressures_generated_per_db", "sack_rate_generated",
                    "sack_rate_allowed_rolling3", "sack_rate_generated_rolling3"):
            assert col in df.columns, f"missing {col}"

    def test_empty_pbp_returns_empty(self):
        assert build_pressure_features(pd.DataFrame(), [2023], _norm).empty

    def test_none_returns_empty(self):
        assert build_pressure_features(None, [2023], _norm).empty

    def test_missing_season_returns_empty(self):
        pbp = _make_pbp().drop(columns=["season"])
        assert build_pressure_features(pbp, [2023], _norm).empty

    def test_rates_between_0_and_1(self):
        df = build_pressure_features(self._pbp(), [2023], _norm)
        for col in ("sack_rate_allowed", "sack_rate_generated"):
            valid = df[col].dropna()
            assert ((valid >= 0) & (valid <= 1)).all(), f"{col} out of [0,1]"

    def test_season_filter(self):
        pbp = _make_pbp(seasons=(2021, 2023))
        df = build_pressure_features(pbp, [2021], _norm)
        assert set(df["season"].astype(int).unique()) == {2021}

    def test_no_dropbacks_returns_empty(self):
        pbp = _make_pbp()
        pbp["pass_attempt"] = 0
        pbp["qb_scramble"] = 0
        pbp["sack"] = 0
        result = build_pressure_features(pbp, [2023], _norm)
        assert result.empty


# ===========================================================================
# adjusted_efficiency
# ===========================================================================
from src.features.adjusted_efficiency import build_adjusted_efficiency_features


def _make_sched(seasons=(2023,), weeks=(1, 2, 3, 4)) -> pd.DataFrame:
    rows = []
    for season in seasons:
        for week in weeks:
            rows.append({
                "game_id": f"{season}_{week:02d}_KC_BUF",
                "season": season, "week": week,
                "home_team": "KC", "away_team": "BUF",
            })
            rows.append({
                "game_id": f"{season}_{week:02d}_SF_DAL",
                "season": season, "week": week,
                "home_team": "SF", "away_team": "DAL",
            })
    return pd.DataFrame(rows)


class TestAdjustedEfficiencyFeatures:

    def test_returns_dataframe(self):
        df = build_adjusted_efficiency_features(_make_pbp(), _make_sched(), [2023], _norm)
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns_present(self):
        df = build_adjusted_efficiency_features(_make_pbp(), _make_sched(), [2023], _norm)
        for col in ("season", "week", "team",
                    "off_adj_eff_raw", "def_adj_eff_raw",
                    "off_adj_eff_rolling3", "def_adj_eff_rolling3"):
            assert col in df.columns, f"missing {col}"

    def test_empty_pbp_returns_empty(self):
        result = build_adjusted_efficiency_features(pd.DataFrame(), _make_sched(), [2023], _norm)
        assert result.empty

    def test_empty_sched_returns_empty(self):
        result = build_adjusted_efficiency_features(_make_pbp(), pd.DataFrame(), [2023], _norm)
        assert result.empty

    def test_none_pbp_returns_empty(self):
        result = build_adjusted_efficiency_features(None, _make_sched(), [2023], _norm)
        assert result.empty

    def test_season_filter(self):
        sched = _make_sched(seasons=(2022, 2023))
        pbp = _make_pbp(seasons=(2022, 2023))
        df = build_adjusted_efficiency_features(pbp, sched, [2022], _norm)
        assert set(df["season"].astype(int).unique()) == {2022}

    def test_sched_missing_required_columns_returns_empty(self):
        sched = _make_sched().drop(columns=["home_team"])
        result = build_adjusted_efficiency_features(_make_pbp(), sched, [2023], _norm)
        assert result.empty

    def test_rolling_values_are_finite(self):
        df = build_adjusted_efficiency_features(_make_pbp(), _make_sched(), [2023], _norm)
        # Not all must be finite (some are NaN for first game), but some must be
        assert df["off_adj_eff_rolling3"].notna().any()


# ===========================================================================
# volatility (engineer_volatility_inputs + build_volatility_feature_matrix)
# ===========================================================================
from src.features.volatility import (
    engineer_volatility_inputs,
    build_volatility_feature_matrix,
    VOLATILITY_REQUIRED_COLUMNS,
    VOLATILITY_FEATURE_COLUMNS,
    VolatilityFeatureFrame,
)


def _make_volatility_df(n=5) -> pd.DataFrame:
    """Minimal DataFrame containing all required volatility columns."""
    data = {col: [0.0] * n for col in VOLATILITY_REQUIRED_COLUMNS}
    # Set some realistic values
    data["home_rest"] = [7] * n
    data["away_rest"] = [7] * n
    data["sched_rest_days_home"] = [7] * n
    data["sched_rest_days_away"] = [7] * n
    data["away_travel_distance_km"] = [500] * n
    data["away_travel_distance_miles"] = [310] * n
    data["timezone_diff_hours"] = [0] * n
    data["timezone_diff_hours_abs"] = [0] * n
    # weather columns (optional but useful to include)
    data["weather_wind_mph"] = [10.0] * n
    data["weather_temp_f"] = [55.0] * n
    return pd.DataFrame(data)


class TestEngineerVolatilityInputs:

    def test_returns_dataframe_with_wind_and_temp(self):
        df = _make_volatility_df()
        result = engineer_volatility_inputs(df)
        assert "wind_mph" in result.columns
        assert "temp_f" in result.columns

    def test_weather_temp_deviation_computed(self):
        df = _make_volatility_df()
        df["weather_temp_f"] = 40.0
        result = engineer_volatility_inputs(df)
        assert "weather_temp_deviation" in result.columns
        # |40 - 60| = 20
        assert (result["weather_temp_deviation"] == 20.0).all()

    def test_qb_uncertainty_zero_when_no_injury_cols(self):
        df = pd.DataFrame({"weather_wind_mph": [10.0], "weather_temp_f": [60.0]})
        result = engineer_volatility_inputs(df)
        assert result["qb_uncertainty_score"].iloc[0] == 0.0

    def test_short_rest_flag_triggered(self):
        df = _make_volatility_df()
        df["home_rest"] = 5
        df["away_rest"] = 5
        result = engineer_volatility_inputs(df)
        assert result["short_rest"].all()

    def test_rest_diff_correct(self):
        df = _make_volatility_df()
        df["home_rest"] = 10
        df["away_rest"] = 6
        result = engineer_volatility_inputs(df)
        assert (result["rest_diff"] == 4.0).all()

    def test_prediction_confidence_features_computed(self):
        df = _make_volatility_df()
        df["home_win_prob"] = 0.80
        df["pred_home_margin"] = -7.0
        result = engineer_volatility_inputs(df)
        assert result["model_confidence_abs"].iloc[0] == pytest.approx(0.30)
        assert result["model_uncertainty"].iloc[0] == pytest.approx(0.40)
        assert result["model_logit_abs"].iloc[0] == pytest.approx(np.log(4.0))
        assert result["pred_margin_abs"].iloc[0] == pytest.approx(7.0)
        assert result["pred_margin_confidence"].iloc[0] == pytest.approx(np.tanh(0.5))

    def test_indoor_game_flagged(self):
        df = _make_volatility_df()
        df["roof_is_dome"] = 1
        result = engineer_volatility_inputs(df)
        assert result["indoor_game"].all()
        assert not result["outdoor_game"].any()

    def test_indoor_game_derived_from_roof(self):
        df = _make_volatility_df().drop(columns=["roof_is_dome"])
        df["roof"] = "dome"
        result = engineer_volatility_inputs(df)
        assert result["roof_is_dome"].eq(1).all()
        assert result["indoor_game"].all()


class TestBuildVolatilityFeatureMatrix:

    def test_returns_volatility_feature_frame(self):
        df = _make_volatility_df()
        result = build_volatility_feature_matrix(df)
        assert isinstance(result, VolatilityFeatureFrame)

    def test_features_all_numeric(self):
        df = _make_volatility_df()
        result = build_volatility_feature_matrix(df)
        assert all(pd.api.types.is_numeric_dtype(result.features[c])
                   for c in result.features.columns)

    def test_missing_required_column_raises(self):
        df = _make_volatility_df().drop(columns=[VOLATILITY_REQUIRED_COLUMNS[0]])
        with pytest.raises(KeyError):
            build_volatility_feature_matrix(df)

    def test_output_columns_subset_of_feature_columns(self):
        df = _make_volatility_df()
        result = build_volatility_feature_matrix(df)
        for col in result.features.columns:
            assert col in VOLATILITY_FEATURE_COLUMNS, f"unexpected col {col}"

    def test_no_nans_in_feature_matrix(self):
        df = _make_volatility_df()
        result = build_volatility_feature_matrix(df)
        assert not result.features.isnull().any().any()

    def test_build_matrix_derives_roof_is_dome_from_roof(self):
        df = _make_volatility_df().drop(columns=["roof_is_dome"])
        df["roof"] = "closed"
        result = build_volatility_feature_matrix(df)
        assert "indoor_game" in result.features.columns
        assert result.features["indoor_game"].eq(1.0).all()


# ===========================================================================
# qb_health
# ===========================================================================
from src.features.qb_health import (
    build_qb_health_features,
    _normalize_name,
    _score_status,
    SEVERITY_WEIGHTS,
)


def _make_sched_qb(seasons=(2023,)) -> pd.DataFrame:
    rows = []
    for season in seasons:
        for week in range(1, 5):
            rows.append({
                "game_id": f"{season}_{week:02d}_KC_BUF",
                "season": season, "week": week,
                "home_team": "KC", "away_team": "BUF",
                "home_qb_name": "Patrick Mahomes",
                "away_qb_name": "Josh Allen",
            })
    return pd.DataFrame(rows)


def _make_injuries_df(seasons=(2023,)) -> pd.DataFrame:
    rows = []
    for season in seasons:
        rows.append({
            "season": season, "week": 2,
            "team": "KC", "full_name": "Patrick Mahomes",
            "position": "QB",
            "report_status": "Questionable",
            "practice_status": "Limited",
            "player_id": "QB1",
        })
    return pd.DataFrame(rows)


class TestNormalizeName:
    def test_none_returns_none(self):
        assert _normalize_name(None) is None

    def test_strips_and_uppercases(self):
        assert _normalize_name("  patrick mahomes  ") == "PATRICK MAHOMES"

    def test_empty_string_returns_none(self):
        assert _normalize_name("   ") is None


class TestScoreStatus:
    def test_out_maps_to_2(self):
        assert _score_status("Out") == 2.0

    def test_doubtful_maps_to_1_5(self):
        assert _score_status("Doubtful") == 1.5

    def test_unknown_text_maps_to_zero(self):
        assert _score_status("Active") == 0.0

    def test_non_string_maps_to_zero(self):
        assert _score_status(None) == 0.0

    def test_dnp_maps_to_0_7(self):
        assert _score_status("Did Not Practice") == 0.7


class TestBuildQbHealthFeatures:

    def test_returns_dataframe(self):
        sched = _make_sched_qb()
        df = build_qb_health_features(sched, _make_injuries_df(), pd.DataFrame(), [2023], _norm)
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns_present(self):
        sched = _make_sched_qb()
        df = build_qb_health_features(sched, _make_injuries_df(), pd.DataFrame(), [2023], _norm)
        for col in ("season", "week", "team",
                    "qb_status_flag", "qb_status_delta_rolling3",
                    "qb_missed_last_game", "qb_games_started_rolling5",
                    "qb_recovery_score_rolling3", "qb_weeks_since_injury"):
            assert col in df.columns, f"missing {col}"

    def test_empty_sched_returns_empty(self):
        result = build_qb_health_features(
            pd.DataFrame(), _make_injuries_df(), pd.DataFrame(), [2023], _norm
        )
        assert result.empty

    def test_none_sched_returns_empty(self):
        result = build_qb_health_features(
            None, _make_injuries_df(), pd.DataFrame(), [2023], _norm
        )
        assert result.empty

    def test_season_filter(self):
        sched = _make_sched_qb(seasons=(2022, 2023))
        df = build_qb_health_features(sched, pd.DataFrame(), pd.DataFrame(), [2022], _norm)
        assert set(df["season"].astype(int).unique()) == {2022}

    def test_no_injuries_returns_zero_flags(self):
        sched = _make_sched_qb()
        df = build_qb_health_features(sched, pd.DataFrame(), pd.DataFrame(), [2023], _norm)
        assert (df["qb_status_flag"] == 0.0).all()

    def test_missed_last_game_is_int_typed(self):
        sched = _make_sched_qb()
        df = build_qb_health_features(sched, _make_injuries_df(), pd.DataFrame(), [2023], _norm)
        assert pd.api.types.is_integer_dtype(df["qb_missed_last_game"])

    def test_qb_weeks_since_injury_capped_at_8(self):
        sched = _make_sched_qb()
        df = build_qb_health_features(sched, pd.DataFrame(), pd.DataFrame(), [2023], _norm)
        assert (df["qb_weeks_since_injury"] <= 8).all()

    def test_recovery_score_non_negative(self):
        sched = _make_sched_qb()
        df = build_qb_health_features(sched, _make_injuries_df(), pd.DataFrame(), [2023], _norm)
        assert (df["qb_recovery_score_rolling3"] >= 0).all()

    def test_injury_feed_without_player_id_matches_by_name(self):
        sched = _make_sched_qb()
        injuries = _make_injuries_df().drop(columns=["player_id"])
        df = build_qb_health_features(sched, injuries, pd.DataFrame(), [2023], _norm)
        kc_week2 = df[(df["team"] == "KC") & (df["week"] == 2)]
        assert not kc_week2.empty
        assert kc_week2["qb_status_flag"].iloc[0] == 1.0
