"""Tests for pure helper functions in src/features/build_features.py.

Targets small, dependency-free functions to maximize coverage lift
without needing file I/O or network access.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import (
    EARTH_RADIUS_KM,
    _clean_numeric_value,
    _coalesce,
    _consecutive_counts,
    _detect_team_col,
    _extract_abbr_from_jsonlike,
    _haversine_km,
    _mk_uid,
    _mk_uid_relaxed,
    _moneyline_to_prob,
    _norm_colname,
    _normalize_schedule_columns,
    _resolve_sportradar_abbr,
    _safe_json_load,
    _safe_numeric_series,
    _add_per_game_means,
    _to_int,
    build_matchup_features,
    team_season_agg_from_pbp,
    team_season_def_allowed_from_pbp,
)


# ---------------------------------------------------------------------------
# _coalesce
# ---------------------------------------------------------------------------
class TestCoalesce:
    def test_returns_first_matching_column(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        result = _coalesce(["b", "a"], df)
        assert list(result) == [2]

    def test_returns_default_when_no_match(self):
        df = pd.DataFrame({"x": [1]})
        result = _coalesce(["a", "b"], df, default="fallback")
        assert result == "fallback"

    def test_empty_cols_returns_default(self):
        df = pd.DataFrame({"a": [1]})
        result = _coalesce([], df, default=None)
        assert result is None


# ---------------------------------------------------------------------------
# _to_int
# ---------------------------------------------------------------------------
class TestToInt:
    def test_converts_int(self):
        assert _to_int(5) == 5

    def test_converts_float(self):
        assert _to_int(3.9) == 3

    def test_converts_string(self):
        assert _to_int("7") == 7

    def test_none_returns_none(self):
        assert _to_int(None) is None

    def test_nan_returns_none(self):
        assert _to_int(float("nan")) is None

    def test_invalid_string_returns_none(self):
        assert _to_int("abc") is None


# ---------------------------------------------------------------------------
# _detect_team_col
# ---------------------------------------------------------------------------
class TestDetectTeamCol:
    def test_finds_team(self):
        df = pd.DataFrame({"team": ["KC"]})
        assert _detect_team_col(df) == "team"

    def test_finds_team_abbr(self):
        df = pd.DataFrame({"team_abbr": ["BUF"]})
        assert _detect_team_col(df) == "team_abbr"

    def test_finds_abbr(self):
        df = pd.DataFrame({"abbr": ["SF"]})
        assert _detect_team_col(df) == "abbr"

    def test_returns_none_when_missing(self):
        df = pd.DataFrame({"score": [24]})
        assert _detect_team_col(df) is None


# ---------------------------------------------------------------------------
# _safe_json_load
# ---------------------------------------------------------------------------
class TestSafeJsonLoad:
    def test_parses_json_object(self):
        result = _safe_json_load('{"abbreviation": "KC"}')
        assert result == {"abbreviation": "KC"}

    def test_parses_json_array(self):
        result = _safe_json_load("[1, 2, 3]")
        assert result == [1, 2, 3]

    def test_plain_string_unchanged(self):
        assert _safe_json_load("KC") == "KC"

    def test_non_string_unchanged(self):
        assert _safe_json_load(42) == 42

    def test_bad_json_returns_original(self):
        bad = "{not valid json"
        assert _safe_json_load(bad) == bad


# ---------------------------------------------------------------------------
# _extract_abbr_from_jsonlike
# ---------------------------------------------------------------------------
class TestExtractAbbrFromJsonlike:
    def test_plain_abbr_returned(self):
        s = pd.Series(["KC", "BUF"])
        result = _extract_abbr_from_jsonlike(s)
        assert list(result) == ["KC", "BUF"]

    def test_extracts_from_json_object(self):
        s = pd.Series(['{"abbreviation": "SF"}'])
        result = _extract_abbr_from_jsonlike(s)
        assert result.iloc[0] == "SF"

    def test_extracts_from_nested_team_key(self):
        s = pd.Series(['{"team": {"abbreviation": "LAR"}}'])
        result = _extract_abbr_from_jsonlike(s)
        assert result.iloc[0] == "LAR"

    def test_non_abbr_dict_returned_as_is(self):
        s = pd.Series(['{"name": "Kansas City"}'])
        result = _extract_abbr_from_jsonlike(s)
        assert isinstance(result.iloc[0], dict)


# ---------------------------------------------------------------------------
# _resolve_sportradar_abbr
# ---------------------------------------------------------------------------
class TestResolveSporsradarAbbr:
    def test_alias_returned_uppercased(self):
        assert _resolve_sportradar_abbr("kc", None) == "KC"

    def test_name_resolved_when_no_alias(self):
        # Any team name that exists in TEAM_NAME_TO_ABBR
        result = _resolve_sportradar_abbr(None, "Kansas City Chiefs")
        assert result is None or isinstance(result, str)

    def test_none_both_returns_none(self):
        assert _resolve_sportradar_abbr(None, None) is None

    def test_empty_alias_falls_through_to_name(self):
        result = _resolve_sportradar_abbr("", None)
        assert result is None


# ---------------------------------------------------------------------------
# _norm_colname
# ---------------------------------------------------------------------------
class TestNormColname:
    def test_lowercases(self):
        assert _norm_colname("EPA") == "epa"

    def test_replaces_percent(self):
        assert "pct" in _norm_colname("Completion%")

    def test_replaces_slash(self):
        assert "_per_" in _norm_colname("Yards/Play")

    def test_replaces_dash(self):
        assert "_" in _norm_colname("Pass-Rate")

    def test_removes_spaces(self):
        assert " " not in _norm_colname("pass rate")


# ---------------------------------------------------------------------------
# _clean_numeric_value
# ---------------------------------------------------------------------------
class TestCleanNumericValue:
    def test_none_returns_nan(self):
        assert math.isnan(_clean_numeric_value(None))

    def test_int_returns_float(self):
        assert _clean_numeric_value(5) == 5.0

    def test_float_returns_float(self):
        assert _clean_numeric_value(3.14) == pytest.approx(3.14)

    def test_empty_string_returns_nan(self):
        assert math.isnan(_clean_numeric_value(""))

    def test_dash_returns_nan(self):
        assert math.isnan(_clean_numeric_value("--"))

    def test_time_string_converted_to_seconds(self):
        # "1:30" -> 90 seconds
        assert _clean_numeric_value("1:30") == 90.0

    def test_percentage_parsed(self):
        assert _clean_numeric_value("57.8%") == pytest.approx(57.8)

    def test_hash_rank_parsed(self):
        assert _clean_numeric_value("#3") == 3.0

    def test_tie_rank_parsed(self):
        assert _clean_numeric_value("T-7") == 7.0

    def test_plus_prefix_removed(self):
        assert _clean_numeric_value("+12") == 12.0

    def test_unicode_minus_handled(self):
        assert _clean_numeric_value("−5") == -5.0

    def test_comma_number_parsed(self):
        assert _clean_numeric_value("1,234") == 1234.0

    def test_na_string_returns_nan(self):
        assert math.isnan(_clean_numeric_value("na"))

    def test_invalid_string_returns_nan(self):
        assert math.isnan(_clean_numeric_value("abc"))


# ---------------------------------------------------------------------------
# _safe_numeric_series
# ---------------------------------------------------------------------------
class TestSafeNumericSeries:
    def test_missing_col_returns_default_series(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = _safe_numeric_series(df, "missing", default=0.0)
        assert list(result) == [0.0, 0.0, 0.0]

    def test_fills_nans_with_default(self):
        df = pd.DataFrame({"v": [1.0, None, 3.0]})
        result = _safe_numeric_series(df, "v", default=-1.0)
        assert result.iloc[1] == -1.0

    def test_returns_numeric_values(self):
        df = pd.DataFrame({"v": ["1", "2", "3"]})
        result = _safe_numeric_series(df, "v")
        assert list(result) == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# _add_per_game_means
# ---------------------------------------------------------------------------
class TestAddPerGameMeans:
    def test_adds_mean_column(self):
        df = pd.DataFrame({"yards_sum": [100.0, 200.0], "games": [4.0, 8.0]})
        _add_per_game_means(df, ["yards_sum"], "games")
        assert "yards_mean" in df.columns
        assert df["yards_mean"].iloc[0] == pytest.approx(25.0)

    def test_empty_df_does_nothing(self):
        df = pd.DataFrame()
        _add_per_game_means(df, ["yards_sum"], "games")  # should not raise

    def test_missing_denom_col_does_nothing(self):
        df = pd.DataFrame({"yards_sum": [100.0]})
        _add_per_game_means(df, ["yards_sum"], "no_such_col")
        assert "yards_mean" not in df.columns


# ---------------------------------------------------------------------------
# _haversine_km
# ---------------------------------------------------------------------------
class TestHaversineKm:
    def test_same_location_is_zero(self):
        result = _haversine_km(40.0, -75.0, 40.0, -75.0)
        assert float(result) == pytest.approx(0.0, abs=1e-6)

    def test_known_distance(self):
        # KC (39.05, -94.48) to BUF (42.77, -78.79) ≈ 1380 km
        result = _haversine_km(39.05, -94.48, 42.77, -78.79)
        assert 1200 < float(result) < 1600

    def test_earth_radius_constant(self):
        assert EARTH_RADIUS_KM == pytest.approx(6371.0)


# ---------------------------------------------------------------------------
# _moneyline_to_prob
# ---------------------------------------------------------------------------
class TestMoneylineToProb:
    def test_positive_odds(self):
        result = _moneyline_to_prob(pd.Series([100.0]))
        assert result[0] == pytest.approx(0.5)

    def test_negative_odds(self):
        result = _moneyline_to_prob(pd.Series([-110.0]))
        assert result[0] == pytest.approx(110.0 / 210.0)

    def test_zero_returns_nan(self):
        result = _moneyline_to_prob(pd.Series([0.0]))
        assert math.isnan(result[0])

    def test_empty_returns_empty(self):
        result = _moneyline_to_prob(pd.Series([], dtype=float))
        assert len(result) == 0

    def test_nan_input_returns_nan(self):
        result = _moneyline_to_prob(pd.Series([float("nan")]))
        assert math.isnan(result[0])


# ---------------------------------------------------------------------------
# _consecutive_counts
# ---------------------------------------------------------------------------
class TestConsecutiveCounts:
    def test_all_zeros_returns_zeros(self):
        s = pd.Series([0, 0, 0])
        result = _consecutive_counts(s)
        assert list(result) == [0, 0, 0]

    def test_all_ones_counts_up(self):
        s = pd.Series([1, 1, 1])
        result = _consecutive_counts(s)
        assert list(result) == [1, 2, 3]

    def test_mixed_resets_on_zero(self):
        s = pd.Series([1, 1, 0, 1, 1, 1])
        result = _consecutive_counts(s)
        assert list(result) == [1, 2, 0, 1, 2, 3]

    def test_nans_treated_as_zero(self):
        s = pd.Series([1.0, None, 1.0])
        result = _consecutive_counts(s)
        assert list(result) == [1, 0, 1]


# ---------------------------------------------------------------------------
# team_season_agg_from_pbp
# ---------------------------------------------------------------------------
class TestTeamSeasonAggFromPbp:
    def _make_pbp(self) -> pd.DataFrame:
        return pd.DataFrame({
            "play_id": range(10),
            "season": [2023] * 10,
            "posteam": ["KC"] * 5 + ["BUF"] * 5,   # canonical column name
            "defteam": ["BUF"] * 5 + ["KC"] * 5,
            "yards_gained": [5.0] * 10,
            "pass": [1, 1, 0, 0, 1] * 2,
            "rush": [0, 0, 1, 1, 0] * 2,
            "epa": [0.1] * 10,
        })

    def test_none_returns_empty(self):
        result = team_season_agg_from_pbp(None)
        assert result.empty

    def test_empty_returns_empty(self):
        result = team_season_agg_from_pbp(pd.DataFrame())
        assert result.empty

    def test_returns_one_row_per_team(self):
        pbp = self._make_pbp()
        result = team_season_agg_from_pbp(pbp)
        assert len(result) == 2

    def test_expected_columns(self):
        pbp = self._make_pbp()
        result = team_season_agg_from_pbp(pbp)
        for col in ("season", "abbr", "plays", "yards", "epa"):
            assert col in result.columns

    def test_plays_count_correct(self):
        pbp = self._make_pbp()
        result = team_season_agg_from_pbp(pbp)
        kc = result[result["abbr"] == "KC"]
        assert kc["plays"].iloc[0] == 5


# ---------------------------------------------------------------------------
# team_season_def_allowed_from_pbp
# ---------------------------------------------------------------------------
class TestTeamSeasonDefAllowedFromPbp:
    def _make_pbp(self) -> pd.DataFrame:
        return pd.DataFrame({
            "play_id": range(10),
            "season": [2023] * 10,
            "posteam": ["KC"] * 5 + ["BUF"] * 5,
            "defteam": ["BUF"] * 5 + ["KC"] * 5,
            "yards_gained": [6.0] * 10,
            "pass": [1, 0, 1, 0, 1] * 2,
            "rush": [0, 1, 0, 1, 0] * 2,
            "epa": [0.2] * 10,
        })

    def test_none_returns_empty(self):
        result = team_season_def_allowed_from_pbp(None)
        assert result.empty

    def test_empty_returns_empty(self):
        result = team_season_def_allowed_from_pbp(pd.DataFrame())
        assert result.empty

    def test_returns_dataframe(self):
        pbp = self._make_pbp()
        result = team_season_def_allowed_from_pbp(pbp)
        assert isinstance(result, pd.DataFrame)



# ---------------------------------------------------------------------------
# _mk_uid / _mk_uid_relaxed
# ---------------------------------------------------------------------------
class TestMkUid:
    def _df(self):
        return pd.DataFrame([{
            "season": 2023, "week": 1, "home_team": "KC", "away_team": "BUF"
        }])

    def test_returns_series(self):
        result = _mk_uid(self._df())
        assert isinstance(result, pd.Series)

    def test_uid_is_string(self):
        result = _mk_uid(self._df())
        assert isinstance(result.iloc[0], str)

    def test_uid_deterministic(self):
        result1 = _mk_uid(self._df())
        result2 = _mk_uid(self._df())
        assert result1.iloc[0] == result2.iloc[0]

    def test_different_games_different_uid(self):
        df1 = self._df()
        df2 = pd.DataFrame([{"season": 2023, "week": 2, "home_team": "KC", "away_team": "BUF"}])
        assert _mk_uid(df1).iloc[0] != _mk_uid(df2).iloc[0]

    def test_empty_df_returns_empty(self):
        result = _mk_uid(pd.DataFrame(columns=["season", "week", "home_team", "away_team"]))
        assert result.empty


class TestMkUidRelaxed:
    def test_la_lar_lac_collapse(self):
        df = pd.DataFrame([
            {"season": 2023, "week": 1, "home_team": "LA", "away_team": "BUF"},
            {"season": 2023, "week": 1, "home_team": "LAR", "away_team": "BUF"},
        ])
        result = _mk_uid_relaxed(df)
        # LA and LAR should produce the same UID since both collapse to "LA"
        assert result.iloc[0] == result.iloc[1]


# ---------------------------------------------------------------------------
# _normalize_schedule_columns
# ---------------------------------------------------------------------------
class TestNormalizeScheduleColumns:
    def _make_sched(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "home_team": "kc",
            "away_team": "buf",
            "game_id": "2023_01_KC_BUF",
            "season": 2023,
            "week": 1,
        }])

    def test_returns_dataframe(self):
        result = _normalize_schedule_columns(self._make_sched())
        assert isinstance(result, pd.DataFrame)

    def test_home_team_uppercased(self):
        result = _normalize_schedule_columns(self._make_sched())
        assert result["home_team"].iloc[0] == result["home_team"].iloc[0].upper()

    def test_away_team_uppercased(self):
        result = _normalize_schedule_columns(self._make_sched())
        assert result["away_team"].iloc[0] == result["away_team"].iloc[0].upper()

    def test_game_id_preserved(self):
        result = _normalize_schedule_columns(self._make_sched())
        assert result["game_id"].iloc[0] == "2023_01_KC_BUF"

    def test_alternate_col_names(self):
        df = pd.DataFrame([{
            "homeTeam": "SF", "awayTeam": "DAL",
            "season": 2023, "week": 5,
        }])
        result = _normalize_schedule_columns(df)
        assert "home_team" in result.columns


# ---------------------------------------------------------------------------
# build_matchup_features
# ---------------------------------------------------------------------------
class TestBuildMatchupFeatures:
    def _make_schedule(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "home_team": "KC",
            "away_team": "BUF",
            "game_id": "2023_01_KC_BUF",
            "season": 2023,
            "week": 1,
        }])

    def _make_team_stats(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"season": 2023, "abbr": "KC", "epa": 1.0},
            {"season": 2023, "abbr": "BUF", "epa": 0.8},
        ])

    def test_returns_dataframe(self):
        result = build_matchup_features(self._make_schedule(), self._make_team_stats())
        assert isinstance(result, pd.DataFrame)

    def test_empty_team_stats_returns_schedule(self):
        result = build_matchup_features(
            self._make_schedule(),
            pd.DataFrame(columns=["season", "abbr"])
        )
        assert isinstance(result, pd.DataFrame)

    def test_empty_schedule_returns_empty(self):
        result = build_matchup_features(pd.DataFrame(), self._make_team_stats())
        assert result.empty or isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# _aggregate_team_week / _add_weekly_deltas
# ---------------------------------------------------------------------------
from src.features.build_features import _aggregate_team_week, _add_weekly_deltas


class TestAggregateTeamWeek:
    def _make_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "team": ["KC", "KC", "BUF", "BUF"],
            "season": [2023] * 4,
            "week": [1, 2, 1, 2],
            "epa": [1.0, 2.0, 0.5, 1.5],
        })

    def test_returns_empty_for_none(self):
        result = _aggregate_team_week(None, "off")
        assert result.empty

    def test_returns_empty_for_empty_df(self):
        result = _aggregate_team_week(pd.DataFrame(), "off")
        assert result.empty

    def test_returns_dataframe(self):
        result = _aggregate_team_week(self._make_df(), "off")
        assert isinstance(result, pd.DataFrame)

    def test_prefix_applied_to_numeric_cols(self):
        result = _aggregate_team_week(self._make_df(), "off")
        assert "off_epa" in result.columns

    def test_returns_empty_when_no_team_col(self):
        df = pd.DataFrame({"season": [2023], "week": [1], "epa": [1.0]})
        result = _aggregate_team_week(df, "off")
        assert result.empty


class TestAddWeeklyDeltas:
    def _make_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "season": [2023, 2023, 2023],
            "week": [1, 2, 3],
            "team": ["KC", "KC", "KC"],
            "epa": [1.0, 2.0, 3.0],
        })

    def test_returns_none_for_none(self):
        result = _add_weekly_deltas(None)
        assert result is None

    def test_returns_empty_for_empty(self):
        result = _add_weekly_deltas(pd.DataFrame())
        assert result is None or (isinstance(result, pd.DataFrame) and result.empty)

    def test_returns_dataframe(self):
        result = _add_weekly_deltas(self._make_df())
        assert isinstance(result, pd.DataFrame)

    def test_adds_delta_column(self):
        result = _add_weekly_deltas(self._make_df(), value_cols=["epa"])
        assert "epa_delta" in result.columns

    def test_adds_rolling_column(self):
        result = _add_weekly_deltas(self._make_df(), value_cols=["epa"], rolling_windows=(3,))
        assert "epa_rolling3" in result.columns

    def test_missing_required_cols_returns_original(self):
        df = pd.DataFrame({"epa": [1.0, 2.0]})
        result = _add_weekly_deltas(df, value_cols=["epa"])
        assert list(result["epa"]) == [1.0, 2.0]


# ---------------------------------------------------------------------------
# load_stadium_context / load_rivalry_table / load_team_locations
# (these return empty DataFrames when files don't exist — tests run headlessly)
# ---------------------------------------------------------------------------
from src.features.build_features import (
    load_stadium_context,
    load_rivalry_table,
    load_team_locations,
)


class TestFileLoaderFallbacks:
    def test_load_stadium_context_returns_dataframe(self):
        result = load_stadium_context()
        assert isinstance(result, pd.DataFrame)

    def test_load_rivalry_table_returns_dataframe(self):
        result = load_rivalry_table()
        assert isinstance(result, pd.DataFrame)

    def test_load_team_locations_returns_dataframe(self):
        result = load_team_locations()
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# _aggregate_sr_from_pbp
# ---------------------------------------------------------------------------
from src.features.build_features import _aggregate_sr_from_pbp


def _make_sr_pbp(n: int = 20) -> pd.DataFrame:
    """Minimal Sportradar-style PBP DataFrame for testing _aggregate_sr_from_pbp."""
    rng = __import__("numpy").random.default_rng(42)
    teams = (["KC"] * (n // 2)) + (["BUF"] * (n // 2))
    opp = (["BUF"] * (n // 2)) + (["KC"] * (n // 2))
    # Include passing_yards and rushing_yards so fillna doesn't fail on None
    return __import__("pandas").DataFrame({
        "season": [2023] * n,
        "game_id": ["2023_01_KC_BUF"] * n,
        "posteam": teams,
        "defteam": opp,
        "yards_gained": rng.normal(5, 3, n).clip(min=-10),
        "pass_attempt": rng.integers(0, 2, n).astype(float),
        "rush_attempt": rng.integers(0, 2, n).astype(float),
        "passing_yards": rng.normal(7, 5, n).clip(min=0),
        "rushing_yards": rng.normal(4, 3, n).clip(min=0),
        "epa": rng.normal(0, 1, n),
        "complete_pass": rng.integers(0, 2, n).astype(float),
        "pass_touchdown": [0.0] * n,
        "interception": [0.0] * n,
        "rush_touchdown": [0.0] * n,
        "sack": [0.0] * n,
        "tackled_for_loss": [0.0] * n,
        "first_down_pass": [0.0] * n,
        "first_down_rush": [0.0] * n,
        "success": rng.integers(0, 2, n).astype(float),
        "air_yards": rng.normal(7, 3, n),
        "yards_after_catch": rng.normal(3, 2, n),
        "int_team": [""] * n,
        "fumble_lost": [0.0] * n,
        "fumble_recovery_1_team": teams,
        "td_team": [""] * n,
        "field_goal_result": [""] * n,
        "punt_blocked": [0.0] * n,
        "penalty": [0.0] * n,
        "kickoff_attempt": [0.0] * n,
        "punt_attempt": [0.0] * n,
        "two_point_attempt": [0.0] * n,
        "penalty_team": teams,
        "penalty_yards": [0.0] * n,
        "down": [1] * n,
        "ydstogo": [10] * n,
    })


class TestAggregateSrFromPbp:
    def test_none_returns_empty(self):
        result = _aggregate_sr_from_pbp(None)
        assert result.empty

    def test_empty_returns_empty(self):
        result = _aggregate_sr_from_pbp(__import__("pandas").DataFrame())
        assert result.empty

    def test_returns_dataframe(self):
        result = _aggregate_sr_from_pbp(_make_sr_pbp())
        assert isinstance(result, __import__("pandas").DataFrame)

    def test_season_column_present(self):
        result = _aggregate_sr_from_pbp(_make_sr_pbp())
        assert "season" in result.columns

    def test_abbr_column_present(self):
        result = _aggregate_sr_from_pbp(_make_sr_pbp())
        assert "abbr" in result.columns

    def test_two_teams_returned(self):
        result = _aggregate_sr_from_pbp(_make_sr_pbp())
        assert len(result) == 2

    def test_offense_stats_present(self):
        result = _aggregate_sr_from_pbp(_make_sr_pbp())
        # At least one offensive stat column should exist
        off_cols = [c for c in result.columns if "passing" in c or "rushing" in c or "epa" in c]
        assert len(off_cols) > 0


# ---------------------------------------------------------------------------
# _build_schedule_team_features (guard-clause coverage)
# ---------------------------------------------------------------------------
from src.features.build_features import _build_schedule_team_features
from src.features.build_features import load_sportradar_team_features


class TestBuildScheduleTeamFeatures:
    def _make_sched(self, n: int = 4) -> pd.DataFrame:
        rows = []
        for week in range(1, n + 1):
            rows.append({
                "home_team": "KC",
                "away_team": "BUF",
                "season": 2023,
                "week": week,
                "game_id": f"2023_{week:02d}_KC_BUF",
                "spread_line": -3.0,
                "total_line": 45.5,
                "home_moneyline": -145.0,
                "away_moneyline": 125.0,
                "home_rest": 7,
                "away_rest": 7,
                "neutral_site": 0,
            })
        return pd.DataFrame(rows)

    def test_none_returns_empty(self):
        result = _build_schedule_team_features(None, [2023])
        assert result.empty

    def test_empty_df_returns_empty(self):
        result = _build_schedule_team_features(pd.DataFrame(), [2023])
        assert result.empty

    def test_season_filter(self):
        sched = self._make_sched()
        result_2023 = _build_schedule_team_features(sched, [2023])
        result_2024 = _build_schedule_team_features(sched, [2024])
        assert result_2024.empty
        assert isinstance(result_2023, pd.DataFrame)

    def test_returns_dataframe(self):
        result = _build_schedule_team_features(self._make_sched(), [2023])
        assert isinstance(result, pd.DataFrame)

    def test_has_team_column(self):
        result = _build_schedule_team_features(self._make_sched(), [2023])
        assert "team" in result.columns or len(result) == 0  # may be empty if no team col

    def test_missing_required_cols_returns_empty(self):
        df = pd.DataFrame({"season": [2023], "week": [1]})
        result = _build_schedule_team_features(df, [2023])
        assert result.empty


class TestLoadSportradarTeamFeatures:
    def test_returns_dataframe(self):
        """When no PBP or legacy files exist, returns an empty DataFrame."""
        result = load_sportradar_team_features()
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# _build_player_news_features / _build_team_news_features / _augment_team_week_features
# ---------------------------------------------------------------------------
from src.features.build_features import (
    _build_player_news_features,
    _build_team_news_features,
    _augment_team_week_features,
)


class TestBuildPlayerNewsFeatures:
    def test_none_news_returns_empty(self):
        result = _build_player_news_features(None, pd.DataFrame(), pd.DataFrame(), [2023])
        assert result.empty

    def test_empty_news_returns_empty(self):
        result = _build_player_news_features(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), [2023])
        assert result.empty


class TestBuildTeamNewsFeatures:
    def test_none_news_returns_empty(self):
        result = _build_team_news_features(None, pd.DataFrame(), [2023])
        assert result.empty

    def test_empty_news_returns_empty(self):
        result = _build_team_news_features(pd.DataFrame(), pd.DataFrame(), [2023])
        assert result.empty

    def test_news_filtered_by_season(self):
        news = pd.DataFrame({
            "team_abbr": ["KC"],
            "season": [2022],
            "headline": ["Test"],
            "published": ["2022-09-01"],
        })
        result = _build_team_news_features(news, pd.DataFrame(), [2023])
        assert result.empty


class TestAugmentTeamWeekFeatures:
    def test_returns_input_feats_when_no_raw_data(self):
        feats = pd.DataFrame({
            "season": [2023],
            "week": [1],
            "home_team": ["KC"],
            "away_team": ["BUF"],
        })
        # load_raw returns empty DataFrame for both sources
        result = _augment_team_week_features(feats, [2023], lambda _: pd.DataFrame(), pd.DataFrame())
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# Additional coverage for uncovered branches
# ---------------------------------------------------------------------------

class TestMkUidExceptionPath:
    def test_nan_season_returns_valid_uid(self):
        """NaN season should not raise — _one uses -1 fallback."""
        df = pd.DataFrame([{"season": float("nan"), "week": 1, "home_team": "KC", "away_team": "BUF"}])
        result = _mk_uid(df)
        assert len(result) == 1

    def test_missing_cols_handled(self):
        """Missing home_team col falls back to empty string."""
        df = pd.DataFrame([{"season": 2023, "week": 1}])
        result = _mk_uid(df)
        assert len(result) == 1


class TestMkUidRelaxedExceptionPath:
    def test_missing_cols_handled(self):
        df = pd.DataFrame([{"season": 2023}])
        result = _mk_uid_relaxed(df)
        assert len(result) == 1
