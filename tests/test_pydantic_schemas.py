"""
Comprehensive tests for src/utils/pydantic_schemas.py

Covers:
  * All 16 schema models (5 NFLverse, 4 ESPN, 3 Sportradar, 3 Weather, 1 Yahoo)
  * Happy-path construction with minimal and maximal field sets
  * Field-level validators: season, week, team abbreviation, game_id format,
    probability bounds, score consistency, completion rate guard-rail, etc.
  * Cross-field model validators
  * Schema versioning registry (SCHEMA_VERSIONS / SCHEMA_MODELS)
  * validate_record() / get_schema_version() helpers
  * extra-field stripping (extra="ignore" config)
  * Coercion of numeric strings (Pydantic v2 strict=False default)
"""

from __future__ import annotations

from typing import Any, Dict

import pytest
from pydantic import ValidationError

from src.utils.pydantic_schemas import (
    # NFLverse
    NFLverseScheduleRecord,
    NFLversePlayRecord,
    NFLverseRosterRecord,
    NFLverseInjuryRecord,
    NFLversePlayerStatsRecord,
    # ESPN
    ESPNPlayerStatRecord,
    ESPNPlayerNewsRecord,
    ESPNTeamNewsRecord,
    ESPNTeamDefenseRecord,
    # Sportradar
    SportradarScheduleRecord,
    SportradarRosterRecord,
    SportradarGameStatsRecord,
    # Weather
    NOAAWeatherRecord,
    VisualCrossingWeatherRecord,
    GameWeatherRecord,
    # Yahoo
    YahooPlayerRecord,
    # Versioning / helpers
    SCHEMA_VERSIONS,
    SCHEMA_MODELS,
    get_schema_version,
    validate_record,
    _NFL_ABBRS,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

VALID_GAME_ID = "2024_01_KC_BUF"
VALID_SEASON = 2024
VALID_WEEK = 1


def _raises_validation(model_cls: type, data: Dict[str, Any]) -> ValidationError:
    """Assert that model_cls(**data) raises ValidationError and return it."""
    with pytest.raises(ValidationError) as exc_info:
        model_cls.model_validate(data)
    return exc_info.value


# ===========================================================================
# NFLverse — Schedule
# ===========================================================================

class TestNFLverseScheduleRecord:

    BASE: Dict[str, Any] = {
        "game_id": VALID_GAME_ID,
        "season": VALID_SEASON,
        "week": VALID_WEEK,
        "home_team": "KC",
        "away_team": "BUF",
    }

    def test_minimal_happy_path(self):
        rec = NFLverseScheduleRecord.model_validate(self.BASE)
        assert rec.game_id == VALID_GAME_ID
        assert rec.season == VALID_SEASON
        assert rec.week == VALID_WEEK
        assert rec.home_team == "KC"
        assert rec.away_team == "BUF"
        assert rec.home_score is None

    def test_full_fields(self):
        data = {**self.BASE, "home_score": 27.0, "away_score": 24.0,
                "game_type": "REG", "spread_line": -3.5, "total_line": 51.0}
        rec = NFLverseScheduleRecord.model_validate(data)
        assert rec.home_score == 27.0
        assert rec.spread_line == -3.5

    def test_extra_fields_ignored(self):
        data = {**self.BASE, "some_unknown_api_column": "value"}
        rec = NFLverseScheduleRecord.model_validate(data)
        assert not hasattr(rec, "some_unknown_api_column")

    def test_bad_game_id_format(self):
        data = {**self.BASE, "game_id": "NOT_A_GAME_ID"}
        _raises_validation(NFLverseScheduleRecord, data)

    def test_bad_season_too_low(self):
        data = {**self.BASE, "season": 1800}
        _raises_validation(NFLverseScheduleRecord, data)

    def test_bad_season_too_high(self):
        data = {**self.BASE, "season": 2200}
        _raises_validation(NFLverseScheduleRecord, data)

    def test_bad_week_zero(self):
        data = {**self.BASE, "week": 0}
        _raises_validation(NFLverseScheduleRecord, data)

    def test_bad_week_too_high(self):
        data = {**self.BASE, "week": 25}
        _raises_validation(NFLverseScheduleRecord, data)

    def test_invalid_team_abbr(self):
        data = {**self.BASE, "home_team": "NOTATEAM"}
        _raises_validation(NFLverseScheduleRecord, data)

    def test_team_abbr_uppercased(self):
        data = {**self.BASE, "home_team": "kc", "away_team": "buf"}
        rec = NFLverseScheduleRecord.model_validate(data)
        assert rec.home_team == "KC"
        assert rec.away_team == "BUF"

    def test_score_asymmetry_rejected(self):
        # Only one score present → model_validator should raise
        data = {**self.BASE, "home_score": 27.0, "away_score": None}
        _raises_validation(NFLverseScheduleRecord, data)

    def test_negative_home_score_rejected(self):
        data = {**self.BASE, "home_score": -1.0, "away_score": 24.0}
        _raises_validation(NFLverseScheduleRecord, data)

    def test_both_null_scores_ok(self):
        rec = NFLverseScheduleRecord.model_validate(self.BASE)
        assert rec.home_score is None and rec.away_score is None

    def test_string_season_coerced(self):
        data = {**self.BASE, "season": "2023"}
        rec = NFLverseScheduleRecord.model_validate(data)
        assert rec.season == 2023


# ===========================================================================
# NFLverse — Play-by-Play
# ===========================================================================

class TestNFLversePlayRecord:

    BASE: Dict[str, Any] = {"game_id": VALID_GAME_ID, "season": VALID_SEASON}

    def test_minimal_happy_path(self):
        rec = NFLversePlayRecord.model_validate(self.BASE)
        assert rec.season == VALID_SEASON

    def test_full_fields(self):
        data = {
            **self.BASE, "week": 1, "posteam": "KC", "defteam": "BUF",
            "play_type": "pass", "yards_gained": 12.0, "pass_attempt": 1,
            "epa": 0.45, "wp": 0.62, "wpa": 0.05, "down": 3, "ydstogo": 7,
            "yardline_100": 45, "game_seconds_remaining": 1800,
        }
        rec = NFLversePlayRecord.model_validate(data)
        assert rec.posteam == "KC"
        assert rec.wp == 0.62

    def test_invalid_wp_above_one(self):
        data = {**self.BASE, "wp": 1.5}
        _raises_validation(NFLversePlayRecord, data)

    def test_invalid_wp_below_zero(self):
        data = {**self.BASE, "wp": -0.1}
        _raises_validation(NFLversePlayRecord, data)

    def test_down_out_of_range(self):
        data = {**self.BASE, "down": 5}
        _raises_validation(NFLversePlayRecord, data)

    def test_yardline_out_of_range(self):
        data = {**self.BASE, "yardline_100": 101}
        _raises_validation(NFLversePlayRecord, data)

    def test_pass_attempt_out_of_range(self):
        data = {**self.BASE, "pass_attempt": 2}
        _raises_validation(NFLversePlayRecord, data)

    def test_nan_team_coerced_to_none(self):
        """Pandas NaN for optional team fields should not raise."""
        import math
        data = {**self.BASE, "posteam": float("nan")}
        rec = NFLversePlayRecord.model_validate(data)
        assert rec.posteam is None

    def test_invalid_team_raises(self):
        data = {**self.BASE, "posteam": "XX"}
        _raises_validation(NFLversePlayRecord, data)

    def test_game_seconds_negative_rejected(self):
        data = {**self.BASE, "game_seconds_remaining": -1}
        _raises_validation(NFLversePlayRecord, data)


# ===========================================================================
# NFLverse — Roster
# ===========================================================================

class TestNFLverseRosterRecord:

    BASE: Dict[str, Any] = {"season": VALID_SEASON, "team": "KC"}

    def test_minimal_happy_path(self):
        rec = NFLverseRosterRecord.model_validate(self.BASE)
        assert rec.team == "KC"

    def test_full_fields(self):
        data = {
            **self.BASE,
            "player_id": "00-0040123",
            "full_name": "Patrick Mahomes",
            "position": "QB",
            "jersey_number": 15,
            "years_exp": 8,
            "height": 74,
            "weight": 230,
            "college": "Texas Tech",
        }
        rec = NFLverseRosterRecord.model_validate(data)
        assert rec.position == "QB"
        assert rec.jersey_number == 15

    def test_jersey_number_out_of_range(self):
        data = {**self.BASE, "jersey_number": 100}
        _raises_validation(NFLverseRosterRecord, data)

    def test_negative_years_exp_rejected(self):
        data = {**self.BASE, "years_exp": -1}
        _raises_validation(NFLverseRosterRecord, data)

    def test_invalid_team_raises(self):
        data = {**self.BASE, "team": "INVALID"}
        _raises_validation(NFLverseRosterRecord, data)


# ===========================================================================
# NFLverse — Injury
# ===========================================================================

class TestNFLverseInjuryRecord:

    BASE: Dict[str, Any] = {"season": VALID_SEASON, "week": VALID_WEEK, "team": "KC"}

    def test_minimal_happy_path(self):
        rec = NFLverseInjuryRecord.model_validate(self.BASE)
        assert rec.week == 1

    def test_full_fields(self):
        data = {
            **self.BASE,
            "player_id": "00-0040123",
            "full_name": "Patrick Mahomes",
            "position": "QB",
            "report_status": "Questionable",
            "practice_status": "Limited",
            "report_primary_injury": "Ankle",
        }
        rec = NFLverseInjuryRecord.model_validate(data)
        assert rec.report_status == "Questionable"

    def test_invalid_week_zero(self):
        data = {**self.BASE, "week": 0}
        _raises_validation(NFLverseInjuryRecord, data)

    def test_invalid_team(self):
        data = {**self.BASE, "team": "XYZ"}
        _raises_validation(NFLverseInjuryRecord, data)


# ===========================================================================
# NFLverse — Player Stats
# ===========================================================================

class TestNFLversePlayerStatsRecord:

    BASE: Dict[str, Any] = {"season": VALID_SEASON}

    def test_minimal_happy_path(self):
        rec = NFLversePlayerStatsRecord.model_validate(self.BASE)
        assert rec.season == VALID_SEASON

    def test_passing_fields(self):
        data = {
            **self.BASE,
            "player_id": "00-0040123", "player_name": "P.Mahomes",
            "recent_team": "KC", "position": "QB",
            "completions": 378, "attempts": 597, "passing_yards": 4183.0,
            "passing_tds": 26, "interceptions": 11, "passing_epa": 52.3,
        }
        rec = NFLversePlayerStatsRecord.model_validate(data)
        assert rec.completions == 378
        assert rec.recent_team == "KC"

    def test_completions_exceeding_attempts_rejected(self):
        data = {**self.BASE, "completions": 400, "attempts": 300}
        _raises_validation(NFLversePlayerStatsRecord, data)

    def test_negative_carries_rejected(self):
        data = {**self.BASE, "carries": -5}
        _raises_validation(NFLversePlayerStatsRecord, data)

    def test_invalid_recent_team(self):
        data = {**self.BASE, "recent_team": "FAKE"}
        _raises_validation(NFLversePlayerStatsRecord, data)


# ===========================================================================
# ESPN — Player Stats
# ===========================================================================

class TestESPNPlayerStatRecord:

    BASE: Dict[str, Any] = {"season": VALID_SEASON, "season_type": 2, "category": "passing"}

    def test_minimal_happy_path(self):
        rec = ESPNPlayerStatRecord.model_validate(self.BASE)
        assert rec.category == "passing"

    def test_category_normalised_to_lowercase(self):
        data = {**self.BASE, "category": "Rushing"}
        rec = ESPNPlayerStatRecord.model_validate(data)
        assert rec.category == "rushing"

    def test_invalid_category(self):
        data = {**self.BASE, "category": "defense"}
        _raises_validation(ESPNPlayerStatRecord, data)

    def test_invalid_season_type(self):
        data = {**self.BASE, "season_type": 1}
        _raises_validation(ESPNPlayerStatRecord, data)

    def test_pct_out_of_range(self):
        data = {**self.BASE, "pct": 110.0}
        _raises_validation(ESPNPlayerStatRecord, data)

    def test_negative_gp_rejected(self):
        data = {**self.BASE, "gp": -1}
        _raises_validation(ESPNPlayerStatRecord, data)

    def test_team_abbr_uppercased_and_stored(self):
        data = {**self.BASE, "team_abbr": "kc"}
        rec = ESPNPlayerStatRecord.model_validate(data)
        assert rec.team_abbr == "KC"

    def test_dash_team_abbr_stored_as_none(self):
        data = {**self.BASE, "team_abbr": "--"}
        rec = ESPNPlayerStatRecord.model_validate(data)
        assert rec.team_abbr is None


# ===========================================================================
# ESPN — Player News
# ===========================================================================

class TestESPNPlayerNewsRecord:

    BASE: Dict[str, Any] = {"player_id": 3139477}

    def test_minimal_happy_path(self):
        rec = ESPNPlayerNewsRecord.model_validate(self.BASE)
        assert rec.player_id == 3139477

    def test_full_fields(self):
        data = {
            **self.BASE,
            "news_id": "news_001", "type": "Injury", "headline": "QB out",
            "published": "2024-10-01T12:00:00Z", "season": 2024,
            "premium": False, "is_live_blog": False,
        }
        rec = ESPNPlayerNewsRecord.model_validate(data)
        assert rec.headline == "QB out"

    def test_zero_player_id_rejected(self):
        data = {**self.BASE, "player_id": 0}
        _raises_validation(ESPNPlayerNewsRecord, data)

    def test_negative_player_id_rejected(self):
        data = {**self.BASE, "player_id": -1}
        _raises_validation(ESPNPlayerNewsRecord, data)

    def test_invalid_season(self):
        data = {**self.BASE, "season": 1800}
        _raises_validation(ESPNPlayerNewsRecord, data)


# ===========================================================================
# ESPN — Team News
# ===========================================================================

class TestESPNTeamNewsRecord:

    BASE: Dict[str, Any] = {
        "season": VALID_SEASON,
        "news_id": "12345",
        "headline": "Chiefs win AFC West",
        "description": "Kansas City clinch division title.",
        "published": "2024-12-15T18:00:00Z",
    }

    def test_minimal_happy_path(self):
        rec = ESPNTeamNewsRecord.model_validate(self.BASE)
        assert rec.news_id == "12345"

    def test_team_abbr_validated(self):
        data = {**self.BASE, "team_abbr": "KC"}
        rec = ESPNTeamNewsRecord.model_validate(data)
        assert rec.team_abbr == "KC"

    def test_invalid_team_abbr(self):
        data = {**self.BASE, "team_abbr": "NOTVALID"}
        _raises_validation(ESPNTeamNewsRecord, data)

    def test_none_team_abbr_ok(self):
        data = {**self.BASE, "team_abbr": None}
        rec = ESPNTeamNewsRecord.model_validate(data)
        assert rec.team_abbr is None

    def test_empty_news_id_rejected(self):
        data = {**self.BASE, "news_id": ""}
        _raises_validation(ESPNTeamNewsRecord, data)

    def test_invalid_season(self):
        data = {**self.BASE, "season": 2200}
        _raises_validation(ESPNTeamNewsRecord, data)


# ===========================================================================
# ESPN — Team Defense
# ===========================================================================

class TestESPNTeamDefenseRecord:

    BASE: Dict[str, Any] = {"season": VALID_SEASON, "team": "KC"}

    def test_minimal_happy_path(self):
        rec = ESPNTeamDefenseRecord.model_validate(self.BASE)
        assert rec.team == "KC"

    def test_full_fields(self):
        data = {
            **self.BASE,
            "games_played": 17, "points_allowed": 317.0,
            "total_yards_allowed": 5234.0, "sacks": 42.0, "interceptions": 12,
        }
        rec = ESPNTeamDefenseRecord.model_validate(data)
        assert rec.points_allowed == 317.0

    def test_negative_sacks_rejected(self):
        data = {**self.BASE, "sacks": -1.0}
        _raises_validation(ESPNTeamDefenseRecord, data)

    def test_invalid_team(self):
        data = {**self.BASE, "team": "INVALID"}
        _raises_validation(ESPNTeamDefenseRecord, data)

    def test_season_type_default(self):
        rec = ESPNTeamDefenseRecord.model_validate(self.BASE)
        assert rec.season_type == 2


# ===========================================================================
# Sportradar — Schedule
# ===========================================================================

class TestSportradarScheduleRecord:

    BASE: Dict[str, Any] = {"game_id": "abc123-def456"}

    def test_minimal_happy_path(self):
        rec = SportradarScheduleRecord.model_validate(self.BASE)
        assert rec.game_id == "abc123-def456"

    def test_full_fields(self):
        data = {
            **self.BASE,
            "season_year": VALID_SEASON, "season_type": "REG", "week": 3,
            "home_alias": "KC", "away_alias": "BUF",
            "home_points": 27, "away_points": 24,
            "venue_name": "GEHA Field", "venue_city": "Kansas City",
            "weather_temp_f": 55.0, "weather_conditions": "Partly Cloudy",
        }
        rec = SportradarScheduleRecord.model_validate(data)
        assert rec.home_points == 27
        assert rec.weather_temp_f == 55.0

    def test_empty_game_id_rejected(self):
        _raises_validation(SportradarScheduleRecord, {"game_id": "  "})

    def test_attendance_negative_rejected(self):
        data = {**self.BASE, "attendance": -100}
        _raises_validation(SportradarScheduleRecord, data)

    def test_weather_temp_too_cold(self):
        data = {**self.BASE, "weather_temp_f": -100.0}
        _raises_validation(SportradarScheduleRecord, data)

    def test_weather_temp_too_hot(self):
        data = {**self.BASE, "weather_temp_f": 200.0}
        _raises_validation(SportradarScheduleRecord, data)

    def test_invalid_week(self):
        data = {**self.BASE, "week": 30}
        _raises_validation(SportradarScheduleRecord, data)

    def test_season_out_of_range(self):
        data = {**self.BASE, "season_year": 1800}
        _raises_validation(SportradarScheduleRecord, data)


# ===========================================================================
# Sportradar — Roster
# ===========================================================================

class TestSportradarRosterRecord:

    BASE: Dict[str, Any] = {}   # all fields optional

    def test_empty_dict_ok(self):
        rec = SportradarRosterRecord.model_validate(self.BASE)
        assert rec.player_id is None

    def test_full_fields(self):
        data = {
            "team_id": "team-001", "team_alias": "KC",
            "player_id": "player-001", "player_name": "Patrick Mahomes",
            "position": "QB", "status": "ACT",
            "height": 74, "weight": 230, "college": "Texas Tech",
            "rookie_year": 2017, "experience": 8,
            "season": VALID_SEASON,
            "roster_total_players": 53, "roster_active_count": 48,
            "roster_injured_pct": 0.09, "roster_active_pct": 0.91,
        }
        rec = SportradarRosterRecord.model_validate(data)
        assert rec.position == "QB"
        assert rec.roster_total_players == 53

    def test_negative_height_rejected(self):
        data = {"height": -1}
        _raises_validation(SportradarRosterRecord, data)

    def test_injured_pct_above_one_rejected(self):
        data = {"roster_injured_pct": 1.5}
        _raises_validation(SportradarRosterRecord, data)

    def test_invalid_rookie_year(self):
        data = {"rookie_year": 1800}
        _raises_validation(SportradarRosterRecord, data)

    def test_invalid_season(self):
        data = {"season": 2200}
        _raises_validation(SportradarRosterRecord, data)


# ===========================================================================
# Sportradar — Game Stats
# ===========================================================================

class TestSportradarGameStatsRecord:

    BASE: Dict[str, Any] = {"game_id": "abc123-def456"}

    def test_minimal_happy_path(self):
        rec = SportradarGameStatsRecord.model_validate(self.BASE)
        assert rec.game_id == "abc123-def456"

    def test_full_fields(self):
        data = {
            **self.BASE,
            "team_id": "team-001", "team_role": "home",
            "scope": "team", "stat_category": "passing",
            "stats": {"yards": 312, "tds": 3},
            "season_year": VALID_SEASON, "week": 5,
        }
        rec = SportradarGameStatsRecord.model_validate(data)
        assert rec.team_role == "home"
        assert rec.stats == {"yards": 312, "tds": 3}

    def test_invalid_team_role(self):
        data = {**self.BASE, "team_role": "neutral"}
        _raises_validation(SportradarGameStatsRecord, data)

    def test_empty_game_id_rejected(self):
        _raises_validation(SportradarGameStatsRecord, {"game_id": ""})

    def test_team_role_case_insensitive(self):
        data = {**self.BASE, "team_role": "HOME"}
        rec = SportradarGameStatsRecord.model_validate(data)
        assert rec.team_role == "home"

    def test_invalid_week(self):
        data = {**self.BASE, "week": 0}
        _raises_validation(SportradarGameStatsRecord, data)


# ===========================================================================
# Weather — NOAA
# ===========================================================================

class TestNOAAWeatherRecord:

    BASE: Dict[str, Any] = {"game_id": VALID_GAME_ID}

    def test_minimal_happy_path(self):
        rec = NOAAWeatherRecord.model_validate(self.BASE)
        assert rec.game_id == VALID_GAME_ID

    def test_full_fields(self):
        data = {
            **self.BASE,
            "season": VALID_SEASON, "week": VALID_WEEK,
            "home_team": "KC", "away_team": "BUF",
            "weather_temp_f": 38.0, "weather_wind_mph": 15.0,
            "weather_humidity_pct": 60.0, "weather_precip_in": 0.1,
            "is_dome": False, "lat": 39.05, "lon": -94.48,
        }
        rec = NOAAWeatherRecord.model_validate(data)
        assert rec.weather_temp_f == 38.0
        assert rec.lat == 39.05

    def test_temp_too_cold(self):
        data = {**self.BASE, "weather_temp_f": -100.0}
        _raises_validation(NOAAWeatherRecord, data)

    def test_wind_negative(self):
        data = {**self.BASE, "weather_wind_mph": -5.0}
        _raises_validation(NOAAWeatherRecord, data)

    def test_humidity_above_100(self):
        data = {**self.BASE, "weather_humidity_pct": 105.0}
        _raises_validation(NOAAWeatherRecord, data)

    def test_lat_out_of_range(self):
        data = {**self.BASE, "lat": 95.0}
        _raises_validation(NOAAWeatherRecord, data)

    def test_lon_out_of_range(self):
        data = {**self.BASE, "lon": -200.0}
        _raises_validation(NOAAWeatherRecord, data)

    def test_invalid_team(self):
        data = {**self.BASE, "home_team": "NOTATEAM"}
        _raises_validation(NOAAWeatherRecord, data)

    def test_empty_game_id_rejected(self):
        _raises_validation(NOAAWeatherRecord, {"game_id": ""})


# ===========================================================================
# Weather — Visual Crossing
# ===========================================================================

class TestVisualCrossingWeatherRecord:

    BASE: Dict[str, Any] = {"game_id": VALID_GAME_ID}

    def test_minimal_happy_path(self):
        rec = VisualCrossingWeatherRecord.model_validate(self.BASE)
        assert rec.game_id == VALID_GAME_ID

    def test_full_fields(self):
        data = {
            **self.BASE,
            "season": VALID_SEASON, "week": 4,
            "home_team": "KC", "away_team": "DEN",
            "temp_f": 42.0, "feelslike_f": 36.0,
            "humidity_pct": 55.0, "windspeed_mph": 12.0,
            "winddir_deg": 270.0, "cloudcover_pct": 30.0,
            "precip_in": 0.0, "snow_in": 0.0,
            "uvindex": 2.0, "conditions": "Clear",
            "is_dome": False,
        }
        rec = VisualCrossingWeatherRecord.model_validate(data)
        assert rec.temp_f == 42.0

    def test_winddir_360_boundary_ok(self):
        data = {**self.BASE, "winddir_deg": 360.0}
        rec = VisualCrossingWeatherRecord.model_validate(data)
        assert rec.winddir_deg == 360.0

    def test_winddir_out_of_range(self):
        data = {**self.BASE, "winddir_deg": 361.0}
        _raises_validation(VisualCrossingWeatherRecord, data)

    def test_uvindex_out_of_range(self):
        data = {**self.BASE, "uvindex": 12.0}
        _raises_validation(VisualCrossingWeatherRecord, data)

    def test_negative_snow_rejected(self):
        data = {**self.BASE, "snow_in": -0.1}
        _raises_validation(VisualCrossingWeatherRecord, data)


# ===========================================================================
# Weather — Game Weather (merged)
# ===========================================================================

class TestGameWeatherRecord:

    BASE: Dict[str, Any] = {"game_id": VALID_GAME_ID}

    def test_minimal_happy_path(self):
        rec = GameWeatherRecord.model_validate(self.BASE)
        assert rec.game_id == VALID_GAME_ID

    def test_full_fields(self):
        data = {
            **self.BASE,
            "season": VALID_SEASON, "week": 10,
            "home_team": "KC", "away_team": "LAR",
            "weather_temp_f": 65.0, "weather_wind_mph": 8.0,
            "weather_humidity_pct": 45.0, "weather_precip_in": 0.0,
            "weather_conditions": "Sunny",
            "weather_is_dome": False,
            "weather_source": "noaa",
        }
        rec = GameWeatherRecord.model_validate(data)
        assert rec.weather_source == "noaa"

    def test_invalid_weather_source(self):
        data = {**self.BASE, "weather_source": "darksky"}
        _raises_validation(GameWeatherRecord, data)

    def test_weather_source_case_normalised(self):
        data = {**self.BASE, "weather_source": "NOAA"}
        rec = GameWeatherRecord.model_validate(data)
        assert rec.weather_source == "noaa"

    def test_dome_flag_ok(self):
        data = {**self.BASE, "weather_is_dome": True}
        rec = GameWeatherRecord.model_validate(data)
        assert rec.weather_is_dome is True

    def test_valid_sources_accepted(self):
        for src in ("noaa", "visualcrossing", "tomorrow", "openweather", "manual"):
            data = {**self.BASE, "weather_source": src}
            rec = GameWeatherRecord.model_validate(data)
            assert rec.weather_source == src


# ===========================================================================
# Yahoo — Player
# ===========================================================================

class TestYahooPlayerRecord:

    BASE: Dict[str, Any] = {}   # all fields optional

    def test_empty_dict_ok(self):
        rec = YahooPlayerRecord.model_validate(self.BASE)
        assert rec.player_id is None

    def test_full_fields(self):
        data = {
            "player_id": 31765, "player_key": "423.p.31765",
            "full_name": "Patrick Mahomes",
            "display_position": "QB",
            "editorial_team_abbr": "KC",
            "status": "A",
            "season": VALID_SEASON, "game_key": "423",
        }
        rec = YahooPlayerRecord.model_validate(data)
        assert rec.full_name == "Patrick Mahomes"
        assert rec.season == VALID_SEASON

    def test_zero_player_id_rejected(self):
        data = {"player_id": 0}
        _raises_validation(YahooPlayerRecord, data)

    def test_negative_player_id_rejected(self):
        data = {"player_id": -100}
        _raises_validation(YahooPlayerRecord, data)

    def test_invalid_season(self):
        data = {"season": 1800}
        _raises_validation(YahooPlayerRecord, data)

    def test_extra_fields_stripped(self):
        data = {"player_id": 12345, "unknown_yahoo_field": "value"}
        rec = YahooPlayerRecord.model_validate(data)
        assert not hasattr(rec, "unknown_yahoo_field")


# ===========================================================================
# Schema versioning and registry
# ===========================================================================

class TestSchemaVersioning:

    EXPECTED_KEYS = {
        # NFLverse
        "nflverse_schedule", "nflverse_pbp", "nflverse_roster",
        "nflverse_injury", "nflverse_player_stats",
        # ESPN
        "espn_player_stats", "espn_player_news",
        "espn_team_news", "espn_team_defense",
        # Sportradar
        "sportradar_schedule", "sportradar_roster", "sportradar_game_stats",
        # Weather
        "noaa_weather", "visualcrossing_weather", "game_weather",
        # Yahoo
        "yahoo_player",
    }

    def test_all_expected_keys_present(self):
        assert self.EXPECTED_KEYS == set(SCHEMA_VERSIONS.keys())

    def test_all_version_strings_are_semver_like(self):
        import re
        pattern = re.compile(r"^\d+\.\d+\.\d+$")
        for key, version in SCHEMA_VERSIONS.items():
            assert pattern.match(version), (
                f"SCHEMA_VERSIONS['{key}']={version!r} is not semver-like"
            )

    def test_schema_models_keys_match_schema_versions(self):
        assert set(SCHEMA_MODELS.keys()) == set(SCHEMA_VERSIONS.keys())

    def test_get_schema_version_known_key(self):
        v = get_schema_version("nflverse_schedule")
        assert v == "1.0.0"

    def test_get_schema_version_unknown_key_raises(self):
        with pytest.raises(KeyError):
            get_schema_version("does_not_exist")

    def test_validate_record_nflverse_schedule(self):
        data = {
            "game_id": VALID_GAME_ID, "season": VALID_SEASON,
            "week": 1, "home_team": "KC", "away_team": "BUF",
        }
        rec = validate_record("nflverse_schedule", data)
        assert isinstance(rec, NFLverseScheduleRecord)

    def test_validate_record_unknown_key_raises(self):
        with pytest.raises(KeyError):
            validate_record("not_a_schema", {})

    def test_validate_record_invalid_data_raises_validation_error(self):
        with pytest.raises(ValidationError):
            validate_record("nflverse_schedule", {"game_id": "bad", "season": 9999,
                                                   "week": 1, "home_team": "XX",
                                                   "away_team": "BUF"})

    def test_all_model_classes_instantiable_with_minimal_data(self):
        """Each registered model class should accept a near-empty dict without crashing."""
        safe_minimal: Dict[str, Dict[str, Any]] = {
            "nflverse_schedule": {"game_id": VALID_GAME_ID, "season": 2024,
                                   "week": 1, "home_team": "KC", "away_team": "BUF"},
            "nflverse_pbp":      {"game_id": VALID_GAME_ID, "season": 2024},
            "nflverse_roster":   {"season": 2024, "team": "KC"},
            "nflverse_injury":   {"season": 2024, "week": 1, "team": "KC"},
            "nflverse_player_stats": {"season": 2024},
            "espn_player_stats": {"season": 2024, "season_type": 2, "category": "passing"},
            "espn_player_news":  {"player_id": 1},
            "espn_team_news":    {"season": 2024, "news_id": "1", "headline": "h",
                                   "description": "d", "published": "2024-01-01"},
            "espn_team_defense": {"season": 2024, "team": "KC"},
            "sportradar_schedule": {"game_id": "abc"},
            "sportradar_roster": {},
            "sportradar_game_stats": {"game_id": "abc"},
            "noaa_weather":      {"game_id": VALID_GAME_ID},
            "visualcrossing_weather": {"game_id": VALID_GAME_ID},
            "game_weather":      {"game_id": VALID_GAME_ID},
            "yahoo_player":      {},
        }
        for key, model_cls in SCHEMA_MODELS.items():
            data = safe_minimal.get(key, {})
            instance = model_cls.model_validate(data)
            assert isinstance(instance, model_cls), f"Failed for key={key}"


# ===========================================================================
# NFL team abbreviation validator (shared helper)
# ===========================================================================

class TestNFLAbbreviations:

    def test_all_32_current_teams_accepted(self):
        current_teams = [
            "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
            "DAL", "DEN", "DET", "GB",  "HOU", "IND", "JAX", "KC",
            "LAC", "LAR", "LV",  "MIA", "MIN", "NE",  "NO",  "NYG",
            "NYJ", "PHI", "PIT", "SF",  "SEA", "TB",  "TEN", "WAS",
        ]
        for abbr in current_teams:
            assert abbr in _NFL_ABBRS, f"{abbr} not in _NFL_ABBRS"

    def test_historical_aliases_included(self):
        for abbr in ("OAK", "SD", "STL", "JAC"):
            assert abbr in _NFL_ABBRS

    def test_invalid_abbrs_not_included(self):
        for abbr in ("XX", "NFL", "AFC", "NFC", "???"):
            assert abbr not in _NFL_ABBRS

    def test_team_validation_via_schedule_lowercase(self):
        """Field validators should uppercase and validate team abbrs."""
        rec = NFLverseScheduleRecord.model_validate({
            "game_id": VALID_GAME_ID, "season": 2024, "week": 1,
            "home_team": "gb", "away_team": "chi",
        })
        assert rec.home_team == "GB"
        assert rec.away_team == "CHI"
