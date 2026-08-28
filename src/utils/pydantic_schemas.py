"""
Pydantic v2 data schemas for NFL prediction pipeline sources.

This module defines row-level Pydantic models for each data provider.
They validate individual records returned by the data fetchers before
ingestion into the pipeline, catching malformed API responses early.

Schema versioning is supported via the ``SCHEMA_VERSIONS`` registry so
callers can discover which version of a schema is active.

Provider groupings
------------------
* NFLverse  – schedule, play-by-play, roster, injury, player stats
* ESPN      – player stats, player news, team news, team defense (stub)
* Sportradar– schedule, roster player, game stats
* Weather   – NOAA observation, Visual Crossing, Tomorrow.io / generic game weather
* Yahoo     – player (fantasy game)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers / shared validators
# ---------------------------------------------------------------------------

# All 32 current NFL team abbreviations plus common historical aliases used
# throughout the codebase (see espn_team_news.py / teams.py).
_NFL_ABBRS: frozenset[str] = frozenset(
    {
        "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
        "DAL", "DEN", "DET", "GB",  "HOU", "IND", "JAX", "KC",
        "LAC", "LAR", "LV",  "MIA", "MIN", "NE",  "NO",  "NYG",
        "NYJ", "PHI", "PIT", "SF",  "SEA", "TB",  "TEN", "WAS",
        # Historical aliases kept for backward-compat with older data
        "OAK", "SD",  "STL", "JAC",
    }
)

_VALID_INJURY_STATUSES: frozenset[str] = frozenset(
    {"Out", "Doubtful", "Questionable", "Limited", "Full", "DNP",
     "IR", "PUP", "NFI", "Did Not Practice", "Full Participation",
     "Limited Participation", "Active", "Inactive"}
)

_VALID_POSITIONS: frozenset[str] = frozenset(
    {"QB", "RB", "WR", "TE", "OL", "OT", "OG", "C", "FB",
     "DL", "DT", "DE", "LB", "ILB", "OLB", "MLB",
     "DB", "CB", "S", "FS", "SS",
     "K", "P", "LS", "KR", "PR", "ST"}
)

_NFLVERSE_GAME_ID_RE = re.compile(r"^\d{4}_\d{2}_[A-Z]{2,4}_[A-Z]{2,4}$")


def _is_na(v: Any) -> bool:
    """Return True for None or any NaN-like value (float nan, pd.NA, pd.NaT, np.nan)."""
    if v is None:
        return True
    try:
        import math
        return isinstance(v, float) and math.isnan(v)
    except (TypeError, ValueError):
        return False


def _validate_nfl_team(value: Optional[str], field_name: str = "team") -> Optional[str]:
    """Return uppercased abbreviation if it looks like a team code; None passes through."""
    if value is None:
        return None
    upper = value.strip().upper()
    if upper not in _NFL_ABBRS:
        raise ValueError(
            f"{field_name}='{value}' is not a recognised NFL team abbreviation. "
            f"Known: {sorted(_NFL_ABBRS)}"
        )
    return upper


def _validate_season(value: int) -> int:
    if not (1920 <= value <= 2100):
        raise ValueError(f"season={value} is outside the plausible range 1920–2100")
    return value


def _validate_week(value: int) -> int:
    if not (1 <= value <= 23):
        raise ValueError(f"week={value} must be 1–23 (includes pre-season through Super Bowl)")
    return value


def _validate_probability(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"probability={value} must be in [0, 1]")
    return value


# ---------------------------------------------------------------------------
# Base model with shared config
# ---------------------------------------------------------------------------

class _NFLBase(BaseModel):
    """Shared Pydantic v2 config: strip extra fields, populate by name."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="ignore",          # silently drop unknown fields from the API
        str_strip_whitespace=True,
        populate_by_name=True,
    )

    #: Schema version tag – concrete subclasses set this in SCHEMA_VERSIONS.
    schema_version: ClassVar[str] = "1.0.0"


# ===========================================================================
# NFLverse schemas (5)
# ===========================================================================

class NFLverseScheduleRecord(_NFLBase):
    """One game row from nfl_data_py.import_schedules()."""

    game_id: str
    season: int
    week: int
    home_team: str
    away_team: str
    home_score: Optional[float] = None
    away_score: Optional[float] = None
    gameday: Optional[str] = None          # ISO-8601 date string or datetime
    game_type: Optional[str] = None        # REG, POST, SB, PRE
    surface: Optional[str] = None
    roof: Optional[str] = None             # outdoors, dome, retractable
    stadium: Optional[str] = None
    location: Optional[str] = None
    div_game: Optional[int] = Field(None, ge=0, le=1)
    overtime: Optional[int] = Field(None, ge=0, le=1)
    spread_line: Optional[float] = None
    total_line: Optional[float] = None
    home_moneyline: Optional[int] = None
    away_moneyline: Optional[int] = None
    result: Optional[float] = None        # home_score - away_score

    @field_validator("season", mode="before")
    @classmethod
    def check_season(cls, v: Any) -> int:
        return _validate_season(int(v))

    @field_validator("week", mode="before")
    @classmethod
    def check_week(cls, v: Any) -> int:
        return _validate_week(int(v))

    @field_validator("home_team", "away_team", mode="before")
    @classmethod
    def check_team_abbr(cls, v: Any) -> str:
        result = _validate_nfl_team(str(v) if v is not None else None, "team")
        if result is None:
            raise ValueError("team abbreviation is required")
        return result

    @field_validator("game_id", mode="before")
    @classmethod
    def check_game_id(cls, v: Any) -> str:
        s = str(v).strip()
        if not _NFLVERSE_GAME_ID_RE.match(s):
            raise ValueError(
                f"game_id='{s}' does not match expected nflverse format YYYY_WW_HOME_AWAY"
            )
        return s

    @model_validator(mode="after")
    def check_scores_consistent(self) -> "NFLverseScheduleRecord":
        if (self.home_score is None) != (self.away_score is None):
            raise ValueError(
                "home_score and away_score must both be present or both be null"
            )
        if self.home_score is not None and self.home_score < 0:
            raise ValueError(f"home_score={self.home_score} must be >= 0")
        if self.away_score is not None and self.away_score < 0:
            raise ValueError(f"away_score={self.away_score} must be >= 0")
        return self


class NFLversePlayRecord(_NFLBase):
    """One play row from nfl_data_py.import_pbp_data()."""

    game_id: str
    season: int
    week: Optional[int] = None
    play_id: Optional[Any] = None
    posteam: Optional[str] = None
    defteam: Optional[str] = None
    play_type: Optional[str] = None       # pass, run, punt, kickoff, …
    yards_gained: Optional[float] = None
    pass_attempt: Optional[int] = Field(None, ge=0, le=1)
    rush_attempt: Optional[int] = Field(None, ge=0, le=1)
    epa: Optional[float] = None
    wp: Optional[float] = None            # win probability (pre-snap)
    wpa: Optional[float] = None           # win prob added
    qb_epa: Optional[float] = None
    air_yards: Optional[float] = None
    yards_after_catch: Optional[float] = None
    down: Optional[int] = Field(None, ge=1, le=4)
    ydstogo: Optional[int] = Field(None, ge=0)
    yardline_100: Optional[int] = Field(None, ge=0, le=100)
    game_seconds_remaining: Optional[int] = Field(None, ge=0, le=3600)
    score_differential: Optional[float] = None
    home_team: Optional[str] = None
    away_team: Optional[str] = None

    @field_validator("season", mode="before")
    @classmethod
    def check_season(cls, v: Any) -> int:
        return _validate_season(int(v))

    @field_validator("week", mode="before")
    @classmethod
    def check_week(cls, v: Any) -> Optional[int]:
        if v is None:
            return None
        return _validate_week(int(v))

    @field_validator("wp", "wpa", mode="before")
    @classmethod
    def check_probability(cls, v: Any) -> Optional[float]:
        if v is None:
            return None
        return _validate_probability(float(v))

    @field_validator("posteam", "defteam", "home_team", "away_team", mode="before")
    @classmethod
    def check_optional_team(cls, v: Any) -> Optional[str]:
        if v is None or (isinstance(v, float)):
            return None
        return _validate_nfl_team(str(v), "team")


class NFLverseRosterRecord(_NFLBase):
    """One player-season row from nfl_data_py.import_rosters()."""

    season: int
    team: str
    player_id: Optional[str] = None
    player_name: Optional[str] = None
    full_name: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    position: Optional[str] = None
    jersey_number: Optional[int] = Field(None, ge=0, le=99)
    status: Optional[str] = None
    years_exp: Optional[int] = Field(None, ge=0)
    birth_date: Optional[str] = None
    height: Optional[int] = Field(None, ge=0)    # inches
    weight: Optional[int] = Field(None, ge=0)    # lbs
    college: Optional[str] = None
    espn_id: Optional[Any] = None
    sportradar_id: Optional[str] = None
    pfr_id: Optional[str] = None
    depth_chart_position: Optional[str] = None
    depth_chart_order: Optional[int] = Field(None, ge=1)

    @field_validator("season", mode="before")
    @classmethod
    def check_season(cls, v: Any) -> int:
        return _validate_season(int(v))

    @field_validator("team", mode="before")
    @classmethod
    def check_team(cls, v: Any) -> str:
        result = _validate_nfl_team(str(v) if v is not None else None, "team")
        if result is None:
            raise ValueError("team is required")
        return result

    @field_validator("position", mode="before")
    @classmethod
    def check_position(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        upper = str(v).strip().upper()
        # Positions are advisory – warn but don't reject unknown ones
        return upper


class NFLverseInjuryRecord(_NFLBase):
    """One injury report row from nfl_data_py.import_injuries()."""

    season: int
    week: int
    team: str
    player_id: Optional[str] = None
    full_name: Optional[str] = None
    position: Optional[str] = None
    report_status: Optional[str] = None      # Out, Doubtful, Questionable, …
    practice_status: Optional[str] = None    # Full, Limited, DNP
    report_primary_injury: Optional[str] = None
    report_secondary_injury: Optional[str] = None
    practice_primary_injury: Optional[str] = None
    practice_secondary_injury: Optional[str] = None

    @field_validator("season", mode="before")
    @classmethod
    def check_season(cls, v: Any) -> int:
        return _validate_season(int(v))

    @field_validator("week", mode="before")
    @classmethod
    def check_week(cls, v: Any) -> int:
        return _validate_week(int(v))

    @field_validator("team", mode="before")
    @classmethod
    def check_team(cls, v: Any) -> str:
        result = _validate_nfl_team(str(v) if v is not None else None, "team")
        if result is None:
            raise ValueError("team is required")
        return result


class NFLversePlayerStatsRecord(_NFLBase):
    """One player-season row from nfl_data_py.import_seasonal_data()."""

    season: int
    season_type: Optional[str] = None       # REG, POST
    player_id: Optional[str] = None
    player_name: Optional[str] = None
    recent_team: Optional[str] = None
    position: Optional[str] = None
    # Passing
    completions: Optional[int] = Field(None, ge=0)
    attempts: Optional[int] = Field(None, ge=0)
    passing_yards: Optional[float] = None
    passing_tds: Optional[int] = Field(None, ge=0)
    interceptions: Optional[int] = Field(None, ge=0)
    sacks: Optional[float] = Field(None, ge=0)
    passing_air_yards: Optional[float] = None
    passing_yards_after_catch: Optional[float] = None
    passing_epa: Optional[float] = None
    # Rushing
    carries: Optional[int] = Field(None, ge=0)
    rushing_yards: Optional[float] = None
    rushing_tds: Optional[int] = Field(None, ge=0)
    rushing_fumbles: Optional[int] = Field(None, ge=0)
    rushing_epa: Optional[float] = None
    # Receiving
    receptions: Optional[int] = Field(None, ge=0)
    targets: Optional[int] = Field(None, ge=0)
    receiving_yards: Optional[float] = None
    receiving_tds: Optional[int] = Field(None, ge=0)
    receiving_air_yards: Optional[float] = None
    receiving_yards_after_catch: Optional[float] = None
    receiving_epa: Optional[float] = None
    # Fantasy
    fantasy_points: Optional[float] = None
    fantasy_points_ppr: Optional[float] = None

    @field_validator("season", mode="before")
    @classmethod
    def check_season(cls, v: Any) -> int:
        return _validate_season(int(v))

    @field_validator("recent_team", mode="before")
    @classmethod
    def check_team(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        return _validate_nfl_team(str(v), "recent_team")

    @model_validator(mode="after")
    def check_completion_rate(self) -> "NFLversePlayerStatsRecord":
        if (
            self.completions is not None
            and self.attempts is not None
            and self.attempts > 0
            and self.completions > self.attempts
        ):
            raise ValueError(
                f"completions={self.completions} cannot exceed attempts={self.attempts}"
            )
        return self


# ===========================================================================
# ESPN schemas (4)
# ===========================================================================

class ESPNPlayerStatRecord(_NFLBase):
    """One player-category row scraped from ESPN player stats pages."""

    season: int
    season_type: int = Field(description="2=regular season, 3=postseason")
    category: str = Field(description="passing | rushing | receiving")
    # Player identity – column names vary by season; mark all optional
    player_name: Optional[str] = None
    team_abbr: Optional[str] = None
    page: Optional[int] = Field(None, ge=1)
    # Common stat columns (snake_case after normalisation)
    gp: Optional[int] = Field(None, ge=0, description="Games played")
    cmp: Optional[Any] = None             # completions (passing)
    att: Optional[Any] = None             # attempts
    pct: Optional[float] = Field(None, ge=0, le=100, description="Completion %")
    yds: Optional[Any] = None             # total yards
    avg: Optional[float] = None
    td: Optional[Any] = None              # touchdowns
    int_: Optional[Any] = Field(None, alias="int", description="Interceptions")
    rtg: Optional[float] = Field(None, ge=0, description="Passer rating")
    rec: Optional[Any] = None             # receptions
    tar: Optional[Any] = None             # targets
    lng: Optional[Any] = None             # longest gain
    car: Optional[Any] = None             # carries

    @field_validator("season", mode="before")
    @classmethod
    def check_season(cls, v: Any) -> int:
        return _validate_season(int(v))

    @field_validator("season_type", mode="before")
    @classmethod
    def check_season_type(cls, v: Any) -> int:
        val = int(v)
        if val not in (2, 3):
            raise ValueError(f"season_type={val} must be 2 (regular) or 3 (postseason)")
        return val

    @field_validator("category", mode="before")
    @classmethod
    def check_category(cls, v: Any) -> str:
        lower = str(v).strip().lower()
        if lower not in ("passing", "rushing", "receiving"):
            raise ValueError(f"category='{lower}' must be passing, rushing, or receiving")
        return lower

    @field_validator("team_abbr", mode="before")
    @classmethod
    def check_team(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        # ESPN sometimes uses full names or "--"; skip hard validation
        s = str(v).strip()
        if not s or s == "--":
            return None
        return s.upper()


class ESPNPlayerNewsRecord(_NFLBase):
    """One news item row from the ESPN fantasy player news endpoint."""

    player_id: int
    news_id: Optional[Any] = None
    type: Optional[str] = None
    headline: Optional[str] = None
    description: Optional[str] = None
    story: Optional[str] = None
    source: Optional[str] = None
    byline: Optional[str] = None
    published: Optional[str] = None
    last_modified: Optional[str] = None
    categorized: Optional[str] = None
    link_web: Optional[str] = None
    link_mobile: Optional[str] = None
    link_api: Optional[str] = None
    premium: Optional[bool] = None
    is_live_blog: Optional[bool] = None
    allow_comments: Optional[bool] = None
    allow_search: Optional[bool] = None
    allow_reactions: Optional[bool] = None
    season: Optional[int] = None

    @field_validator("player_id", mode="before")
    @classmethod
    def check_player_id(cls, v: Any) -> int:
        val = int(v)
        if val <= 0:
            raise ValueError(f"player_id={val} must be a positive integer")
        return val

    @field_validator("season", mode="before")
    @classmethod
    def check_season(cls, v: Any) -> Optional[int]:
        if v is None:
            return None
        return _validate_season(int(v))


class ESPNTeamNewsRecord(_NFLBase):
    """One article row from the ESPN team news endpoint (espn_team_news.py)."""

    season: int
    news_id: str
    headline: str
    description: str
    published: str
    last_modified: Optional[str] = None
    article_type: Optional[str] = None
    byline: Optional[str] = None
    link_web: Optional[str] = None
    link_mobile: Optional[str] = None
    source: Optional[str] = None
    team_abbr: Optional[str] = None

    @field_validator("season", mode="before")
    @classmethod
    def check_season(cls, v: Any) -> int:
        return _validate_season(int(v))

    @field_validator("team_abbr", mode="before")
    @classmethod
    def check_team(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        return _validate_nfl_team(str(v), "team_abbr")

    @field_validator("news_id", mode="before")
    @classmethod
    def check_news_id(cls, v: Any) -> str:
        s = str(v).strip()
        if not s:
            raise ValueError("news_id must not be empty")
        return s


class ESPNTeamDefenseRecord(_NFLBase):
    """Placeholder schema for ESPN team-level defensive metrics.

    ESPN team-defense scraping was removed (espn_team_defense.py now stubs
    out), but downstream consumers may still produce rows in this shape from
    PBP-derived stats.
    """

    season: int
    team: str
    season_type: int = Field(2, description="2=regular season, 3=postseason")
    games_played: Optional[int] = Field(None, ge=0)
    points_allowed: Optional[float] = Field(None, ge=0)
    total_yards_allowed: Optional[float] = None
    passing_yards_allowed: Optional[float] = None
    rushing_yards_allowed: Optional[float] = None
    sacks: Optional[float] = Field(None, ge=0)
    interceptions: Optional[int] = Field(None, ge=0)
    fumbles_recovered: Optional[int] = Field(None, ge=0)
    touchdowns_allowed: Optional[int] = Field(None, ge=0)
    yards_per_play_allowed: Optional[float] = Field(None, ge=0)

    @field_validator("season", mode="before")
    @classmethod
    def check_season(cls, v: Any) -> int:
        return _validate_season(int(v))

    @field_validator("team", mode="before")
    @classmethod
    def check_team(cls, v: Any) -> str:
        result = _validate_nfl_team(str(v) if v is not None else None, "team")
        if result is None:
            raise ValueError("team is required")
        return result


# ===========================================================================
# Sportradar schemas (3)
# ===========================================================================

class SportradarScheduleRecord(_NFLBase):
    """One game row produced by sportradar_transform.transform_schedule()."""

    game_id: str
    season_year: Optional[int] = None
    season_type: Optional[str] = None     # REG, POST, PRE
    week: Optional[int] = None
    week_title: Optional[str] = None
    game_status: Optional[str] = None     # scheduled, closed, inprogress, …
    scheduled: Optional[str] = None       # ISO-8601
    attendance: Optional[int] = Field(None, ge=0)
    conference_game: Optional[bool] = None
    home_id: Optional[str] = None
    home_name: Optional[str] = None
    home_alias: Optional[str] = None
    home_points: Optional[int] = Field(None, ge=0)
    away_id: Optional[str] = None
    away_name: Optional[str] = None
    away_alias: Optional[str] = None
    away_points: Optional[int] = Field(None, ge=0)
    venue_id: Optional[str] = None
    venue_name: Optional[str] = None
    venue_city: Optional[str] = None
    venue_state: Optional[str] = None
    venue_capacity: Optional[int] = Field(None, ge=0)
    weather_temp_f: Optional[float] = Field(None, ge=-50, le=150)
    weather_conditions: Optional[str] = None
    broadcast_network: Optional[str] = None
    source_path: Optional[str] = None

    @field_validator("season_year", mode="before")
    @classmethod
    def check_season(cls, v: Any) -> Optional[int]:
        if v is None:
            return None
        return _validate_season(int(v))

    @field_validator("week", mode="before")
    @classmethod
    def check_week(cls, v: Any) -> Optional[int]:
        if v is None:
            return None
        return _validate_week(int(v))

    @field_validator("game_id", mode="before")
    @classmethod
    def check_game_id(cls, v: Any) -> str:
        s = str(v).strip()
        if not s:
            raise ValueError("game_id must not be empty")
        return s


class SportradarRosterRecord(_NFLBase):
    """One player row produced by sportradar_transform.transform_rosters()."""

    team_id: Optional[str] = None
    team_name: Optional[str] = None
    team_market: Optional[str] = None
    team_alias: Optional[str] = None
    player_id: Optional[str] = None
    player_name: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    jersey: Optional[str] = None
    position: Optional[str] = None
    status: Optional[str] = None          # ACT, IR, PUP, …
    birth_date: Optional[str] = None
    height: Optional[int] = Field(None, ge=0)   # inches
    weight: Optional[int] = Field(None, ge=0)   # lbs
    college: Optional[str] = None
    rookie_year: Optional[int] = None
    experience: Optional[int] = Field(None, ge=0)
    sr_id: Optional[str] = None
    snapshot_ts: Optional[str] = None
    source_path: Optional[str] = None
    # Summary columns (roster_status_summary table)
    season: Optional[int] = None
    roster_total_players: Optional[int] = Field(None, ge=0)
    roster_active_count: Optional[int] = Field(None, ge=0)
    roster_injured_count: Optional[int] = Field(None, ge=0)
    roster_injured_pct: Optional[float] = Field(None, ge=0, le=1)
    roster_active_pct: Optional[float] = Field(None, ge=0, le=1)
    roster_active_skill_count: Optional[int] = Field(None, ge=0)

    @field_validator("season", mode="before")
    @classmethod
    def check_season(cls, v: Any) -> Optional[int]:
        if v is None:
            return None
        return _validate_season(int(v))

    @field_validator("rookie_year", mode="before")
    @classmethod
    def check_rookie_year(cls, v: Any) -> Optional[int]:
        if v is None:
            return None
        yr = int(v)
        if not (1920 <= yr <= 2100):
            raise ValueError(f"rookie_year={yr} is outside plausible range")
        return yr


class SportradarGameStatsRecord(_NFLBase):
    """One statistic row from sportradar_transform for game-level stats."""

    game_id: str
    team_id: Optional[str] = None
    team_role: Optional[str] = None       # home | away
    scope: Optional[str] = None           # team | player
    stat_category: Optional[str] = None
    stats: Optional[Dict[str, Any]] = None
    season_year: Optional[int] = None
    week: Optional[int] = None

    @field_validator("game_id", mode="before")
    @classmethod
    def check_game_id(cls, v: Any) -> str:
        s = str(v).strip()
        if not s:
            raise ValueError("game_id must not be empty")
        return s

    @field_validator("team_role", mode="before")
    @classmethod
    def check_team_role(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        lower = str(v).strip().lower()
        if lower not in ("home", "away"):
            raise ValueError(f"team_role='{lower}' must be 'home' or 'away'")
        return lower

    @field_validator("season_year", mode="before")
    @classmethod
    def check_season(cls, v: Any) -> Optional[int]:
        if v is None:
            return None
        return _validate_season(int(v))

    @field_validator("week", mode="before")
    @classmethod
    def check_week(cls, v: Any) -> Optional[int]:
        if v is None:
            return None
        return _validate_week(int(v))


# ===========================================================================
# Weather schemas (3)
# ===========================================================================

class NOAAWeatherRecord(_NFLBase):
    """One observation row produced by src.data.noaa.build_noaa_weather()."""

    game_id: str
    season: Optional[int] = None
    week: Optional[int] = None
    home_team: Optional[str] = None
    away_team: Optional[str] = None
    kickoff_utc: Optional[str] = None          # ISO-8601 datetime
    station_id: Optional[str] = None
    weather_temp_f: Optional[float] = Field(None, ge=-80, le=150)
    weather_wind_mph: Optional[float] = Field(None, ge=0, le=250)
    weather_humidity_pct: Optional[float] = Field(None, ge=0, le=100)
    weather_precip_in: Optional[float] = Field(None, ge=0)
    weather_wind_dir: Optional[str] = None
    weather_conditions: Optional[str] = None
    is_dome: Optional[bool] = None
    lat: Optional[float] = Field(None, ge=-90, le=90)
    lon: Optional[float] = Field(None, ge=-180, le=180)

    @field_validator("game_id", mode="before")
    @classmethod
    def check_game_id(cls, v: Any) -> str:
        s = str(v).strip()
        if not s:
            raise ValueError("game_id must not be empty")
        return s

    @field_validator("season", mode="before")
    @classmethod
    def check_season(cls, v: Any) -> Optional[int]:
        if v is None:
            return None
        return _validate_season(int(v))

    @field_validator("week", mode="before")
    @classmethod
    def check_week(cls, v: Any) -> Optional[int]:
        if v is None:
            return None
        return _validate_week(int(v))

    @field_validator("home_team", "away_team", mode="before")
    @classmethod
    def check_team(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        return _validate_nfl_team(str(v), "team")


class VisualCrossingWeatherRecord(_NFLBase):
    """One row from src.data.visualcrossing.build_visualcrossing_weather()."""

    game_id: str
    season: Optional[int] = None
    week: Optional[int] = None
    home_team: Optional[str] = None
    away_team: Optional[str] = None
    kickoff_utc: Optional[str] = None
    # Core weather observations / forecasts
    temp_f: Optional[float] = Field(None, ge=-80, le=150)
    feelslike_f: Optional[float] = Field(None, ge=-100, le=180)
    humidity_pct: Optional[float] = Field(None, ge=0, le=100)
    windspeed_mph: Optional[float] = Field(None, ge=0, le=250)
    winddir_deg: Optional[float] = Field(None, ge=0, le=360)
    visibility_miles: Optional[float] = Field(None, ge=0)
    cloudcover_pct: Optional[float] = Field(None, ge=0, le=100)
    precip_in: Optional[float] = Field(None, ge=0)
    snow_in: Optional[float] = Field(None, ge=0)
    uvindex: Optional[float] = Field(None, ge=0, le=11)
    conditions: Optional[str] = None
    description: Optional[str] = None
    is_dome: Optional[bool] = None

    @field_validator("game_id", mode="before")
    @classmethod
    def check_game_id(cls, v: Any) -> str:
        s = str(v).strip()
        if not s:
            raise ValueError("game_id must not be empty")
        return s

    @field_validator("season", mode="before")
    @classmethod
    def check_season(cls, v: Any) -> Optional[int]:
        if v is None:
            return None
        return _validate_season(int(v))

    @field_validator("week", mode="before")
    @classmethod
    def check_week(cls, v: Any) -> Optional[int]:
        if v is None:
            return None
        return _validate_week(int(v))

    @field_validator("home_team", "away_team", mode="before")
    @classmethod
    def check_team(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        return _validate_nfl_team(str(v), "team")


class GameWeatherRecord(_NFLBase):
    """Canonical merged game-weather row used by the feature pipeline.

    Produced by src.data.weather.build_game_weather() which blends NOAA /
    Visual Crossing / Tomorrow.io sources into a single flat record keyed on
    game_id.
    """

    game_id: str
    season: Optional[int] = None
    week: Optional[int] = None
    home_team: Optional[str] = None
    away_team: Optional[str] = None
    # Canonical feature columns (prefixed weather_*)
    weather_temp_f: Optional[float] = Field(None, ge=-80, le=150)
    weather_wind_mph: Optional[float] = Field(None, ge=0, le=250)
    weather_humidity_pct: Optional[float] = Field(None, ge=0, le=100)
    weather_precip_in: Optional[float] = Field(None, ge=0)
    weather_snow_in: Optional[float] = Field(None, ge=0)
    weather_conditions: Optional[str] = None
    weather_is_dome: Optional[bool] = None
    weather_source: Optional[str] = None    # noaa | visualcrossing | tomorrow

    @field_validator("game_id", mode="before")
    @classmethod
    def check_game_id(cls, v: Any) -> str:
        s = str(v).strip()
        if not s:
            raise ValueError("game_id must not be empty")
        return s

    @field_validator("season", mode="before")
    @classmethod
    def check_season(cls, v: Any) -> Optional[int]:
        if v is None:
            return None
        return _validate_season(int(v))

    @field_validator("week", mode="before")
    @classmethod
    def check_week(cls, v: Any) -> Optional[int]:
        if v is None:
            return None
        return _validate_week(int(v))

    @field_validator("home_team", "away_team", mode="before")
    @classmethod
    def check_team(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        return _validate_nfl_team(str(v), "team")

    @field_validator("weather_source", mode="before")
    @classmethod
    def check_source(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        lower = str(v).strip().lower()
        valid = {"noaa", "visualcrossing", "tomorrow", "openweather", "manual"}
        if lower not in valid:
            raise ValueError(f"weather_source='{lower}' must be one of {sorted(valid)}")
        return lower


# ===========================================================================
# Yahoo schema (1)
# ===========================================================================

class YahooPlayerRecord(_NFLBase):
    """One player row from the Yahoo Fantasy Sports API (game/{key}/players)."""

    player_id: Optional[int] = None
    player_key: Optional[str] = None
    full_name: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    editorial_team_abbr: Optional[str] = None   # team abbreviation as Yahoo reports it
    editorial_team_full_name: Optional[str] = None
    display_position: Optional[str] = None
    eligible_positions: Optional[Any] = None    # list of position codes
    uniform_number: Optional[str] = None
    status: Optional[str] = None                # A (active), IR, NA, …
    injury_note: Optional[str] = None
    is_undroppable: Optional[bool] = None
    position_type: Optional[str] = None
    season: Optional[int] = None
    game_key: Optional[str] = None

    @field_validator("player_id", mode="before")
    @classmethod
    def check_player_id(cls, v: Any) -> Optional[int]:
        if v is None:
            return None
        val = int(v)
        if val <= 0:
            raise ValueError(f"player_id={val} must be a positive integer")
        return val

    @field_validator("season", mode="before")
    @classmethod
    def check_season(cls, v: Any) -> Optional[int]:
        if v is None:
            return None
        return _validate_season(int(v))


# ===========================================================================
# Additional schema models (player actuals, SportsDataIO)
# ===========================================================================


class PlayerActualsRecord(_NFLBase):
    """Per-game player actuals derived from NFLverse play-by-play aggregation.

    Produced by ``src/data/player_actuals.py``.
    """

    game_id: str
    season: int
    week: int
    team_alias: str
    player_id: Optional[str] = None
    player_name: Optional[str] = None
    stat_category: str = Field(description="passing | rushing | receiving | defense")
    stats: Optional[str] = None          # JSON blob e.g. '{"yards": 120.0}'
    value: Optional[float] = None        # convenience numeric extracted from stats

    @field_validator("season", mode="before")
    @classmethod
    def _chk_season(cls, v: Any) -> int:
        return _validate_season(int(v))

    @field_validator("week", mode="before")
    @classmethod
    def _chk_week(cls, v: Any) -> int:
        return _validate_week(int(v))

    @field_validator("team_alias", mode="before")
    @classmethod
    def _chk_team(cls, v: Any) -> str:
        result = _validate_nfl_team(str(v) if v is not None else None, "team_alias")
        return result if result is not None else str(v)

    @field_validator("game_id", mode="before")
    @classmethod
    def _chk_game_id(cls, v: Any) -> str:
        s = str(v).strip()
        if not _NFLVERSE_GAME_ID_RE.match(s):
            raise ValueError(f"game_id '{s}' does not match expected format YYYY_WW_HOME_AWAY")
        return s

    @field_validator("stat_category", mode="before")
    @classmethod
    def _chk_stat_category(cls, v: Any) -> str:
        valid = {"passing", "rushing", "receiving", "defense", "kicking"}
        s = str(v).lower().strip()
        if s not in valid:
            raise ValueError(f"stat_category '{s}' not in {valid}")
        return s


class SportsDataIORecord(_NFLBase):
    """Generic record from the SportsDataIO NFL API.

    SportsDataIO serves many distinct feeds (teams, stadiums, schedules,
    standings, projections, betting_futures, draft_picks, free_agents).
    This schema captures the common envelope fields present across responses,
    plus a flexible ``data`` dict for feed-specific payload.
    """

    feed_name: str = Field(description="API feed type (teams, schedules, standings, …)")
    season: Optional[int] = None
    week: Optional[int] = None
    team: Optional[str] = None
    game_id: Optional[str] = None
    player_id: Optional[Any] = None
    name: Optional[str] = None
    data: Optional[Dict[str, Any]] = None

    @field_validator("season", mode="before")
    @classmethod
    def _chk_season(cls, v: Any) -> Optional[int]:
        if _is_na(v):
            return None
        return _validate_season(int(v))

    @field_validator("week", mode="before")
    @classmethod
    def _chk_week(cls, v: Any) -> Optional[int]:
        if _is_na(v):
            return None
        return _validate_week(int(v))

    @field_validator("team", mode="before")
    @classmethod
    def _chk_team(cls, v: Any) -> Optional[str]:
        return _validate_nfl_team(str(v) if v is not None else None, "team")


# ===========================================================================
# DataFrame-level validation
# ===========================================================================


@dataclass
class ValidationReport:
    """Result of validating a DataFrame against a Pydantic schema.

    Attributes:
        schema_key:         Registry key used for validation.
        total_rows:         Total rows in the input DataFrame.
        valid_rows:         Rows that passed validation.
        invalid_rows:       Rows that failed validation.
        pass_rate:          Fraction of rows that passed (0.0–1.0).
        errors:             List of ``(row_index, field, message)`` triples.
        field_error_counts: Mapping of field name → number of failing rows.
    """

    schema_key: str
    total_rows: int
    valid_rows: int
    invalid_rows: int
    pass_rate: float
    errors: List[Tuple[int, str, str]] = field(default_factory=list)
    field_error_counts: Dict[str, int] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"ValidationReport({self.schema_key}): "
            f"{self.valid_rows}/{self.total_rows} rows valid "
            f"({self.pass_rate:.1%})"
        )

    @property
    def passed(self) -> bool:
        """True when every row passed validation."""
        return self.invalid_rows == 0


def validate_dataframe(
    df: pd.DataFrame,
    schema_key: str,
    *,
    warn_threshold: float = 0.05,
    log_errors: bool = True,
) -> ValidationReport:
    """Validate every row of *df* against the registered Pydantic schema.

    Never raises; returns a :class:`ValidationReport` so callers can decide
    how to handle failures (log, alert, skip rows, etc.).

    Args:
        df:              DataFrame to validate.
        schema_key:      Key into :data:`SCHEMA_MODELS`, e.g. ``"nflverse_schedule"``.
        warn_threshold:  Log a WARNING when the error rate exceeds this fraction.
        log_errors:      Emit per-row DEBUG log lines for validation failures.

    Returns:
        :class:`ValidationReport` with pass rate and field-level error counts.

    Raises:
        KeyError: When *schema_key* is not in the registry.
    """
    model_cls: Type[_NFLBase] = SCHEMA_MODELS[schema_key]

    errors: List[Tuple[int, str, str]] = []
    field_error_counts: Dict[str, int] = {}
    valid_rows = 0

    for idx, row in enumerate(df.to_dict(orient="records")):
        try:
            model_cls.model_validate(row)
            valid_rows += 1
        except ValidationError as exc:
            for err in exc.errors():
                field_name = ".".join(str(loc) for loc in err["loc"]) or "_root_"
                msg = err["msg"]
                errors.append((idx, field_name, msg))
                field_error_counts[field_name] = field_error_counts.get(field_name, 0) + 1
                if log_errors:
                    logger.debug(
                        "[schema:%s] row %d field '%s': %s",
                        schema_key, idx, field_name, msg,
                    )

    total = len(df)
    invalid = total - valid_rows
    pass_rate = valid_rows / total if total > 0 else 1.0

    report = ValidationReport(
        schema_key=schema_key,
        total_rows=total,
        valid_rows=valid_rows,
        invalid_rows=invalid,
        pass_rate=pass_rate,
        errors=errors,
        field_error_counts=field_error_counts,
    )

    if total > 0 and (1.0 - pass_rate) > warn_threshold:
        logger.warning(
            "[schema:%s] high error rate %.1f%% (%d/%d rows failed). "
            "Top fields: %s",
            schema_key,
            (1.0 - pass_rate) * 100,
            invalid,
            total,
            sorted(field_error_counts.items(), key=lambda x: -x[1])[:5],
        )
    else:
        logger.info(
            "[schema:%s] validation complete — %d/%d rows valid (%.1f%%)",
            schema_key, valid_rows, total, pass_rate * 100,
        )

    return report


# ===========================================================================
# Schema versioning registry
# ===========================================================================

SCHEMA_VERSIONS: Dict[str, str] = {
    # NFLverse
    "nflverse_schedule":      NFLverseScheduleRecord.schema_version,
    "nflverse_pbp":           NFLversePlayRecord.schema_version,
    "nflverse_roster":        NFLverseRosterRecord.schema_version,
    "nflverse_injury":        NFLverseInjuryRecord.schema_version,
    "nflverse_player_stats":  NFLversePlayerStatsRecord.schema_version,
    # ESPN
    "espn_player_stats":      ESPNPlayerStatRecord.schema_version,
    "espn_player_news":       ESPNPlayerNewsRecord.schema_version,
    "espn_team_news":         ESPNTeamNewsRecord.schema_version,
    "espn_team_defense":      ESPNTeamDefenseRecord.schema_version,
    # Sportradar
    "sportradar_schedule":    SportradarScheduleRecord.schema_version,
    "sportradar_roster":      SportradarRosterRecord.schema_version,
    "sportradar_game_stats":  SportradarGameStatsRecord.schema_version,
    # Weather
    "noaa_weather":           NOAAWeatherRecord.schema_version,
    "visualcrossing_weather": VisualCrossingWeatherRecord.schema_version,
    "game_weather":           GameWeatherRecord.schema_version,
    # Yahoo
    "yahoo_player":           YahooPlayerRecord.schema_version,
    # Player actuals / SportsDataIO
    "player_actuals":         PlayerActualsRecord.schema_version,
    "sportsdataio":           SportsDataIORecord.schema_version,
}

# Convenience map from schema key → model class for dynamic lookup
SCHEMA_MODELS: Dict[str, type[_NFLBase]] = {
    "nflverse_schedule":      NFLverseScheduleRecord,
    "nflverse_pbp":           NFLversePlayRecord,
    "nflverse_roster":        NFLverseRosterRecord,
    "nflverse_injury":        NFLverseInjuryRecord,
    "nflverse_player_stats":  NFLversePlayerStatsRecord,
    "espn_player_stats":      ESPNPlayerStatRecord,
    "espn_player_news":       ESPNPlayerNewsRecord,
    "espn_team_news":         ESPNTeamNewsRecord,
    "espn_team_defense":      ESPNTeamDefenseRecord,
    "sportradar_schedule":    SportradarScheduleRecord,
    "sportradar_roster":      SportradarRosterRecord,
    "sportradar_game_stats":  SportradarGameStatsRecord,
    "noaa_weather":           NOAAWeatherRecord,
    "visualcrossing_weather": VisualCrossingWeatherRecord,
    "game_weather":           GameWeatherRecord,
    "yahoo_player":           YahooPlayerRecord,
    "player_actuals":         PlayerActualsRecord,
    "sportsdataio":           SportsDataIORecord,
}


def get_schema_version(schema_key: str) -> str:
    """Return the current version string for *schema_key*.

    Raises :class:`KeyError` if the key is not registered.
    """
    return SCHEMA_VERSIONS[schema_key]


def validate_record(schema_key: str, data: Dict[str, Any]) -> _NFLBase:
    """Validate a single raw record dict against the named schema.

    Args:
        schema_key: Registry key, e.g. ``"nflverse_schedule"``.
        data: Raw dict from the data fetcher.

    Returns:
        Validated Pydantic model instance.

    Raises:
        KeyError: If *schema_key* is not registered.
        pydantic.ValidationError: If the record fails validation.
    """
    model_cls = SCHEMA_MODELS[schema_key]
    return model_cls.model_validate(data)
