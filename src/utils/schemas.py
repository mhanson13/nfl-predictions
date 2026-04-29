"""
Data schema validation using Pandera.

This module defines schemas for validating DataFrames throughout the pipeline,
ensuring data quality and catching issues early.
"""

from typing import Optional, Any
import pandas as pd
import pandera as pa  # type: ignore
from pandera import Column, DataFrameSchema, Check  # type: ignore


# =========================== Schedule Schema ===========================

schedule_schema = DataFrameSchema(
    {
        "season": Column(int, Check.greater_than_or_equal_to(2000)),
        "week": Column(int, Check.in_range(1, 22)),
        "game_id": Column(str, nullable=False),
        "home_team": Column(str, Check(lambda s: s.str.len() <= 4)),
        "away_team": Column(str, Check(lambda s: s.str.len() <= 4)),
        "home_score": Column(float, nullable=True),
        "away_score": Column(float, nullable=True),
        "gameday": Column(pd.DatetimeTZDtype(tz="UTC"), nullable=True),
    },
    strict=False,  # Allow additional columns
    coerce=True,   # Coerce types when possible
)


# =========================== Play-by-Play Schema ===========================

pbp_schema = DataFrameSchema(
    {
        "season": Column(int, Check.greater_than_or_equal_to(2000)),
        "game_id": Column(str, nullable=False),
        "play_id": Column(nullable=True),
        "posteam": Column(str, nullable=True),
        "defteam": Column(str, nullable=True),
        "yards_gained": Column(float, nullable=True),
        "pass_attempt": Column(float, nullable=True, checks=Check.isin([0, 1])),
        "rush_attempt": Column(float, nullable=True, checks=Check.isin([0, 1])),
        "epa": Column(float, nullable=True),
    },
    strict=False,
    coerce=True,
)


# =========================== Team Stats Schema ===========================

team_stats_schema = DataFrameSchema(
    {
        "season": Column(int, Check.greater_than_or_equal_to(2000)),
        "abbr": Column(str, Check(lambda s: s.str.len() <= 4)),
    },
    strict=False,  # Many dynamic columns
    coerce=True,
)


# =========================== Weather Schema ===========================

weather_schema = DataFrameSchema(
    {
        "game_id": Column(str, nullable=False),
        "season": Column(int, nullable=True),
        "week": Column(int, nullable=True),
        "weather_temp_f": Column(float, nullable=True, checks=Check.in_range(-50, 150)),
        "weather_wind_mph": Column(float, nullable=True, checks=Check.greater_than_or_equal_to(0)),
        "weather_humidity_pct": Column(float, nullable=True, checks=Check.in_range(0, 100)),
    },
    strict=False,
    coerce=True,
)


# =========================== Injury Schema ===========================

injury_schema = DataFrameSchema(
    {
        "season": Column(int, Check.greater_than_or_equal_to(2000)),
        "week": Column(int, Check.in_range(1, 22)),
        "team": Column(str, nullable=False),
        "player_id": Column(nullable=True),
        "status": Column(str, nullable=True),
    },
    strict=False,
    coerce=True,
)


# =========================== Feature Schema ===========================

matchup_features_schema = DataFrameSchema(
    {
        "season": Column(int, Check.greater_than_or_equal_to(2000)),
        "week": Column(int, Check.in_range(1, 22)),
        "game_id": Column(str, nullable=False),
        "home_team": Column(str, nullable=False),
        "away_team": Column(str, nullable=False),
        "home_score": Column(float, nullable=True),
        "away_score": Column(float, nullable=True),
        "home_margin": Column(float, nullable=True),
        "home_win": Column(float, nullable=True, checks=Check.isin([0, 1])),
    },
    strict=False,  # Many feature columns
    coerce=True,
)


# =========================== Prediction Schema ===========================

prediction_schema = DataFrameSchema(
    {
        "game_id": Column(str, nullable=False),
        "season": Column(int, Check.greater_than_or_equal_to(2000)),
        "week": Column(int, Check.in_range(1, 22)),
        "home_team": Column(str, nullable=False),
        "away_team": Column(str, nullable=False),
        "predicted_home_win_prob": Column(
            float, 
            checks=[Check.in_range(0, 1)],
            nullable=False
        ),
        "predicted_spread": Column(float, nullable=True),
        "confidence": Column(
            str,
            nullable=True,
            checks=Check.isin(["low", "medium", "high"])
        ),
    },
    strict=False,
    coerce=True,
)


# =========================== Validation Functions ===========================

def validate_dataframe(
    df: pd.DataFrame,
    schema: Any,  # DataFrameSchema or dict
    name: str = "DataFrame",
    raise_on_error: bool = False
) -> tuple[bool, Optional[str]]:
    """
    Validate a DataFrame against a schema.
    
    Args:
        df: DataFrame to validate
        schema: Pandera schema to validate against or dict of column types
        name: Name of the DataFrame for error messages
        raise_on_error: Whether to raise exception on validation failure
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        if isinstance(schema, dict):
            # Simple dict-based validation
            for col, dtype in schema.items():
                if col not in df.columns:
                    raise ValueError(f"Missing column: {col}")
            return True, None
        else:
            # Pandera schema validation
            schema.validate(df, lazy=True)
            return True, None
    except pa.errors.SchemaErrors as e:  # type: ignore
        error_msg = f"{name} validation failed:\n{e.failure_cases}"
        if raise_on_error:
            raise ValueError(error_msg) from e
        return False, error_msg
    except Exception as e:
        error_msg = f"{name} validation failed: {e}"
        if raise_on_error:
            raise ValueError(error_msg) from e
        return False, error_msg


def validate_schedule(df: pd.DataFrame, raise_on_error: bool = False) -> tuple[bool, Optional[str]]:
    """Validate schedule DataFrame."""
    return validate_dataframe(df, schedule_schema, "Schedule", raise_on_error)


def validate_pbp(df: pd.DataFrame, raise_on_error: bool = False) -> tuple[bool, Optional[str]]:
    """Validate play-by-play DataFrame."""
    return validate_dataframe(df, pbp_schema, "Play-by-Play", raise_on_error)


def validate_team_stats(df: pd.DataFrame, raise_on_error: bool = False) -> tuple[bool, Optional[str]]:
    """Validate team stats DataFrame."""
    return validate_dataframe(df, team_stats_schema, "Team Stats", raise_on_error)


def validate_weather(df: pd.DataFrame, raise_on_error: bool = False) -> tuple[bool, Optional[str]]:
    """Validate weather DataFrame."""
    return validate_dataframe(df, weather_schema, "Weather", raise_on_error)


def validate_injuries(df: pd.DataFrame, raise_on_error: bool = False) -> tuple[bool, Optional[str]]:
    """Validate injuries DataFrame."""
    return validate_dataframe(df, injury_schema, "Injuries", raise_on_error)


def validate_features(df: pd.DataFrame, raise_on_error: bool = False) -> tuple[bool, Optional[str]]:
    """Validate matchup features DataFrame."""
    return validate_dataframe(df, matchup_features_schema, "Features", raise_on_error)


def validate_predictions(df: pd.DataFrame, raise_on_error: bool = False) -> tuple[bool, Optional[str]]:
    """Validate predictions DataFrame."""
    return validate_dataframe(df, prediction_schema, "Predictions", raise_on_error)


# =========================== Data Quality Checks ===========================

def check_data_freshness(df: pd.DataFrame, date_column: str = "gameday", max_age_days: int = 7) -> bool:
    """
    Check if data is fresh (not too old).
    
    Args:
        df: DataFrame with date column
        date_column: Name of the date column
        max_age_days: Maximum age in days
    
    Returns:
        True if data is fresh, False otherwise
    """
    if date_column not in df.columns:
        return False
    
    latest_date = pd.to_datetime(df[date_column]).max()
    age_days = (pd.Timestamp.now(tz="UTC") - latest_date).days
    
    return age_days <= max_age_days


def check_completeness(df: pd.DataFrame, required_columns: list[str], min_fill_rate: float = 0.8) -> dict:
    """
    Check data completeness for required columns.
    
    Args:
        df: DataFrame to check
        required_columns: List of required column names
        min_fill_rate: Minimum acceptable fill rate (0-1)
    
    Returns:
        Dictionary with completeness metrics per column
    """
    results = {}
    for col in required_columns:
        if col not in df.columns:
            results[col] = {"present": False, "fill_rate": 0.0, "passes": False}
        else:
            fill_rate = df[col].notna().mean()
            results[col] = {
                "present": True,
                "fill_rate": fill_rate,
                "passes": fill_rate >= min_fill_rate
            }
    return results


def detect_outliers(df: pd.DataFrame, column: str, n_std: float = 3.0) -> pd.Series:  # type: ignore
    """
    Detect outliers using standard deviation method.
    
    Args:
        df: DataFrame
        column: Column name to check
        n_std: Number of standard deviations for outlier threshold
    
    Returns:
        Boolean Series indicating outliers
    """
    if column not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    
    values = pd.to_numeric(df[column], errors="coerce")
    mean: float = float(values.mean())  # type: ignore
    std: float = float(values.std())  # type: ignore
    
    return (values - mean).abs() > (n_std * std)  # type: ignore

# Made with Bob
