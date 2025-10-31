from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

VOLATILITY_OPTIONAL_COLUMNS: list[str] = [
    "weather_wind_mph",
    "weather_wind_mph_wx",
    "weather_temp_f",
    "weather_temp_f_wx",
    "weather_temp_kickoff_f",
    "weather_temp_kickoff_f_wx",
    "wind",
    "temp",
]

VOLATILITY_REQUIRED_COLUMNS: list[str] = [
    "home_rest",
    "away_rest",
    "sched_rest_days_home",
    "sched_rest_days_away",
    "sched_back_to_back_travel_home",
    "sched_back_to_back_travel_away",
    "roof_is_dome",
    "home_indoor",
    "away_indoor",
    "away_travel_distance_km",
    "away_travel_distance_miles",
    "timezone_diff_hours",
    "timezone_diff_hours_abs",
    "away_travel_east",
    "away_travel_west",
    "travel_km_short_rest",
    "travel_km_back_to_back",
    "travel_km_per_rest_day",
    "timezone_diff_short_rest",
    "inj_qb_questionable_home",
    "inj_qb_questionable_away",
    "inj_qb_doubtful_home",
    "inj_qb_doubtful_away",
    "inj_qb_out_home",
    "inj_qb_out_away",
    "inj_qb_reserve_home",
    "inj_qb_reserve_away",
    "inj_practice_limited_home",
    "inj_practice_limited_away",
    "inj_practice_limited_rolling3_home",
    "inj_practice_limited_rolling3_away",
    "inj_qb_questionable_rolling3_home",
    "inj_qb_questionable_rolling3_away",
    "inj_qb_doubtful_rolling3_home",
    "inj_qb_doubtful_rolling3_away",
    "inj_qb_reserve_rolling3_home",
    "inj_qb_reserve_rolling3_away",
]

VOLATILITY_FEATURE_COLUMNS: list[str] = [
    "wind_mph",
    "temp_f",
    "weather_temp_deviation",
    "wind_temp_interaction",
    "weather_temp_delta",
    "wind_mph_delta",
    "qb_uncertainty_score",
    "qb_uncertain",
    "qb_uncertainty_recent",
    "qb_uncertainty_trend",
    "short_rest",
    "back_to_back_travel",
    "indoor_game",
    "outdoor_game",
    "home_rest",
    "away_rest",
    "sched_rest_days_home",
    "sched_rest_days_away",
    "sched_back_to_back_travel_home",
    "sched_back_to_back_travel_away",
    "away_travel_distance_km",
    "away_travel_distance_miles",
    "timezone_diff_hours",
    "timezone_diff_hours_abs",
    "away_travel_east",
    "away_travel_west",
    "travel_km_short_rest",
    "travel_km_back_to_back",
    "travel_km_per_rest_day",
    "timezone_diff_short_rest",
    "travel_timezone_product",
    "travel_short_rest_flag",
    "travel_rest_pressure",
    "rest_diff",
    "rest_days_diff",
]


@dataclass
class VolatilityFeatureFrame:
    features: pd.DataFrame
    enriched: pd.DataFrame


def _coalesce(df: pd.DataFrame, columns: Iterable[str], default=np.nan) -> pd.Series:
    cols = [c for c in columns if c in df.columns]
    if not cols:
        return pd.Series(default, index=df.index)
    data = df[cols].copy()
    return data.bfill(axis=1).iloc[:, 0].fillna(default)


def engineer_volatility_inputs(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["wind_mph"] = pd.to_numeric(
        _coalesce(enriched, ["weather_wind_mph", "weather_wind_mph_wx", "wind"]), errors="coerce"
    )
    enriched["temp_f"] = pd.to_numeric(
        _coalesce(
            enriched,
            [
                "weather_temp_kickoff_f",
                "weather_temp_kickoff_f_wx",
                "weather_temp_f",
                "weather_temp_f_wx",
                "temp",
            ],
        ),
        errors="coerce",
    )
    enriched["weather_temp_deviation"] = enriched["temp_f"].sub(60.0).abs()
    enriched["wind_temp_interaction"] = enriched["wind_mph"].fillna(0.0) * enriched["weather_temp_deviation"].fillna(0.0)
    baseline_temp = pd.to_numeric(_coalesce(enriched, ["weather_temp_f", "temp"]), errors="coerce")
    enriched["weather_temp_delta"] = enriched["temp_f"] - baseline_temp
    baseline_wind = pd.to_numeric(_coalesce(enriched, ["wind", "weather_wind_mph"]), errors="coerce")
    enriched["wind_mph_delta"] = enriched["wind_mph"] - baseline_wind

    qb_current = [
        "inj_qb_questionable_home",
        "inj_qb_questionable_away",
        "inj_qb_doubtful_home",
        "inj_qb_doubtful_away",
        "inj_qb_out_home",
        "inj_qb_out_away",
        "inj_qb_reserve_home",
        "inj_qb_reserve_away",
    ]
    qb_rolling = [
        "inj_qb_questionable_rolling3_home",
        "inj_qb_questionable_rolling3_away",
        "inj_qb_doubtful_rolling3_home",
        "inj_qb_doubtful_rolling3_away",
        "inj_qb_reserve_rolling3_home",
        "inj_qb_reserve_rolling3_away",
        "inj_practice_limited_rolling3_home",
        "inj_practice_limited_rolling3_away",
    ]
    qb_cols = [c for c in qb_current + qb_rolling if c in enriched.columns]
    if qb_cols:
        qb_df = enriched[qb_cols].fillna(0)
        current_cols = qb_df.columns.intersection(qb_current)
        rolling_cols = qb_df.columns.intersection(qb_rolling)
        current_sum = qb_df[current_cols].sum(axis=1) if len(current_cols) else pd.Series(0, index=qb_df.index)
        rolling_sum = qb_df[rolling_cols].sum(axis=1) if len(rolling_cols) else pd.Series(0, index=qb_df.index)
        enriched["qb_uncertain"] = (current_sum > 0) | (rolling_sum > 0)
        enriched["qb_uncertainty_score"] = current_sum + rolling_sum
        enriched["qb_uncertainty_recent"] = current_sum
        enriched["qb_uncertainty_trend"] = current_sum - rolling_sum
    else:
        enriched["qb_uncertain"] = False
        enriched["qb_uncertainty_score"] = 0.0
        enriched["qb_uncertainty_recent"] = 0.0
        enriched["qb_uncertainty_trend"] = 0.0

    rest_cols = ["home_rest", "away_rest"]
    if all(c in enriched.columns for c in rest_cols):
        rest_vals = enriched[rest_cols].astype(float)
        enriched["short_rest"] = rest_vals.min(axis=1) <= 6
    else:
        enriched["short_rest"] = False

    travel_cols = ["sched_back_to_back_travel_home", "sched_back_to_back_travel_away"]
    if all(c in enriched.columns for c in travel_cols):
        travel_vals = enriched[travel_cols].fillna(0)
        enriched["back_to_back_travel"] = (travel_vals > 0).any(axis=1)
    else:
        enriched["back_to_back_travel"] = False

    indoor_cols = ["roof_is_dome", "home_indoor", "away_indoor"]
    indoor_vals = enriched[indoor_cols].fillna(0) if all(c in enriched.columns for c in indoor_cols) else pd.DataFrame(
        0, index=enriched.index, columns=indoor_cols
    )
    enriched["indoor_game"] = indoor_vals.astype(bool).any(axis=1)
    enriched["outdoor_game"] = ~enriched["indoor_game"]

    if all(c in enriched.columns for c in rest_cols):
        enriched["rest_diff"] = enriched["home_rest"].astype(float) - enriched["away_rest"].astype(float)
    else:
        enriched["rest_diff"] = 0.0
    if "sched_rest_days_home" in enriched.columns and "sched_rest_days_away" in enriched.columns:
        enriched["rest_days_diff"] = (
            enriched["sched_rest_days_home"].astype(float) - enriched["sched_rest_days_away"].astype(float)
        )
    else:
        enriched["rest_days_diff"] = 0.0

    distance_series = enriched["away_travel_distance_km"] if "away_travel_distance_km" in enriched.columns else pd.Series(0, index=enriched.index)
    tz_series = enriched["timezone_diff_hours_abs"] if "timezone_diff_hours_abs" in enriched.columns else pd.Series(0, index=enriched.index)
    enriched["travel_timezone_product"] = distance_series.fillna(0).astype(float) * tz_series.fillna(0).astype(float)
    short_rest_series = (
        enriched["travel_km_short_rest"] if "travel_km_short_rest" in enriched.columns else pd.Series(0, index=enriched.index)
    )
    enriched["travel_short_rest_flag"] = (short_rest_series.fillna(0).astype(float) > 0).astype(int)
    away_rest_series = enriched["away_rest"] if "away_rest" in enriched.columns else pd.Series(0, index=enriched.index)
    enriched["travel_rest_pressure"] = distance_series.fillna(0).astype(float) * np.clip(7.0 - away_rest_series.fillna(0).astype(float), 0.0, None)

    return enriched


def build_volatility_feature_matrix(df: pd.DataFrame) -> VolatilityFeatureFrame:
    missing_required = [c for c in VOLATILITY_REQUIRED_COLUMNS if c not in df.columns]
    if missing_required:
        raise KeyError(f"DataFrame missing required volatility columns: {missing_required}")

    enriched = engineer_volatility_inputs(df)

    feature_df = enriched.reindex(columns=VOLATILITY_FEATURE_COLUMNS).copy()

    bool_cols = ["qb_uncertain", "short_rest", "back_to_back_travel", "indoor_game", "outdoor_game", "travel_short_rest_flag"]
    for col in bool_cols:
        if col in feature_df.columns:
            feature_df[col] = feature_df[col].fillna(False).astype(int)

    feature_df = feature_df.fillna(0.0).astype(float)
    return VolatilityFeatureFrame(features=feature_df, enriched=enriched)
