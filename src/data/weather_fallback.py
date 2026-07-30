# Copyright (c) 2025 Matt Hanson
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Static climatological weather fallback for NFL stadiums.

Provides last-resort weather estimates when all live sources (Visual Crossing,
NOAA, Tomorrow.io) fail to return data for a game. Values are monthly
climatological normals derived from 30-year NOAA climate averages for each
stadium's city.

Usage
-----
    from src.data.weather_fallback import get_fallback_weather

    row = get_fallback_weather("DEN", game_date=date(2024, 10, 6))
    # Returns: {"weather_temp_f": 56.0, "weather_wind_mph": 10.2, "weather_precip_prob": 0.18}
"""

from __future__ import annotations

from datetime import date
from typing import Optional


# ---------------------------------------------------------------------------
# Monthly climatological normals per team/stadium (30-year averages).
# Format: {TEAM_ABBR: [Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec]}
# Each month entry: (avg_temp_f, avg_wind_mph, avg_precip_prob)
# ---------------------------------------------------------------------------
_CLIMATE: dict[str, list[tuple[float, float, float]]] = {
    # (avg_high+avg_low)/2 °F, avg wind mph, monthly precip days / 30
    "ARI": [(54,7,.15),(57,7,.14),(63,8,.13),(72,8,.09),(81,8,.06),(91,9,.07),(95,11,.22),(93,10,.25),(88,9,.19),(76,8,.12),(63,7,.11),(54,7,.14)],
    "ATL": [(43,9,.33),(47,9,.32),(55,9,.30),(64,8,.28),(72,8,.29),(79,8,.32),(82,8,.37),(81,7,.31),(75,7,.29),(64,7,.24),(54,8,.29),(45,9,.31)],
    "BAL": [(34,10,.34),(37,10,.30),(46,11,.32),(57,10,.30),(67,9,.31),(76,9,.30),(81,8,.31),(79,8,.30),(72,8,.28),(60,8,.25),(50,9,.29),(38,10,.33)],
    "BUF": [(26,13,.41),(27,12,.39),(36,13,.38),(49,12,.34),(60,11,.33),(70,10,.31),(75,9,.30),(73,9,.31),(65,10,.34),(53,11,.37),(42,12,.42),(30,13,.44)],
    "CAR": [(41,9,.30),(45,9,.29),(53,9,.29),(63,8,.27),(71,7,.29),(79,7,.32),(83,6,.35),(82,6,.33),(75,7,.28),(64,7,.24),(54,8,.28),(44,8,.29)],
    "CHI": [(25,12,.38),(29,12,.37),(40,13,.38),(53,12,.36),(63,11,.36),(73,10,.36),(79,9,.36),(77,9,.35),(69,10,.34),(57,11,.36),(44,12,.39),(30,12,.40)],
    "CIN": [(31,10,.33),(35,10,.32),(45,11,.34),(57,10,.33),(67,9,.36),(76,8,.33),(80,7,.33),(78,7,.31),(71,8,.28),(59,8,.26),(48,9,.31),(35,10,.34)],
    "CLE": [(27,13,.37),(30,13,.36),(40,13,.39),(52,12,.37),(63,11,.37),(72,10,.34),(77,9,.34),(75,9,.33),(67,9,.32),(55,10,.34),(45,11,.40),(32,12,.41)],
    "DAL": [(45,12,.27),(50,12,.27),(59,13,.26),(68,13,.28),(76,12,.36),(85,12,.33),(89,11,.26),(89,10,.22),(82,11,.26),(71,11,.28),(59,12,.27),(48,12,.27)],
    "DEN": [(30,8,.19),(34,9,.20),(42,10,.24),(52,11,.28),(62,11,.31),(72,10,.24),(78,9,.29),(76,9,.25),(67,10,.23),(54,9,.21),(41,9,.22),(32,8,.19)],
    "DET": [(24,12,.37),(27,12,.36),(37,13,.38),(50,12,.36),(61,11,.35),(71,10,.35),(76,9,.34),(74,9,.34),(66,9,.31),(54,10,.34),(43,11,.38),(29,12,.40)],
    "GB":  [(17,13,.41),(21,13,.39),(31,13,.40),(46,12,.37),(57,11,.36),(67,10,.36),(73,9,.36),(70,9,.36),(62,10,.37),(50,11,.39),(37,12,.42),(23,13,.44)],
    "HOU": [(51,10,.32),(55,10,.31),(63,11,.29),(71,10,.30),(78,10,.36),(85,10,.38),(88,9,.34),(88,8,.33),(83,9,.39),(74,9,.35),(63,10,.31),(54,10,.31)],
    "IND": [(28,11,.36),(33,11,.34),(44,12,.37),(56,11,.37),(66,10,.40),(75,9,.38),(79,8,.38),(77,8,.34),(70,9,.29),(57,10,.29),(46,11,.34),(32,11,.37)],
    "JAX": [(53,9,.28),(56,9,.27),(63,9,.26),(70,9,.26),(77,9,.29),(82,9,.43),(84,8,.47),(84,7,.46),(81,8,.41),(74,8,.29),(65,9,.26),(56,9,.27)],
    "KC":  [(28,12,.28),(34,13,.28),(45,14,.34),(58,13,.37),(68,12,.40),(78,11,.38),(84,10,.34),(82,10,.31),(73,11,.33),(61,11,.33),(47,12,.31),(33,12,.29)],
    "LAC": [(58,7,.20),(60,7,.18),(62,7,.16),(65,7,.10),(68,7,.06),(72,7,.04),(76,6,.02),(77,6,.04),(76,6,.06),(71,6,.10),(65,7,.14),(59,7,.19)],
    "LAR": [(58,7,.20),(60,7,.18),(62,7,.16),(65,7,.10),(68,7,.06),(72,7,.04),(76,6,.02),(77,6,.04),(76,6,.06),(71,6,.10),(65,7,.14),(59,7,.19)],
    "LA":  [(58,7,.20),(60,7,.18),(62,7,.16),(65,7,.10),(68,7,.06),(72,7,.04),(76,6,.02),(77,6,.04),(76,6,.06),(71,6,.10),(65,7,.14),(59,7,.19)],
    "LV":  [(46,8,.08),(51,9,.09),(58,10,.09),(68,11,.07),(78,11,.04),(89,12,.04),(95,10,.07),(93,9,.08),(85,9,.06),(72,8,.05),(57,8,.07),(47,8,.08)],
    "MIA": [(68,10,.21),(69,10,.19),(72,10,.20),(76,10,.22),(80,10,.32),(83,11,.42),(85,11,.44),(85,10,.44),(84,11,.44),(80,10,.34),(75,10,.22),(70,10,.20)],
    "MIN": [(14,11,.40),(19,11,.37),(32,12,.39),(48,12,.37),(61,11,.39),(71,10,.39),(77,9,.38),(74,9,.36),(64,10,.38),(51,11,.38),(35,11,.41),(20,11,.42)],
    "NE":  [(29,14,.39),(31,14,.36),(39,14,.39),(50,13,.37),(60,12,.35),(70,11,.33),(76,10,.31),(75,10,.31),(67,11,.31),(56,12,.34),(46,13,.39),(33,14,.41)],
    "NO":  [(51,9,.34),(55,9,.32),(62,10,.31),(70,9,.31),(77,9,.35),(83,8,.47),(85,7,.49),(86,7,.49),(82,8,.44),(73,8,.30),(63,9,.30),(54,9,.33)],
    "NYG": [(32,12,.35),(35,12,.32),(43,13,.37),(54,12,.34),(64,11,.34),(74,10,.32),(80,9,.29),(78,9,.31),(71,10,.29),(59,11,.29),(49,12,.34),(36,12,.36)],
    "NYJ": [(32,12,.35),(35,12,.32),(43,13,.37),(54,12,.34),(64,11,.34),(74,10,.32),(80,9,.29),(78,9,.31),(71,10,.29),(59,11,.29),(49,12,.34),(36,12,.36)],
    "PHI": [(34,11,.33),(37,11,.30),(46,12,.33),(57,11,.31),(67,10,.33),(76,9,.31),(82,9,.30),(80,8,.30),(73,9,.27),(61,9,.24),(51,10,.29),(38,11,.33)],
    "PIT": [(29,11,.34),(33,11,.32),(43,12,.36),(55,11,.35),(65,10,.38),(73,9,.35),(78,8,.34),(76,8,.33),(68,9,.29),(56,9,.27),(46,10,.33),(33,11,.36)],
    "SEA": [(40,9,.42),(43,9,.36),(47,9,.35),(51,9,.29),(57,8,.24),(62,8,.17),(67,7,.12),(68,7,.13),(63,8,.19),(54,9,.32),(46,9,.42),(41,9,.46)],
    "SF":  [(49,10,.30),(52,11,.26),(54,12,.27),(57,12,.19),(60,13,.12),(63,14,.06),(65,13,.02),(65,12,.03),(64,12,.08),(60,11,.15),(54,10,.27),(49,10,.31)],
    "TB":  [(60,10,.24),(62,10,.23),(67,10,.24),(73,10,.24),(79,9,.29),(84,9,.42),(86,8,.47),(87,7,.46),(84,8,.40),(77,8,.27),(69,9,.23),(63,9,.22)],
    "TEN": [(37,9,.32),(42,9,.31),(51,10,.32),(62,9,.33),(70,8,.37),(78,8,.36),(82,7,.34),(81,7,.30),(74,8,.27),(62,8,.24),(52,9,.30),(40,9,.32)],
    "WAS": [(35,10,.31),(38,10,.28),(47,11,.31),(58,10,.30),(68,9,.32),(77,9,.29),(82,8,.29),(80,8,.30),(73,9,.27),(61,9,.24),(51,10,.29),(39,10,.32)],
}

# Dome teams: weather is irrelevant for home games (should already be zeroed by indoor_mask,
# but fallback returns neutral values to avoid interfering with that logic).
_DOME_TEAMS = {"ARI", "ATL", "DAL", "DET", "HOU", "IND", "LAC", "LAR", "LA", "LV", "MIN", "NO"}


def get_fallback_weather(
    home_team: str,
    game_date: Optional[date] = None,
    month: Optional[int] = None,
) -> dict[str, float]:
    """
    Return static climatological weather estimates for a game.

    Used as a last-resort fallback when all live weather sources (Visual Crossing,
    NOAA, Tomorrow.io) fail to return data for a game.

    Parameters
    ----------
    home_team : str
        NFL team abbreviation for the home team (e.g. "DEN", "MIA").
    game_date : date, optional
        Game date. Used to extract the month if ``month`` is not provided.
    month : int, optional
        Calendar month (1-12). Takes precedence over ``game_date``.

    Returns
    -------
    dict
        Keys: ``weather_temp_f``, ``weather_wind_mph``, ``weather_precip_prob``.
        Returns neutral indoor values (70°F, 0 mph, 0.0 prob) for dome teams.
    """
    team = str(home_team).strip().upper()

    # Dome teams: return neutral indoor conditions
    if team in _DOME_TEAMS:
        return {"weather_temp_f": 70.0, "weather_wind_mph": 0.0, "weather_precip_prob": 0.0}

    # Resolve month
    if month is None and game_date is not None:
        month = game_date.month
    if month is None:
        month = 10  # NFL season default (October)

    month_idx = max(0, min(11, int(month) - 1))  # 0-indexed, clamped

    climate = _CLIMATE.get(team, _CLIMATE["DAL"])  # DAL as generic neutral fallback
    temp_f, wind_mph, precip_prob = climate[month_idx]

    return {
        "weather_temp_f": float(temp_f),
        "weather_wind_mph": float(wind_mph),
        "weather_precip_prob": float(precip_prob),
    }
