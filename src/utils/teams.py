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

"""Centralized team name and abbreviation mappings for NFL teams."""

from __future__ import annotations

from typing import Any, Dict, Optional

# Common alternate abbreviations mapped to canonical form
ALT_ABBR_MAP: Dict[str, Optional[str]] = {
    # Common alternates → canonical
    "JAC": "JAX",
    "WSH": "WAS",
    "ARZ": "ARI",
    "KAN": "KC",
    "NOR": "NO",
    "TAM": "TB",
    "GNB": "GB",
    "SFO": "SF",
    "NWE": "NE",
    "RAM": "LAR",
    "SD": "LAC",
    "STL": "LAR",
    "OAK": "LV",
    # Ambiguous historical "LA" (Rams/Chargers) — try resolving via team name if provided
    "LA": None,
}

# Full team names mapped to canonical abbreviations
TEAM_NAME_TO_ABBR: Dict[str, str] = {
    # Current names
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LAR",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
    # Historical → modern
    "Oakland Raiders": "LV",
    "San Diego Chargers": "LAC",
    "St. Louis Rams": "LAR",
    "Washington Football Team": "WAS",
    "Washington Redskins": "WAS",
    "Phoenix Cardinals": "ARI",
    "Los Angeles Raiders": "LV",
}


def normalize_team_abbr(val: Any, fallback_name: Any = None) -> Any:
    """
    Normalize a team abbreviation to its canonical form.
    
    Handles cases like JAC→JAX, WSH→WAS, etc. If the abbreviation is 'LA',
    attempts to resolve via fallback team name to LAR/LAC; otherwise keeps 'LA'.
    
    Args:
        val: Team abbreviation to normalize (typically a string).
        fallback_name: Optional team name to help resolve ambiguous abbreviations.
        
    Returns:
        Canonical team abbreviation, or the original value if not a string or not found.
        
    Examples:
        >>> normalize_team_abbr("JAC")
        'JAX'
        >>> normalize_team_abbr("LA", "Los Angeles Rams")
        'LAR'
        >>> normalize_team_abbr("LA", "Los Angeles Chargers")
        'LAC'
    """
    if not isinstance(val, str):
        return val
    
    v = val.strip().upper()
    
    if v in ALT_ABBR_MAP:
        canon = ALT_ABBR_MAP[v]
        if canon is not None:
            return canon
        
        # v == 'LA' case — try resolve via name
        if isinstance(fallback_name, str):
            name = fallback_name.lower()
            if "rams" in name:
                return "LAR"
            if "chargers" in name:
                return "LAC"
        return v
    
    return v


def get_team_abbr_from_name(team_name: str) -> Optional[str]:
    """
    Get the canonical team abbreviation from a full team name.
    
    Args:
        team_name: Full team name (e.g., "Arizona Cardinals").
        
    Returns:
        Canonical abbreviation (e.g., "ARI"), or None if not found.
        
    Examples:
        >>> get_team_abbr_from_name("Arizona Cardinals")
        'ARI'
        >>> get_team_abbr_from_name("Oakland Raiders")
        'LV'
    """
    return TEAM_NAME_TO_ABBR.get(team_name)

# Made with Bob
