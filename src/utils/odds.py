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

from __future__ import annotations

import math
from typing import Optional, Sequence

import requests


ODDS_BASE_URL = "https://api.the-odds-api.com/v4/sports/{sport}/odds/"


def american_to_decimal(odds: float) -> float:
    """Convert American odds to decimal odds."""
    odds = float(odds)
    if not math.isfinite(odds) or odds == 0:
        raise ValueError("American odds must be a non-zero finite value")
    if odds > 0:
        return 1.0 + (odds / 100.0)
    return 1.0 + (100.0 / abs(odds))


def decimal_to_american(decimal_odds: float) -> int:
    """Convert decimal odds to rounded American odds."""
    decimal_odds = float(decimal_odds)
    if not math.isfinite(decimal_odds) or decimal_odds <= 1.0:
        raise ValueError("Decimal odds must be greater than 1.0")
    if decimal_odds >= 2.0:
        return int(round((decimal_odds - 1.0) * 100.0))
    return int(round(-100.0 / (decimal_odds - 1.0)))


def implied_probability(decimal_odds: float) -> float:
    """Return the implied win probability for decimal odds."""
    decimal_odds = float(decimal_odds)
    if not math.isfinite(decimal_odds) or decimal_odds <= 1.0:
        raise ValueError("Decimal odds must be greater than 1.0")
    return 1.0 / decimal_odds


def calculate_ev(win_probability: float, decimal_odds: float, stake: float = 1.0) -> float:
    """Calculate expected value for a stake at decimal odds."""
    win_probability = float(win_probability)
    decimal_odds = float(decimal_odds)
    stake = float(stake)
    if not math.isfinite(win_probability) or not 0.0 <= win_probability <= 1.0:
        raise ValueError("Win probability must be between 0.0 and 1.0")
    if not math.isfinite(decimal_odds) or decimal_odds <= 1.0:
        raise ValueError("Decimal odds must be greater than 1.0")
    if not math.isfinite(stake) or stake < 0:
        raise ValueError("Stake must be a non-negative finite value")

    profit_if_win = stake * (decimal_odds - 1.0)
    return (win_probability * profit_if_win) - ((1.0 - win_probability) * stake)


def fetch_odds(
    api_key: str,
    sport: str = "americanfootball_nfl",
    regions: str = "us",
    markets: Sequence[str] = ("spreads",),
    odds_format: str = "american",
    date_format: str = "iso",
    timeout: int = 15,
) -> Optional[list[dict]]:
    """
    Fetch odds data from The Odds API.

    Parameters are aligned with https://the-odds-api.com/liveapi/guides/v4/#parameters-2
    """
    if not api_key:
        return None

    url = ODDS_BASE_URL.format(sport=sport)
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": ",".join(markets),
        "oddsFormat": odds_format,
        "dateFormat": date_format,
    }
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            return data
        return None
    except Exception:
        return None
