from __future__ import annotations

import requests
from typing import Iterable, Optional, Sequence


ODDS_BASE_URL = "https://api.the-odds-api.com/v4/sports/{sport}/odds/"


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
