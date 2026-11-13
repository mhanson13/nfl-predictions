"""Xweather (formerly AerisWeather) client for game-window weather.

Uses the ``conditions`` endpoint, which serves hourly data from January 2004
through +15 days from one model. That makes the same source usable for training
rows (past games) and live rows (forecasts). Docs:
https://www.xweather.com/docs/weather-api/endpoints/conditions

Budget (free plan: 15,000 accesses/month, 100/minute): each calendar day in a
``from``/``to`` range costs one access, so a kickoff +/-3h window costs 1-2.
Raw responses are cached under ``data/raw/xweather/`` so a game is never paid
for twice, and every fetch stops at ``max_requests``.

Credentials: ``XWEATHER_CLIENT_API_KEY`` / ``XWEATHER_CLIENT_API_SECRET`` in
``secrets.env`` or the environment (``XWEATHER_CLIENT_ID`` / ``_SECRET`` and any
capitalization also work).
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests

API_BASE = "https://data.api.xweather.com"
RAW_CACHE_DIR = Path("data/raw/xweather")
MIN_INTERVAL_S = 0.65  # stays under 100 requests/minute
DEFAULT_MAX_REQUESTS = 300

# Accepts XWEATHER_CLIENT_ID / XWEATHER_CLIENT_API_KEY (id) and
# XWEATHER_CLIENT_SECRET / XWEATHER_CLIENT_API_SECRET (secret), any case.
_ID_RE = re.compile(r"^x[\s_-]*weather[\s_-]*client[\s_-]*(api[\s_-]*)?(id|key)$", re.IGNORECASE)
_SECRET_RE = re.compile(r"^x[\s_-]*weather[\s_-]*client[\s_-]*(api[\s_-]*)?secret$", re.IGNORECASE)


def get_xweather_credentials() -> tuple[Optional[str], Optional[str]]:
    """Find the client id/secret in the environment or secrets.env (case-insensitive names)."""
    from src.utils import secrets as secrets_mod

    candidates: dict[str, str] = {}
    candidates.update({k: v for k, v in secrets_mod._load_env_file().items() if v})
    candidates.update({k: v for k, v in os.environ.items() if v})
    client_id = next((v for k, v in candidates.items() if _ID_RE.match(k.strip())), None)
    client_secret = next((v for k, v in candidates.items() if _SECRET_RE.match(k.strip())), None)
    return client_id, client_secret


@dataclass
class XweatherClient:
    client_id: str
    client_secret: str
    max_requests: int = DEFAULT_MAX_REQUESTS
    cache_dir: Path = RAW_CACHE_DIR
    session: requests.Session = field(default_factory=requests.Session)
    requests_made: int = 0
    last_rate_headers: dict[str, str] = field(default_factory=dict)
    _last_call: float = 0.0

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', key)}.json"

    def conditions(self, lat: float, lon: float, start: datetime, end: datetime, *, key: str) -> Optional[dict]:
        """Hourly conditions for [start, end] at lat/lon; cached by ``key``."""
        path = self._cache_path(key)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        if self.requests_made >= self.max_requests:
            raise BudgetExhausted(f"Xweather request budget of {self.max_requests} reached")
        wait = MIN_INTERVAL_S - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        url = f"{API_BASE}/conditions/{lat:.4f},{lon:.4f}"
        params = {
            "from": _epoch(start),
            "to": _epoch(end),
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }
        for attempt in range(3):
            self._last_call = time.monotonic()
            response = self.session.get(url, params=params, timeout=30)
            self.requests_made += 1
            self.last_rate_headers = {k: v for k, v in response.headers.items() if k.lower().startswith("x-ratelimit")}
            if response.status_code == 429 and attempt < 2:
                time.sleep(15 * (attempt + 1))
                continue
            break
        if response.status_code != 200:
            print(f"[xweather] HTTP {response.status_code} for {key}: {response.text[:200]}")
            return None
        payload = response.json()
        if not payload.get("success", False):
            print(f"[xweather] error for {key}: {payload.get('error')}")
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return payload


class BudgetExhausted(RuntimeError):
    pass


def _epoch(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return str(int(dt.timestamp()))


def _num(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(out) else out


def periods_from_payload(payload: Optional[dict]) -> list[dict]:
    if not payload:
        return []
    response = payload.get("response")
    items = response if isinstance(response, list) else [response] if isinstance(response, dict) else []
    periods: list[dict] = []
    for item in items:
        if isinstance(item, dict):
            periods.extend(p for p in item.get("periods", []) if isinstance(p, dict))
    return periods


def aggregate_window(periods: list[dict], kickoff: datetime, window_hours: float = 3.0) -> dict:
    """Aggregate hourly periods within kickoff +/- window into Visual-Crossing-style columns."""
    if kickoff.tzinfo is None:
        kickoff = kickoff.replace(tzinfo=timezone.utc)
    rows = []
    for period in periods:
        ts = _num(period.get("timestamp"))
        if ts is None:
            continue
        when = datetime.fromtimestamp(ts, tz=timezone.utc)
        if abs((when - kickoff).total_seconds()) <= window_hours * 3600:
            rows.append((when, period))
    if not rows:
        return {}

    def series(name: str) -> pd.Series:
        return pd.Series([_num(p.get(name)) for _, p in rows], dtype=float)

    nearest = min(rows, key=lambda item: abs((item[0] - kickoff).total_seconds()))[1]
    precip = series("precipIN")
    snow = series("snowIN")
    out = {
        "xw_temp_f": series("tempF").mean(),
        "xw_temp_kickoff_f": _num(nearest.get("tempF")),
        "xw_feelslike_f": series("feelslikeF").mean(),
        "xw_dewpoint_f": series("dewpointF").mean(),
        "xw_humidity_pct": series("humidity").mean(),
        "xw_wind_mph": series("windSpeedMPH").mean(),
        "xw_windgust_mph": series("windGustMPH").mean(),
        "xw_wind_dir_deg": series("windDirDEG").mean(),
        "xw_visibility_mi": series("visibilityMI").mean(),
        "xw_cloud_cover_pct": series("sky").mean(),
        "xw_precip_in_total": precip.sum(min_count=1),
        "xw_snow_in_total": snow.sum(min_count=1),
        "xw_pop_max_pct": series("pop").max(),  # forecasts only; null for archive
        "xw_weather": nearest.get("weatherPrimary") or nearest.get("weather"),
        "xw_hours": len(rows),
    }
    return {k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in out.items()}


def fetch_game_window(
    client: XweatherClient,
    game_id: str,
    lat: float,
    lon: float,
    kickoff: datetime,
    *,
    window_hours: float = 3.0,
    cache_tag: str = "",
) -> dict:
    """Fetch and aggregate one game. ``cache_tag`` separates forecast snapshots (e.g. a date)."""
    start = kickoff - timedelta(hours=window_hours)
    end = kickoff + timedelta(hours=window_hours)
    # place + time are part of the key so a corrected venue or kickoff never reuses a stale response
    key = f"{game_id}_{lat:.3f}_{lon:.3f}_{int(kickoff.timestamp())}{'_' + cache_tag if cache_tag else ''}"
    payload = client.conditions(lat, lon, start, end, key=key)
    agg = aggregate_window(periods_from_payload(payload), kickoff, window_hours)
    if agg:
        agg.update({"game_id": game_id, "xw_fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    return agg
