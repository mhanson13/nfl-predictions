"""Backfill historical *pick-time* (early-week) game lines from The Odds API.

History in ``matchup_features`` only carries closing lines, but picks are made
early in the week. This tool takes one bulk snapshot per NFL week (default:
Wednesday 16:00 UTC, i.e. noon ET, before the week's first kickoff) of the
moneyline and spread for every game, so walk-forward evaluation can measure
and train against the line that was actually available at pick time.

Cost control (the historical endpoint is billed per call):
  * one bulk call per week returns every game (cost = 10 x markets x regions);
  * the first call's ``x-requests-last`` header is checked against the
    expected cost and the run stops if it differs;
  * ``--max-credits`` caps the total spend of a run;
  * snapshots already on disk are skipped, so reruns resume for free.

Usage::

    python -m src.data.odds_api_pick_time_lines --seasons 2025 2026 --max-credits 1000
    python -m src.data.odds_api_pick_time_lines --seasons 2020 ... --dry-run
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import requests

from src.data.odds_api_historical import (
    DEFAULT_RETRIES,
    DEFAULT_SLEEP,
    DEFAULT_SPORT_KEY,
    REQUEST_TIMEOUT,
    _flatten_event_odds,
    _request_with_retry,
)
from src.utils.io import RAW_DIR
from src.utils.odds import get_odds_api_paid_key
from src.utils.teams import get_team_abbr_from_name

OUTPUT_PATH = RAW_DIR / "oddsapi_pick_time_game_lines.parquet"
SCHEDULE_PATH = RAW_DIR / "nfl_schedules.parquet"
DEFAULT_MARKETS = ("h2h", "spreads")
DEFAULT_BOOKMAKERS = ("draftkings", "fanduel")  # <= 10 bookmakers bills as one region
SNAPSHOT_WEEKDAY = 2  # Wednesday
SNAPSHOT_HOUR_UTC = 16
MIN_LEAD = timedelta(hours=2)
HISTORICAL_START = datetime(2020, 6, 6, tzinfo=timezone.utc)  # Odds API historical coverage
# nflverse schedule abbreviations differ from the canonical map for the Rams.
_SCHEDULE_ABBR = {"LAR": "LA"}


def expected_call_cost(markets: Sequence[str], n_regions: int = 1) -> int:
    return 10 * len(markets) * n_regions


def _kickoff_utc(schedule: pd.DataFrame) -> pd.Series:
    local = pd.to_datetime(
        schedule["gameday"].astype(str) + " " + schedule["gametime"].fillna("13:00").astype(str),
        format="%Y-%m-%d %H:%M",
        errors="coerce",
    )
    return local.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert("UTC")


def week_snapshot_times(schedule: pd.DataFrame, seasons: Iterable[int]) -> pd.DataFrame:
    """One pick-time snapshot per (season, week): the Wednesday 16:00 UTC before the first kickoff."""
    sched = schedule[schedule["season"].isin(list(seasons))].copy()
    sched["kickoff_utc"] = _kickoff_utc(sched)
    first = sched.groupby(["season", "week"], as_index=False)["kickoff_utc"].min()
    rows = []
    for rec in first.itertuples(index=False):
        kickoff = rec.kickoff_utc.to_pydatetime()
        day = kickoff.replace(hour=SNAPSHOT_HOUR_UTC, minute=0, second=0, microsecond=0)
        day -= timedelta(days=(day.weekday() - SNAPSHOT_WEEKDAY) % 7)
        if day > kickoff - MIN_LEAD:
            day -= timedelta(days=7)
        rows.append({"season": int(rec.season), "week": int(rec.week), "first_kickoff_utc": kickoff, "snapshot_utc": day})
    out = pd.DataFrame(rows)
    now = datetime.now(timezone.utc)
    return out[(out["snapshot_utc"] >= HISTORICAL_START) & (out["snapshot_utc"] < now)].reset_index(drop=True)


def fetch_snapshot(
    session: requests.Session,
    api_key: str,
    snapshot_utc: datetime,
    *,
    markets: Sequence[str],
    bookmakers: Sequence[str],
    sport_key: str = DEFAULT_SPORT_KEY,
) -> tuple[pd.DataFrame, dict[str, str]]:
    date = snapshot_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    params = {
        "apiKey": api_key,
        "markets": ",".join(markets),
        "bookmakers": ",".join(bookmakers),
        "oddsFormat": "american",
        "dateFormat": "iso",
        "date": date,
    }
    payload, quota = _request_with_retry(
        session,
        f"/historical/sports/{sport_key}/odds",
        params=params,
        retries=DEFAULT_RETRIES,
        sleep=DEFAULT_SLEEP,
        timeout=REQUEST_TIMEOUT,
    )
    events = payload.get("data") if isinstance(payload, dict) else payload
    frames = []
    for event in events or []:
        wrapped = {
            "timestamp": payload.get("timestamp") if isinstance(payload, dict) else None,
            "previous_timestamp": payload.get("previous_timestamp") if isinstance(payload, dict) else None,
            "next_timestamp": payload.get("next_timestamp") if isinstance(payload, dict) else None,
            "data": event,
        }
        frames.append(
            _flatten_event_odds(
                wrapped,
                requested_date=date,
                requested_markets=markets,
                requested_bookmakers=bookmakers,
                requested_regions=[],
                quota_headers=quota,
            )
        )
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return frame, quota


def _abbr(name: object) -> Optional[str]:
    abbr = get_team_abbr_from_name(str(name)) if isinstance(name, str) else None
    return _SCHEDULE_ABBR.get(abbr, abbr) if abbr else None


def attach_game_ids(lines: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Map snapshot outcomes to schedule ``game_id`` for the requested (season, week)."""
    if lines.empty:
        return lines.assign(game_id=pd.Series(dtype=str))
    out = lines.copy()
    out["home_abbr"] = out["home_team"].map(_abbr)
    out["away_abbr"] = out["away_team"].map(_abbr)
    keys = schedule[["season", "week", "home_team", "away_team", "game_id"]].rename(
        columns={"home_team": "home_abbr", "away_team": "away_abbr"}
    )
    return out.merge(keys, left_on=["requested_season", "requested_week", "home_abbr", "away_abbr"],
                     right_on=["season", "week", "home_abbr", "away_abbr"], how="inner").drop(columns=["season", "week"])


def summarize_game_lines(lines: pd.DataFrame) -> pd.DataFrame:
    """One row per game: consensus (median across books) no-vig home prob and home spread at pick time."""
    if lines.empty:
        return pd.DataFrame(columns=["game_id", "pick_novig_home_prob", "pick_spread_home", "pick_snapshot_utc"])
    df = lines.copy()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["point"] = pd.to_numeric(df["point"], errors="coerce")
    df["implied"] = np.where(df["price"] > 0, 100.0 / (df["price"] + 100.0), -df["price"] / (-df["price"] + 100.0))
    df["side"] = np.where(df["outcome_name"] == df["home_team"], "home", np.where(df["outcome_name"] == df["away_team"], "away", None))
    h2h = df[df["market_key"] == "h2h"].pivot_table(index=["game_id", "bookmaker_key"], columns="side", values="implied")
    h2h = (h2h["home"] / (h2h["home"] + h2h["away"])).groupby("game_id").median().rename("pick_novig_home_prob")
    spreads = df[(df["market_key"] == "spreads") & (df["side"] == "home")]
    # Odds API home "point" is the handicap (negative = home favoured); flip to the expected home margin.
    spread = (-spreads.groupby("game_id")["point"].median()).rename("pick_spread_home")
    snap = df.groupby("game_id")["snapshot_timestamp"].first().rename("pick_snapshot_utc")
    return pd.concat([h2h, spread, snap], axis=1).reset_index()


def run(
    seasons: Sequence[int],
    *,
    max_credits: int,
    markets: Sequence[str] = DEFAULT_MARKETS,
    bookmakers: Sequence[str] = DEFAULT_BOOKMAKERS,
    output: Path = OUTPUT_PATH,
    dry_run: bool = False,
) -> pd.DataFrame:
    schedule = pd.read_parquet(SCHEDULE_PATH)
    plan = week_snapshot_times(schedule, seasons)
    existing = pd.read_parquet(output) if output.exists() else pd.DataFrame()
    done: set[tuple[int, int]] = set()
    if not existing.empty:
        done = set(zip(existing["requested_season"].astype(int), existing["requested_week"].astype(int)))
    todo = plan[[(s, w) not in done for s, w in zip(plan["season"], plan["week"])]]
    cost = expected_call_cost(markets)
    logging.info("%d weeks planned, %d already on disk, %d to fetch (~%d credits)", len(plan), len(plan) - len(todo), len(todo), cost * len(todo))
    if dry_run:
        return todo
    api_key = get_odds_api_paid_key()
    session = requests.Session()
    spent = 0
    frames = [existing] if not existing.empty else []
    try:
        for rec in todo.itertuples(index=False):
            if spent + cost > max_credits:
                logging.warning("Stopping: next call would exceed --max-credits=%d (spent %d).", max_credits, spent)
                break
            raw, quota = fetch_snapshot(session, api_key, rec.snapshot_utc, markets=markets, bookmakers=bookmakers)
            last = int(quota.get("x_requests_last") or 0)
            spent += last
            logging.info("%s wk%02d snapshot %s: %d rows, cost %s, remaining %s",
                         rec.season, rec.week, rec.snapshot_utc, len(raw), last, quota.get("x_requests_remaining"))
            raw = raw.assign(requested_season=rec.season, requested_week=rec.week,
                             requested_snapshot_utc=rec.snapshot_utc.isoformat(), first_kickoff_utc=rec.first_kickoff_utc.isoformat())
            frames.append(attach_game_ids(raw, schedule))
            if last != cost:
                logging.error("Call cost %s differs from expected %s; stopping so the budget can be re-checked.", last, cost)
                break
    finally:
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            combined.to_parquet(output, index=False)
            logging.info("Wrote %s (%d rows, %d games)", output, len(combined), combined["game_id"].nunique())
    return pd.read_parquet(output) if output.exists() else pd.DataFrame()


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seasons", type=int, nargs="+", required=True)
    parser.add_argument("--max-credits", type=int, required=True, help="Hard cap on Odds API credits for this run.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--dry-run", action="store_true", help="Show the snapshot plan and estimated cost only.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    todo = run(args.seasons, max_credits=args.max_credits, output=args.output, dry_run=args.dry_run)
    if args.dry_run:
        print(todo.to_string(index=False))


if __name__ == "__main__":
    main()
