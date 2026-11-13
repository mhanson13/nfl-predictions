"""Where are we in the NFL week? Resolve the slate to predict and the run phase.

The schedule (nflverse ``nfl_schedules.parquet``) lists ``gameday`` + ``gametime``
in US/Eastern. A game counts as *started* once its kickoff has passed and as
*final* once it has a score or is more than ``GAME_DURATION`` past kickoff (the
local schedule may be a day stale, so the clock is the fallback).

Phases for the target slate (the earliest week with a game not yet final):

* ``early_week``   - no game started and the first kickoff is more than
  ``PRE_KICKOFF_WINDOW`` away (Tue/Wed: the picks-of-record run);
* ``pre_kickoff``  - no game started, first kickoff within the window (Thu before TNF);
* ``in_progress``  - at least one game started (TNF/Sun/Mon); started games are
  dropped from the slate by predict_upcoming;
* ``offseason``    - nothing left on the schedule.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

EASTERN = ZoneInfo("America/New_York")
GAME_DURATION = timedelta(hours=4)
PRE_KICKOFF_WINDOW = timedelta(hours=36)
DEFAULT_SCHEDULE_PATH = Path("data/raw/nfl_schedules.parquet")

PHASE_EARLY_WEEK = "early_week"
PHASE_PRE_KICKOFF = "pre_kickoff"
PHASE_IN_PROGRESS = "in_progress"
PHASE_OFFSEASON = "offseason"


@dataclass
class WeekContext:
    now_et: str
    phase: str
    season: Optional[int] = None
    week: Optional[int] = None
    game_type: Optional[str] = None
    games_total: int = 0
    games_started: int = 0
    games_final: int = 0
    first_kickoff_et: Optional[str] = None
    last_kickoff_et: Optional[str] = None
    hours_to_first_kickoff: Optional[float] = None
    previous_week: Optional[int] = None
    previous_week_scored: Optional[bool] = None
    notes: list[str] = field(default_factory=list)

    @property
    def is_full_slate(self) -> bool:
        return self.phase in (PHASE_EARLY_WEEK, PHASE_PRE_KICKOFF)

    def as_dict(self) -> dict:
        out = asdict(self)
        out["is_full_slate"] = self.is_full_slate
        return out


def _kickoffs(schedule: pd.DataFrame) -> pd.Series:
    stamp = pd.to_datetime(
        schedule["gameday"].astype(str) + " " + schedule["gametime"].fillna("13:00").astype(str),
        errors="coerce",
    )
    return stamp.dt.tz_localize(EASTERN, ambiguous="NaT", nonexistent="shift_forward")


def resolve_week_context(schedule: pd.DataFrame, now: Optional[datetime] = None) -> WeekContext:
    """Return the slate to predict and the phase of the week at ``now``."""
    now_et = (now or datetime.now(tz=EASTERN)).astimezone(EASTERN)
    frame = schedule[["season", "week", "game_type", "gameday", "gametime", "home_score", "away_score"]].copy()
    frame["kickoff"] = _kickoffs(frame)
    frame = frame.dropna(subset=["kickoff"])
    frame["started"] = frame["kickoff"] <= now_et
    scored = frame["home_score"].notna() & frame["away_score"].notna()
    frame["final"] = scored | (frame["kickoff"] + GAME_DURATION <= now_et)

    pending = frame[~frame["final"]]
    ctx = WeekContext(now_et=now_et.isoformat(timespec="minutes"), phase=PHASE_OFFSEASON)
    if pending.empty:
        ctx.notes.append("No unfinished games on the local schedule.")
        return ctx

    season, week = pending.sort_values("kickoff")[["season", "week"]].iloc[0]
    slate = frame[(frame["season"] == season) & (frame["week"] == week)]
    first, last = slate["kickoff"].min(), slate["kickoff"].max()
    ctx.season, ctx.week = int(season), int(week)
    ctx.game_type = str(slate["game_type"].mode().iloc[0])
    ctx.games_total = int(len(slate))
    ctx.games_started = int(slate["started"].sum())
    ctx.games_final = int(slate["final"].sum())
    ctx.first_kickoff_et = first.isoformat(timespec="minutes")
    ctx.last_kickoff_et = last.isoformat(timespec="minutes")
    ctx.hours_to_first_kickoff = round((first - now_et).total_seconds() / 3600.0, 1)

    if ctx.games_started:
        ctx.phase = PHASE_IN_PROGRESS
        ctx.notes.append(f"{ctx.games_started}/{ctx.games_total} games already kicked off; slate is partial.")
    elif first - now_et <= PRE_KICKOFF_WINDOW:
        ctx.phase = PHASE_PRE_KICKOFF
    else:
        ctx.phase = PHASE_EARLY_WEEK

    prev = frame[(frame["season"] == season) & (frame["week"] < week)]
    if not prev.empty:
        prev_week = int(prev["week"].max())
        prev_games = prev[prev["week"] == prev_week]
        ctx.previous_week = prev_week
        ctx.previous_week_scored = bool(
            (prev_games["home_score"].notna() & prev_games["away_score"].notna()).all()
        )
        if not ctx.previous_week_scored:
            ctx.notes.append(
                f"Week {prev_week} scores are not in the local schedule yet; the nflverse refresh should fill them."
            )
    if ctx.game_type and ctx.game_type != "REG":
        ctx.notes.append(f"Postseason slate ({ctx.game_type}).")
    return ctx


def load_week_context(
    schedule_path: Path = DEFAULT_SCHEDULE_PATH, now: Optional[datetime] = None
) -> WeekContext:
    return resolve_week_context(pd.read_parquet(schedule_path), now=now)
