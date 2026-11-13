from datetime import datetime

import numpy as np
import pandas as pd

from src.utils.nfl_calendar import (
    EASTERN,
    PHASE_EARLY_WEEK,
    PHASE_IN_PROGRESS,
    PHASE_OFFSEASON,
    PHASE_PRE_KICKOFF,
    resolve_week_context,
)


def _schedule():
    rows = [
        # week 9: Thu, Sun, Mon - all scored
        (2026, 9, "2026-11-05", "20:15", 10, 33),
        (2026, 9, "2026-11-08", "13:00", 17, 34),
        (2026, 9, "2026-11-09", "20:15", 6, 23),
        # week 10: Thu, Sun x2, Mon - unscored
        (2026, 10, "2026-11-12", "20:15", np.nan, np.nan),
        (2026, 10, "2026-11-15", "13:00", np.nan, np.nan),
        (2026, 10, "2026-11-15", "16:25", np.nan, np.nan),
        (2026, 10, "2026-11-16", "20:15", np.nan, np.nan),
    ]
    frame = pd.DataFrame(rows, columns=["season", "week", "gameday", "gametime", "home_score", "away_score"])
    frame["game_type"] = "REG"
    return frame


def _at(text):
    return datetime.fromisoformat(text).replace(tzinfo=EASTERN)


def test_tuesday_is_early_week_for_next_slate():
    ctx = resolve_week_context(_schedule(), now=_at("2026-11-10T10:00"))
    assert (ctx.season, ctx.week, ctx.phase) == (2026, 10, PHASE_EARLY_WEEK)
    assert ctx.is_full_slate and ctx.games_started == 0
    assert ctx.previous_week == 9 and ctx.previous_week_scored is True


def test_thursday_morning_is_pre_kickoff():
    ctx = resolve_week_context(_schedule(), now=_at("2026-11-12T09:00"))
    assert ctx.phase == PHASE_PRE_KICKOFF and ctx.week == 10


def test_friday_after_tnf_is_in_progress():
    ctx = resolve_week_context(_schedule(), now=_at("2026-11-13T07:00"))
    assert ctx.phase == PHASE_IN_PROGRESS and ctx.week == 10
    assert ctx.games_started == 1 and not ctx.is_full_slate


def test_monday_night_game_keeps_week_until_final():
    ctx = resolve_week_context(_schedule(), now=_at("2026-11-16T21:00"))
    assert ctx.week == 10 and ctx.phase == PHASE_IN_PROGRESS
    # four hours after the last kickoff the week is treated as final even without scores
    ctx = resolve_week_context(_schedule(), now=_at("2026-11-17T01:00"))
    assert ctx.phase == PHASE_OFFSEASON  # no later weeks in this toy schedule


def test_monday_before_mnf_without_scores_notes_stale_schedule():
    sched = _schedule()
    sched.loc[sched["week"] == 9, ["home_score", "away_score"]] = np.nan
    ctx = resolve_week_context(sched, now=_at("2026-11-10T10:00"))
    assert ctx.week == 10 and ctx.previous_week_scored is False
    assert any("Week 9 scores" in note for note in ctx.notes)
