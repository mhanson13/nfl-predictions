from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.injury_coverage import complete_injury_grid


def _games() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2023, 2023, 2023, 2023, 2024],
            "week": [18, 18, 19, 20, 1],
            "home_team": ["BAL", "SF", "DET", "BAL", "BAL"],
            "away_team": ["PIT", "LA", "LA", "HOU", "KC"],
        }
    )


def _inj() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2023, 2023, 2023, 2023, 2023],
            "week": [18, 18, 18, 19, 19],
            "team": ["BAL", "SF", "LA", "DET", "LA"],  # PIT absent in wk18; no wk20 / 2024 reports
            "inj_out": [2.0, 1.0, 3.0, 4.0, 5.0],
        }
    )


def _value(frame, season, week, team):
    row = frame[(frame.season == season) & (frame.week == week) & (frame.team == team)]
    return row["inj_out"].iloc[0]


def test_absent_team_in_reported_week_is_zero():
    out = complete_injury_grid(_inj(), _games(), ["inj_out"])
    assert _value(out, 2023, 18, "PIT") == 0.0


def test_missing_week_carries_forward_latest_prior_report():
    out = complete_injury_grid(_inj(), _games(), ["inj_out"])
    assert _value(out, 2023, 20, "BAL") == 2.0  # bye in wk19 -> wk18 report
    assert np.isnan(_value(out, 2023, 20, "HOU"))  # no earlier report this season


def test_carry_forward_never_crosses_seasons():
    out = complete_injury_grid(_inj(), _games(), ["inj_out"])
    assert np.isnan(_value(out, 2024, 1, "BAL"))
    assert _value(out, 2023, 19, "LA") == 5.0  # real reports untouched
