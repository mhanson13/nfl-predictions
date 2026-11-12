from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from src.data import odds_api_pick_time_lines as ptl


def _schedule() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["2025_01_DAL_PHI", "2025_01_LA_DET"],
            "season": [2025, 2025],
            "week": [1, 1],
            "gameday": ["2025-09-04", "2025-09-07"],
            "gametime": ["20:20", "13:00"],
            "home_team": ["PHI", "DET"],
            "away_team": ["DAL", "LA"],
        }
    )


def test_snapshot_is_wednesday_noon_et_before_first_kickoff():
    plan = ptl.week_snapshot_times(_schedule(), [2025])
    assert len(plan) == 1
    snap = plan.loc[0, "snapshot_utc"]
    assert snap == datetime(2025, 9, 3, 16, tzinfo=timezone.utc)
    assert plan.loc[0, "first_kickoff_utc"] > snap


def _payload_rows() -> pd.DataFrame:
    event = lambda home, away, hp, ap, hpt: {  # noqa: E731
        "id": home, "home_team": home, "away_team": away, "commence_time": "2025-09-05T00:20:00Z",
        "bookmakers": [{"key": "draftkings", "markets": [
            {"key": "h2h", "outcomes": [{"name": home, "price": hp}, {"name": away, "price": ap}]},
            {"key": "spreads", "outcomes": [{"name": home, "price": -110, "point": hpt}, {"name": away, "price": -110, "point": -hpt}]},
        ]}],
    }
    frames = []
    for ev in (event("Philadelphia Eagles", "Dallas Cowboys", -300, 250, -7.0),
               event("Detroit Lions", "Los Angeles Rams", 120, -140, 2.5)):
        frames.append(ptl._flatten_event_odds({"timestamp": "2025-09-03T16:00:00Z", "data": ev},
                                              requested_date="x", requested_markets=["h2h"], requested_bookmakers=[], requested_regions=[]))
    return pd.concat(frames).assign(requested_season=2025, requested_week=1)


def test_lines_map_to_game_ids_and_summarize():
    lines = ptl.attach_game_ids(_payload_rows(), _schedule())
    assert set(lines["game_id"]) == {"2025_01_DAL_PHI", "2025_01_LA_DET"}  # LAR -> nflverse LA
    summary = ptl.summarize_game_lines(lines).set_index("game_id")
    assert summary.loc["2025_01_DAL_PHI", "pick_spread_home"] == 7.0  # home favoured by 7
    assert summary.loc["2025_01_LA_DET", "pick_spread_home"] == -2.5
    p = summary.loc["2025_01_DAL_PHI", "pick_novig_home_prob"]
    assert p == pytest.approx((300 / 400) / (300 / 400 + 100 / 350))


def test_expected_cost():
    assert ptl.expected_call_cost(("h2h", "spreads")) == 20
