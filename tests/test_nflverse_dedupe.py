from __future__ import annotations

import pandas as pd

from src.data import nflverse
from src.data.nflverse import _dedupe_schedule_rows, fetch_player_stats


def test_dedupe_schedule_rows_keeps_latest_game_id_row():
    df = pd.DataFrame(
        [
            {"game_id": "2026_01_BAL_IND", "season": 2026, "week": 1, "home_team": "IND", "away_team": "BAL", "value": 1},
            {"game_id": "2026_01_BAL_IND", "season": 2026, "week": 1, "home_team": "IND", "away_team": "BAL", "value": 2},
        ]
    )

    result = _dedupe_schedule_rows(df)

    assert len(result) == 1
    assert result["value"].iloc[0] == 2


def test_fetch_player_stats_keeps_successful_seasons_when_one_weekly_fetch_fails(monkeypatch):
    monkeypatch.setattr(nflverse, "download_player_stats", lambda seasons: None)

    class FakeNFL:
        @staticmethod
        def import_weekly_data(seasons, downcast=True, thread_requests=False):
            season = seasons[0]
            if season == 2026:
                raise RuntimeError("HTTP Error 404: Not Found")
            return pd.DataFrame(
                {
                    "season": [season],
                    "week": [1],
                    "player_id": [f"00-{season}"],
                    "position_group": ["QB"],
                }
            )

    monkeypatch.setattr(nflverse, "_try_import_nfl", lambda: FakeNFL)

    result = fetch_player_stats([2024, 2026])

    assert result is not None
    assert result["season"].tolist() == [2024]


def test_fetch_player_stats_prefers_release_files_and_normalizes_columns(monkeypatch):
    release_df = pd.DataFrame(
        {
            "season": [2026],
            "week": [2],
            "player_id": ["00-test"],
            "team": ["DEN"],
            "season_type": ["REG"],
        }
    )

    class FakeNFL:
        @staticmethod
        def import_weekly_data(seasons, downcast=True, thread_requests=False):
            raise AssertionError("nfl_data_py should not be called when release files are available")

    monkeypatch.setattr(nflverse, "download_player_stats", lambda seasons: release_df)
    monkeypatch.setattr(nflverse, "_try_import_nfl", lambda: FakeNFL)

    result = fetch_player_stats([2026])

    assert result is not None
    assert result["recent_team"].tolist() == ["DEN"]
    assert result["team"].tolist() == ["DEN"]
    assert result["season_type"].tolist() == [2]
