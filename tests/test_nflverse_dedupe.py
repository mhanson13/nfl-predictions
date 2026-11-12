from __future__ import annotations

import pandas as pd

from src.data import nflverse
from src.data.nflverse import _dedupe_schedule_rows, _merge_and_write, fetch_player_stats
from src.utils.io import read_df


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


def test_merge_and_write_normalizes_mixed_schedule_id_columns(tmp_path):
    existing = pd.DataFrame(
        [
            {
                "game_id": "2025_01_DAL_PHI",
                "season": 2025,
                "week": 1,
                "home_team": "PHI",
                "away_team": "DAL",
                "old_game_id": 2025090400,
            }
        ]
    )
    new_df = pd.DataFrame(
        [
            {
                "game_id": "2026_01_BAL_IND",
                "season": 2026,
                "week": 1,
                "home_team": "IND",
                "away_team": "BAL",
                "old_game_id": "2026090900",
            }
        ]
    )
    path = tmp_path / "nfl_schedules.parquet"

    output_path = _merge_and_write(path, existing, new_df)
    result = read_df(output_path)

    assert output_path.suffix == ".parquet"
    assert result["old_game_id"].astype("string").tolist() == ["2025090400", "2026090900"]


def test_merge_and_write_normalizes_mixed_pbp_text_columns(tmp_path):
    existing = pd.DataFrame(
        [
            {
                "game_id": "2025_01_DAL_PHI",
                "season": 2025,
                "week": 1,
                "old_game_id": 2025090400,
                "play_clock": 0,
                "play_id": 1,
            }
        ]
    )
    new_df = pd.DataFrame(
        [
            {
                "game_id": "2026_01_BAL_IND",
                "season": 2026,
                "week": 1,
                "old_game_id": "2026090900",
                "play_clock": "00:12",
                "play_id": 1,
            }
        ]
    )
    path = tmp_path / "nfl_pbp.parquet"

    output_path = _merge_and_write(path, existing, new_df)
    result = read_df(output_path)

    assert output_path.suffix == ".parquet"
    assert result["old_game_id"].astype("string").tolist() == ["2025090400", "2026090900"]
    assert result["play_clock"].astype("string").tolist() == ["0", "00:12"]


def test_fetch_pbp_prefers_release_files(monkeypatch):
    release_df = pd.DataFrame(
        {
            "season": [2026],
            "game_id": ["2026_01_BAL_IND"],
            "play_id": [1],
        }
    )

    monkeypatch.setattr(nflverse, "download_pbp", lambda seasons: release_df)
    monkeypatch.setattr(
        nflverse,
        "_try_import_nfl",
        lambda: (_ for _ in ()).throw(AssertionError("nfl_data_py should not be called")),
    )

    result = nflverse.fetch_pbp([2026])

    assert result.equals(release_df)


def test_fetch_pbp_falls_back_to_nfl_data_py_when_release_missing(monkeypatch):
    monkeypatch.setattr(nflverse, "download_pbp", lambda seasons: None)

    class FakeNFL:
        @staticmethod
        def import_pbp_data(seasons, downcast=True):
            return pd.DataFrame({"gameId": ["2026_01_BAL_IND"], "play_id": [1]})

    monkeypatch.setattr(nflverse, "_try_import_nfl", lambda: FakeNFL)

    result = nflverse.fetch_pbp([2026])

    assert result["season"].tolist() == [2026]
    assert result["game_id"].tolist() == ["2026_01_BAL_IND"]


def test_fetch_schedules_prefers_release_files(monkeypatch):
    release_df = pd.DataFrame(
        {
            "season": [2026],
            "week": [1],
            "game_id": ["2026_01_BAL_IND"],
        }
    )

    monkeypatch.setattr(nflverse, "download_schedules", lambda seasons: release_df)
    monkeypatch.setattr(
        nflverse,
        "_try_import_nfl",
        lambda: (_ for _ in ()).throw(AssertionError("nfl_data_py should not be called")),
    )

    result = nflverse.fetch_schedules([2026])

    assert result.equals(release_df)


def test_download_schedules_filters_multi_season_release(monkeypatch):
    release_df = pd.DataFrame(
        {
            "season": [2025, 2026],
            "week": [1, 1],
            "game_id": ["2025_01_DAL_PHI", "2026_01_BAL_IND"],
        }
    )

    monkeypatch.setattr(nflverse, "_try_read_url", lambda url: release_df if url.endswith("games.parquet") else None)

    result = nflverse.download_schedules([2026])

    assert result is not None
    assert result["game_id"].tolist() == ["2026_01_BAL_IND"]


def test_fetch_snap_counts_prefers_release_files(monkeypatch):
    release_df = pd.DataFrame(
        {
            "season": [2026],
            "week": [1],
            "team": ["BAL"],
            "offense_snaps": [67],
        }
    )

    monkeypatch.setattr(nflverse, "download_snap_counts", lambda seasons: release_df)
    monkeypatch.setattr(
        nflverse,
        "_try_import_nfl",
        lambda: (_ for _ in ()).throw(AssertionError("nfl_data_py should not be called")),
    )

    result = nflverse.fetch_snap_counts([2026])

    assert result is not None
    assert result.equals(release_df)


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
