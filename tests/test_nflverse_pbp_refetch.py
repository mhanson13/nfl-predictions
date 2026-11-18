"""A refetched in-season PBP slice replaces the cached one instead of piling up revisions."""
import pandas as pd
import pytest

from src.data import nflverse


def test_refetched_season_replaces_cached_rows(tmp_path, monkeypatch):
    written = {}
    monkeypatch.setattr(nflverse, "write_df", lambda df, path: written.setdefault("df", df) is not None and path)
    cached = pd.DataFrame({"season": [2025, 2026, 2026], "game_id": ["g25", "g26", "g26"], "play_id": [1, 1, 2],
                           "epa": [0.1, 0.50, -0.2]})
    revised = pd.DataFrame({"season": [2026, 2026, 2026], "game_id": ["g26", "g26", "g26b"], "play_id": [1, 2, 1],
                            "epa": [0.55, -0.25, 0.3]})  # nflverse revised EPA and added a game
    nflverse._merge_and_write(tmp_path / "nfl_pbp.parquet", cached, revised)
    out = written["df"]
    assert len(out) == 4  # 1 (2025, untouched) + 3 (2026, replaced)
    assert out[["game_id", "play_id"]].duplicated().sum() == 0
    assert out.loc[out.season == 2025, "epa"].tolist() == [0.1]
    assert sorted(out.loc[out.season == 2026, "epa"].tolist()) == [-0.25, 0.3, 0.55]


@pytest.mark.parametrize("name", ["nfl_player_stats.parquet", "nfl_injuries.parquet"])
def test_player_stats_and_injuries_replace_the_refetched_season(tmp_path, monkeypatch, name):
    written = {}
    monkeypatch.setattr(nflverse, "write_df", lambda df, path: written.setdefault("df", df) is not None and path)
    cached = pd.DataFrame({"season": [2025, 2026, 2026], "week": [1, 1, 1], "player_id": ["a", "b", "b"],
                           "gsis_id": ["a", "b", "b"], "team": ["X", "Y", "Y"], "value": [1.0, 2.0, 2.5]})
    revised = pd.DataFrame({"season": [2026, 2026], "week": [1, 2], "player_id": ["b", "b"],
                            "gsis_id": ["b", "b"], "team": ["Y", "Y"], "value": [2.6, 3.0]})
    nflverse._merge_and_write(tmp_path / name, cached, revised)
    out = written["df"]
    assert sorted(out.loc[out.season == 2026, "value"].tolist()) == [2.6, 3.0]
    assert out.loc[out.season == 2025, "value"].tolist() == [1.0]

