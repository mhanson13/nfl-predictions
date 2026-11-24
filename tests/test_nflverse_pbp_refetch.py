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


@pytest.mark.parametrize("name", ["nfl_player_stats.parquet"])
def test_player_stats_replace_the_refetched_season(tmp_path, monkeypatch, name):
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


def test_injuries_still_append_until_dedupe_is_validated(tmp_path, monkeypatch):
    # Deduping injuries scored worse in the 2026 walk-forward; kept out on purpose.
    assert "nfl_injuries.parquet" not in nflverse.REPLACE_ON_REFETCH


def test_refetch_dedupes_new_rows_and_keeps_cached_order(tmp_path, monkeypatch):
    written = {}
    monkeypatch.setattr(nflverse, "write_df", lambda df, path: written.setdefault("df", df) is not None and path)
    cached = pd.DataFrame({"season": [2024, 2025, 2026], "game_id": ["g24", "g25", "g26"], "play_id": [1, 1, 1],
                           "epa": [0.1, 0.2, 0.3]})
    # the refetch itself repeats a row: it must still be deduped
    revised = pd.DataFrame({"season": [2026, 2026, 2026], "game_id": ["g26", "g26", "g26"], "play_id": [1, 1, 2],
                            "epa": [0.35, 0.35, -0.1]})
    nflverse._merge_and_write(tmp_path / "nfl_pbp.parquet", cached, revised)
    out = written["df"]
    assert out["game_id"].tolist() == ["g24", "g25", "g26", "g26"]
    assert out["epa"].tolist() == [0.1, 0.2, 0.35, -0.1]


def test_drop_refetched_seasons_works_in_place():
    # In place, so the caller's (1M+ row) frame is not copied.
    cached = pd.DataFrame({"season": [2025, 2026, 2026], "x": [1, 2, 3]})
    out = nflverse._drop_refetched_seasons(cached, pd.DataFrame({"season": [2026]}))
    assert out is cached and cached["x"].tolist() == [1] and cached.index.tolist() == [0]


def test_normalize_pbp_columns_leaves_the_input_alone():
    df = pd.DataFrame({"old_game_id": [2026111500, "2026111501"], "epa": [0.1, 0.2]})
    out = nflverse._normalize_pbp_columns(df)
    assert out["old_game_id"].tolist() == ["2026111500", "2026111501"]
    assert df["old_game_id"].tolist() == [2026111500, "2026111501"]  # input not mutated
