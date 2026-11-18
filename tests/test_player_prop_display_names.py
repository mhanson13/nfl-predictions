from __future__ import annotations

import pandas as pd

from src.player_props.display_names import apply_display_names, load_display_names


def _write(tmp_path):
    stats = pd.DataFrame(
        {
            "season": [2025, 2026, 2026],
            "week": [5, 9, 10],
            "player_id": ["00-1", "00-1", "00-2"],
            "player_display_name": ["Old Name", "Justin Herbert", "Emmett"],
        }
    )
    rosters = pd.DataFrame(
        {
            "season": [2026, 2026],
            "week": [10, 10],
            "gsis_id": ["00-2", "00-3"],
            "full_name": ["Emmett Johnson", "Theo Wease Jr."],
        }
    )
    stats_path = tmp_path / "stats.parquet"
    rosters_path = tmp_path / "rosters.parquet"
    stats.to_parquet(stats_path)
    rosters.to_parquet(rosters_path)
    return stats_path, rosters_path


def test_output_names_use_full_display_names(tmp_path):
    names = load_display_names(*_write(tmp_path))
    props = pd.DataFrame(
        {
            "player_id": ["00-1", "00-2", "00-3", "00-9", None],
            "player_name": ["J.Herbert", "E.Johnson", "T.Wease", "X.Unknown", "No Id"],
            "market": ["qb_passing_yards"] * 5,
        }
    )
    out = apply_display_names(props, names)
    assert out["player_name"].tolist() == [
        "Justin Herbert",  # latest player-stats display name
        "Emmett Johnson",  # first-name-only stats row ignored; roster full name used
        "Theo Wease Jr.",  # roster fallback
        "X.Unknown",  # unknown id keeps its name
        "No Id",
    ]
    assert props["player_name"].iloc[0] == "J.Herbert"  # input not mutated


def test_missing_sources_leave_names_alone(tmp_path):
    names = load_display_names(tmp_path / "nope.parquet", tmp_path / "nope2.parquet")
    props = pd.DataFrame({"player_id": ["00-1"], "player_name": ["J.Herbert"]})
    assert apply_display_names(props, names)["player_name"].tolist() == ["J.Herbert"]


def test_projection_csv_writer_uses_full_names(tmp_path, monkeypatch):
    import src.predict.predict_upcoming as pu

    monkeypatch.setattr(pu, "_DISPLAY_NAMES", pd.Series({"00-1": "Lamar Jackson"}))
    frame = pd.DataFrame({"player_id": ["00-1", "00-2"], "player_name": ["L.Jackson", "T.Huntley"]})
    path = tmp_path / "predictions_players_qb.csv"
    pu._write_player_projection_csv(frame, str(path), ["player_id", "player_name"], "QB")
    assert pd.read_csv(path)["player_name"].tolist() == ["Lamar Jackson", "T.Huntley"]
