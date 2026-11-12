from __future__ import annotations

import pandas as pd

from src.data import player_actuals


def test_load_pbp_uses_csv_fallback_when_parquet_missing(monkeypatch, tmp_path):
    pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 9,
                "game_id": "2026_09_DAL_HOU",
            }
        ]
    ).to_csv(tmp_path / "nfl_pbp.csv", index=False)
    monkeypatch.setattr(player_actuals, "RAW_DIR", tmp_path)

    result = player_actuals._load_pbp(["season", "week", "game_id"], debug=False)

    assert result["game_id"].tolist() == ["2026_09_DAL_HOU"]
