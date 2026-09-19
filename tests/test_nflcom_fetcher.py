from __future__ import annotations

import pandas as pd

from src.data.nflcom import _prepare_team_stats_frame


def test_prepare_team_stats_frame_repairs_headers_and_writes_parquet(tmp_path):
    raw = pd.DataFrame(
        {
            "T e a m": ["Chiefs", "Bills"],
            "L n g": [52, "T-12"],
            "y e a r": [2026, 2026],
            "c a t e g o r y": ["rushing", "rushing"],
        }
    )

    prepared = _prepare_team_stats_frame(raw)

    assert "Team" in prepared.columns
    assert "Lng" in prepared.columns
    assert "year" in prepared.columns
    assert prepared["Lng"].tolist() == ["52", "T-12"]

    prepared.to_parquet(tmp_path / "nflcom_teamstats_rushing.parquet", index=False)
