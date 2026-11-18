from datetime import datetime, timezone

import pandas as pd

from src.data import nflverse


def _df(status="Limited"):
    return pd.DataFrame(
        {
            "season": [2025, 2026, 2026],
            "week": [18, 10, 10],
            "gsis_id": ["a", "b", "c"],
            "team": ["X", "Y", "Z"],
            "practice_status": ["Full", status, "Full"],
        }
    )


T1 = datetime(2026, 11, 19, 15, 0, tzinfo=timezone.utc)
T2 = datetime(2026, 11, 19, 15, 15, tzinfo=timezone.utc)
T3 = datetime(2026, 11, 20, 21, 0, tzinfo=timezone.utc)


def test_each_season_saved_with_utc_timestamp_and_identical_downloads_skipped(tmp_path):
    first = nflverse.log_injury_download(_df(), tmp_path, fetched_at=T1)
    assert sorted(p.name for p in first) == ["injuries_2025_20261119T150000Z.parquet", "injuries_2026_20261119T150000Z.parquet"]

    # Same content in a different row order: nothing saved.
    assert nflverse.log_injury_download(_df().iloc[::-1], tmp_path, fetched_at=T2) == []

    # A revised 2026 report: only 2026 saved.
    third = nflverse.log_injury_download(_df(status="Full"), tmp_path, fetched_at=T3)
    assert [p.name for p in third] == ["injuries_2026_20261120T210000Z.parquet"]

    manifest = pd.read_csv(tmp_path / "manifest.csv")
    assert manifest[["season", "rows"]].values.tolist() == [[2025, 1], [2026, 2], [2026, 2]]
    assert manifest["fetched_at_utc"].tolist()[-1] == "2026-11-20T21:00:00+00:00"
    saved = pd.read_parquet(tmp_path / "injuries_2026_20261120T210000Z.parquet")
    assert saved["practice_status"].tolist() == ["Full", "Full"]


def test_merge_and_write_logs_injuries_and_survives_log_errors(tmp_path, monkeypatch):
    written = {}
    monkeypatch.setattr(nflverse, "write_df", lambda df, path: written.setdefault("df", df) is not None and path)
    nflverse._merge_and_write(tmp_path / "nfl_injuries.parquet", pd.DataFrame(), _df())
    assert (tmp_path / "history" / "nfl_injuries" / "manifest.csv").exists()

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(nflverse, "log_injury_download", boom)
    out = nflverse._merge_and_write(tmp_path / "nfl_injuries.parquet", pd.DataFrame(), _df())
    assert out == tmp_path / "nfl_injuries.parquet"  # refresh still written


def test_other_datasets_are_not_logged(tmp_path, monkeypatch):
    monkeypatch.setattr(nflverse, "write_df", lambda df, path: path)
    nflverse._merge_and_write(tmp_path / "nfl_rosters.parquet", pd.DataFrame(), _df())
    assert not (tmp_path / "history").exists()
