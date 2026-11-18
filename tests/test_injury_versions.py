from datetime import datetime, timezone

import pandas as pd

from src.data import nflverse
from src.data.injury_versions import injuries_as_of_pick

SCHEDULE = pd.DataFrame(
    {
        "game_id": ["2026_11_BUF_HOU", "2026_11_KC_DEN"],
        "season": [2026, 2026],
        "week": [11, 11],
        "gameday": ["2026-11-19", "2026-11-22"],  # Thu and Sun
        "gametime": ["20:15", "13:00"],
        "home_team": ["HOU", "DEN"],
        "away_team": ["BUF", "KC"],
    }
)


def _version(status_buf, status_den):
    return pd.DataFrame(
        {
            "season": [2026, 2026],
            "week": [11, 11],
            "team": ["BUF", "DEN"],
            "gsis_id": ["b1", "d1"],
            "practice_status": [status_buf, status_den],
        }
    )


def _log(tmp_path):
    hist = tmp_path / "hist"
    nflverse.log_injury_download(_version("Limited", "Limited"), hist, fetched_at=datetime(2026, 11, 18, 20, tzinfo=timezone.utc))
    # Friday: BUF (played Thursday) gets a post-game revision, DEN gets its final report.
    nflverse.log_injury_download(_version("Full", "Full"), hist, fetched_at=datetime(2026, 11, 21, 20, tzinfo=timezone.utc))
    return hist


def test_each_game_uses_the_version_before_its_pick(tmp_path):
    hist = _log(tmp_path)
    cache = _version("Full", "Full")  # cache = latest download
    out, summary = injuries_as_of_pick(
        cache, SCHEDULE, history_dir=hist, records_dir=tmp_path / "none", now=datetime(2026, 11, 25, tzinfo=timezone.utc)
    )
    status = out.set_index("team")["practice_status"].to_dict()
    assert status["BUF"] == "Limited"  # Thursday game: Friday's revision came after the pick
    assert status["DEN"] == "Full"  # Sunday game: Friday's report was available
    assert summary.from_log == 4 and summary.from_cache == 0


def test_pick_of_record_time_overrides_the_kickoff_fallback(tmp_path):
    hist = _log(tmp_path)
    records = tmp_path / "records"
    records.mkdir()
    # DEN's pick of record was recorded Thursday, before Friday's report.
    pd.DataFrame({"game_id": ["2026_11_KC_DEN"], "recorded_at": ["2026-11-20T15:00:00Z"]}).to_csv(
        records / "picks_of_record_2026.csv", index=False
    )
    out, _ = injuries_as_of_pick(
        _version("Full", "Full"), SCHEDULE, history_dir=hist, records_dir=records, now=datetime(2026, 11, 25, tzinfo=timezone.utc)
    )
    assert out.set_index("team").loc["DEN", "practice_status"] == "Limited"


def test_live_run_uses_everything_downloaded_so_far(tmp_path):
    hist = _log(tmp_path)
    out, _ = injuries_as_of_pick(
        _version("Full", "Full"), SCHEDULE, history_dir=hist, records_dir=tmp_path / "none", now=datetime(2026, 11, 22, 15, tzinfo=timezone.utc)
    )
    assert out.set_index("team").loc["DEN", "practice_status"] == "Full"


def test_without_a_log_the_cache_is_unchanged(tmp_path):
    cache = _version("Full", "Full")
    out, summary = injuries_as_of_pick(cache, SCHEDULE, history_dir=tmp_path / "missing", records_dir=tmp_path)
    pd.testing.assert_frame_equal(out, cache)
    assert summary.from_log == 0


def test_versions_before_logging_fall_back_to_cache(tmp_path):
    hist = tmp_path / "hist"
    nflverse.log_injury_download(_version("Full", "Full"), hist, fetched_at=datetime(2026, 11, 23, tzinfo=timezone.utc))
    out, summary = injuries_as_of_pick(
        _version("Q", "Q"), SCHEDULE, history_dir=hist, records_dir=tmp_path / "none", now=datetime(2026, 11, 25, tzinfo=timezone.utc)
    )
    assert out["practice_status"].tolist() == ["Q", "Q"]  # both picks predate the first logged download
    assert summary.from_cache == 4
