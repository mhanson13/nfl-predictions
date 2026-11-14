from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.evaluation import picks_of_record as por

KICK_THU = datetime(2026, 11, 13, 1, 15, tzinfo=timezone.utc)   # Thu 8:15 pm ET
KICK_SUN = datetime(2026, 11, 15, 18, 0, tzinfo=timezone.utc)   # Sun 1:00 pm ET


def _preds(prob_thu=0.60, prob_sun=0.55, mkt_sun=0.52):
    return pd.DataFrame(
        {
            "season": [2026, 2026],
            "week": [10, 10],
            "game_id": ["2026_10_A_B", "2026_10_C_D"],
            "home_team": ["B", "D"],
            "away_team": ["A", "C"],
            "kickoff": [KICK_THU.isoformat(), KICK_SUN.isoformat()],
            "home_win_prob": [prob_thu, prob_sun],
            "home_pick": ["B", "D"],
            "market_home_implied_prob_novig": [0.58, mkt_sun],
            "odds_dk_last_update": ["2026-11-12T20:00:00Z", "2026-11-12T20:00:00Z"],
        }
    )


def _rows(preds, at, run_id, hashes=None):
    return por.build_ledger_rows(preds, recorded_at=at, run_id=run_id, slot="t", git_sha="abc", feature_hashes=hashes)


def test_games_already_kicked_off_are_never_recorded():
    sat = datetime(2026, 11, 14, 15, 0, tzinfo=timezone.utc)
    rows = _rows(_preds(), sat, "sat")
    assert rows["game_id"].tolist() == ["2026_10_C_D"]


def test_lock_takes_latest_row_before_deadline_and_ignores_later_rows(tmp_path):
    path = tmp_path / "ledger.csv"
    thu = KICK_THU - timedelta(hours=3)
    sun = KICK_SUN - timedelta(hours=3)
    sun_late = KICK_SUN - timedelta(minutes=30)  # inside the 2h lock window
    por.append_ledger(_rows(_preds(prob_sun=0.55), thu, "thu"), path)
    por.append_ledger(_rows(_preds(prob_sun=0.62), sun, "sun"), path)
    por.append_ledger(_rows(_preds(prob_sun=0.90), sun_late, "late"), path)
    rec = por.resolve_records(pd.read_csv(path), now=KICK_SUN + timedelta(hours=4)).set_index("game_id")

    assert rec.loc["2026_10_A_B", "status"] == "locked" and rec.loc["2026_10_A_B", "run_id"] == "thu"
    sun_game = rec.loc["2026_10_C_D"]
    assert sun_game["status"] == "locked" and sun_game["run_id"] == "sun"
    assert sun_game["home_win_prob"] == pytest.approx(0.62)  # the post-deadline 0.90 never wins
    assert sun_game["n_versions"] == 3
    assert sun_game["prob_change_since_first"] == pytest.approx(0.07)


def test_provisional_before_deadline_and_late_when_only_inside_window(tmp_path):
    path = tmp_path / "ledger.csv"
    early = tmp_path / "early.csv"
    por.append_ledger(_rows(_preds(), KICK_SUN - timedelta(hours=5), "early"), early)
    pending = por.resolve_records(pd.read_csv(early), now=KICK_SUN - timedelta(hours=4)).set_index("game_id")
    assert pending.loc["2026_10_C_D", "status"] == "provisional"

    por.append_ledger(_rows(_preds(), KICK_SUN - timedelta(hours=1), "only_late"), path)
    after = por.resolve_records(pd.read_csv(path), now=KICK_SUN + timedelta(hours=1)).set_index("game_id")
    assert after.loc["2026_10_C_D", "status"] == "late"


def test_ledger_is_append_only_and_idempotent(tmp_path):
    path = tmp_path / "ledger.csv"
    at = KICK_THU - timedelta(hours=5)
    assert por.append_ledger(_rows(_preds(), at, "r1"), path) == 2
    first = path.read_text()
    assert por.append_ledger(_rows(_preds(prob_thu=0.1), at, "r1"), path) == 0  # same run never rewrites
    assert path.read_text() == first
    assert por.append_ledger(_rows(_preds(), at + timedelta(hours=1), "r2"), path) == 2
    assert path.read_text().startswith(first)


def test_static_feature_change_is_flagged(tmp_path):
    path = tmp_path / "ledger.csv"
    gid = "2026_10_C_D"
    feats = pd.DataFrame({"game_id": [gid], "pass_epa_per_db_rolling3_diff": [0.1], "sched_spread_close_diff": [-3.0],
                          "inj_out_diff": [1.0]})
    h1 = por.static_feature_hashes(feats, [gid])
    # Pick-time inputs (spread, injuries) changing is fine.
    h2 = por.static_feature_hashes(feats.assign(sched_spread_close_diff=-4.5, inj_out_diff=3.0), [gid])
    assert h1[gid] == h2[gid]
    h3 = por.static_feature_hashes(feats.assign(pass_epa_per_db_rolling3_diff=0.2), [gid])
    assert h3[gid] != h1[gid]

    por.append_ledger(_rows(_preds(), KICK_THU - timedelta(hours=4), "thu", h1), path)
    por.append_ledger(_rows(_preds(), KICK_SUN - timedelta(hours=4), "sun", h3), path)
    rec = por.resolve_records(pd.read_csv(path), now=KICK_SUN).set_index("game_id")
    assert bool(rec.loc[gid, "static_features_changed"]) is True
    assert "non-pick-time features changed" in por.summarize(rec.reset_index())


def test_results_scored_against_pick_time_market():
    rec = pd.DataFrame({"game_id": ["g"], "home_win_prob": [0.7], "market_home_implied_prob_novig": [0.6], "status": ["locked"]})
    sched = pd.DataFrame({"game_id": ["g"], "home_score": [20.0], "away_score": [17.0]})
    out = por.attach_results(rec, sched).iloc[0]
    assert out["pick_correct"] == 1.0
    assert out["brier"] == pytest.approx(0.09) and out["market_brier"] == pytest.approx(0.16)


def test_record_refuses_stale_predictions(tmp_path):
    preds_path = tmp_path / "predictions.csv"
    _preds().to_csv(preds_path, index=False)
    future_start = datetime.now(timezone.utc) + timedelta(hours=1)
    with pytest.raises(RuntimeError, match="stale"):
        por.record(2026, 10, future_start, predictions_path=preds_path, out_dir=tmp_path)


def test_record_writes_ledger(tmp_path):
    preds_path = tmp_path / "predictions.csv"
    _preds().to_csv(preds_path, index=False)
    start = datetime.now(timezone.utc) - timedelta(minutes=5)
    now = KICK_THU - timedelta(hours=3)
    n = por.record(2026, 10, start, now=now, predictions_path=preds_path,
                   features_path=tmp_path / "missing.parquet", out_dir=tmp_path, slot="thu")
    led = pd.read_csv(por.ledger_path(2026, tmp_path))
    assert n == 2 and set(led["slot"]) == {"thu"}
    assert led["odds_last_update"].iloc[0] == "2026-11-12T20:00:00Z"


def test_pipeline_job_only_for_live_numeric_weeks():
    from tools import run_pipeline as rp

    started = datetime(2026, 11, 15, 15, 0, tzinfo=timezone.utc)
    assert rp.build_picks_of_record_job(live_run=False, season=2026, prediction_week="10", run_started_at=started) is None
    assert rp.build_picks_of_record_job(live_run=True, season=2026, prediction_week="auto", run_started_at=started) is None
    job = rp.build_picks_of_record_job(live_run=True, season=2026, prediction_week="10", run_started_at=started)
    assert job.name == "picks_of_record" and "2026-11-15T15:00:00+00:00" in job.command


def test_picks_files_published_but_never_strict_required(tmp_path):
    from tools import publish_mtb_csvs as pub

    plan = pub.build_artifact_plan(season=2026, week=10)
    picks = [a for a in plan if "picks_of_record" in str(a.destination)]
    assert len(picks) == 2 and all(a.never_required and not a.week_archive for a in picks)
