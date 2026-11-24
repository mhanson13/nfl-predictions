import pandas as pd
import pytest

from src.evaluation import pick_time_vs_close as ptc


def _records():
    base = {"season": 2026, "status": "locked", "kickoff_utc": "2026-11-15T18:00:00Z",
            "recorded_at": "2026-11-15T15:00:00Z"}
    return pd.DataFrame([
        # model likes home more than the pick-time market; the close moves toward it
        {**base, "week": 10, "game_id": "g1", "home_team": "H1", "away_team": "A1", "home_win_prob": 0.60,
         "market_home_implied_prob_novig": 0.50, "market_home_spread": -1.0, "home_won": 1.0},
        # model likes away more; the close moves toward home (against the model)
        {**base, "week": 10, "game_id": "g2", "home_team": "H2", "away_team": "A2", "home_win_prob": 0.30,
         "market_home_implied_prob_novig": 0.40, "market_home_spread": 3.0, "home_won": 0.0},
        # not graded yet: left out
        {**base, "week": 11, "game_id": "g3", "home_team": "H3", "away_team": "A3", "home_win_prob": 0.5,
         "market_home_implied_prob_novig": 0.5, "market_home_spread": 0.0, "home_won": None},
    ])


def _schedule():
    return pd.DataFrame({
        "game_id": ["g1", "g2", "g3"],
        "home_moneyline": [-150.0, 110.0, -110.0], "away_moneyline": [130.0, -130.0, -110.0],
        "spread_line": [2.5, -2.0, 0.0],  # nflverse: expected home margin
    })


def test_novig_home_prob():
    p = ptc.novig_home_prob(pd.Series([-110.0, 150.0]), pd.Series([-110.0, -180.0]))
    assert p.iloc[0] == pytest.approx(0.5)
    assert p.iloc[1] == pytest.approx((100 / 250) / (100 / 250 + 180 / 280))


def test_detail_signs_and_clv():
    d = ptc.build_detail(_records(), _schedule()).set_index("game_id")
    assert list(d.index) == ["g1", "g2"]  # ungraded g3 dropped
    g1, g2 = d.loc["g1"], d.loc["g2"]
    assert g1["close_home_spread"] == -2.5 and g1["spread_move"] == -1.5
    assert g1["hours_before_kickoff"] == 3.0
    assert g1["market_prob_move"] > 0 and g1["clv_prob"] > 0 and g1["moved_toward_model"] == 1.0
    assert g2["model_lean"] == "away" and g2["market_prob_move"] > 0 and g2["clv_prob"] < 0
    assert g2["moved_toward_model"] == 0.0
    assert g1["model_brier"] == pytest.approx(0.16) and g1["model_correct"] == 1.0


def test_summary_and_report():
    detail = ptc.build_detail(_records(), _schedule())
    summary = ptc.build_summary(detail)
    s = summary.iloc[0]
    assert s["n_games"] == 2 and s["share_moved_toward_model"] == 0.5
    assert s["model_minus_close_market_brier"] == pytest.approx(s["model_brier"] - s["close_market_brier"])
    assert list(summary["scope"]) == ["all", "week_10"]
    text = ptc.format_report(summary, detail)
    assert "2 graded picks of record (weeks 10)" in text and "closing market" in text


def test_missing_close_is_excluded_from_summary():
    sched = _schedule().assign(home_moneyline=[None, 110.0, -110.0])
    detail = ptc.build_detail(_records(), sched)
    assert not detail.set_index("game_id").loc["g1", "close_available"]
    assert ptc.build_summary(detail).iloc[0]["n_games"] == 1


def test_run_writes_outputs(tmp_path):
    (tmp_path / "rec").mkdir()
    _records().to_csv(tmp_path / "rec" / "picks_of_record_2026.csv", index=False)
    _schedule().assign(season=2026).to_parquet(tmp_path / "sched.parquet")
    ptc.run(2026, records_dir=tmp_path / "rec", schedule_path=tmp_path / "sched.parquet", out_dir=tmp_path / "out")
    assert (tmp_path / "out" / "pick_time_vs_close_detail.csv").exists()
    assert len(pd.read_csv(tmp_path / "out" / "pick_time_vs_close_summary.csv")) == 2
