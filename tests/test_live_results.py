from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.live_results import (
    build_and_write_live_results,
    build_prediction_vs_actual,
    load_finalized_actuals,
    load_prediction_source,
    prepare_prediction_frame,
)


def _prediction_rows() -> list[dict[str, object]]:
    return [
        {
            "season": 2026,
            "week": 7,
            "game_id": "2026_07_BUF_NYJ",
            "home_team": "NYJ",
            "away_team": "BUF",
            "home_win_prob": 0.40,
            "pred_home_margin": -4.0,
            "market_home_moneyline": 150,
            "market_away_moneyline": -170,
            "market_home_margin": -3.5,
            "market_spread_selected_side": "away",
            "volatility_label": "stable",
            "market_policy_tier": "no_bet",
        },
        {
            "season": 2026,
            "week": 7,
            "game_id": "2026_07_KC_DEN",
            "home_team": "DEN",
            "away_team": "KC",
            "home_win_prob": 0.62,
            "pred_home_margin": 3.0,
            "market_home_moneyline": -130,
            "market_away_moneyline": 110,
            "market_home_margin": 2.5,
            "market_spread_selected_side": "home",
            "volatility_label": "watch",
            "market_policy_tier": "review",
        },
    ]


def _actual_rows() -> list[dict[str, object]]:
    return [
        {
            "season": 2026,
            "week": 7,
            "game_id": "2026_07_BUF_NYJ",
            "home_team": "NYJ",
            "away_team": "BUF",
            "home_score": 17.0,
            "away_score": 24.0,
            "home_margin": -7.0,
        },
        {
            "season": 2026,
            "week": 7,
            "game_id": "2026_07_KC_DEN",
            "home_team": "DEN",
            "away_team": "KC",
            "home_score": 20.0,
            "away_score": 23.0,
            "home_margin": -3.0,
        },
    ]


def _write_mtb_archive(root: Path, rows: list[dict[str, object]] | None = None) -> Path:
    path = root / "seasons" / "2026" / "week_07" / "predictions.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows if rows is not None else _prediction_rows()).to_csv(path, index=False)
    return path


def _write_actuals(path: Path) -> None:
    pd.DataFrame(_actual_rows()).to_parquet(path, index=False)


def _write_clv_detail(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 7,
                "game_id": "2026_07_BUF_NYJ",
                "model_pick_side": "away",
                "model_pick_team": "BUF",
                "model_pick_prob": 0.60,
                "model_side_pick_moneyline": -170,
                "model_side_close_moneyline": -190,
                "model_side_pick_implied_prob_novig": 0.60,
                "model_side_close_implied_prob_novig": 0.63,
                "moneyline_clv_decimal_delta": 0.0619,
                "moneyline_clv_prob_delta": 0.03,
                "moneyline_positive_clv": True,
                "model_vs_pick_prob_edge": 0.00,
                "model_vs_close_prob_edge": -0.03,
                "pick_selected_spread": -3.5,
                "close_selected_spread": -4.0,
                "spread_clv_points": 0.5,
                "spread_positive_clv": True,
                "clv_status": "movement_available",
            },
            {
                "season": 2026,
                "week": 7,
                "game_id": "2026_07_KC_DEN",
                "model_pick_side": "away",
                "model_pick_team": "KC",
                "moneyline_clv_prob_delta": 0.02,
                "spread_clv_points": 0.5,
                "clv_status": "movement_available",
            },
        ]
    ).to_csv(path, index=False)


def test_load_prediction_source_prefers_mtb_archive(tmp_path: Path):
    mtb_root = tmp_path / "mtb_csvs"
    archive_path = _write_mtb_archive(mtb_root)
    locked_dir = tmp_path / "predictions_log"
    locked_path = locked_dir / "2026" / "week_07_predictions.csv"
    locked_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{**_prediction_rows()[0], "home_win_prob": 0.99}]).to_csv(
        locked_path,
        index=False,
    )

    source = load_prediction_source(
        season=2026,
        week=7,
        mtb_csv_root=mtb_root,
        predictions_log_dir=locked_dir,
    )

    assert source.source_type == "mtb_archive"
    assert source.path == archive_path.resolve()
    assert len(source.frame) == 2
    assert source.frame.loc[source.frame["game_id"].eq("2026_07_BUF_NYJ"), "home_win_prob"].iloc[0] == 0.40


def test_load_prediction_source_falls_back_to_locked_log(tmp_path: Path):
    locked_dir = tmp_path / "predictions_log"
    locked_path = locked_dir / "2026" / "week_07_predictions.csv"
    locked_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(_prediction_rows()).to_csv(locked_path, index=False)

    source = load_prediction_source(
        season=2026,
        week=7,
        mtb_csv_root=tmp_path / "missing_mtb",
        predictions_log_dir=locked_dir,
    )

    assert source.source_type == "locked_log"
    assert source.path == locked_path.resolve()


def test_prepare_prediction_frame_rejects_duplicate_games():
    rows = [_prediction_rows()[0], _prediction_rows()[0]]

    with pytest.raises(ValueError, match="Duplicate prediction snapshot"):
        prepare_prediction_frame(pd.DataFrame(rows), season=2026, week=7)


def test_build_prediction_vs_actual_grades_games(tmp_path: Path):
    mtb_root = tmp_path / "mtb_csvs"
    _write_mtb_archive(mtb_root)
    features_path = tmp_path / "matchup_features.parquet"
    _write_actuals(features_path)
    source = load_prediction_source(season=2026, week=7, mtb_csv_root=mtb_root)
    actuals = load_finalized_actuals(features_path, season=2026, week=7)

    clv_path = tmp_path / "analysis" / "market_clv_detail.csv"
    _write_clv_detail(clv_path)
    clv_detail = pd.read_csv(clv_path)

    detail = build_prediction_vs_actual(source, actuals, clv_detail=clv_detail)

    buf = detail[detail["game_id"].eq("2026_07_BUF_NYJ")].iloc[0]
    assert buf["model_pick_team"] == "BUF"
    assert bool(buf["su_correct"]) is True
    assert buf["su_result"] == "win"
    assert buf["selected_spread_result"] == "win"
    assert buf["moneyline_profit_unit"] == pytest.approx(100 / 170)
    assert bool(buf["clv_detail_available"]) is True
    assert buf["clv_status"] == "movement_available"
    assert buf["moneyline_clv_prob_delta"] == pytest.approx(0.03)
    assert bool(buf["moneyline_positive_clv"]) is True
    assert buf["spread_clv_points"] == pytest.approx(0.5)

    kc = detail[detail["game_id"].eq("2026_07_KC_DEN")].iloc[0]
    assert kc["model_pick_team"] == "DEN"
    assert bool(kc["su_correct"]) is False
    assert kc["su_result"] == "loss"
    assert kc["selected_spread_result"] == "loss"
    assert kc["moneyline_profit_unit"] == pytest.approx(-1.0)
    assert bool(kc["clv_detail_available"]) is True
    assert kc["clv_status"] == "unavailable_model_side_mismatch"
    assert pd.isna(kc["moneyline_clv_prob_delta"])
    assert pd.isna(kc["spread_clv_points"])
    assert detail["brier_component"].notna().all()
    assert detail["log_loss_component"].notna().all()


def test_build_and_write_live_results_writes_weekly_and_cumulative_without_mutating_source(tmp_path: Path):
    mtb_root = tmp_path / "mtb_csvs"
    archive_path = _write_mtb_archive(mtb_root)
    source_before = archive_path.read_text(encoding="utf-8")
    features_path = tmp_path / "matchup_features.parquet"
    _write_actuals(features_path)
    clv_path = tmp_path / "analysis" / "market_clv_detail.csv"
    _write_clv_detail(clv_path)
    output_dir = tmp_path / "evaluation"

    artifacts = build_and_write_live_results(
        season=2026,
        week=7,
        features_path=features_path,
        output_dir=output_dir,
        market_clv_detail_path=clv_path,
        mtb_csv_root=mtb_root,
    )
    build_and_write_live_results(
        season=2026,
        week=7,
        features_path=features_path,
        output_dir=output_dir,
        market_clv_detail_path=clv_path,
        mtb_csv_root=mtb_root,
    )

    assert archive_path.read_text(encoding="utf-8") == source_before
    assert artifacts.detail_path.exists()
    assert artifacts.summary_path.exists()
    assert artifacts.segments_path is not None and artifacts.segments_path.exists()
    cumulative = pd.read_csv(artifacts.cumulative_detail_path)
    assert len(cumulative) == 2
    summary = pd.read_csv(artifacts.cumulative_summary_path)
    assert len(summary) == 1
    assert summary["n_games"].iloc[0] == 2
    assert np.isfinite(summary["brier"].iloc[0])
    assert summary["moneyline_clv_tracked"].iloc[0] == 1
    assert summary["avg_moneyline_clv_prob_delta"].iloc[0] == pytest.approx(0.03)
    assert summary["positive_moneyline_clv_rate"].iloc[0] == pytest.approx(1.0)


# ---- picks of record, cumulative replacement, completed-week grading ----

from datetime import datetime, timedelta, timezone  # noqa: E402

from src.evaluation import live_results as lr  # noqa: E402
from src.evaluation import picks_of_record as por  # noqa: E402

KICK_DEN = datetime(2026, 10, 25, 20, 25, tzinfo=timezone.utc)


def _ledger_preds(prob_den: float) -> pd.DataFrame:
    rows = pd.DataFrame(_prediction_rows())
    rows["kickoff"] = [(KICK_DEN - timedelta(hours=3)).isoformat(), KICK_DEN.isoformat()]
    rows["market_home_spread"] = -rows["market_home_margin"]
    rows.loc[rows["game_id"] == "2026_07_KC_DEN", "home_win_prob"] = prob_den
    rows["home_pick"] = np.where(rows["home_win_prob"] >= 0.5, rows["home_team"], rows["away_team"])
    rows["market_home_implied_prob_novig"] = 0.5
    return rows.drop(columns=["market_spread_selected_side"])


def _write_ledger(picks_dir: Path, entries: list[tuple[pd.DataFrame, datetime, str]]) -> None:
    path = por.ledger_path(2026, picks_dir)
    for preds, at, run_id in entries:
        rows = por.build_ledger_rows(preds, recorded_at=at, run_id=run_id, slot=run_id, git_sha="t")
        por.append_ledger(rows, path)


def test_week_with_ledger_grades_locked_pick_not_first_snapshot(tmp_path: Path):
    # MTB archive = first snapshot of the week: DEN 0.62 (pick DEN, who lost).
    mtb_root = tmp_path / "mtb_csvs"
    _write_mtb_archive(mtb_root)
    picks_dir = tmp_path / "picks"
    _write_ledger(
        picks_dir,
        [
            (_ledger_preds(0.62), KICK_DEN - timedelta(days=3), "thu"),
            (_ledger_preds(0.45), KICK_DEN - timedelta(hours=3), "sun"),     # locked pick: KC
            (_ledger_preds(0.70), KICK_DEN - timedelta(minutes=90), "late"),  # inside lock window
        ],
    )
    features_path = tmp_path / "features.parquet"
    _write_actuals(features_path)

    artifacts = build_and_write_live_results(
        season=2026, week=7, features_path=features_path, output_dir=tmp_path / "eval",
        market_clv_detail_path=tmp_path / "none.csv", mtb_csv_root=mtb_root, picks_dir=picks_dir,
        now=KICK_DEN + timedelta(days=1),
    )
    detail = pd.read_csv(artifacts.detail_path).set_index("game_id")
    den = detail.loc["2026_07_KC_DEN"]
    assert den["prediction_source_type"] == "picks_of_record"
    assert den["pick_status"] == "locked" and den["pick_slot"] == "sun"
    assert den["home_win_prob"] == pytest.approx(0.45)
    assert den["model_pick_team"] == "KC" and den["su_result"] == "win"
    # Spread side derived from the locked pick: margin 3.0 vs market 2.5 -> home.
    assert den["market_spread_selected_side"] == "home"
    assert set(detail["prediction_source_type"]) == {"picks_of_record"}


def test_games_without_pick_of_record_fall_back_to_mtb_archive(tmp_path: Path):
    mtb_root = tmp_path / "mtb_csvs"
    _write_mtb_archive(mtb_root)
    picks_dir = tmp_path / "picks"
    only_den = _ledger_preds(0.45)
    _write_ledger(picks_dir, [(only_den[only_den["game_id"] == "2026_07_KC_DEN"], KICK_DEN - timedelta(hours=3), "sun")])
    source = load_prediction_source(season=2026, week=7, mtb_csv_root=mtb_root, picks_dir=picks_dir,
                                    now=KICK_DEN + timedelta(days=1))
    by_game = source.frame.set_index("game_id")["prediction_source_type"].to_dict()
    assert by_game == {"2026_07_KC_DEN": "picks_of_record", "2026_07_BUF_NYJ": "mtb_archive"}


def test_cumulative_keeps_all_weeks_and_rerun_replaces_week(tmp_path: Path):
    mtb_root = tmp_path / "mtb_csvs"
    _write_mtb_archive(mtb_root)
    wk8 = [dict(r, week=8, game_id=r["game_id"].replace("_07_", "_08_")) for r in _prediction_rows()]
    path8 = mtb_root / "seasons" / "2026" / "week_08" / "predictions.csv"
    path8.parent.mkdir(parents=True)
    pd.DataFrame(wk8).to_csv(path8, index=False)
    schedule = pd.DataFrame(
        _actual_rows() + [dict(r, week=8, game_id=r["game_id"].replace("_07_", "_08_")) for r in _actual_rows()]
    )
    out = tmp_path / "eval"
    kwargs = dict(output_dir=out, mtb_csv_root=mtb_root, market_clv_detail_path=tmp_path / "none.csv",
                  picks_dir=tmp_path / "no_picks", predictions_log_dir=tmp_path / "no_log")
    grades =lr.grade_completed_weeks(season=2026, schedule=schedule, **kwargs)
    assert [(g.week, g.status, g.games) for g in grades] == [(7, "graded", 2), (8, "graded", 2)]
    # Re-grade week 7 from a different source: rows replaced, not duplicated.
    lr.grade_week(season=2026, week=7, schedule=schedule, prediction_file=_write_mtb_archive(tmp_path / "x"), **kwargs)
    cumulative = pd.read_csv(out / lr.CUMULATIVE_DETAIL_NAME)
    assert len(cumulative) == 4 and cumulative.groupby("week").size().to_dict() == {7: 2, 8: 2}
    assert set(cumulative.loc[cumulative["week"] == 7, "prediction_source_type"]) == {"explicit_file"}
    summary = pd.read_csv(out / lr.CUMULATIVE_SUMMARY_NAME)
    assert sorted(summary["week"]) == [7, 8]
    record = lr.season_record(out, 2026)
    assert record["games"] == 4 and record["su_wins"] + record["su_losses"] == 4


def test_incomplete_week_is_not_graded(tmp_path: Path):
    schedule = pd.DataFrame(_actual_rows())
    schedule.loc[1, ["home_score", "away_score"]] = np.nan
    grade = lr.grade_week(season=2026, week=7, schedule=schedule, output_dir=tmp_path)
    assert grade.status == "not_final" and "1/2" in grade.detail
    assert not (tmp_path / lr.CUMULATIVE_DETAIL_NAME).exists()


def test_cli_require_complete_exit_code(tmp_path: Path):
    schedule_path = tmp_path / "sched.parquet"
    schedule = pd.DataFrame(_actual_rows())
    schedule.loc[0, "home_score"] = np.nan
    schedule.to_parquet(schedule_path, index=False)
    rc = lr.main(["--season", "2026", "--week", "7", "--require-complete", "--schedule-path", str(schedule_path),
                  "--output-dir", str(tmp_path / "eval"), "--report-json", str(tmp_path / "r.json")])
    assert rc == lr.EXIT_NOT_FINAL and "not graded" in (tmp_path / "r.json").read_text()


def test_published_override_beats_mtb_archive(tmp_path: Path):
    mtb_root = tmp_path / "mtb_csvs"
    _write_mtb_archive(mtb_root)
    log_dir = tmp_path / "log"
    override = lr.published_override_csv_path(log_dir, 2026, 7)
    override.parent.mkdir(parents=True)
    rows = pd.DataFrame(_prediction_rows())
    rows["home_win_prob"] = [0.55, 0.30]
    rows.to_csv(override, index=False)
    source = load_prediction_source(season=2026, week=7, mtb_csv_root=mtb_root, predictions_log_dir=log_dir,
                                    picks_dir=tmp_path / "no_picks")
    assert source.source_type == "published_override"
    assert source.frame.set_index("game_id")["home_win_prob"].to_dict() == {"2026_07_BUF_NYJ": 0.55, "2026_07_KC_DEN": 0.30}


def test_final_game_without_pick_gets_score_row_with_blank_grades(tmp_path: Path):
    # Archive has only BUF-NYJ; KC-DEN was final but had no pick published before kickoff.
    mtb_root = tmp_path / "mtb_csvs"
    _write_mtb_archive(mtb_root, [r for r in _prediction_rows() if r["game_id"] == "2026_07_BUF_NYJ"])
    schedule = pd.DataFrame(_actual_rows())
    out = tmp_path / "eval"
    known = tmp_path / "known"
    known.mkdir()
    (known / "no_pick_of_record_2026.csv").write_text("game_id,reason\n2026_07_KC_DEN,test\n", encoding="utf-8")
    grade = lr.grade_week(season=2026, week=7, schedule=schedule, output_dir=out, mtb_csv_root=mtb_root,
                          market_clv_detail_path=tmp_path / "none.csv", picks_dir=tmp_path / "no_picks",
                          predictions_log_dir=tmp_path / "no_log", known_no_pick_dir=known)
    assert grade.status == "graded" and grade.games == 1 and "1/2" in grade.detail
    assert grade.alerts == []  # listed game: no alert

    detail = pd.read_csv(out / lr.WEEKLY_DETAIL_TEMPLATE.format(week=7)).set_index("game_id")
    assert list(detail.index) == ["2026_07_BUF_NYJ", "2026_07_KC_DEN"]
    den = detail.loc["2026_07_KC_DEN"]
    assert den["prediction_source_type"] == lr.NO_PICK_SOURCE
    assert den["home_score"] == 20.0 and pd.notna(den["away_score"]) and pd.notna(den["actual_winner"])
    for col in ("model_pick_team", "home_win_prob", "su_correct", "su_result", "moneyline_result",
                "spread_result_points", "selected_spread_result", "brier_component"):
        assert pd.isna(den[col]), col
    assert detail.loc["2026_07_BUF_NYJ", "su_result"] in {"win", "loss"}

    summary = pd.read_csv(out / lr.WEEKLY_SUMMARY_TEMPLATE.format(week=7))
    assert summary.loc[0, "n_games"] == 1
    record = lr.season_record(out, 2026)
    assert record["games"] == 1 and record["no_pick_games"] == 1
    assert record["su_wins"] + record["su_losses"] + record["su_pushes"] == 1
    assert "1 final games with no pick of record" in lr.format_report([grade], record, 2026)


def test_unlisted_no_pick_game_raises_alert(tmp_path: Path, capsys):
    # KC-DEN is final, missing from the archive, and not on the known list: it still gets a
    # score-only row that doesn't count, but grading must alert (it may be a newly dropped game).
    mtb_root = tmp_path / "mtb_csvs"
    _write_mtb_archive(mtb_root, [r for r in _prediction_rows() if r["game_id"] == "2026_07_BUF_NYJ"])
    schedule = pd.DataFrame(_actual_rows())
    out = tmp_path / "eval"
    known = tmp_path / "known"
    known.mkdir()
    (known / "no_pick_of_record_2026.csv").write_text("game_id,reason\n2026_05_TB_DAL,test\n", encoding="utf-8")
    grade = lr.grade_week(season=2026, week=7, schedule=schedule, output_dir=out, mtb_csv_root=mtb_root,
                          market_clv_detail_path=tmp_path / "none.csv", picks_dir=tmp_path / "no_picks",
                          predictions_log_dir=tmp_path / "no_log", known_no_pick_dir=known)
    expected = "ALERT: final game with no pick of record, not on the known list -- check the archive history"
    assert grade.status == "graded" and grade.games == 1
    assert len(grade.alerts) == 1 and grade.alerts[0].startswith(expected) and "2026_07_KC_DEN" in grade.alerts[0]
    assert expected in capsys.readouterr().out

    detail = pd.read_csv(out / lr.WEEKLY_DETAIL_TEMPLATE.format(week=7)).set_index("game_id")
    assert detail.loc["2026_07_KC_DEN", "prediction_source_type"] == lr.NO_PICK_SOURCE
    assert pd.isna(detail.loc["2026_07_KC_DEN", "su_result"])
    record = lr.season_record(out, 2026)
    assert record["games"] == 1 and record["no_pick_games"] == 1
    report = lr.format_report([grade], record, 2026)
    assert expected in report

    # The results email flags it in the subject and carries the line in the body.
    from tools import dispatcher

    sent = {}
    import src.utils.alerts as alerts

    original = alerts.send_alert
    alerts.send_alert = lambda subject, body: sent.update(subject=subject, body=body)
    try:
        dispatcher.send_grading_alert(2026, 7, {"status": "success", "report": report}, final=False)
    finally:
        alerts.send_alert = original
    assert "RESULTS ALERT" in sent["subject"] and expected in sent["body"]
