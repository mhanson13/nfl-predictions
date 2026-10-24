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
