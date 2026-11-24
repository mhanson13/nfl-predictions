from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analysis.model_metrics_export import build_model_metrics_rows, build_season_to_date_rows


def _game(season, week, gid, prob, home_won, *, pick=True, tie=False, spread=None, ml_profit=None):
    predicted_home = prob >= 0.5
    actual = np.nan if tie else float(home_won)
    su = "push" if tie else ("win" if predicted_home == bool(home_won) else "loss")
    return {
        "season": season,
        "week": week,
        "game_id": gid,
        "model_pick_team": ("HOME" if predicted_home else "AWAY") if pick else "",
        "home_win_prob": prob if pick else np.nan,
        "predicted_home_win": predicted_home if pick else np.nan,
        "actual_home_win": actual if pick else np.nan,
        "su_result": su if pick else "",
        "su_correct": (np.nan if tie else float(su == "win")) if pick else np.nan,
        "brier_component": ((prob - actual) ** 2) if pick and not tie else np.nan,
        "log_loss_component": (-np.log(prob if actual else 1 - prob)) if pick and not tie else np.nan,
        "abs_margin_error": 3.0 if pick else np.nan,
        "sq_margin_error": 9.0 if pick else np.nan,
        "selected_spread_result": spread or "",
        "moneyline_result": su if pick else "",
        "moneyline_profit_unit": ml_profit if ml_profit is not None else np.nan,
    }


def _detail(path: Path) -> Path:
    rows = [
        _game(2026, 2, "g1", 0.70, 1, spread="win", ml_profit=0.5),
        _game(2026, 2, "g2", 0.60, 0, spread="loss", ml_profit=-1.0),
        _game(2026, 3, "g3", 0.30, 0, spread="push", ml_profit=0.8),
        _game(2026, 3, "g4", 0.55, 1, tie=True),
        _game(2026, 3, "g5", 0.50, 1, pick=False),  # no pick of record
        _game(2026, 4, "g6", 0.80, 1, spread="win", ml_profit=0.25),  # after the export week
        _game(2025, 3, "old", 0.90, 0, spread="loss", ml_profit=-1.0),  # other season
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_season_to_date_rows_count_only_picks_of_record_up_to_week(tmp_path):
    path = _detail(tmp_path / "live_prediction_vs_actual.csv")

    rows = build_season_to_date_rows(season=2026, week=3, generated_at="t", live_results_detail_path=path)
    by_key = {r.metric_key: r for r in rows}

    record = by_key["live_season_su_record"]
    assert record.display_value == "2-1-1"
    assert record.sample_size == 4
    assert record.section == "live_published_season"
    assert "weeks 2-3" in record.description
    assert "1 final games with no pick of record are not counted" in record.description
    assert by_key["live_season_accuracy"].display_value == "66.7%"  # tie not counted
    assert by_key["live_season_ats_record"].display_value == "1-1-1"
    assert by_key["live_season_ats_hit_rate"].display_value == "50.0%"
    assert by_key["live_season_ats_hit_rate"].sample_size == 2
    assert by_key["live_season_moneyline_units"].display_value == "+0.30"
    assert by_key["live_season_moneyline_units"].sample_size == 3
    assert by_key["live_season_moneyline_roi"].display_value == "10.0%"
    assert [r.sort_order for r in rows] == sorted(r.sort_order for r in rows)
    assert all(101 <= r.sort_order <= 109 for r in rows)


def test_season_to_date_rows_empty_without_detail(tmp_path):
    assert build_season_to_date_rows(
        season=2026, week=3, generated_at="t", live_results_detail_path=tmp_path / "missing.csv"
    ) == []


def test_season_to_date_rows_skip_market_rows_without_lines(tmp_path):
    path = tmp_path / "d.csv"
    pd.DataFrame([_game(2026, 2, "g1", 0.7, 1), _game(2026, 2, "g2", 0.4, 1)]).to_csv(path, index=False)

    keys = {r.metric_key for r in build_season_to_date_rows(
        season=2026, week=2, generated_at="t", live_results_detail_path=path)}

    assert "live_season_su_record" in keys
    assert not keys & {"live_season_ats_record", "live_season_ats_hit_rate",
                       "live_season_moneyline_units", "live_season_moneyline_roi"}


def test_build_model_metrics_rows_includes_season_to_date_section(tmp_path):
    path = _detail(tmp_path / "live_prediction_vs_actual.csv")
    missing = tmp_path / "missing.csv"

    metrics = build_model_metrics_rows(
        season=2026,
        week=4,
        generated_at="2026-10-01T00:00:00+00:00",
        overall_metrics_path=missing,
        market_benchmark_summary_path=missing,
        market_clv_summary_path=missing,
        market_disagreement_roi_path=missing,
        market_probability_edge_calibration_path=missing,
        market_blend_summary_path=missing,
        market_blend_policy_path=missing,
        market_spread_roi_path=missing,
        market_spread_edge_bins_path=missing,
        market_spread_key_numbers_path=missing,
        market_edge_policy_path=missing,
        live_results_summary_dir=tmp_path,
        live_results_detail_path=path,
    )

    season_rows = metrics[metrics["section"] == "live_published_season"].set_index("metric_key")
    assert season_rows.loc["live_season_su_record", "display_value"] == "3-1-1"
    assert season_rows.loc["live_season_su_record", "source_artifact"] == path.as_posix()
