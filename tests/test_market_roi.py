from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from analysis import market_roi


def _predictions_row(
    *,
    home_win_prob: float,
    pred_home_margin: float,
    home_margin: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 2,
                "game_id": "2026_02_AWAY_HOME",
                "home_team": "HOME",
                "away_team": "AWAY",
                "home_win_prob": home_win_prob,
                "pred_home_margin": pred_home_margin,
                "home_margin": home_margin,
            }
        ]
    )


def _moneyline_stability_benchmark(season_rois: dict[int, float], bets_per_season: int = 20) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for season, roi in season_rois.items():
        for idx in range(bets_per_season):
            records.append(
                {
                    "season": season,
                    "favorite_agreement": False,
                    "prob_edge_model_side": 0.03,
                    "home_tie": False,
                    "model_side_actual_win": idx % 2 == 0,
                    "model_side_profit": float(roi),
                }
            )
    return pd.DataFrame.from_records(records)


def test_prepare_dataset_treats_spread_line_as_market_home_margin(monkeypatch):
    odds = pd.DataFrame(
        [
            {
                "game_id": "2026_02_AWAY_HOME",
                "home_moneyline": -271,
                "away_moneyline": 220,
                "spread_line": 6.5,
                "home_spread_odds": -110,
                "away_spread_odds": -110,
            }
        ]
    )
    monkeypatch.setattr(market_roi, "load_market_lines", lambda columns: odds[list(columns)])

    dataset = market_roi.prepare_dataset(
        _predictions_row(home_win_prob=0.60, pred_home_margin=3.5, home_margin=1.0)
    )
    row = dataset.iloc[0]

    assert row["market_home_margin"] == 6.5
    assert row["spread_edge_points"] == pytest.approx(-3.0)
    assert row["spread_result_points"] == pytest.approx(-5.5)
    assert bool(row["home_cover"]) is False
    assert bool(row["away_cover"]) is True
    assert row["home_implied_prob_novig"] + row["away_implied_prob_novig"] == pytest.approx(1.0)


def test_summarize_spread_uses_corrected_edge_side(monkeypatch):
    odds = pd.DataFrame(
        [
            {
                "game_id": "2026_02_AWAY_HOME",
                "home_moneyline": -271,
                "away_moneyline": 220,
                "spread_line": 6.5,
                "home_spread_odds": -110,
                "away_spread_odds": -110,
            }
        ]
    )
    monkeypatch.setattr(market_roi, "load_market_lines", lambda columns: odds[list(columns)])
    dataset = market_roi.prepare_dataset(
        _predictions_row(home_win_prob=0.60, pred_home_margin=3.5, home_margin=1.0)
    )

    summary = market_roi.summarize_spread(dataset, thresholds=[2.0], sigma_margin_error=10.0)
    away = summary[(summary["bet_side"] == "away") & (summary["threshold"] == 2.0)].iloc[0]
    home = summary[(summary["bet_side"] == "home") & (summary["threshold"] == 2.0)].iloc[0]

    assert away["n_bets"] == 1
    assert away["wins"] == 1
    assert away["losses"] == 0
    assert away["roi"] == pytest.approx(100.0 / 110.0)
    assert home["n_bets"] == 0


def test_market_benchmark_disagreement_summary_and_roi(monkeypatch):
    odds = pd.DataFrame(
        [
            {
                "game_id": "2026_02_AWAY_HOME",
                "home_moneyline": 120,
                "away_moneyline": -140,
                "spread_line": -1.5,
                "home_spread_odds": -110,
                "away_spread_odds": -110,
            }
        ]
    )
    monkeypatch.setattr(market_roi, "load_market_lines", lambda columns: odds[list(columns)])
    dataset = market_roi.prepare_dataset(
        _predictions_row(home_win_prob=0.60, pred_home_margin=2.0, home_margin=3.0)
    )

    benchmark = market_roi.build_market_benchmark(dataset)
    row = benchmark.iloc[0]

    assert bool(row["model_home_pick"]) is True
    assert bool(row["vegas_home_pick"]) is False
    assert bool(row["favorite_agreement"]) is False
    assert bool(row["model_correct"]) is True
    assert bool(row["vegas_correct"]) is False
    assert row["prob_edge_model_side"] > 0.15

    summary = market_roi.summarize_market_benchmark(benchmark)
    disagree = summary[(summary["scope"] == "overall") & (summary["segment"] == "disagree")].iloc[0]
    assert disagree["n_games"] == 1
    assert disagree["model_accuracy"] == pytest.approx(1.0)
    assert disagree["vegas_accuracy"] == pytest.approx(0.0)
    assert disagree["model_side_roi"] == pytest.approx(1.2)

    roi = market_roi.summarize_disagreement_moneyline_roi(benchmark, thresholds=[0.15]).iloc[0]
    assert roi["n_bets"] == 1
    assert roi["wins"] == 1
    assert roi["roi"] == pytest.approx(1.2)


def test_market_benchmark_attaches_matching_clv_detail(monkeypatch):
    odds = pd.DataFrame(
        [
            {
                "game_id": "2026_02_AWAY_HOME",
                "home_moneyline": 120,
                "away_moneyline": -140,
                "spread_line": -1.5,
                "home_spread_odds": -110,
                "away_spread_odds": -110,
            }
        ]
    )
    monkeypatch.setattr(market_roi, "load_market_lines", lambda columns: odds[list(columns)])
    dataset = market_roi.prepare_dataset(
        _predictions_row(home_win_prob=0.60, pred_home_margin=2.0, home_margin=3.0)
    )
    benchmark = market_roi.build_market_benchmark(dataset)
    clv_detail = pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 2,
                "game_id": "2026_02_AWAY_HOME",
                "model_pick_side": "home",
                "model_pick_team": "HOME",
                "model_pick_prob": 0.60,
                "model_side_pick_moneyline": 130,
                "model_side_close_moneyline": 120,
                "model_side_pick_implied_prob_novig": 0.43,
                "model_side_close_implied_prob_novig": 0.45,
                "moneyline_clv_decimal_delta": 0.10,
                "moneyline_clv_prob_delta": 0.02,
                "moneyline_positive_clv": True,
                "model_vs_pick_prob_edge": 0.17,
                "model_vs_close_prob_edge": 0.15,
                "pick_selected_spread": 2.0,
                "close_selected_spread": 1.5,
                "spread_clv_points": 0.5,
                "spread_positive_clv": True,
                "clv_status": "movement_available",
            }
        ]
    )

    enriched = market_roi.attach_market_clv_detail(benchmark, clv_detail)
    row = enriched.iloc[0]

    assert bool(row["clv_detail_available"]) is True
    assert row["clv_status"] == "movement_available"
    assert row["clv_model_pick_side"] == "home"
    assert row["clv_model_side_pick_moneyline"] == 130
    assert row["clv_model_side_close_moneyline"] == 120
    assert row["moneyline_clv_prob_delta"] == pytest.approx(0.02)
    assert bool(row["moneyline_positive_clv"]) is True
    assert row["spread_clv_points"] == pytest.approx(0.5)


def test_market_benchmark_rejects_clv_detail_when_model_side_changed(monkeypatch):
    odds = pd.DataFrame(
        [
            {
                "game_id": "2026_02_AWAY_HOME",
                "home_moneyline": 120,
                "away_moneyline": -140,
                "spread_line": -1.5,
                "home_spread_odds": -110,
                "away_spread_odds": -110,
            }
        ]
    )
    monkeypatch.setattr(market_roi, "load_market_lines", lambda columns: odds[list(columns)])
    dataset = market_roi.prepare_dataset(
        _predictions_row(home_win_prob=0.60, pred_home_margin=2.0, home_margin=3.0)
    )
    benchmark = market_roi.build_market_benchmark(dataset)
    clv_detail = pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 2,
                "game_id": "2026_02_AWAY_HOME",
                "model_pick_side": "away",
                "moneyline_clv_prob_delta": 0.02,
                "spread_clv_points": 0.5,
                "clv_status": "movement_available",
            }
        ]
    )

    enriched = market_roi.attach_market_clv_detail(benchmark, clv_detail)
    row = enriched.iloc[0]

    assert bool(row["clv_detail_available"]) is True
    assert row["clv_status"] == "unavailable_model_side_mismatch"
    assert pd.isna(row["moneyline_clv_prob_delta"])
    assert pd.isna(row["spread_clv_points"])


def test_market_utility_segments_include_disagreement_and_edge_bins(monkeypatch):
    odds = pd.DataFrame(
        [
            {
                "game_id": "2026_02_AWAY_HOME",
                "home_moneyline": 120,
                "away_moneyline": -140,
                "spread_line": -1.5,
                "home_spread_odds": -110,
                "away_spread_odds": -110,
            }
        ]
    )
    monkeypatch.setattr(market_roi, "load_market_lines", lambda columns: odds[list(columns)])
    dataset = market_roi.prepare_dataset(
        _predictions_row(home_win_prob=0.60, pred_home_margin=2.0, home_margin=3.0)
    )
    benchmark = market_roi.build_market_benchmark(dataset)

    segments = market_roi.summarize_market_utility_segments(benchmark)

    disagreement = segments[
        segments["segment_family"].eq("agreement") & segments["segment"].eq("disagree")
    ].iloc[0]
    probability_edge = segments[
        segments["segment_family"].eq("probability_edge") & segments["segment"].eq("15%+")
    ].iloc[0]

    assert disagreement["n_bets"] == 1
    assert disagreement["wins"] == 1
    assert probability_edge["n_bets"] == 1
    assert probability_edge["model_side_roi"] == pytest.approx(1.2)


def test_moneyline_disagreement_edge_bins_are_bounded(monkeypatch):
    odds = pd.DataFrame(
        [
            {
                "game_id": "2026_02_AWAY_HOME",
                "home_moneyline": 120,
                "away_moneyline": -140,
                "spread_line": -1.5,
                "home_spread_odds": -110,
                "away_spread_odds": -110,
            }
        ]
    )
    monkeypatch.setattr(market_roi, "load_market_lines", lambda columns: odds[list(columns)])
    dataset = market_roi.prepare_dataset(
        _predictions_row(home_win_prob=0.60, pred_home_margin=2.0, home_margin=3.0)
    )
    benchmark = market_roi.build_market_benchmark(dataset)

    bins = market_roi.summarize_disagreement_moneyline_edge_bins(benchmark)
    row = bins[(bins["scope"] == "overall") & (bins["edge_bin"] == "15%+")].iloc[0]
    empty_row = bins[(bins["scope"] == "overall") & (bins["edge_bin"] == "8-10%")].iloc[0]

    assert row["n_bets"] == 1
    assert row["wins"] == 1
    assert row["roi"] == pytest.approx(1.2)
    assert empty_row["n_bets"] == 0


def test_moneyline_disagreement_segments_slice_side_odds_and_clv():
    benchmark = pd.DataFrame(
        [
            {
                "season": 2025,
                "favorite_agreement": False,
                "model_home_pick": True,
                "home_tie": False,
                "model_side_actual_win": 1,
                "model_side_profit": 1.2,
                "model_pick_prob": 0.60,
                "vegas_model_side_prob": 0.45,
                "prob_edge_model_side": 0.15,
                "model_side_moneyline": 120,
                "model_side_decimal": 2.2,
                "clv_status": "movement_available",
                "moneyline_clv_prob_delta": 0.01,
                "moneyline_positive_clv": True,
            },
            {
                "season": 2025,
                "favorite_agreement": False,
                "model_home_pick": False,
                "home_tie": False,
                "model_side_actual_win": 0,
                "model_side_profit": -1.0,
                "model_pick_prob": 0.57,
                "vegas_model_side_prob": 0.48,
                "prob_edge_model_side": 0.09,
                "model_side_moneyline": 150,
                "model_side_decimal": 2.5,
                "clv_status": "movement_available",
                "moneyline_clv_prob_delta": -0.02,
                "moneyline_positive_clv": False,
            },
            {
                "season": 2025,
                "favorite_agreement": False,
                "model_home_pick": True,
                "home_tie": True,
                "model_side_actual_win": 0,
                "model_side_profit": 0.0,
                "model_pick_prob": 0.54,
                "vegas_model_side_prob": 0.52,
                "prob_edge_model_side": 0.02,
                "model_side_moneyline": -125,
                "model_side_decimal": 1.8,
                "clv_status": "unavailable_missing_clv_detail",
                "moneyline_clv_prob_delta": pd.NA,
                "moneyline_positive_clv": pd.NA,
            },
            {
                "season": 2025,
                "favorite_agreement": True,
                "model_home_pick": True,
                "home_tie": False,
                "model_side_actual_win": 1,
                "model_side_profit": 0.8,
                "model_pick_prob": 0.70,
                "vegas_model_side_prob": 0.65,
                "prob_edge_model_side": 0.05,
                "model_side_moneyline": -130,
                "model_side_decimal": 1.77,
                "clv_status": "movement_available",
                "moneyline_clv_prob_delta": 0.03,
                "moneyline_positive_clv": True,
            },
        ]
    )

    segments = market_roi.summarize_disagreement_moneyline_segments(benchmark)

    overall = segments[
        segments["segment_family"].eq("all") & segments["segment"].eq("all")
    ].iloc[0]
    home_side = segments[
        segments["segment_family"].eq("model_pick_side") & segments["segment"].eq("home")
    ].iloc[0]
    underdog = segments[
        segments["segment_family"].eq("model_side_moneyline_status")
        & segments["segment"].eq("underdog")
    ].iloc[0]
    positive_clv = segments[
        segments["segment_family"].eq("moneyline_clv_result")
        & segments["segment"].eq("positive_clv")
    ].iloc[0]

    assert overall["n_bets"] == 3
    assert overall["wins"] == 1
    assert overall["losses"] == 1
    assert overall["pushes"] == 1
    assert overall["roi"] == pytest.approx((1.2 - 1.0) / 3.0)
    assert overall["positive_moneyline_clv_rate"] == pytest.approx(0.5)
    assert home_side["n_bets"] == 2
    assert underdog["n_bets"] == 2
    assert positive_clv["n_bets"] == 1
    assert positive_clv["roi"] == pytest.approx(1.2)


def test_market_probability_edge_calibration_compares_model_to_vegas_same_side():
    benchmark = pd.DataFrame(
        [
            {
                "season": 2026,
                "model_pick_prob": 0.60,
                "vegas_model_side_prob": 0.55,
                "prob_edge_model_side": 0.05,
                "home_tie": False,
                "model_side_actual_win": 1,
                "favorite_agreement": False,
                "model_side_profit": 0.8,
                "model_side_moneyline": 125,
            },
            {
                "season": 2026,
                "model_pick_prob": 0.60,
                "vegas_model_side_prob": 0.65,
                "prob_edge_model_side": -0.05,
                "home_tie": False,
                "model_side_actual_win": 0,
                "favorite_agreement": True,
                "model_side_profit": -1.0,
                "model_side_moneyline": -130,
            },
        ]
    )

    calibration = market_roi.summarize_market_probability_edge_calibration(benchmark)
    overall = calibration[
        (calibration["scope"] == "overall")
        & (calibration["bin_family"] == "all")
        & (calibration["bin_label"] == "all")
    ].iloc[0]
    model_bin = calibration[
        (calibration["scope"] == "overall")
        & (calibration["bin_family"] == "model_probability")
        & (calibration["bin_label"] == "60-65%")
    ].iloc[0]

    assert overall["n_games"] == 2
    assert overall["actual_model_side_win_rate"] == pytest.approx(0.5)
    assert overall["avg_model_pick_prob"] == pytest.approx(0.6)
    assert overall["avg_vegas_model_side_prob"] == pytest.approx(0.6)
    assert overall["model_brier"] == pytest.approx(0.26)
    assert overall["vegas_brier"] == pytest.approx(0.3125)
    assert overall["brier_delta_model_minus_vegas"] == pytest.approx(-0.0525)
    assert model_bin["n_games"] == 2


def test_model_confidence_calibration_slices_agreement_and_logloss():
    benchmark = pd.DataFrame(
        [
            {
                "season": 2026,
                "model_pick_prob": 0.62,
                "vegas_model_side_prob": 0.56,
                "prob_edge_model_side": 0.06,
                "home_tie": False,
                "model_side_actual_win": 1,
                "favorite_agreement": True,
                "model_side_profit": 0.8,
                "model_side_moneyline": -125,
            },
            {
                "season": 2026,
                "model_pick_prob": 0.64,
                "vegas_model_side_prob": 0.48,
                "prob_edge_model_side": 0.16,
                "home_tie": False,
                "model_side_actual_win": 0,
                "favorite_agreement": False,
                "model_side_profit": -1.0,
                "model_side_moneyline": 150,
            },
            {
                "season": 2026,
                "model_pick_prob": 0.72,
                "vegas_model_side_prob": 0.60,
                "prob_edge_model_side": 0.12,
                "home_tie": False,
                "model_side_actual_win": 1,
                "favorite_agreement": False,
                "model_side_profit": 1.2,
                "model_side_moneyline": 120,
            },
            {
                "season": 2026,
                "model_pick_prob": 0.74,
                "vegas_model_side_prob": 0.70,
                "prob_edge_model_side": 0.04,
                "home_tie": True,
                "model_side_actual_win": 0,
                "favorite_agreement": True,
                "model_side_profit": 0.0,
                "model_side_moneyline": -160,
            },
        ]
    )

    calibration = market_roi.summarize_model_confidence_calibration(benchmark)
    all_60_65 = calibration[
        (calibration["scope"] == "overall")
        & (calibration["agreement_segment"] == "all")
        & (calibration["confidence_bin"] == "60-65%")
    ].iloc[0]
    disagree_70 = calibration[
        (calibration["scope"] == "overall")
        & (calibration["agreement_segment"] == "disagree")
        & (calibration["confidence_bin"] == "70%+")
    ].iloc[0]

    expected_brier = ((0.62 - 1.0) ** 2 + (0.64 - 0.0) ** 2) / 2
    expected_logloss = -((np.log(0.62) + np.log(1.0 - 0.64)) / 2)

    assert all_60_65["n_games"] == 2
    assert all_60_65["n_graded"] == 2
    assert all_60_65["accuracy"] == pytest.approx(0.5)
    assert all_60_65["avg_model_pick_prob"] == pytest.approx(0.63)
    assert all_60_65["calibration_error"] == pytest.approx(0.13)
    assert all_60_65["brier"] == pytest.approx(expected_brier)
    assert all_60_65["log_loss"] == pytest.approx(expected_logloss)
    assert all_60_65["favorite_agreement_rate"] == pytest.approx(0.5)
    assert disagree_70["n_games"] == 1
    assert disagree_70["model_side_roi"] == pytest.approx(1.2)


def test_model_confidence_recalibration_flags_overconfident_disagreement_bins():
    calibration = pd.DataFrame(
        [
            {
                "scope": "overall",
                "season": "all",
                "agreement_segment": "disagree",
                "confidence_bin": "55-60%",
                "min_confidence": 0.55,
                "max_confidence": 0.60,
                "n_games": 80,
                "n_graded": 80,
                "actual_model_side_win_rate": 0.46,
                "avg_model_pick_prob": 0.57,
                "calibration_error": 0.11,
            },
            {
                "scope": "overall",
                "season": "all",
                "agreement_segment": "disagree",
                "confidence_bin": "60-65%",
                "min_confidence": 0.60,
                "max_confidence": 0.65,
                "n_games": 70,
                "n_graded": 70,
                "actual_model_side_win_rate": 0.61,
                "avg_model_pick_prob": 0.63,
                "calibration_error": 0.02,
            },
        ]
    )

    recalibration = market_roi.summarize_model_confidence_recalibration(
        calibration,
        min_graded=25,
        max_overestimate=0.07,
        confidence_min_prob=0.55,
    )
    overconfident = recalibration[recalibration["confidence_bin"].eq("55-60%")].iloc[0]
    calibrated = recalibration[recalibration["confidence_bin"].eq("60-65%")].iloc[0]

    assert overconfident["recalibrated_model_pick_prob"] == pytest.approx(0.46)
    assert overconfident["raw_minus_recalibrated_prob"] == pytest.approx(0.11)
    assert bool(overconfident["recommended_for_confidence"]) is False
    assert overconfident["calibration_status"] == "overconfident"
    assert bool(calibrated["recommended_for_confidence"]) is True
    assert calibrated["calibration_status"] == "calibrated_for_confidence"


def test_model_confidence_calibration_gate_blocks_overconfident_disagreement_bins():
    recalibration = pd.DataFrame(
        [
            {
                "scope": "overall",
                "season": "all",
                "agreement_segment": "disagree",
                "confidence_bin": "55-60%",
                "min_confidence": 0.55,
                "n_graded": 80,
                "calibration_error": 0.11,
                "recalibrated_model_pick_prob": 0.46,
                "raw_minus_recalibrated_prob": 0.11,
            },
            {
                "scope": "overall",
                "season": "all",
                "agreement_segment": "disagree",
                "confidence_bin": "60-65%",
                "min_confidence": 0.60,
                "n_graded": 70,
                "calibration_error": 0.03,
                "recalibrated_model_pick_prob": 0.60,
                "raw_minus_recalibrated_prob": 0.03,
            },
        ]
    )

    gate = market_roi.summarize_model_confidence_calibration_gate(
        recalibration,
        min_graded=25,
        max_overestimate=0.07,
        confidence_min_prob=0.55,
    )

    assert bool(gate["pass"]) is False
    assert gate["status"] == "blocked_model_confidence_overconfidence"
    assert gate["evaluated_bins"] == 2
    assert gate["failed_bins"] == "55-60%"
    assert gate["worst_bin"] == "55-60%"
    assert gate["worst_error"] == pytest.approx(0.11)


def test_moneyline_disagreement_stability_passes_with_multiseason_profitability():
    benchmark = _moneyline_stability_benchmark(
        {
            2018: 0.10,
            2019: 0.02,
            2020: 0.03,
            2021: -0.05,
            2022: 0.06,
            2023: 0.04,
        }
    )

    stability = market_roi.summarize_disagreement_moneyline_season_stability(
        benchmark,
        thresholds=[0.02],
        min_season_bets=20,
        min_qualified_seasons=5,
        min_positive_season_rate=0.55,
        min_recent_qualified_seasons=2,
        min_recent_positive_season_rate=0.50,
    )
    row = stability.iloc[0]

    assert row["qualified_seasons"] == 6
    assert row["positive_qualified_season_rate"] == pytest.approx(5 / 6)
    assert bool(row["stability_pass"]) is True
    assert row["stability_status"] == "stable"


def test_moneyline_disagreement_stability_fails_weak_season_rate():
    benchmark = _moneyline_stability_benchmark(
        {
            2018: 0.10,
            2019: -0.02,
            2020: -0.03,
            2021: -0.05,
            2022: 0.06,
            2023: -0.04,
        }
    )

    stability = market_roi.summarize_disagreement_moneyline_season_stability(
        benchmark,
        thresholds=[0.02],
        min_season_bets=20,
        min_qualified_seasons=5,
        min_positive_season_rate=0.55,
        min_recent_qualified_seasons=2,
        min_recent_positive_season_rate=0.50,
    )
    row = stability.iloc[0]

    assert row["positive_qualified_season_rate"] == pytest.approx(2 / 6)
    assert bool(row["stability_pass"]) is False
    assert row["stability_status"] == "weak_positive_season_rate"


def test_spread_cover_calibration_reports_overconfidence(monkeypatch):
    odds = pd.DataFrame(
        [
            {
                "game_id": "2026_02_AWAY_HOME",
                "home_moneyline": -271,
                "away_moneyline": 220,
                "spread_line": 6.5,
                "home_spread_odds": -110,
                "away_spread_odds": -110,
            }
        ]
    )
    monkeypatch.setattr(market_roi, "load_market_lines", lambda columns: odds[list(columns)])
    dataset = market_roi.prepare_dataset(
        _predictions_row(home_win_prob=0.60, pred_home_margin=3.5, home_margin=1.0)
    )

    calibration = market_roi.summarize_spread_cover_calibration(dataset, sigma_margin_error=10.0)
    row = calibration[
        (calibration["scope"] == "overall")
        & (calibration["bet_side"] == "all")
        & (calibration["cover_prob_bin"] == "60%-65%")
    ].iloc[0]

    assert row["n_bets"] == 1
    assert row["wins"] == 1
    assert row["actual_cover_rate"] == pytest.approx(1.0)
    assert row["avg_cover_prob"] == pytest.approx(market_roi.standard_normal_cdf(0.3), abs=1e-6)
    assert row["calibration_error"] < 0.0


def test_market_edge_policy_does_not_recommend_negative_disagreement_roi():
    disagreement_roi = pd.DataFrame(
        [
            {
                "threshold": 0.02,
                "n_bets": 120,
                "wins": 54,
                "losses": 66,
                "pushes": 0,
                "hit_rate": 0.45,
                "avg_prob_edge_model_side": 0.12,
                "roi": -0.02,
            }
        ]
    )
    spread_roi = pd.DataFrame(
        [
            {
                "bet_side": "away",
                "threshold": 2.0,
                "n_bets": 150,
                "wins": 82,
                "losses": 68,
                "pushes": 0,
                "hit_rate": 82 / 150,
                "roi": 0.04,
                "avg_edge_points": 3.1,
                "avg_expected_profit": 0.08,
            }
        ]
    )
    spread_edge_bins = pd.DataFrame(
        [
            {
                "scope": "overall",
                "edge_bin": "2-3",
                "min_edge": 2.0,
                "n_bets": 140,
                "wins": 76,
                "losses": 64,
                "pushes": 0,
                "hit_rate": 76 / 140,
                "roi": 0.03,
                "avg_abs_edge_points": 2.4,
            }
        ]
    )

    policy = market_roi.build_market_edge_policy(
        disagreement_roi,
        spread_roi,
        spread_edge_bins,
        min_bets=100,
        min_roi=0.0,
        min_probability_edge=0.02,
        min_spread_edge=2.0,
        generated_at="2026-09-27T00:00:00+00:00",
    )

    moneyline = policy[policy["signal"].eq("moneyline_disagreement")].iloc[0]
    spread = policy[policy["signal"].eq("spread_threshold")].iloc[0]
    edge_bin = policy[policy["signal"].eq("spread_edge_bin")].iloc[0]

    assert bool(moneyline["recommended"]) is False
    assert moneyline["recommendation_tier"] == "no_bet"
    assert bool(spread["recommended"]) is False
    assert spread["recommendation_tier"] == "no_bet"
    assert bool(edge_bin["recommended"]) is True


def test_market_edge_policy_downgrades_positive_moneyline_without_stability():
    disagreement_roi = pd.DataFrame(
        [
            {
                "threshold": 0.02,
                "n_bets": 120,
                "wins": 64,
                "losses": 56,
                "pushes": 0,
                "hit_rate": 64 / 120,
                "avg_prob_edge_model_side": 0.04,
                "roi": 0.06,
            }
        ]
    )
    stability = pd.DataFrame(
        [
            {
                "threshold": 0.02,
                "stability_pass": False,
                "stability_status": "weak_positive_season_rate",
                "qualified_seasons": 6,
                "positive_qualified_season_rate": 2 / 6,
                "recent_positive_qualified_season_rate": 1 / 3,
                "severe_negative_qualified_season_rate": 0.0,
                "min_season_bets": 20,
                "min_qualified_seasons": 5,
            }
        ]
    )

    policy = market_roi.build_market_edge_policy(
        disagreement_roi,
        pd.DataFrame(),
        pd.DataFrame(),
        moneyline_stability=stability,
        min_bets=100,
        min_roi=0.0,
        min_probability_edge=0.02,
        min_spread_edge=2.0,
        generated_at="2026-09-27T00:00:00+00:00",
    )
    moneyline = policy[policy["signal"].eq("moneyline_disagreement")].iloc[0]

    assert bool(moneyline["recommended"]) is False
    assert moneyline["recommendation_tier"] == "review"
    assert bool(moneyline["stability_pass"]) is False
    assert moneyline["stability_status"] == "weak_positive_season_rate"


def test_market_edge_policy_uses_best_stable_moneyline_threshold():
    disagreement_roi = pd.DataFrame(
        [
            {
                "threshold": 0.02,
                "n_bets": 120,
                "wins": 64,
                "losses": 56,
                "pushes": 0,
                "hit_rate": 64 / 120,
                "avg_prob_edge_model_side": 0.04,
                "roi": 0.05,
            },
            {
                "threshold": 0.08,
                "n_bets": 120,
                "wins": 70,
                "losses": 50,
                "pushes": 0,
                "hit_rate": 70 / 120,
                "avg_prob_edge_model_side": 0.10,
                "roi": 0.10,
            },
        ]
    )
    stability = pd.DataFrame(
        [
            {
                "threshold": 0.02,
                "stability_pass": True,
                "stability_status": "stable",
                "qualified_seasons": 6,
                "positive_qualified_season_rate": 4 / 6,
                "recent_positive_qualified_season_rate": 1.0,
                "severe_negative_qualified_season_rate": 0.0,
                "min_season_bets": 20,
                "min_qualified_seasons": 5,
            },
            {
                "threshold": 0.08,
                "stability_pass": False,
                "stability_status": "insufficient_recent_qualified_seasons",
                "qualified_seasons": 5,
                "positive_qualified_season_rate": 4 / 5,
                "recent_positive_qualified_season_rate": 1.0,
                "severe_negative_qualified_season_rate": 0.0,
                "min_season_bets": 20,
                "min_qualified_seasons": 5,
            },
        ]
    )

    policy = market_roi.build_market_edge_policy(
        disagreement_roi,
        pd.DataFrame(),
        pd.DataFrame(),
        moneyline_stability=stability,
        min_bets=100,
        min_roi=0.0,
        min_probability_edge=0.02,
        min_spread_edge=2.0,
        generated_at="2026-09-27T00:00:00+00:00",
    )
    moneyline = policy[policy["signal"].eq("moneyline_disagreement")].iloc[0]

    assert bool(moneyline["recommended"]) is True
    assert moneyline["threshold"] == pytest.approx(0.02)
    assert moneyline["stability_status"] == "stable"


def test_market_edge_policy_downgrades_moneyline_when_clv_sample_is_sparse():
    disagreement_roi = pd.DataFrame(
        [
            {
                "threshold": 0.02,
                "n_bets": 120,
                "wins": 64,
                "losses": 56,
                "pushes": 0,
                "hit_rate": 64 / 120,
                "avg_prob_edge_model_side": 0.04,
                "roi": 0.06,
            }
        ]
    )
    clv_summary = pd.DataFrame(
        [
            {
                "scope": "moneyline_threshold",
                "segment": "prob_edge >= 0.02",
                "threshold": 0.02,
                "n_games": 6,
                "moneyline_clv_available": 6,
                "avg_moneyline_clv_prob_delta": 0.008,
                "positive_moneyline_clv_rate": 4 / 6,
            }
        ]
    )

    policy = market_roi.build_market_edge_policy(
        disagreement_roi,
        pd.DataFrame(),
        pd.DataFrame(),
        moneyline_clv_summary=clv_summary,
        min_bets=100,
        min_roi=0.0,
        min_probability_edge=0.02,
        min_spread_edge=2.0,
        clv_min_tracked_games=10,
        generated_at="2026-09-27T00:00:00+00:00",
    )
    moneyline = policy[policy["signal"].eq("moneyline_disagreement")].iloc[0]

    assert bool(moneyline["recommended"]) is False
    assert moneyline["recommendation_tier"] == "review"
    assert moneyline["clv_status"] == "insufficient_tracked_games"
    assert moneyline["clv_tracked_games"] == 6
    assert moneyline["clv_positive_rate"] == pytest.approx(4 / 6)


def test_market_edge_policy_allows_moneyline_when_clv_gate_passes():
    disagreement_roi = pd.DataFrame(
        [
            {
                "threshold": 0.02,
                "n_bets": 120,
                "wins": 64,
                "losses": 56,
                "pushes": 0,
                "hit_rate": 64 / 120,
                "avg_prob_edge_model_side": 0.04,
                "roi": 0.06,
            }
        ]
    )
    clv_summary = pd.DataFrame(
        [
            {
                "scope": "moneyline_threshold",
                "segment": "prob_edge >= 0.02",
                "threshold": 0.02,
                "n_games": 12,
                "moneyline_clv_available": 12,
                "avg_moneyline_clv_prob_delta": 0.009,
                "positive_moneyline_clv_rate": 7 / 12,
            }
        ]
    )

    policy = market_roi.build_market_edge_policy(
        disagreement_roi,
        pd.DataFrame(),
        pd.DataFrame(),
        moneyline_clv_summary=clv_summary,
        min_bets=100,
        min_roi=0.0,
        min_probability_edge=0.02,
        min_spread_edge=2.0,
        clv_min_tracked_games=10,
        clv_min_positive_rate=0.52,
        generated_at="2026-09-27T00:00:00+00:00",
    )
    moneyline = policy[policy["signal"].eq("moneyline_disagreement")].iloc[0]

    assert bool(moneyline["recommended"]) is True
    assert moneyline["recommendation_tier"] == "strong"
    assert bool(moneyline["clv_pass"]) is True
    assert moneyline["clv_status"] == "clv_pass"


def test_market_edge_policy_blocks_moneyline_when_segment_gate_fails():
    disagreement_roi = pd.DataFrame(
        [
            {
                "threshold": 0.02,
                "n_bets": 160,
                "wins": 82,
                "losses": 78,
                "pushes": 0,
                "hit_rate": 82 / 160,
                "avg_prob_edge_model_side": 0.06,
                "roi": 0.04,
            }
        ]
    )
    segments = pd.DataFrame(
        [
            {
                "scope": "overall",
                "segment_family": "probability_edge",
                "segment": "2-5%",
                "n_bets": 55,
                "roi": 0.07,
            },
            {
                "scope": "overall",
                "segment_family": "probability_edge",
                "segment": "5-8%",
                "n_bets": 60,
                "roi": -0.04,
            },
            {
                "scope": "overall",
                "segment_family": "probability_edge",
                "segment": "8-10%",
                "n_bets": 45,
                "roi": 0.02,
            },
            {
                "scope": "overall",
                "segment_family": "probability_edge",
                "segment": "10-15%",
                "n_bets": 70,
                "roi": -0.20,
            },
        ]
    )

    policy = market_roi.build_market_edge_policy(
        disagreement_roi,
        pd.DataFrame(),
        pd.DataFrame(),
        moneyline_segments=segments,
        min_bets=100,
        min_roi=0.0,
        min_probability_edge=0.02,
        min_spread_edge=2.0,
        generated_at="2026-09-27T00:00:00+00:00",
    )
    moneyline = policy[policy["signal"].eq("moneyline_disagreement")].iloc[0]

    assert bool(moneyline["recommended"]) is False
    assert moneyline["recommendation_tier"] == "no_bet"
    assert bool(moneyline["segment_gate_pass"]) is False
    assert moneyline["segment_gate_status"] == "weak_probability_edge_segment"
    assert moneyline["segment_gate_failed_segments"] == "5-8%"
    assert moneyline["segment_gate_worst_segment"] == "5-8%"
    assert moneyline["segment_gate_worst_roi"] == pytest.approx(-0.04)


def test_market_edge_policy_blocks_moneyline_when_confidence_gate_fails():
    disagreement_roi = pd.DataFrame(
        [
            {
                "threshold": 0.02,
                "n_bets": 160,
                "wins": 86,
                "losses": 74,
                "pushes": 0,
                "hit_rate": 86 / 160,
                "avg_prob_edge_model_side": 0.06,
                "roi": 0.04,
            }
        ]
    )
    recalibration = pd.DataFrame(
        [
            {
                "scope": "overall",
                "season": "all",
                "agreement_segment": "disagree",
                "confidence_bin": "55-60%",
                "min_confidence": 0.55,
                "n_graded": 80,
                "calibration_error": 0.11,
                "recalibrated_model_pick_prob": 0.46,
                "raw_minus_recalibrated_prob": 0.11,
            }
        ]
    )

    policy = market_roi.build_market_edge_policy(
        disagreement_roi,
        pd.DataFrame(),
        pd.DataFrame(),
        model_confidence_recalibration=recalibration,
        min_bets=100,
        min_roi=0.0,
        min_probability_edge=0.02,
        min_spread_edge=2.0,
        model_confidence_min_graded=25,
        model_confidence_max_overestimate=0.07,
        model_confidence_min_prob=0.55,
        generated_at="2026-09-27T00:00:00+00:00",
    )
    moneyline = policy[policy["signal"].eq("moneyline_disagreement")].iloc[0]

    assert bool(moneyline["recommended"]) is False
    assert moneyline["recommendation_tier"] == "no_bet"
    assert bool(moneyline["model_confidence_gate_pass"]) is False
    assert moneyline["model_confidence_gate_status"] == "blocked_model_confidence_overconfidence"
    assert moneyline["model_confidence_gate_failed_bins"] == "55-60%"
    assert moneyline["model_confidence_gate_worst_error"] == pytest.approx(0.11)


def test_market_edge_policy_limits_spread_edge_bin_to_two_to_three_points():
    disagreement_roi = pd.DataFrame(
        [
            {
                "threshold": 0.02,
                "n_bets": 120,
                "wins": 64,
                "losses": 56,
                "pushes": 0,
                "hit_rate": 64 / 120,
                "avg_prob_edge_model_side": 0.12,
                "roi": 0.03,
            }
        ]
    )
    spread_roi = pd.DataFrame()
    spread_edge_bins = pd.DataFrame(
        [
            {
                "scope": "overall",
                "edge_bin": "2-3",
                "min_edge": 2.0,
                "n_bets": 140,
                "wins": 74,
                "losses": 66,
                "pushes": 0,
                "hit_rate": 74 / 140,
                "roi": 0.02,
                "avg_abs_edge_points": 2.4,
            },
            {
                "scope": "overall",
                "edge_bin": "3-5",
                "min_edge": 3.0,
                "n_bets": 220,
                "wins": 125,
                "losses": 95,
                "pushes": 0,
                "hit_rate": 125 / 220,
                "roi": 0.10,
                "avg_abs_edge_points": 3.8,
            },
        ]
    )

    policy = market_roi.build_market_edge_policy(
        disagreement_roi,
        spread_roi,
        spread_edge_bins,
        min_bets=100,
        min_roi=0.0,
        min_probability_edge=0.02,
        min_spread_edge=2.0,
        generated_at="2026-09-27T00:00:00+00:00",
    )

    edge_bin = policy[policy["signal"].eq("spread_edge_bin")].iloc[0]

    assert bool(edge_bin["recommended"]) is True
    assert edge_bin["edge_bin"] == "2-3"
    assert edge_bin["roi"] == pytest.approx(0.02)


def test_market_edge_policy_blocks_spread_edge_bin_when_cover_calibration_fails():
    disagreement_roi = pd.DataFrame()
    spread_roi = pd.DataFrame()
    spread_edge_bins = pd.DataFrame(
        [
            {
                "scope": "overall",
                "edge_bin": "2-3",
                "min_edge": 2.0,
                "n_bets": 140,
                "wins": 74,
                "losses": 66,
                "pushes": 0,
                "hit_rate": 74 / 140,
                "roi": 0.02,
                "avg_abs_edge_points": 2.4,
            }
        ]
    )
    spread_calibration = pd.DataFrame(
        [
            {
                "scope": "overall",
                "season": "all",
                "bet_side": "all",
                "cover_prob_bin": "65%-70%",
                "min_cover_prob": 0.65,
                "max_cover_prob": 0.70,
                "n_bets": 120,
                "actual_cover_rate": 0.49,
                "avg_cover_prob": 0.67,
                "calibration_error": 0.18,
            }
        ]
    )
    recalibration = market_roi.summarize_spread_cover_recalibration(
        spread_calibration,
        min_bets=50,
        max_abs_calibration_error=0.05,
        confidence_min_cover_prob=0.65,
    )

    policy = market_roi.build_market_edge_policy(
        disagreement_roi,
        spread_roi,
        spread_edge_bins,
        spread_cover_recalibration=recalibration,
        min_bets=100,
        min_roi=0.0,
        min_probability_edge=0.02,
        min_spread_edge=2.0,
        generated_at="2026-09-27T00:00:00+00:00",
    )

    edge_bin = policy[policy["signal"].eq("spread_edge_bin")].iloc[0]

    assert bool(edge_bin["recommended"]) is False
    assert edge_bin["recommendation_tier"] == "review"
    assert edge_bin["spread_cover_calibration_status"] == "blocked_spread_cover_overconfidence"
    assert edge_bin["spread_cover_calibration_worst_bin"] == "65%-70%"


def test_spread_edge_bins_report_selected_side(monkeypatch):
    odds = pd.DataFrame(
        [
            {
                "game_id": "2026_02_AWAY_HOME",
                "home_moneyline": -271,
                "away_moneyline": 220,
                "spread_line": 6.5,
                "home_spread_odds": -110,
                "away_spread_odds": -110,
            }
        ]
    )
    monkeypatch.setattr(market_roi, "load_market_lines", lambda columns: odds[list(columns)])
    dataset = market_roi.prepare_dataset(
        _predictions_row(home_win_prob=0.60, pred_home_margin=3.5, home_margin=1.0)
    )

    bins = market_roi.summarize_spread_edge_bins(dataset)
    row = bins[(bins["scope"] == "overall") & (bins["edge_bin"] == "3-5")].iloc[0]

    assert row["n_bets"] == 1
    assert row["home_bets"] == 0
    assert row["away_bets"] == 1
    assert row["wins"] == 1
    assert row["roi"] == pytest.approx(100.0 / 110.0)


def test_spread_key_numbers_report_crossed_key_for_selected_side(monkeypatch):
    odds = pd.DataFrame(
        [
            {
                "game_id": "2026_02_AWAY_HOME",
                "home_moneyline": -271,
                "away_moneyline": 220,
                "spread_line": 6.5,
                "home_spread_odds": -110,
                "away_spread_odds": -110,
            }
        ]
    )
    monkeypatch.setattr(market_roi, "load_market_lines", lambda columns: odds[list(columns)])
    dataset = market_roi.prepare_dataset(
        _predictions_row(home_win_prob=0.60, pred_home_margin=3.5, home_margin=1.0)
    )

    key_numbers = market_roi.summarize_spread_key_numbers(dataset, key_numbers=[6.0])
    row = key_numbers[
        (key_numbers["scope"] == "overall")
        & (key_numbers["key_number"] == 6.0)
        & (key_numbers["edge_bucket"] == "3-5")
        & (key_numbers["segment"] == "edge_crosses_key")
    ].iloc[0]

    assert row["n_bets"] == 1
    assert row["home_bets"] == 0
    assert row["away_bets"] == 1
    assert row["wins"] == 1
    assert row["avg_market_abs_line"] == pytest.approx(6.5)
    assert row["avg_model_abs_line"] == pytest.approx(3.5)
    assert row["roi"] == pytest.approx(100.0 / 110.0)
