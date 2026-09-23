from __future__ import annotations

import pytest
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
