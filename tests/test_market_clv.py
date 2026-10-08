from __future__ import annotations

import pandas as pd
import pytest

from analysis import market_clv


def test_build_market_clv_detail_calculates_moneyline_and_spread_movement():
    picks = pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 5,
                "game_id": "2026_05_AWAY_HOME",
                "home_team": "HOME",
                "away_team": "AWAY",
                "home_win_prob": 0.42,
                "market_home_moneyline": 120,
                "market_away_moneyline": -140,
                "market_home_spread": 3.5,
                "market_away_spread": -3.5,
                "market_moneyline_favorite_disagreement": True,
                "market_moneyline_policy_recommendation_flag": True,
                "market_moneyline_policy_status": "strong_moneyline_disagreement",
                "market_policy_primary_signal": "moneyline_disagreement",
                "market_policy_tier": "strong",
                "market_policy_status": "strong_moneyline_disagreement",
            }
        ]
    )
    close = pd.DataFrame(
        [
            {
                "game_id": "2026_05_AWAY_HOME",
                "home_moneyline": 105,
                "away_moneyline": -160,
                "spread_line": -4.0,
            }
        ]
    )

    detail = market_clv.build_market_clv_detail(picks, close)
    row = detail.iloc[0]

    assert row["model_pick_side"] == "away"
    assert row["model_side_pick_moneyline"] == -140
    assert row["model_side_close_moneyline"] == -160
    assert row["moneyline_clv_decimal_delta"] > 0.0
    assert bool(row["moneyline_positive_clv"]) is True
    assert row["pick_selected_spread"] == pytest.approx(-3.5)
    assert row["close_selected_spread"] == pytest.approx(-4.0)
    assert row["spread_clv_points"] == pytest.approx(0.5)
    assert bool(row["spread_positive_clv"]) is True
    assert row["clv_status"] == "movement_available"


def test_summarize_market_clv_reports_policy_segments():
    detail = pd.DataFrame(
        [
            {
                "moneyline_clv_decimal_delta": 0.10,
                "moneyline_clv_prob_delta": 0.02,
                "moneyline_positive_clv": True,
                "model_vs_pick_prob_edge": 0.03,
                "model_vs_close_prob_edge": 0.03,
                "spread_clv_points": 0.5,
                "spread_positive_clv": True,
                "moneyline_favorite_disagreement": True,
                "moneyline_policy_recommendation_flag": True,
                "spread_policy_recommendation_flag": False,
                "policy_status": "strong_moneyline_disagreement",
                "clv_status": "movement_available",
            },
            {
                "moneyline_clv_decimal_delta": -0.05,
                "moneyline_clv_prob_delta": -0.01,
                "moneyline_positive_clv": False,
                "model_vs_pick_prob_edge": 0.01,
                "model_vs_close_prob_edge": 0.01,
                "spread_clv_points": None,
                "spread_positive_clv": None,
                "moneyline_favorite_disagreement": False,
                "moneyline_policy_recommendation_flag": False,
                "spread_policy_recommendation_flag": False,
                "policy_status": "no_bet",
                "clv_status": "movement_available",
            },
        ]
    )

    summary = market_clv.summarize_market_clv(detail)
    overall = summary[(summary["scope"] == "overall") & (summary["segment"] == "all")].iloc[0]
    policy = summary[
        (summary["scope"] == "overall") & (summary["segment"] == "moneyline_policy_recommended")
    ].iloc[0]
    threshold = summary[
        (summary["scope"] == "moneyline_threshold") & (pd.to_numeric(summary["threshold"]).round(2).eq(0.02))
    ].iloc[0]

    assert overall["n_games"] == 2
    assert overall["positive_moneyline_clv_rate"] == pytest.approx(0.5)
    assert policy["n_games"] == 1
    assert policy["avg_moneyline_clv_prob_delta"] == pytest.approx(0.02)
    assert threshold["n_games"] == 1
    assert threshold["avg_moneyline_clv_prob_delta"] == pytest.approx(0.02)
