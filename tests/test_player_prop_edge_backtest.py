from __future__ import annotations

import pandas as pd

from src.player_props.edge_backtest import (
    build_edge_backtest,
    build_edge_recommendation_policy,
    build_market_threshold_summary,
    summarize_edge_backtest,
)


def _line_row(
    *,
    player_id: str,
    sportsbook: str,
    line: float,
    over_odds: float,
    under_odds: float,
    actual_value: float,
) -> dict[str, object]:
    return {
        "season": 2025,
        "week": 1,
        "game_id": "2025_01_ARI_NO",
        "market": "qb_passing_yards",
        "sportsbook": sportsbook,
        "player_id": player_id,
        "model_player_name": f"Player {player_id}",
        "line_player_name": f"Player {player_id}",
        "team": "ARI",
        "opponent": "NO",
        "position": "QB",
        "line": line,
        "over_odds": over_odds,
        "under_odds": under_odds,
        "model_projection": 100.0,
        "actual_value": actual_value,
        "actual_over_line": actual_value > line,
        "actual_under_line": actual_value < line,
        "actual_push": actual_value == line,
        "line_is_comparable": True,
        "line_updated_at": "2025-09-07T16:45:00Z",
        "match_method": "test",
    }


def _probability_row(
    *,
    player_id: str,
    sportsbook: str,
    line: float,
    prob_over_model: float,
    prob_over_market: float,
) -> dict[str, object]:
    return {
        "season": 2025,
        "week": 1,
        "game_id": "2025_01_ARI_NO",
        "market": "qb_passing_yards",
        "sportsbook": sportsbook,
        "player_id": player_id,
        "line": line,
        "prob_over_model": prob_over_model,
        "prob_over_market": prob_over_market,
        "prob_edge_vs_market": prob_over_model - prob_over_market,
        "calibration_method": "market_residual_blend",
        "calibration_sample_size": 500,
    }


def test_edge_backtest_grades_ev_and_best_price_rows():
    lines = pd.DataFrame(
        [
            _line_row(player_id="p1", sportsbook="draftkings", line=100.5, over_odds=100, under_odds=-120, actual_value=130),
            _line_row(player_id="p2", sportsbook="draftkings", line=100.5, over_odds=-110, under_odds=100, actual_value=80),
            _line_row(player_id="p3", sportsbook="draftkings", line=100.5, over_odds=-110, under_odds=-110, actual_value=80),
            _line_row(player_id="p3", sportsbook="fanduel", line=101.5, over_odds=-115, under_odds=120, actual_value=80),
        ]
    )
    probabilities = pd.DataFrame(
        [
            _probability_row(player_id="p1", sportsbook="draftkings", line=100.5, prob_over_model=0.60, prob_over_market=0.50),
            _probability_row(player_id="p2", sportsbook="draftkings", line=100.5, prob_over_model=0.35, prob_over_market=0.50),
            _probability_row(player_id="p3", sportsbook="draftkings", line=100.5, prob_over_model=0.40, prob_over_market=0.50),
            _probability_row(player_id="p3", sportsbook="fanduel", line=101.5, prob_over_model=0.35, prob_over_market=0.48),
        ]
    )

    detail = build_edge_backtest(
        lines,
        probabilities,
        ev_thresholds=[0.0],
        prob_edge_thresholds=[0.05],
        generated_at="2026-09-25T00:00:00+00:00",
    )
    summary = summarize_edge_backtest(detail, generated_at="2026-09-25T00:00:00+00:00")
    market_thresholds = build_market_threshold_summary(summary)

    per_book_ev = detail[
        detail["selection_scope"].eq("per_book") & detail["strategy"].eq("ev") & detail["threshold"].eq(0.0)
    ]
    assert len(per_book_ev) == 4
    assert set(per_book_ev["bet_side"]) == {"over", "under"}
    assert per_book_ev["profit"].sum() > 4.0

    best_p3 = detail[
        detail["selection_scope"].eq("best_price")
        & detail["strategy"].eq("ev")
        & detail["player_id"].eq("p3")
    ].iloc[0]
    assert best_p3["sportsbook"] == "fanduel"
    assert best_p3["bet_side"] == "under"
    assert best_p3["profit"] == 1.2

    overall = summary[
        summary["selection_scope"].eq("per_book")
        & summary["strategy"].eq("ev")
        & summary["scope"].eq("all")
    ].iloc[0]
    assert overall["n_bets"] == 4
    assert overall["wins"] == 4
    assert overall["roi"] > 1.0
    assert not market_thresholds.empty


def test_edge_backtest_uses_strict_positive_thresholds():
    lines = pd.DataFrame(
        [
            _line_row(player_id="p1", sportsbook="draftkings", line=100.5, over_odds=100, under_odds=100, actual_value=130),
        ]
    )
    probabilities = pd.DataFrame(
        [
            _probability_row(player_id="p1", sportsbook="draftkings", line=100.5, prob_over_model=0.50, prob_over_market=0.50),
        ]
    )

    detail = build_edge_backtest(
        lines,
        probabilities,
        ev_thresholds=[0.0],
        prob_edge_thresholds=[0.0],
        generated_at="2026-09-25T00:00:00+00:00",
    )

    assert detail.empty


def test_build_edge_recommendation_policy_selects_positive_market_thresholds():
    summary = pd.DataFrame(
        [
            {
                "selection_scope": "best_price",
                "strategy": "prob_edge",
                "scope": "market",
                "market": "anytime_td",
                "sportsbook": pd.NA,
                "threshold": 0.02,
                "n_bets": 125,
                "wins": 20,
                "losses": 105,
                "pushes": 0,
                "win_rate": 0.16,
                "avg_odds": 600.0,
                "avg_model_probability": 0.20,
                "avg_market_probability": 0.08,
                "avg_prob_edge": 0.12,
                "avg_edge_ev": 1.2,
                "roi": 0.12,
                "avg_profit_per_bet": 0.12,
                "generated_at": "2026-09-25T00:00:00+00:00",
            },
            {
                "selection_scope": "best_price",
                "strategy": "prob_edge",
                "scope": "market",
                "market": "def_sacks",
                "sportsbook": pd.NA,
                "threshold": 0.02,
                "n_bets": 125,
                "wins": 40,
                "losses": 85,
                "pushes": 0,
                "win_rate": 0.32,
                "avg_odds": 200.0,
                "avg_model_probability": 0.35,
                "avg_market_probability": 0.25,
                "avg_prob_edge": 0.10,
                "avg_edge_ev": 0.2,
                "roi": -0.05,
                "avg_profit_per_bet": -0.05,
                "generated_at": "2026-09-25T00:00:00+00:00",
            },
        ]
    )

    policy = build_edge_recommendation_policy(summary, min_bets=100, min_roi=0.0, min_threshold=0.02)

    assert policy["market"].tolist() == ["anytime_td"]
    row = policy.iloc[0]
    assert row["threshold"] == 0.02
    assert row["recommendation_tier"] == "strong"
