from __future__ import annotations

import pandas as pd

from analysis import market_disagreement_diagnostics as diag


def _benchmark_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for season in [2020, 2021, 2022]:
        for idx in range(6):
            home_win = idx < 4
            rows.append(
                {
                    "season": season,
                    "week": idx + 1,
                    "game_id": f"{season}_{idx}_AWY_HOM",
                    "home_team": "HOM",
                    "away_team": "AWY",
                    "home_win_prob": 0.58,
                    "home_implied_prob_novig": 0.43,
                    "home_moneyline": 150,
                    "away_moneyline": -170,
                    "home_margin": 7.0 if home_win else -3.0,
                    "model_home_pick": True,
                    "vegas_home_pick": False,
                    "favorite_agreement": False,
                    "model_correct": home_win,
                    "vegas_correct": not home_win,
                    "model_pick_prob": 0.58,
                    "vegas_model_side_prob": 0.43,
                    "prob_edge_model_side": 0.15,
                    "model_side_moneyline": 150,
                    "model_side_profit": 1.5 if home_win else -1.0,
                    "vegas_side_profit": -1.0 if home_win else 100.0 / 170.0,
                    "actual_winner": "HOM" if home_win else "AWY",
                    "pred_home_margin": 4.0,
                    "market_home_margin": -2.0,
                    "spread_edge_points": 6.0,
                    "clv_detail_available": True,
                    "moneyline_positive_clv": idx % 2 == 0,
                    "moneyline_clv_prob_delta": 0.01 if idx % 2 == 0 else -0.01,
                }
            )
        rows.append(
            {
                "season": season,
                "week": 7,
                "game_id": f"{season}_agree_AWY_HOM",
                "home_team": "HOM",
                "away_team": "AWY",
                "home_win_prob": 0.62,
                "home_implied_prob_novig": 0.59,
                "home_moneyline": -140,
                "away_moneyline": 125,
                "home_margin": 10.0,
                "model_home_pick": True,
                "vegas_home_pick": True,
                "favorite_agreement": True,
                "model_correct": True,
                "vegas_correct": True,
                "model_pick_prob": 0.62,
                "vegas_model_side_prob": 0.59,
                "prob_edge_model_side": 0.03,
                "model_side_moneyline": -140,
                "model_side_profit": 100.0 / 140.0,
                "vegas_side_profit": 100.0 / 140.0,
            }
        )
    return pd.DataFrame(rows)


def _blend_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for season in [2020, 2021, 2022]:
        for idx in range(6):
            blend_home_pick = idx < 5
            home_win = idx < 4
            blend_win = blend_home_pick == home_win
            rows.append(
                {
                    "season": season,
                    "week": idx + 1,
                    "game_id": f"{season}_{idx}_AWY_HOM",
                    "blend_home_pick": blend_home_pick,
                    "blend_correct": blend_win,
                    "blend_side_profit": 1.5 if blend_win and blend_home_pick else (-1.0 if blend_home_pick else 100.0 / 170.0),
                    "blend_pick_prob": 0.54,
                    "prob_edge_blend_side": 0.06,
                    "blend_side_moneyline": 150 if blend_home_pick else -170,
                }
            )
    return pd.DataFrame(rows)


def test_prepare_disagreement_games_filters_agreements_and_adds_segments():
    games = diag.prepare_disagreement_games(_benchmark_rows(), _blend_rows())

    assert len(games) == 18
    assert set(games["model_side"]) == {"home"}
    assert set(games["prob_edge_bin"]) == {"15%+"}
    assert set(games["spread_edge_alignment"]) == {"aligns_model"}
    assert set(games["blend_followed"]) == {"model", "vegas"}


def test_summarize_disagreement_diagnostics_reports_roi_and_blend_metrics():
    games = diag.prepare_disagreement_games(_benchmark_rows(), _blend_rows())
    summary = diag.summarize_disagreement_diagnostics(games)

    overall = summary[
        summary["scope"].eq("overall") & summary["segment_family"].eq("all") & summary["segment"].eq("all")
    ].iloc[0]

    assert overall["n_games"] == 18
    assert round(overall["model_accuracy"], 3) == 0.667
    assert overall["model_roi"] > 0.0
    assert overall["model_roi_delta_vs_vegas"] > 0.0
    assert overall["blend_n_games"] == 18


def test_build_disagreement_recommendations_uses_season_stability():
    games = diag.prepare_disagreement_games(_benchmark_rows(), _blend_rows())
    summary = diag.summarize_disagreement_diagnostics(games)
    recommendations = diag.build_disagreement_recommendations(
        summary,
        min_games=10,
        min_season_games=5,
        min_qualified_seasons=2,
        min_positive_season_rate=0.55,
        generated_at="2026-10-11T00:00:00+00:00",
    )

    overall = recommendations[
        recommendations["segment_family"].eq("all") & recommendations["segment"].eq("all")
    ].iloc[0]

    assert overall["action"] == "candidate_model_side_segment"
    assert overall["recommendation_tier"] == "research"
    assert overall["qualified_seasons"] == 3
