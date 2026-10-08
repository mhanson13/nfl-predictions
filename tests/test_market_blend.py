from __future__ import annotations

import pandas as pd

from analysis import market_blend


def _benchmark_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for season in [2020, 2021, 2022, 2023]:
        for idx in range(6):
            home_win = idx % 2 == 0
            home_prob = 0.62 if home_win else 0.38
            vegas_prob = 0.56 if home_win else 0.44
            rows.append(
                {
                    "season": season,
                    "week": idx + 1,
                    "game_id": f"{season}_{idx}_AWY_HOM",
                    "home_team": "HOM",
                    "away_team": "AWY",
                    "home_win_prob": home_prob,
                    "away_win_prob": 1.0 - home_prob,
                    "home_implied_prob_novig": vegas_prob,
                    "away_implied_prob_novig": 1.0 - vegas_prob,
                    "home_moneyline": -120 if home_win else 125,
                    "away_moneyline": 110 if home_win else -115,
                    "home_margin": 7.0 if home_win else -4.0,
                    "pred_home_margin": 3.0 if home_win else -2.5,
                    "market_home_margin": 1.5 if home_win else -1.0,
                    "spread_edge_points": 1.5 if home_win else -1.5,
                    "model_side_profit": (100.0 / 120.0) if home_win else (100.0 / 115.0),
                    "vegas_side_profit": (100.0 / 120.0) if home_win else (100.0 / 115.0),
                    "model_side_moneyline": -120 if home_win else -115,
                    "vegas_side_moneyline": -120 if home_win else -115,
                    "model_pick_prob": home_prob if home_win else 1.0 - home_prob,
                    "vegas_model_side_prob": vegas_prob if home_win else 1.0 - vegas_prob,
                    "prob_edge_model_side": abs(home_prob - vegas_prob),
                    "favorite_agreement": True,
                }
            )
    return pd.DataFrame(rows)


def test_walk_forward_blend_uses_prior_seasons_only():
    dataset = market_blend.prepare_blend_dataset(_benchmark_rows())
    scored = market_blend.walk_forward_blend(dataset, min_train_seasons=2)

    assert set(scored["season"]) == {2022, 2023}
    assert (scored["fold_train_end_season"] < scored["season"]).all()
    assert scored["blend_home_win_prob"].between(0.0, 1.0).all()


def test_summarize_blend_backtest_reports_model_vegas_and_blend_metrics():
    dataset = market_blend.prepare_blend_dataset(_benchmark_rows())
    scored = market_blend.walk_forward_blend(dataset, min_train_seasons=2)
    summary = market_blend.summarize_blend_backtest(scored)

    overall = summary[(summary["scope"] == "overall") & (summary["segment"] == "all")].iloc[0]

    assert overall["n_games"] == 12
    assert overall["blend_brier"] >= 0.0
    assert "blend_brier_delta_vs_vegas" in overall
    assert "blend_roi_delta_vs_model" in overall


def test_build_blend_policy_requires_blend_to_beat_model_and_vegas():
    summary = pd.DataFrame(
        [
            {
                "scope": "overall",
                "season": "all",
                "segment": "all",
                "n_games": 300,
                "model_brier": 0.21,
                "vegas_brier": 0.20,
                "blend_brier": 0.22,
                "blend_brier_delta_vs_model": 0.01,
                "blend_brier_delta_vs_vegas": 0.02,
                "model_side_roi": 0.01,
                "vegas_side_roi": 0.00,
                "blend_side_roi": -0.01,
                "blend_roi_delta_vs_model": -0.02,
                "blend_roi_delta_vs_vegas": -0.01,
            }
        ]
    )

    policy = market_blend.build_blend_policy(summary, min_games=250, generated_at="2026-10-07T00:00:00+00:00")

    assert not bool(policy[policy["signal"].eq("market_blend_probability")].iloc[0]["recommended"])
    assert not bool(policy[policy["signal"].eq("market_blend_moneyline")].iloc[0]["recommended"])
