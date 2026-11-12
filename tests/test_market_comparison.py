from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation import market_comparison


def test_no_vig_probability_removes_overround():
    features = pd.DataFrame({"sched_implied_prob_home": [0.55, 0.30], "sched_implied_prob_away": [0.50, 0.75]})

    prob = market_comparison.no_vig_home_probability(features)

    assert np.allclose(prob.to_numpy(), [0.55 / 1.05, 0.30 / 1.05])


def test_compare_to_market_scores_identical_rows_only():
    history = pd.DataFrame(
        {
            "season": [2024] * 6,
            "week": [1, 2, 3, 9, 10, 20],
            "game_id": list("abcdef"),
            "home_margin": [3.0, -3.0, 7.0, -7.0, 0.0, 10.0],
            "home_win_prob_raw": [0.7, 0.4, 0.6, 0.3, 0.5, 0.8],
            "home_win_prob": [0.65, 0.45, 0.6, 0.35, 0.5, 0.75],
        }
    )
    features = pd.DataFrame(
        {
            "game_id": list("abcdef"),
            "sched_implied_prob_home": [0.6, 0.45, np.nan, 0.4, 0.5, 0.7],
            "sched_implied_prob_away": [0.45, 0.6, np.nan, 0.65, 0.55, 0.35],
        }
    )

    report = market_comparison.compare_to_market(market_comparison.attach_market_probability(history, features))

    overall = report[report["scope"].eq("overall")].set_index("source")
    # Game c has no market line and game e is a tie, so both are excluded for every source.
    assert set(overall["n_games"]) == {4}
    assert {"model_raw", "model_final", "market_novig"} == set(overall.index)
    assert np.isnan(overall.loc["market_novig"].get("brier_minus_market", np.nan))
    assert np.isclose(
        overall.loc["model_raw", "brier_minus_market"],
        overall.loc["model_raw", "brier"] - overall.loc["market_novig", "brier"],
    )
    assert "postseason" in set(report.loc[report["scope"].eq("week_bucket"), "segment"])
