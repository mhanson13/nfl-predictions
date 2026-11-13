import numpy as np
import pandas as pd

from src.evaluation.market_bias_monitor import build_monitor


def _features(ids, market):
    # implied probabilities with vig; the monitor removes it
    return pd.DataFrame(
        {"game_id": ids, "sched_implied_prob_home": np.array(market) * 1.04, "sched_implied_prob_away": (1 - np.array(market)) * 1.04}
    )


def test_current_slate_systematic_home_markdown_is_watched():
    ids = [f"g{i}" for i in range(10)]
    market = [0.6] * 10
    current = pd.DataFrame({"game_id": ids, "season": 2026, "week": 10, "home_win_prob": [0.57] * 10})
    history = pd.DataFrame(columns=["game_id", "season", "week", "home_win_prob", "home_score", "away_score"])
    report = build_monitor(history, _features(ids, market), current).set_index("scope")
    row = report.loc["current_slate"]
    assert row["status"] == "watch"
    assert np.isclose(row["mean_gap"], -0.03) and row["share_home_below_market"] == 1.0


def test_history_shift_that_costs_brier_is_alert_and_helpful_shift_is_not():
    rng = np.random.default_rng(0)
    n = 400
    ids = [f"h{i}" for i in range(n)]
    market = np.full(n, 0.6)
    home_win = rng.permutation(np.arange(n) < int(0.6 * n))  # market is exactly right
    history = pd.DataFrame(
        {
            "game_id": ids,
            "season": 2026,
            "week": np.repeat(np.arange(1, 11), n // 10),
            "home_win_prob": 0.6 - 0.08 + rng.normal(0, 0.01, n),  # model marks home down 8 pts
            "home_score": np.where(home_win, 24, 17),
            "away_score": np.where(home_win, 17, 24),
        }
    )
    report = build_monitor(history, _features(ids, market), None).set_index("scope")
    assert report.loc["season", "status"] == "alert"
    assert report.loc["season", "brier_cost_of_shift"] > 0
    assert report.loc["all_history", "status"] == "info"

    # when home teams really do win less, the same shift helps and is only watched
    home_win = rng.permutation(np.arange(n) < int(0.5 * n))
    history["home_score"] = np.where(home_win, 24, 17)
    history["away_score"] = np.where(home_win, 17, 24)
    report = build_monitor(history, _features(ids, market), None).set_index("scope")
    assert report.loc["season", "status"] == "watch"
    assert report.loc["season", "brier_cost_of_shift"] < 0
