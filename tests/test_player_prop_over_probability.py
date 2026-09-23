from __future__ import annotations

import pandas as pd

from src.player_props.over_probability import (
    add_walk_forward_over_probabilities,
    attach_current_over_probabilities,
    build_calibration_bins,
    summarize_over_probability,
)


def _historical_lines() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": 2025,
                "week": 1,
                "game_id": "2025_01_CAR_ATL",
                "market": "qb_passing_yards",
                "sportsbook": "draftkings",
                "player_id": "qb-1",
                "model_player_name": "QB One",
                "line": 245.5,
                "model_projection": 250.0,
                "actual_value": 270.0,
                "implied_probability": 0.5,
            },
            {
                "season": 2025,
                "week": 1,
                "game_id": "2025_01_DEN_LV",
                "market": "qb_passing_yards",
                "sportsbook": "draftkings",
                "player_id": "qb-2",
                "model_player_name": "QB Two",
                "line": 235.5,
                "model_projection": 250.0,
                "actual_value": 230.0,
                "implied_probability": 0.5,
            },
            {
                "season": 2025,
                "week": 2,
                "game_id": "2025_02_CAR_ATL",
                "market": "qb_passing_yards",
                "sportsbook": "fanduel",
                "player_id": "qb-3",
                "model_player_name": "QB Three",
                "line": 235.5,
                "model_projection": 260.0,
                "actual_value": 280.0,
                "implied_probability": 0.52,
            },
            {
                "season": 2025,
                "week": 1,
                "game_id": "2025_01_CAR_ATL",
                "market": "rb_rushing_yards",
                "sportsbook": "draftkings",
                "player_id": "rb-1",
                "model_player_name": "RB One",
                "line": 62.5,
                "model_projection": 60.0,
                "actual_value": 58.0,
                "implied_probability": 0.44,
            },
        ]
    )


def test_add_walk_forward_over_probabilities_uses_prior_market_residuals():
    scored = add_walk_forward_over_probabilities(
        _historical_lines(),
        min_samples=2,
        shrinkage=0,
        probability_cap=1.0,
    )

    week1_qb = scored[(scored["market"].eq("qb_passing_yards")) & scored["week"].eq(1)]
    week2_qb = scored[(scored["market"].eq("qb_passing_yards")) & scored["week"].eq(2)].iloc[0]
    rb = scored[scored["market"].eq("rb_rushing_yards")].iloc[0]

    assert set(week1_qb["calibration_method"]) == {"market_implied_fallback"}
    assert week2_qb["calibration_method"] == "residual_cdf"
    assert week2_qb["calibration_sample_size"] == 2
    assert week2_qb["prob_over_model"] == 1.0
    assert round(week2_qb["prob_edge_vs_market"], 2) == 0.48
    assert rb["calibration_method"] == "market_implied_fallback"
    assert rb["prob_over_model"] == 0.44


def test_over_probability_metrics_and_bins_are_written_shape():
    scored = add_walk_forward_over_probabilities(
        _historical_lines(),
        min_samples=2,
        shrinkage=0,
        probability_cap=1.0,
    )

    metrics = summarize_over_probability(scored, generated_at="2026-09-22T00:00:00+00:00")
    bins = build_calibration_bins(scored, generated_at="2026-09-22T00:00:00+00:00")

    assert set(metrics["scope"]) == {"all", "market"}
    assert "model_brier" in metrics.columns
    assert not bins.empty
    assert bins["n"].sum() >= len(scored)


def test_attach_current_over_probabilities_updates_lined_rows_only():
    current = pd.DataFrame(
        [
            {
                "season": 2025,
                "week": 3,
                "game_id": "2025_03_CAR_ATL",
                "market": "qb_passing_yards",
                "player_name": "QB Four",
                "projection": 260.0,
                "model_projection": 260.0,
                "line": 250.5,
                "implied_probability": 0.51,
                "prob_over": float("nan"),
            },
            {
                "season": 2025,
                "week": 3,
                "game_id": "2025_03_DEN_LV",
                "market": "wrte_receiving_yards",
                "player_name": "WR One",
                "projection": 55.0,
                "model_projection": 55.0,
                "line": float("nan"),
                "implied_probability": float("nan"),
                "prob_over": float("nan"),
            },
        ]
    )

    attached = attach_current_over_probabilities(
        current,
        _historical_lines(),
        season=2025,
        week=3,
        min_samples=2,
        shrinkage=0,
    )

    lined = attached.iloc[0]
    unlined = attached.iloc[1]
    assert lined["prob_over_method"] == "residual_cdf"
    assert lined["prob_over_sample_size"] == 3
    assert lined["prob_over"] > 0.5
    assert lined["prob_edge"] == lined["prob_over"] - 0.51
    assert pd.isna(unlined["prob_over"])
