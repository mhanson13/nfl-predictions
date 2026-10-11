from __future__ import annotations

import pandas as pd
import pytest

from analysis import market_confidence_recalibration as recal


def _row(
    *,
    season: int,
    week: int,
    game_id: str,
    home_prob: float,
    vegas_home_prob: float,
    home_win: bool,
    home_moneyline: int = 120,
    away_moneyline: int = -140,
) -> dict[str, object]:
    home_pick = home_prob >= 0.5
    vegas_home_pick = vegas_home_prob >= 0.5
    model_side_win = home_win if home_pick else not home_win
    vegas_side_win = home_win if vegas_home_pick else not home_win
    model_ml = home_moneyline if home_pick else away_moneyline
    vegas_ml = home_moneyline if vegas_home_pick else away_moneyline

    def profit(win: bool, odds: int) -> float:
        if not win:
            return -1.0
        return odds / 100.0 if odds > 0 else 100.0 / abs(odds)

    return {
        "season": season,
        "week": week,
        "game_id": game_id,
        "home_team": "HOME",
        "away_team": "AWAY",
        "home_win_prob": home_prob,
        "away_win_prob": 1.0 - home_prob,
        "home_implied_prob_novig": vegas_home_prob,
        "away_implied_prob_novig": 1.0 - vegas_home_prob,
        "home_moneyline": home_moneyline,
        "away_moneyline": away_moneyline,
        "home_margin": 3.0 if home_win else -3.0,
        "model_home_pick": home_pick,
        "vegas_home_pick": vegas_home_pick,
        "favorite_agreement": home_pick == vegas_home_pick,
        "model_side_profit": profit(model_side_win, model_ml),
        "vegas_side_profit": profit(vegas_side_win, vegas_ml),
        "prob_edge_model_side": (home_prob if home_pick else 1.0 - home_prob)
        - (vegas_home_prob if home_pick else 1.0 - vegas_home_prob),
        "pred_home_margin": 1.0,
        "market_home_margin": -1.0,
        "spread_edge_points": 2.0,
    }


def _benchmark_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for season in [2020, 2021]:
        rows.extend(
            [
                _row(season=season, week=1, game_id=f"{season}_01_A", home_prob=0.58, vegas_home_prob=0.45, home_win=False),
                _row(season=season, week=2, game_id=f"{season}_02_A", home_prob=0.57, vegas_home_prob=0.44, home_win=False),
                _row(season=season, week=3, game_id=f"{season}_03_A", home_prob=0.56, vegas_home_prob=0.43, home_win=True),
                _row(season=season, week=4, game_id=f"{season}_04_A", home_prob=0.62, vegas_home_prob=0.60, home_win=True),
            ]
        )
    rows.extend(
        [
            _row(season=2022, week=1, game_id="2022_01_A", home_prob=0.58, vegas_home_prob=0.45, home_win=False),
            _row(season=2022, week=2, game_id="2022_02_A", home_prob=0.57, vegas_home_prob=0.44, home_win=True),
            _row(season=2022, week=3, game_id="2022_03_A", home_prob=0.62, vegas_home_prob=0.60, home_win=True),
        ]
    )
    return pd.DataFrame(rows)


def test_walk_forward_confidence_recalibration_uses_prior_seasons_only():
    dataset = recal.prepare_dataset(_benchmark_rows())
    detail = recal.walk_forward_recalibration_backtest(dataset, min_train_seasons=2, min_bin_games=2)

    assert set(detail["season"]) == {2022}
    assert (detail["fold_train_end_season"] < detail["season"]).all()
    assert set(detail["strategy"]) == {
        "current_model",
        "vegas_novig",
        "vegas_blend_50_50",
        "confidence_shrink",
        "disagreement_shrink",
    }

    shrunk = detail[
        detail["strategy"].eq("confidence_shrink")
        & detail["game_id"].eq("2022_01_A")
    ].iloc[0]

    assert shrunk["calibration_source"] == "segment_confidence_bin"
    assert shrunk["calibration_sample_size"] == 6
    assert shrunk["calibration_actual_rate"] == pytest.approx(1 / 3)
    assert shrunk["strategy_home_win_prob"] == pytest.approx(1 / 3)


def test_recalibration_summary_and_decision_compare_against_current_model():
    dataset = recal.prepare_dataset(_benchmark_rows())
    detail = recal.walk_forward_recalibration_backtest(dataset, min_train_seasons=2, min_bin_games=2)
    summary = recal.summarize_recalibration_backtest(detail)
    decision = recal.build_recalibration_decision(
        summary,
        min_games=1,
        min_disagreement_games=1,
        generated_at="2026-10-11T00:00:00+00:00",
    )

    overall = summary[
        summary["scope"].eq("overall")
        & summary["segment"].eq("all")
        & summary["strategy"].eq("confidence_shrink")
    ].iloc[0]
    candidate = decision[decision["candidate_strategy"].eq("confidence_shrink")].iloc[0]

    assert overall["n_games"] == 3
    assert overall["brier"] >= 0.0
    assert candidate["overall_brier_delta_vs_current"] < 0.0
    assert bool(candidate["passes_overall_probability_gate"]) is False
    assert candidate["decision"] == "base_probability_research_only"
