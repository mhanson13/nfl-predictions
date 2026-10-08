from __future__ import annotations

import pandas as pd
import pytest

from src.player_props.confidence_tiers import build_confidence_tier_metrics, main


def _features() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 1,
                "game_id": "2026_01_CAR_ATL",
                "team": "ATL",
                "player_id": "p-low",
                "market": "anytime_td",
                "player_games_prior": 1,
            },
            {
                "season": 2026,
                "week": 1,
                "game_id": "2026_01_CAR_ATL",
                "team": "ATL",
                "player_id": "p-med",
                "market": "anytime_td",
                "player_games_prior": 8,
            },
            {
                "season": 2026,
                "week": 1,
                "game_id": "2026_01_CAR_ATL",
                "team": "ATL",
                "player_id": "p-high",
                "market": "anytime_td",
                "player_games_prior": 24,
            },
        ]
    )


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 1,
                "game_id": "2026_01_CAR_ATL",
                "team": "ATL",
                "player_id": "p-low",
                "market": "anytime_td",
                "model_projection": 0.2,
                "actual_value": 0.0,
                "actual_over_zero": False,
                "prob_over_zero": 0.25,
            },
            {
                "season": 2026,
                "week": 1,
                "game_id": "2026_01_CAR_ATL",
                "team": "ATL",
                "player_id": "p-med",
                "market": "anytime_td",
                "model_projection": 0.6,
                "actual_value": 1.0,
                "actual_over_zero": True,
                "prob_over_zero": 0.55,
            },
            {
                "season": 2026,
                "week": 1,
                "game_id": "2026_01_CAR_ATL",
                "team": "ATL",
                "player_id": "p-high",
                "market": "anytime_td",
                "model_projection": 0.8,
                "actual_value": 1.0,
                "actual_over_zero": True,
                "prob_over_zero": 0.70,
            },
        ]
    )


def test_build_confidence_tier_metrics_joins_feature_sample_sizes():
    metrics = build_confidence_tier_metrics(
        model_predictions=_predictions(),
        feature_frames=[_features()],
        generated_at="2026-10-08T00:00:00+00:00",
    )

    tiers = metrics[metrics["scope"].eq("confidence_tier")].set_index("confidence_tier")
    assert set(tiers.index) == {"low", "medium", "high"}
    assert tiers.loc["low", "n_rows"] == 1
    assert tiers.loc["medium", "avg_sample_size"] == 8
    assert tiers.loc["high", "avg_sample_size"] == 24

    market_high = metrics[
        metrics["scope"].eq("market_confidence_tier")
        & metrics["market"].eq("anytime_td")
        & metrics["confidence_tier"].eq("high")
    ].iloc[0]
    assert market_high["mae"] == pytest.approx(0.2)


def test_build_confidence_tier_metrics_includes_historical_line_scope():
    line_eval = _predictions().copy()
    line_eval["line"] = 0.5
    line_eval["model_pick_correct"] = [True, True, False]
    line_eval["abs_model_edge"] = [0.3, 0.1, 0.3]
    line_eval["abs_model_error"] = [0.2, 0.4, 0.2]
    line_eval["abs_line_error"] = [0.5, 0.5, 0.5]

    metrics = build_confidence_tier_metrics(
        model_predictions=_predictions(),
        feature_frames=[_features()],
        line_eval=line_eval,
        generated_at="2026-10-08T00:00:00+00:00",
    )

    historical = metrics[
        metrics["evaluation_scope"].eq("historical_lines")
        & metrics["scope"].eq("overall")
    ].iloc[0]
    assert historical["n_rows"] == 3
    assert historical["model_pick_accuracy"] == 2 / 3


def test_confidence_tiers_cli_writes_metrics(tmp_path):
    predictions_path = tmp_path / "model_predictions.csv"
    features_path = tmp_path / "features.parquet"
    line_eval_path = tmp_path / "line_eval.csv"
    output_path = tmp_path / "confidence.csv"
    _predictions().to_csv(predictions_path, index=False)
    _features().to_parquet(features_path, index=False)
    pd.DataFrame().to_csv(line_eval_path, index=False)

    rc = main(
        [
            "--model-predictions",
            str(predictions_path),
            "--offense-features",
            str(features_path),
            "--defense-features",
            str(tmp_path / "missing_defense.parquet"),
            "--novelty-features",
            str(tmp_path / "missing_novelty.parquet"),
            "--line-eval",
            str(line_eval_path),
            "--output",
            str(output_path),
        ]
    )

    assert rc == 0
    written = pd.read_csv(output_path)
    assert "confidence_tier" in written.columns
    assert "high" in set(written["confidence_tier"])
