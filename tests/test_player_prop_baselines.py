from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.player_props.evaluate_baselines import (
    collect_baseline_projections,
    discover_projection_files,
    evaluate_baselines,
    main,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _labels() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 1,
                "game_id": "2026_01_ATL_CAR",
                "team": "ATL",
                "player_id": "qb-1",
                "market": "qb_passing_yards",
                "actual_value": 275.0,
                "actual_over_zero": True,
            },
            {
                "season": 2026,
                "week": 1,
                "game_id": "2026_01_ATL_CAR",
                "team": "ATL",
                "player_id": "rb-1",
                "market": "rb_rushing_yards",
                "actual_value": 88.0,
                "actual_over_zero": True,
            },
            {
                "season": 2026,
                "week": 1,
                "game_id": "2026_01_ATL_CAR",
                "team": "CAR",
                "player_id": "wr-1",
                "market": "wrte_receiving_yards",
                "actual_value": 102.0,
                "actual_over_zero": True,
            },
            {
                "season": 2026,
                "week": 1,
                "game_id": "2026_01_ATL_CAR",
                "team": "CAR",
                "player_id": "dl-1",
                "market": "def_sacks",
                "actual_value": 1.0,
                "actual_over_zero": True,
            },
        ]
    )


def _write_projection_files(prediction_dir: Path) -> None:
    _write_csv(
        prediction_dir / "w1_predictions_players_qb.csv",
        [
            {
                "season": 2026,
                "week": 1,
                "game_id": "2026_01_ATL_CAR",
                "team": "ATL",
                "player_id": "qb-1",
                "player_name": "QB One",
                "games_sampled": 10,
                "player_rank": 1,
                "projected_passing_yards": 260.0,
            }
        ],
    )
    _write_csv(
        prediction_dir / "predictions_players_qb.csv",
        [
            {
                "season": 2026,
                "week": 1,
                "game_id": "2026_01_ATL_CAR",
                "team": "ATL",
                "player_id": "qb-1",
                "player_name": "QB One",
                "games_sampled": 11,
                "player_rank": 1,
                "projected_passing_yards": 999.0,
            }
        ],
    )
    _write_csv(
        prediction_dir / "w1_predictions_players_offense.csv",
        [
            {
                "season": 2026,
                "week": 1,
                "game_id": "2026_01_ATL_CAR",
                "team": "ATL",
                "player_id": "rb-1",
                "player_name": "RB One",
                "games_sampled": 9,
                "player_rank": 1,
                "projected_rushing_yards": 80.0,
                "projected_receiving_yards": 5.0,
            },
            {
                "season": 2026,
                "week": 1,
                "game_id": "2026_01_ATL_CAR",
                "team": "CAR",
                "player_id": "wr-1",
                "player_name": "WR One",
                "games_sampled": 9,
                "player_rank": 1,
                "projected_rushing_yards": 2.0,
                "projected_receiving_yards": 90.0,
            },
        ],
    )
    _write_csv(
        prediction_dir / "w1_predictions_players_defense.csv",
        [
            {
                "season": 2026,
                "week": 1,
                "game_id": "2026_01_ATL_CAR",
                "team": "CAR",
                "player_id": "dl-1",
                "player_name": "DL One",
                "games_sampled": 8,
                "player_rank": 1,
                "projected_sacks": 0.7,
            }
        ],
    )
    _write_csv(
        prediction_dir / "w1_week1_experiment_predictions_players_qb.csv",
        [
            {
                "season": 2026,
                "week": 1,
                "game_id": "2026_01_ATL_CAR",
                "team": "ATL",
                "player_id": "qb-1",
                "projected_passing_yards": 123.0,
            }
        ],
    )


def test_discover_projection_files_keeps_only_canonical_archives(tmp_path):
    prediction_dir = tmp_path / "predictions"
    _write_projection_files(prediction_dir)

    files = discover_projection_files(prediction_dir)

    names = {Path(path).name for path in files["path"]}
    assert "w1_predictions_players_qb.csv" in names
    assert "predictions_players_qb.csv" in names
    assert "w1_week1_experiment_predictions_players_qb.csv" not in names


def test_evaluate_baselines_scores_first_markets_and_prefers_archives(tmp_path):
    prediction_dir = tmp_path / "predictions"
    _write_projection_files(prediction_dir)

    projections = collect_baseline_projections(prediction_dir)
    joined, metrics = evaluate_baselines(
        labels=_labels(),
        projections=projections,
        generated_at="2026-09-21T00:00:00+00:00",
        exclude_from_season=2026,
        exclude_from_week=2,
    )

    qb = joined[(joined["market"] == "qb_passing_yards") & (joined["player_id"] == "qb-1")].iloc[0]
    assert qb["baseline_projection"] == 260.0
    assert qb["source_type"] == "archive"
    assert qb["abs_error"] == 15.0

    overall = metrics[metrics["scope"].eq("overall")]
    qb_metrics = overall[overall["market"].eq("qb_passing_yards")].iloc[0]
    assert qb_metrics["n_scored"] == 1
    assert qb_metrics["mae"] == 15.0

    rb_metrics = overall[overall["market"].eq("rb_rushing_yards")].iloc[0]
    assert rb_metrics["n_candidate_predictions"] == 2
    assert rb_metrics["n_scored"] == 1
    assert rb_metrics["label_coverage"] == 1.0
    assert rb_metrics["prediction_scored_rate"] == 0.5

    sack_metrics = overall[overall["market"].eq("def_sacks")].iloc[0]
    expected_prob = 1.0 - np.exp(-0.7)
    assert sack_metrics["n_scored"] == 1
    assert np.isclose(sack_metrics["prob_over_zero_mean"], expected_prob)
    assert np.isfinite(sack_metrics["brier_over_zero"])


def test_evaluate_baselines_cli_writes_outputs(tmp_path):
    prediction_dir = tmp_path / "predictions"
    labels_path = tmp_path / "labels.parquet"
    output_dir = tmp_path / "evaluation"
    _write_projection_files(prediction_dir)
    _labels().to_parquet(labels_path, index=False)

    rc = main(
        [
            "--labels",
            str(labels_path),
            "--prediction-dir",
            str(prediction_dir),
            "--output-dir",
            str(output_dir),
            "--exclude-from-season",
            "2026",
            "--exclude-from-week",
            "2",
        ]
    )

    assert rc == 0
    predictions = pd.read_csv(output_dir / "baseline_predictions.csv")
    metrics = pd.read_csv(output_dir / "baseline_metrics.csv")
    assert not predictions.empty
    assert set(metrics["market"]) >= {
        "qb_passing_yards",
        "rb_rushing_yards",
        "wrte_receiving_yards",
        "def_sacks",
    }
