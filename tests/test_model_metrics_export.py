from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis.model_metrics_export import build_model_metrics_rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_build_model_metrics_rows_exports_about_page_metrics(tmp_path):
    overall_path = tmp_path / "predictions" / "evaluation" / "overall_metrics.csv"
    benchmark_path = tmp_path / "analysis" / "market_benchmark_summary.csv"
    disagreement_path = tmp_path / "analysis" / "market_disagreement_roi.csv"
    spread_path = tmp_path / "analysis" / "market_roi_spread.csv"
    edge_bins_path = tmp_path / "analysis" / "market_roi_spread_edge_bins.csv"

    _write_csv(
        overall_path,
        [
            {
                "n_games": 2250,
                "accuracy": 0.6538,
                "f1": 0.6958,
                "auc": 0.7071,
                "brier": 0.2181,
                "log_loss": 0.6266,
                "mae_margin": 10.235,
                "rmse_margin": 13.274,
            }
        ],
    )
    _write_csv(
        benchmark_path,
        [
            {
                "scope": "overall",
                "season": "all",
                "segment": "all",
                "n_games": 2250,
                "model_accuracy": 0.6538,
                "vegas_accuracy": 0.6622,
                "favorite_agreement_rate": 0.8751,
                "model_side_roi": -0.0262,
            },
            {
                "scope": "overall",
                "season": "all",
                "segment": "disagree",
                "n_games": 281,
                "model_accuracy": 0.4662,
                "vegas_accuracy": 0.5338,
                "favorite_agreement_rate": 0.0,
                "model_side_roi": 0.0116,
            },
        ],
    )
    _write_csv(
        disagreement_path,
        [{"threshold": 0.0, "n_bets": 281, "roi": 0.0116, "hit_rate": 0.4662}],
    )
    _write_csv(
        spread_path,
        [
            {
                "bet_side": "home",
                "threshold": 3.0,
                "n_bets": 246,
                "roi": 0.0476,
                "hit_rate": 0.5466,
            }
        ],
    )
    _write_csv(
        edge_bins_path,
        [
            {
                "scope": "overall",
                "edge_bin": "3-5",
                "n_bets": 477,
                "roi": 0.059,
                "hit_rate": 0.5522,
            }
        ],
    )

    metrics = build_model_metrics_rows(
        season=2026,
        week=3,
        generated_at="2026-09-21T20:00:00+00:00",
        overall_metrics_path=overall_path,
        market_benchmark_summary_path=benchmark_path,
        market_disagreement_roi_path=disagreement_path,
        market_spread_roi_path=spread_path,
        market_spread_edge_bins_path=edge_bins_path,
        start_season=2017,
        exclude_from_season=2026,
        exclude_from_week=3,
    )

    by_key = metrics.set_index("metric_key")
    assert by_key.loc["accuracy", "display_value"] == "65.4%"
    assert by_key.loc["vegas_favorite_accuracy", "display_value"] == "66.2%"
    assert by_key.loc["model_vegas_disagreement_games", "display_value"] == "281"
    assert by_key.loc["best_spread_roi", "display_value"] == "4.8%"
    assert by_key.loc["best_spread_edge_bin", "display_value"] == "3-5 pts"
    assert by_key.loc["evaluation_start_season", "value"] == 2017
