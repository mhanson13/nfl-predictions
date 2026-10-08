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
    clv_path = tmp_path / "analysis" / "market_clv_summary.csv"
    disagreement_path = tmp_path / "analysis" / "market_disagreement_roi.csv"
    probability_path = tmp_path / "analysis" / "market_probability_edge_calibration.csv"
    blend_summary_path = tmp_path / "analysis" / "market_blend_summary.csv"
    blend_policy_path = tmp_path / "analysis" / "market_blend_policy.csv"
    spread_path = tmp_path / "analysis" / "market_roi_spread.csv"
    edge_bins_path = tmp_path / "analysis" / "market_roi_spread_edge_bins.csv"
    key_numbers_path = tmp_path / "analysis" / "market_spread_key_numbers.csv"
    policy_path = tmp_path / "analysis" / "market_edge_policy.csv"

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
        clv_path,
        [
            {
                "scope": "overall",
                "segment": "moneyline_disagreement",
                "n_games": 10,
                "moneyline_clv_available": 10,
                "avg_moneyline_clv_prob_delta": 0.012,
                "positive_moneyline_clv_rate": 0.6,
            },
            {
                "scope": "overall",
                "segment": "moneyline_policy_recommended",
                "n_games": 4,
                "moneyline_clv_available": 4,
                "avg_moneyline_clv_prob_delta": 0.021,
                "positive_moneyline_clv_rate": 0.75,
            },
        ],
    )
    _write_csv(
        probability_path,
        [
            {
                "scope": "overall",
                "season": "all",
                "bin_family": "all",
                "bin_label": "all",
                "n_games": 2250,
                "model_brier": 0.220,
                "vegas_brier": 0.218,
                "brier_delta_model_minus_vegas": 0.002,
                "model_calibration_error": 0.015,
                "vegas_calibration_error": 0.005,
            }
        ],
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
        blend_summary_path,
        [
            {
                "scope": "overall",
                "season": "all",
                "segment": "all",
                "n_games": 1200,
                "blend_brier_delta_vs_model": -0.003,
                "blend_brier_delta_vs_vegas": 0.004,
                "blend_side_roi": -0.012,
            }
        ],
    )
    _write_csv(
        blend_policy_path,
        [
            {
                "signal": "market_blend_probability",
                "recommended": False,
                "recommendation_tier": "no_bet",
                "selection": "blend home_win_prob",
                "n_games": 1200,
            },
            {
                "signal": "market_blend_moneyline",
                "recommended": False,
                "recommendation_tier": "no_bet",
                "selection": "blend selected side",
                "n_games": 1200,
            },
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
    _write_csv(
        key_numbers_path,
        [
            {
                "scope": "overall",
                "season": "all",
                "key_number": 3.0,
                "edge_bucket": "2-3",
                "segment": "key_window",
                "n_bets": 80,
                "roi": 0.012,
            }
        ],
    )
    _write_csv(
        policy_path,
        [
            {
                "signal": "moneyline_disagreement",
                "recommended": False,
                "recommendation_tier": "no_bet",
                "selection": "prob_edge >= 0.02",
                "n_bets": 281,
                "roi": -0.008,
                "stability_status": "stable",
                "stability_qualified_seasons": 7,
                "stability_positive_season_rate": 0.714,
                "stability_recent_positive_season_rate": 1.0,
                "stability_severe_negative_season_rate": 0.143,
                "clv_status": "clv_pass",
                "clv_tracked_games": 12,
                "clv_avg_prob_delta": 0.012,
                "clv_positive_rate": 0.583,
            },
            {
                "signal": "spread_threshold",
                "recommended": True,
                "recommendation_tier": "lean",
                "selection": "home edge > 3 pts",
                "n_bets": 246,
                "roi": 0.0476,
            },
            {
                "signal": "spread_edge_bin",
                "recommended": False,
                "recommendation_tier": "review",
                "selection": "2-3 pts",
                "n_bets": 368,
                "roi": 0.012,
                "spread_cover_calibration_pass": False,
                "spread_cover_calibration_status": "blocked_spread_cover_overconfidence",
                "spread_cover_calibration_max_abs_error": 0.184,
                "spread_cover_calibration_worst_bin": "65%-70%",
            },
        ],
    )

    metrics = build_model_metrics_rows(
        season=2026,
        week=3,
        generated_at="2026-09-21T20:00:00+00:00",
        overall_metrics_path=overall_path,
        market_benchmark_summary_path=benchmark_path,
        market_clv_summary_path=clv_path,
        market_disagreement_roi_path=disagreement_path,
        market_probability_edge_calibration_path=probability_path,
        market_blend_summary_path=blend_summary_path,
        market_blend_policy_path=blend_policy_path,
        market_spread_roi_path=spread_path,
        market_spread_edge_bins_path=edge_bins_path,
        market_spread_key_numbers_path=key_numbers_path,
        market_edge_policy_path=policy_path,
        start_season=2017,
        exclude_from_season=2026,
        exclude_from_week=3,
    )

    by_key = metrics.set_index("metric_key")
    assert by_key.loc["accuracy", "display_value"] == "65.4%"
    assert by_key.loc["vegas_favorite_accuracy", "display_value"] == "66.2%"
    assert by_key.loc["model_vegas_disagreement_games", "display_value"] == "281"
    assert by_key.loc["clv_moneyline_disagreement_avg_prob_delta", "display_value"] == "1.2%"
    assert by_key.loc["clv_moneyline_policy_positive_rate", "display_value"] == "75.0%"
    assert by_key.loc["model_minus_vegas_probability_brier_delta", "display_value"] == "0.002"
    assert by_key.loc["market_blend_brier_delta_vs_model", "display_value"] == "-0.003"
    assert by_key.loc["market_blend_brier_delta_vs_vegas", "display_value"] == "0.004"
    assert by_key.loc["market_blend_moneyline_roi", "display_value"] == "-1.2%"
    assert by_key.loc["market_blend_probability_policy", "display_value"] == "No bet"
    assert by_key.loc["moneyline_disagreement_policy", "display_value"] == "No bet"
    assert by_key.loc["moneyline_disagreement_policy_stability_status", "display_value"] == "stable"
    assert by_key.loc["moneyline_disagreement_policy_stability_qualified_seasons", "display_value"] == "7"
    assert by_key.loc["moneyline_disagreement_policy_stability_positive_season_rate", "display_value"] == "71.4%"
    assert by_key.loc["moneyline_disagreement_policy_clv_status", "display_value"] == "clv_pass"
    assert by_key.loc["moneyline_disagreement_policy_clv_tracked_games", "display_value"] == "12"
    assert by_key.loc["moneyline_disagreement_policy_clv_avg_prob_delta", "display_value"] == "1.2%"
    assert by_key.loc["moneyline_disagreement_policy_clv_positive_rate", "display_value"] == "58.3%"
    assert by_key.loc["spread_threshold_policy", "display_value"] == "home edge > 3 pts"
    assert by_key.loc["spread_threshold_policy_roi", "display_value"] == "4.8%"
    assert by_key.loc["spread_edge_bin_policy", "display_value"] == "No bet"
    assert by_key.loc["spread_edge_bin_policy_cover_calibration_status", "display_value"] == "blocked_spread_cover_overconfidence"
    assert by_key.loc["spread_edge_bin_policy_cover_calibration_max_abs_error", "display_value"] == "18.4%"
    assert by_key.loc["best_spread_roi", "display_value"] == "4.8%"
    assert by_key.loc["best_spread_edge_bin", "display_value"] == "3-5 pts"
    assert by_key.loc["spread_key_3_policy_bucket_roi", "display_value"] == "1.2%"
    assert by_key.loc["evaluation_start_season", "value"] == 2017
