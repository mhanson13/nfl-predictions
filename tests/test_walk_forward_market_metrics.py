from __future__ import annotations

import pandas as pd

from analysis.model_metrics_export import build_model_metrics_rows, build_walk_forward_market_rows


def _report(path):
    rows = [
        ("overall", "all", "model_raw", 2380, 0.2106, 0.7240, -0.0012),
        ("overall", "all", "model_final", 2380, 0.2104, 0.72435, -0.00088),
        ("overall", "all", "market_novig", 2380, 0.2106, 0.72523, None),
        ("season", "2026", "model_final", 161, 0.2088, 0.73835, -0.00417),
        ("season", "2026", "market_novig", 161, 0.2080, 0.74251, None),
        ("week_bucket", "wk01-04", "model_final", 567, 0.2204, 0.6935, -0.0017),
    ]
    pd.DataFrame(rows, columns=["scope", "segment", "source", "n_games", "brier", "auc", "auc_minus_market"]).to_csv(
        path, index=False
    )
    return path


def test_rows_compare_model_final_with_the_no_vig_market(tmp_path):
    rows = build_walk_forward_market_rows(season=2026, week=12, generated_at="t",
                                          report_path=_report(tmp_path / "wf.csv"))
    by_key = {r.metric_key: r for r in rows}

    assert by_key["wf_vs_market_games"].display_value == "2,380"
    assert by_key["wf_model_auc"].display_value == "0.7244"  # model_final, not model_raw
    assert by_key["wf_market_auc"].display_value == "0.7252"
    assert by_key["wf_auc_minus_market"].display_value == "-0.0009"
    assert by_key["wf_model_brier"].display_value == "0.2104"
    assert by_key["wf_season_model_auc"].sample_size == 161
    assert by_key["wf_season_auc_minus_market"].display_value == "-0.0042"
    assert by_key["wf_season_market_auc"].label == "2026 Vegas AUC (no-vig close)"
    assert {r.section for r in rows} == {"walk_forward_vs_market"}
    assert sorted(r.sort_order for r in rows) == list(range(161, 170))


def test_season_rows_skipped_when_the_export_season_is_missing(tmp_path):
    rows = build_walk_forward_market_rows(season=2027, week=1, generated_at="t",
                                          report_path=_report(tmp_path / "wf.csv"))
    keys = {r.metric_key for r in rows}
    assert "wf_model_auc" in keys and not any(k.startswith("wf_season_") for k in keys)


def test_missing_or_malformed_report_gives_no_rows(tmp_path):
    assert build_walk_forward_market_rows(season=2026, week=1, generated_at="t",
                                          report_path=tmp_path / "missing.csv") == []
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"scope": ["overall"]}).to_csv(bad, index=False)
    assert build_walk_forward_market_rows(season=2026, week=1, generated_at="t", report_path=bad) == []


def test_build_model_metrics_rows_includes_the_section(tmp_path):
    missing = tmp_path / "missing.csv"
    metrics = build_model_metrics_rows(
        season=2026, week=12, generated_at="t",
        overall_metrics_path=missing, market_benchmark_summary_path=missing, market_clv_summary_path=missing,
        market_disagreement_roi_path=missing, market_probability_edge_calibration_path=missing,
        market_blend_summary_path=missing, market_blend_policy_path=missing, market_spread_roi_path=missing,
        market_spread_edge_bins_path=missing, market_spread_key_numbers_path=missing,
        market_edge_policy_path=missing, live_results_summary_dir=tmp_path, live_results_detail_path=missing,
        walk_forward_market_report_path=_report(tmp_path / "wf.csv"),
    )
    section = metrics[metrics["section"] == "walk_forward_vs_market"]
    assert len(section) == 9
    assert not metrics["sort_order"].duplicated().any()
