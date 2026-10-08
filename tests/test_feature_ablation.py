from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.feature_ablation import build_ablation_report, build_feature_sets


def test_build_feature_sets_separates_bdl_family_and_sparse_sack_rates():
    cols = [
        "sched_spread_close_diff",
        "bdl_total_yards_rolling3_diff",
        "bdl_sack_rate_allowed_prev_diff",
    ]

    feature_sets = build_feature_sets(cols)

    assert feature_sets["baseline_no_bdl"] == ["sched_spread_close_diff"]
    assert feature_sets["baseline_plus_bdl"] == cols
    assert feature_sets["baseline_plus_bdl_no_sparse_sack_rate"] == [
        "sched_spread_close_diff",
        "bdl_total_yards_rolling3_diff",
    ]


def test_build_ablation_report_writes_expected_rows():
    rng = np.random.default_rng(7)
    n = 120
    baseline_home = rng.normal(0, 1, n)
    baseline_away = rng.normal(0, 1, n)
    bdl_home = rng.normal(0, 1, n)
    bdl_away = rng.normal(0, 1, n)
    signal = (baseline_home - baseline_away) + 0.8 * (bdl_home - bdl_away)

    frame = pd.DataFrame(
        {
            "season": [2024] * n,
            "week": np.tile(np.arange(1, 13), 10),
            "sched_spread_close_home": baseline_home,
            "sched_spread_close_away": baseline_away,
            "bdl_total_yards_rolling3_home": bdl_home,
            "bdl_total_yards_rolling3_away": bdl_away,
            "bdl_sack_rate_allowed_prev_home": rng.normal(0.06, 0.02, n),
            "bdl_sack_rate_allowed_prev_away": rng.normal(0.06, 0.02, n),
            "home_win": (signal > np.median(signal)).astype(int),
            "home_margin": signal * 7.0 + rng.normal(0, 2, n),
        }
    )

    report = build_ablation_report(
        frame,
        cv_folds=2,
        use_gpu=False,
        winprob_estimators=10,
        spread_estimators=10,
    )

    assert set(report["target"]) == {"win_prob", "spread"}
    assert set(report["feature_set"]) == {
        "baseline_no_bdl",
        "baseline_plus_bdl",
        "baseline_plus_bdl_no_sparse_sack_rate",
    }
    assert len(report) == 6
    assert report["primary_delta_vs_baseline"].notna().all()
    assert (
        report.loc[report["feature_set"].eq("baseline_plus_bdl"), "bdl_feature_count"] > 0
    ).all()
    assert (
        report.loc[report["feature_set"].eq("baseline_no_bdl"), "primary_delta_vs_baseline"]
        == 0.0
    ).all()
