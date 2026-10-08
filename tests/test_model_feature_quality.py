from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.models import train


def _flags(summary: pd.DataFrame, feature: str) -> set[str]:
    value = summary.loc[summary["feature"].eq(feature), "flags"].iloc[0]
    return set(str(value).split())


def test_report_feature_quality_separates_expected_sparse_high_nulls(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(train, "EVAL_DIR", tmp_path)
    frame = pd.DataFrame(
        {
            "inj_questionable_pct_change_diff": [np.nan, np.nan, np.nan, 0.0, 1.0],
            "weather_severe_risk_pct": [np.nan, np.nan, np.nan, 5.0, 35.0],
            "weather_precip_prob": [np.nan, np.nan, np.nan, 0.1, 0.7],
            "sched_prev_is_away_pct_change_diff": [np.nan, np.nan, np.nan, 0.0, 1.0],
            "bdl_sack_rate_allowed_prev_diff": [np.nan, np.nan, np.nan, 0.02, 0.08],
            "custom_feed_gap": [np.nan, np.nan, np.nan, 2.0, 4.0],
            "constant_feature": [1.0, 1.0, 1.0, 1.0, 1.0],
            "normal_feature": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    caplog.set_level(logging.WARNING, logger="src.models.train")
    summary = train.report_feature_quality(
        frame,
        list(frame.columns),
        context="spread",
        warn_null_threshold=0.40,
    )

    assert {"EXPECTED_SPARSE", "HIGH_NULL_EXPECTED"}.issubset(
        _flags(summary, "inj_questionable_pct_change_diff")
    )
    assert {"EXPECTED_SPARSE", "HIGH_NULL_EXPECTED"}.issubset(
        _flags(summary, "weather_severe_risk_pct")
    )
    assert {"EXPECTED_SPARSE", "HIGH_NULL_EXPECTED"}.issubset(
        _flags(summary, "weather_precip_prob")
    )
    assert {"EXPECTED_SPARSE", "HIGH_NULL_EXPECTED"}.issubset(
        _flags(summary, "sched_prev_is_away_pct_change_diff")
    )
    assert {"EXPECTED_SPARSE", "HIGH_NULL_EXPECTED"}.issubset(
        _flags(summary, "bdl_sack_rate_allowed_prev_diff")
    )
    assert "HIGH_NULL" in _flags(summary, "custom_feed_gap")
    assert "NEAR_CONSTANT" in _flags(summary, "constant_feature")

    warning_text = "\n".join(record.getMessage() for record in caplog.records)
    assert "custom_feed_gap" in warning_text
    assert "constant_feature" in warning_text
    assert "inj_questionable_pct_change_diff" not in warning_text
    assert "weather_severe_risk_pct" not in warning_text
    assert "weather_precip_prob" not in warning_text
    assert "sched_prev_is_away_pct_change_diff" not in warning_text
    assert "bdl_sack_rate_allowed_prev_diff" not in warning_text
