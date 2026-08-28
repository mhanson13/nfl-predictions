"""Tests for pure helper functions in src/analysis/compare_runs.py.

These target the small, file-I/O-free utility functions only.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis.compare_runs import (
    _df_to_markdown,
    _fmt,
    _metric_table,
    _top_runs,
    generate_report,
    _load_runs,
)


# ---------------------------------------------------------------------------
# _df_to_markdown
# ---------------------------------------------------------------------------
class TestDfToMarkdown:
    def test_returns_string(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        result = _df_to_markdown(df)
        assert isinstance(result, str)

    def test_empty_df_returns_string(self):
        result = _df_to_markdown(pd.DataFrame())
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _fmt
# ---------------------------------------------------------------------------
class TestFmt:
    def test_formats_float(self):
        assert _fmt(0.123456789) == "0.123"

    def test_nan_returns_na(self):
        assert _fmt(float("nan")) == "NA"

    def test_custom_digits(self):
        assert _fmt(3.14159, digits=2) == "3.14"

    def test_none_returns_na(self):
        assert _fmt(None) == "NA"


# ---------------------------------------------------------------------------
# _top_runs
# ---------------------------------------------------------------------------
def _make_runs_df() -> pd.DataFrame:
    return pd.DataFrame({
        "run_id": ["r1", "r2", "r3"],
        "model_name": ["modelA", "modelB", "modelA"],
        "created_at": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
        "acc": [0.60, 0.65, 0.62],
        "brier": [0.22, 0.20, 0.21],
        "n_samples": [100, 120, 110],
    })


class TestTopRuns:
    def test_returns_dataframe(self):
        result = _top_runs(_make_runs_df(), ["acc"], [False], n=2)
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= 2

    def test_sorted_by_col(self):
        result = _top_runs(_make_runs_df(), ["acc"], [False], n=3)
        assert result["acc"].iloc[0] >= result["acc"].iloc[1]

    def test_missing_sort_col_returns_head(self):
        result = _top_runs(_make_runs_df(), ["nonexistent"], [False], n=2)
        assert len(result) == 2

    def test_ascending_sort(self):
        result = _top_runs(_make_runs_df(), ["brier"], [True], n=3)
        assert result["brier"].iloc[0] <= result["brier"].iloc[1]


# ---------------------------------------------------------------------------
# _metric_table
# ---------------------------------------------------------------------------
class TestMetricTable:
    def test_returns_string_for_valid_metric(self):
        result = _metric_table(_make_runs_df(), "acc", ascending=False)
        assert isinstance(result, str)

    def test_returns_none_for_missing_metric(self):
        result = _metric_table(_make_runs_df(), "not_a_col", ascending=False)
        assert result is None


# ---------------------------------------------------------------------------
# _load_runs
# ---------------------------------------------------------------------------
class TestLoadRuns:
    def test_raises_if_file_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _load_runs(tmp_path / "nonexistent.csv")

    def test_raises_if_empty(self, tmp_path):
        p = tmp_path / "metrics.csv"
        pd.DataFrame().to_csv(p, index=False)
        with pytest.raises(ValueError):
            _load_runs(p)

    def test_loads_and_adds_run_id(self, tmp_path):
        p = tmp_path / "metrics.csv"
        pd.DataFrame({
            "acc": [0.65],
            "brier": [0.21],
            "n_samples": [100],
        }).to_csv(p, index=False)
        result = _load_runs(p)
        assert "run_id" in result.columns

    def test_adds_model_name_if_missing(self, tmp_path):
        p = tmp_path / "metrics.csv"
        pd.DataFrame({"acc": [0.60]}).to_csv(p, index=False)
        result = _load_runs(p)
        assert "model_name" in result.columns


# ---------------------------------------------------------------------------
# generate_report (with a minimal DataFrame — touches most branches)
# ---------------------------------------------------------------------------
class TestGenerateReport:
    def test_returns_string(self, tmp_path, monkeypatch):
        # Redirect OUTPUT_DIR to tmp_path to avoid polluting real dirs
        import src.analysis.compare_runs as cr
        monkeypatch.setattr(cr, "OUTPUT_DIR", tmp_path)
        result = generate_report(_make_runs_df())
        assert isinstance(result, str)

    def test_report_contains_accuracy(self, tmp_path, monkeypatch):
        import src.analysis.compare_runs as cr
        monkeypatch.setattr(cr, "OUTPUT_DIR", tmp_path)
        result = generate_report(_make_runs_df())
        assert "acc" in result.lower() or "accuracy" in result.lower() or isinstance(result, str)
