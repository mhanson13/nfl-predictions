"""Tests for pure helper functions in src/analysis/update_readme_metrics.py."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from src.analysis.update_readme_metrics import (
    _baseline_run,
    _best_run,
    _build_metrics_table,
    _compose_section,
    _format_value,
    _load_runs,
    _replace_section,
    read_text_with_fallback,
    write_text_with_fallback,
)


# ---------------------------------------------------------------------------
# _format_value
# ---------------------------------------------------------------------------
class TestFormatValue:
    def test_formats_float(self):
        assert _format_value(0.6543) == "0.654"

    def test_nan_returns_dash(self):
        assert _format_value(float("nan")) == "-"

    def test_none_returns_dash(self):
        assert _format_value(None) == "-"

    def test_custom_digits(self):
        assert _format_value(3.14159, digits=2) == "3.14"


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

    def test_loads_and_normalizes_accuracy_alias(self, tmp_path):
        p = tmp_path / "metrics.csv"
        pd.DataFrame({"accuracy": [0.62], "brier": [0.22]}).to_csv(p, index=False)
        result = _load_runs(p)
        assert "acc" in result.columns

    def test_adds_run_id_if_missing(self, tmp_path):
        p = tmp_path / "metrics.csv"
        pd.DataFrame({"acc": [0.60]}).to_csv(p, index=False)
        result = _load_runs(p)
        assert "run_id" in result.columns


# ---------------------------------------------------------------------------
# _best_run
# ---------------------------------------------------------------------------
def _make_runs(n: int = 3) -> pd.DataFrame:
    all_vals = {
        "run_id": [f"r{i}" for i in range(n)],
        "model_name": ["modelA"] * n,
        "created_at": pd.to_datetime([f"2025-01-{i+1:02d}" for i in range(n)]),
        "auc": [0.60, 0.65, 0.63][:n],
        "brier": [0.22, 0.20, 0.21][:n],
        "logloss": [0.55, 0.50, 0.52][:n],
        "acc": [0.60, 0.65, 0.62][:n],
        "mae": [5.0, 4.5, 4.8][:n],
        "rmse": [7.0, 6.5, 6.8][:n],
        "n_samples": [100, 120, 110][:n],
    }
    return pd.DataFrame(all_vals)


class TestBestRun:
    def test_returns_series(self):
        result = _best_run(_make_runs())
        assert isinstance(result, pd.Series)

    def test_picks_highest_auc(self):
        result = _best_run(_make_runs())
        assert result["auc"] == 0.65

    def test_raises_if_no_metric_cols(self):
        df = pd.DataFrame({"run_id": ["r1"], "model_name": ["m"]})
        with pytest.raises(ValueError):
            _best_run(df)


# ---------------------------------------------------------------------------
# _baseline_run
# ---------------------------------------------------------------------------
class TestBaselineRun:
    def test_returns_series_or_none(self):
        result = _baseline_run(_make_runs(), best_idx=0)
        assert result is None or isinstance(result, pd.Series)

    def test_returns_none_for_single_row(self):
        df = _make_runs(n=1)
        result = _baseline_run(df, best_idx=df.index[0])
        assert result is None

    def test_returns_different_from_best(self):
        df = _make_runs(n=3)
        best_idx = df.index[1]
        result = _baseline_run(df, best_idx)
        if result is not None:
            assert result.name != best_idx


# ---------------------------------------------------------------------------
# _build_metrics_table
# ---------------------------------------------------------------------------
class TestBuildMetricsTable:
    def test_returns_string(self):
        best = _make_runs().iloc[0]
        result = _build_metrics_table(best)
        assert isinstance(result, str)

    def test_contains_metric_labels(self):
        best = _make_runs().iloc[0]
        result = _build_metrics_table(best)
        for label in ("Accuracy", "AUC", "Brier", "LogLoss"):
            assert label in result


# ---------------------------------------------------------------------------
# _compose_section
# ---------------------------------------------------------------------------
class TestComposeSection:
    def test_returns_string(self):
        best = _make_runs().iloc[1]
        result = _compose_section(best, None)
        assert isinstance(result, str)

    def test_contains_model_performance_header(self):
        best = _make_runs().iloc[1]
        result = _compose_section(best, None)
        assert "## Model Performance" in result

    def test_with_baseline(self):
        df = _make_runs()
        best = df.iloc[1]
        baseline = df.iloc[0]
        result = _compose_section(best, baseline)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _replace_section
# ---------------------------------------------------------------------------
class TestReplaceSection:
    def test_replaces_existing_section(self):
        text = "# NFL\n\n## Model Performance\nOld content.\n\n## Other\n..."
        result = _replace_section(text, "## Model Performance\nNew content.\n\n")
        assert "New content" in result
        assert "Old content" not in result

    def test_inserts_after_title_when_missing(self):
        text = "# NFL Predictions\n\nSome intro."
        result = _replace_section(text, "## Model Performance\nNew.\n\n")
        assert "## Model Performance" in result

    def test_empty_text_inserts_section(self):
        result = _replace_section("", "## Model Performance\n")
        assert "## Model Performance" in result


# ---------------------------------------------------------------------------
# read_text_with_fallback / write_text_with_fallback
# ---------------------------------------------------------------------------
class TestReadWriteTextWithFallback:
    def test_raises_if_file_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_text_with_fallback(tmp_path / "nope.md")

    def test_reads_utf8_file(self, tmp_path):
        p = tmp_path / "file.md"
        p.write_text("hello world", encoding="utf-8")
        assert read_text_with_fallback(p) == "hello world"

    def test_write_and_read_roundtrip(self, tmp_path):
        p = tmp_path / "out.md"
        write_text_with_fallback(p, "Test content")
        assert read_text_with_fallback(p) == "Test content"
