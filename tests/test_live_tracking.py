#!/usr/bin/env python
"""Test live prediction tracking system."""

import pytest

import analysis.live_tracking as live_tracking
from analysis.live_tracking import (
    get_season_dir,
    get_prediction_filename,
    get_actuals_filename,
    get_metrics_filename,
    list_tracked_weeks,
)

def test_filename_generation():
    """Test filename generation functions."""
    print("Testing filename generation...")

    assert get_prediction_filename(2025, 1) == "week_01_predictions.csv"
    assert get_prediction_filename(2025, 10) == "week_10_predictions.csv"
    assert get_actuals_filename(2025, 1) == "week_01_actuals.csv"
    assert get_metrics_filename(2025, 1) == "week_01_metrics.json"

    print("  [PASS] Filename generation")

def test_season_directory():
    """Test season directory creation."""
    print("\nTesting season directory...")

    # Use predictions_log directory
    season_dir = get_season_dir(2025)
    assert season_dir.exists()
    assert season_dir.name == "2025"
    assert season_dir.parent.name == "predictions_log"

    print(f"  Season directory: {season_dir}")
    print("  [PASS] Season directory creation")

def test_list_tracked_weeks():
    """Test listing tracked weeks."""
    print("\nTesting list_tracked_weeks...")

    df = list_tracked_weeks()
    print(f"  Found {len(df)} tracked weeks")

    if not df.empty:
        print(f"  Columns: {list(df.columns)}")
        print(f"  Sample:\n{df.head()}")

    print("  [PASS] List tracked weeks")

def test_workflow_simulation():
    """Simulate the workflow without actual data."""
    print("\nTesting workflow simulation...")

    # Test that functions exist and are callable
    from analysis.live_tracking import (
        lock_predictions,
        fetch_actuals,
        calculate_metrics,
        generate_report,
        weekly_cycle,
    )

    print("  Functions available:")
    print("    - lock_predictions")
    print("    - fetch_actuals")
    print("    - calculate_metrics")
    print("    - generate_report")
    print("    - weekly_cycle")

    print("  [PASS] Workflow functions available")


def test_lock_predictions_rejects_duplicate_game_ids(tmp_path, monkeypatch):
    source = tmp_path / "predictions.csv"
    pd = pytest.importorskip("pandas")
    pd.DataFrame([
        {"season": 2026, "week": 1, "game_id": "2026_01_BAL_IND", "home_win_prob": 0.55},
        {"season": 2026, "week": 1, "game_id": "2026_01_BAL_IND", "home_win_prob": 0.58},
    ]).to_csv(source, index=False)
    monkeypatch.setattr(live_tracking, "PREDICTIONS_LOG_DIR", tmp_path / "predictions_log")

    with pytest.raises(ValueError, match="duplicate game_id"):
        live_tracking.lock_predictions(2026, 1, source)


def test_lock_predictions_rejects_constant_probabilities(tmp_path, monkeypatch):
    source = tmp_path / "predictions.csv"
    pd = pytest.importorskip("pandas")
    pd.DataFrame([
        {"season": 2026, "week": 1, "game_id": "2026_01_BAL_IND", "home_win_prob": 0.15, "home_win_prob_model_raw": 0.016},
        {"season": 2026, "week": 1, "game_id": "2026_01_CLE_JAX", "home_win_prob": 0.15, "home_win_prob_model_raw": 0.024},
    ]).to_csv(source, index=False)
    monkeypatch.setattr(live_tracking, "PREDICTIONS_LOG_DIR", tmp_path / "predictions_log")

    with pytest.raises(ValueError, match="home_win_prob is constant"):
        live_tracking.lock_predictions(2026, 1, source)


def test_lock_predictions_rejects_collapsed_diagnostic_probability(tmp_path, monkeypatch):
    source = tmp_path / "predictions.csv"
    pd = pytest.importorskip("pandas")
    pd.DataFrame([
        {"season": 2026, "week": 1, "game_id": "2026_01_BAL_IND", "home_win_prob": 0.15, "home_win_prob_calibrated": 0.0},
        {"season": 2026, "week": 1, "game_id": "2026_01_CLE_JAX", "home_win_prob": 0.2725, "home_win_prob_calibrated": 0.0},
    ]).to_csv(source, index=False)
    monkeypatch.setattr(live_tracking, "PREDICTIONS_LOG_DIR", tmp_path / "predictions_log")

    with pytest.raises(ValueError, match="home_win_prob_calibrated"):
        live_tracking.lock_predictions(2026, 1, source)


def test_lock_predictions_success_uses_ascii_status(tmp_path, monkeypatch, capsys):
    source = tmp_path / "predictions.csv"
    pd = pytest.importorskip("pandas")
    pd.DataFrame([
        {
            "season": 2026,
            "week": 2,
            "game_id": "2026_02_BAL_IND",
            "home_win_prob": 0.41,
            "home_win_prob_calibrated": 0.39,
            "home_win_prob_capped": 0.39,
            "home_win_prob_raw": 0.39,
        },
        {
            "season": 2026,
            "week": 2,
            "game_id": "2026_02_CLE_JAX",
            "home_win_prob": 0.58,
            "home_win_prob_calibrated": 0.57,
            "home_win_prob_capped": 0.57,
            "home_win_prob_raw": 0.57,
        },
    ]).to_csv(source, index=False)
    monkeypatch.setattr(live_tracking, "PREDICTIONS_LOG_DIR", tmp_path / "predictions_log")

    output = live_tracking.lock_predictions(2026, 2, source)

    assert output.exists()
    assert "[ok] Locked 2 predictions" in capsys.readouterr().out


def test_lock_predictions_existing_file_noninteractive_aborts_cleanly(tmp_path, monkeypatch, capsys):
    source = tmp_path / "predictions.csv"
    pd = pytest.importorskip("pandas")
    pd.DataFrame([
        {"season": 2026, "week": 2, "game_id": "2026_02_BAL_IND", "home_win_prob": 0.41},
        {"season": 2026, "week": 2, "game_id": "2026_02_CLE_JAX", "home_win_prob": 0.58},
    ]).to_csv(source, index=False)
    monkeypatch.setattr(live_tracking, "PREDICTIONS_LOG_DIR", tmp_path / "predictions_log")
    output = live_tracking.lock_predictions(2026, 2, source)

    def _raise_eof(_prompt: str) -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", _raise_eof)
    second_output = live_tracking.lock_predictions(2026, 2, source)

    assert second_output == output
    assert "Stdin is not interactive" in capsys.readouterr().out
