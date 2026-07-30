#!/usr/bin/env python
"""Test live prediction tracking system."""

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
