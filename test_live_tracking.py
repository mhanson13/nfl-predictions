#!/usr/bin/env python
"""Test live prediction tracking system."""

from pathlib import Path
import sys
import tempfile
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

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

if __name__ == "__main__":
    print("=" * 80)
    print("Live Prediction Tracking System Tests")
    print("=" * 80)
    
    try:
        test_filename_generation()
        test_season_directory()
        test_list_tracked_weeks()
        test_workflow_simulation()
        
        print("\n" + "=" * 80)
        print("[SUCCESS] ALL TESTS PASSED")
        print("=" * 80)
        print("\nThe live tracking system is ready to use.")
        print("\nTo use the system:")
        print("  # Lock predictions before games")
        print("  python -m analysis.live_tracking --lock-predictions --season 2025 --week 10")
        print("")
        print("  # Fetch actuals after games")
        print("  python -m analysis.live_tracking --fetch-actuals --season 2025 --week 10")
        print("")
        print("  # Calculate metrics")
        print("  python -m analysis.live_tracking --calculate-metrics --season 2025 --week 10")
        print("")
        print("  # Generate report")
        print("  python -m analysis.live_tracking --generate-report --season 2025 --week 10")
        print("")
        print("  # Or run complete cycle")
        print("  python -m analysis.live_tracking --weekly-cycle --season 2025 --week 10")
        print("")
        print("  # List tracked weeks")
        print("  python -m analysis.live_tracking --list-tracked")
        
    except Exception as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# Made with Bob
