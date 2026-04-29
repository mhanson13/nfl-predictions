#!/usr/bin/env python
"""Test calibration monitoring."""

from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from analysis.calibration_monitor import (
    decompose_brier_score,
    calculate_calibration_metrics,
    calculate_probability_bins,
    detect_calibration_drift,
    CalibrationMetrics,
)

def test_brier_decomposition():
    """Test Brier score decomposition."""
    print("Testing Brier score decomposition...")
    
    # Perfect predictions
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_pred = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    
    brier, calibration, resolution, uncertainty = decompose_brier_score(y_true, y_pred)
    
    print(f"  Perfect predictions:")
    print(f"    Brier: {brier:.4f} (should be 0)")
    print(f"    Calibration: {calibration:.4f}")
    print(f"    Resolution: {resolution:.4f}")
    print(f"    Uncertainty: {uncertainty:.4f}")
    
    assert brier < 0.01, "Perfect predictions should have Brier ~0"
    
    # Random predictions (50% for all)
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_pred = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    
    brier, calibration, resolution, uncertainty = decompose_brier_score(y_true, y_pred)
    
    print(f"  Random predictions (all 0.5):")
    print(f"    Brier: {brier:.4f}")
    print(f"    Calibration: {calibration:.4f}")
    print(f"    Resolution: {resolution:.4f} (should be ~0)")
    print(f"    Uncertainty: {uncertainty:.4f}")
    
    assert resolution < 0.01, "Random predictions should have resolution ~0"
    
    print("  [PASS] Brier score decomposition")


def test_calibration_metrics():
    """Test calibration metrics calculation."""
    print("\nTesting calibration metrics...")
    
    # Well-calibrated predictions
    np.random.seed(42)
    n = 1000
    y_pred = np.random.uniform(0.2, 0.8, n)
    y_true = (np.random.random(n) < y_pred).astype(int)
    
    metrics = calculate_calibration_metrics(y_true, y_pred)
    
    print(f"  Well-calibrated predictions (n={n}):")
    print(f"    Brier Score: {metrics.brier_score:.4f}")
    print(f"    Calibration Error: {metrics.calibration_error:.4f}")
    print(f"    Max Calibration Error: {metrics.max_calibration_error:.4f}")
    print(f"    Mean Predicted: {metrics.mean_predicted_prob:.3f}")
    print(f"    Mean Actual: {metrics.mean_actual_prob:.3f}")
    
    assert metrics.n_predictions == n
    assert 0 <= metrics.brier_score <= 1
    assert metrics.calibration_error < 0.1, "Should be reasonably calibrated"
    
    print("  [PASS] Calibration metrics")


def test_probability_bins():
    """Test probability bin calculation."""
    print("\nTesting probability bins...")
    
    # Create predictions across probability range
    y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_pred = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    
    bins = calculate_probability_bins(y_true, y_pred, n_bins=5)
    
    print(f"  Created {len(bins)} probability bins")
    
    for i, bin_data in enumerate(bins):
        print(f"    Bin {i+1}: {bin_data.bin_range[0]:.2f}-{bin_data.bin_range[1]:.2f}, "
              f"n={bin_data.n_predictions}, "
              f"pred={bin_data.mean_predicted:.3f}, "
              f"actual={bin_data.mean_actual:.3f}, "
              f"error={bin_data.calibration_error:.3f}")
    
    assert len(bins) > 0, "Should create at least one bin"
    assert all(b.n_predictions > 0 for b in bins), "All bins should have predictions"
    
    print("  [PASS] Probability bins")


def test_drift_detection():
    """Test calibration drift detection."""
    print("\nTesting drift detection...")
    
    # Baseline metrics (good calibration)
    baseline = CalibrationMetrics(
        n_predictions=1000,
        brier_score=0.200,
        brier_calibration=0.010,
        brier_resolution=0.045,
        brier_uncertainty=0.245,
        mean_predicted_prob=0.520,
        mean_actual_prob=0.515,
        calibration_error=0.015,
        max_calibration_error=0.040,
    )
    
    # Current metrics (slight drift)
    current = CalibrationMetrics(
        n_predictions=1000,
        brier_score=0.225,  # +0.025 drift
        brier_calibration=0.015,
        brier_resolution=0.040,
        brier_uncertainty=0.245,
        mean_predicted_prob=0.530,
        mean_actual_prob=0.510,
        calibration_error=0.025,  # +0.010 drift
        max_calibration_error=0.055,
    )
    
    # Detect drift with 0.02 threshold
    alerts = detect_calibration_drift(current, baseline, threshold=0.02)
    
    print(f"  Detected {len(alerts)} drift alert(s)")
    
    for alert in alerts:
        print(f"    {alert.metric}: {alert.drift:+.4f} ({alert.severity})")
        print(f"      Current: {alert.current_value:.4f}, Baseline: {alert.baseline_value:.4f}")
    
    assert len(alerts) > 0, "Should detect drift"
    assert any(a.metric == 'brier_score' for a in alerts), "Should detect Brier drift"
    
    # No drift case
    no_drift = CalibrationMetrics(
        n_predictions=1000,
        brier_score=0.205,  # +0.005 (below threshold)
        brier_calibration=0.012,
        brier_resolution=0.043,
        brier_uncertainty=0.245,
        mean_predicted_prob=0.518,
        mean_actual_prob=0.517,
        calibration_error=0.018,
        max_calibration_error=0.042,
    )
    
    no_alerts = detect_calibration_drift(no_drift, baseline, threshold=0.02)
    
    print(f"  No drift case: {len(no_alerts)} alert(s)")
    assert len(no_alerts) == 0, "Should not detect drift below threshold"
    
    print("  [PASS] Drift detection")


def test_module_functions():
    """Test that all module functions are available."""
    print("\nTesting module functions...")
    
    from analysis.calibration_monitor import (
        load_predictions,
        monitor_calibration,
        check_drift,
        generate_report,
        plot_reliability_diagram,
    )
    
    print("  Functions available:")
    print("    - decompose_brier_score")
    print("    - calculate_calibration_metrics")
    print("    - calculate_probability_bins")
    print("    - detect_calibration_drift")
    print("    - load_predictions")
    print("    - monitor_calibration")
    print("    - check_drift")
    print("    - generate_report")
    print("    - plot_reliability_diagram")
    
    print("  [PASS] Module functions available")


if __name__ == "__main__":
    print("=" * 80)
    print("Calibration Monitoring Tests")
    print("=" * 80)
    
    try:
        test_brier_decomposition()
        test_calibration_metrics()
        test_probability_bins()
        test_drift_detection()
        test_module_functions()
        
        print("\n" + "=" * 80)
        print("[SUCCESS] ALL TESTS PASSED")
        print("=" * 80)
        print("\nThe calibration monitoring system is ready to use.")
        print("\nTo use the system:")
        print("  # Monitor single week")
        print("  python -m analysis.calibration_monitor --season 2025 --week 10 --generate-report")
        print("")
        print("  # Monitor full season")
        print("  python -m analysis.calibration_monitor --season 2025 --generate-report")
        print("")
        print("  # Check for drift")
        print("  python -m analysis.calibration_monitor --season 2025 --check-drift --threshold 0.02")
        print("")
        print("  # Generate reliability diagram")
        print("  python -m analysis.calibration_monitor --season 2025 --plot-reliability")
        print("")
        print("Note: Requires predictions from live_tracking or matchup_features.parquet")
        
    except Exception as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# Made with Bob
