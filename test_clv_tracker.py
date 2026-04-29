#!/usr/bin/env python
"""Test CLV tracker module."""

from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from analysis.clv_tracker import (
    american_to_implied_prob,
    calculate_clv,
)

def test_american_to_implied_prob():
    """Test American odds to implied probability conversion."""
    print("Testing American odds conversion...")
    
    # Favorite odds
    assert abs(american_to_implied_prob(-150) - 0.60) < 0.01
    assert abs(american_to_implied_prob(-200) - 0.6667) < 0.01
    assert abs(american_to_implied_prob(-110) - 0.5238) < 0.01
    
    # Underdog odds
    assert abs(american_to_implied_prob(+150) - 0.40) < 0.01
    assert abs(american_to_implied_prob(+200) - 0.3333) < 0.01
    assert abs(american_to_implied_prob(+110) - 0.4762) < 0.01
    
    # Even odds
    assert abs(american_to_implied_prob(+100) - 0.50) < 0.01
    
    print("  [PASS] American odds conversion")

def test_calculate_clv():
    """Test CLV calculation."""
    print("\nTesting CLV calculation...")
    
    # Positive CLV (model found value)
    clv1 = calculate_clv(0.65, -150)  # Model 65%, market 60%
    assert abs(clv1 - 0.05) < 0.01
    print(f"  Model 65% vs -150 odds: CLV = {clv1:.4f} (expected ~0.05)")
    
    # Negative CLV (market sharper)
    clv2 = calculate_clv(0.55, -150)  # Model 55%, market 60%
    assert abs(clv2 - (-0.05)) < 0.01
    print(f"  Model 55% vs -150 odds: CLV = {clv2:.4f} (expected ~-0.05)")
    
    # Zero CLV (model matches market)
    clv3 = calculate_clv(0.60, -150)  # Model 60%, market 60%
    assert abs(clv3) < 0.01
    print(f"  Model 60% vs -150 odds: CLV = {clv3:.4f} (expected ~0.00)")
    
    print("  [PASS] CLV calculation")

def test_clv_interpretation():
    """Test CLV interpretation examples."""
    print("\nTesting CLV interpretation...")
    
    # Strong positive CLV
    clv_strong = calculate_clv(0.70, -150)
    print(f"  Strong edge: Model 70% vs -150 odds = CLV {clv_strong:.4f}")
    assert clv_strong > 0.05, "Should show strong positive CLV"
    
    # Weak positive CLV
    clv_weak = calculate_clv(0.62, -150)
    print(f"  Weak edge: Model 62% vs -150 odds = CLV {clv_weak:.4f}")
    assert 0 < clv_weak < 0.05, "Should show weak positive CLV"
    
    # Underdog value
    clv_dog = calculate_clv(0.45, +150)
    print(f"  Underdog value: Model 45% vs +150 odds = CLV {clv_dog:.4f}")
    assert clv_dog > 0, "Should find value on underdog"
    
    print("  [PASS] CLV interpretation")

def test_module_functions():
    """Test that all module functions are available."""
    print("\nTesting module functions...")
    
    from analysis.clv_tracker import (
        load_predictions_with_odds,
        calculate_moneyline_clv,
        calculate_spread_clv,
        analyze_clv,
        generate_clv_report,
    )
    
    print("  Functions available:")
    print("    - load_predictions_with_odds")
    print("    - calculate_moneyline_clv")
    print("    - calculate_spread_clv")
    print("    - analyze_clv")
    print("    - generate_clv_report")
    
    print("  [PASS] Module functions available")

if __name__ == "__main__":
    print("=" * 80)
    print("CLV Tracker Module Tests")
    print("=" * 80)
    
    try:
        test_american_to_implied_prob()
        test_calculate_clv()
        test_clv_interpretation()
        test_module_functions()
        
        print("\n" + "=" * 80)
        print("[SUCCESS] ALL TESTS PASSED")
        print("=" * 80)
        print("\nThe CLV tracker is ready to use.")
        print("\nTo use the system:")
        print("  # Calculate CLV for specific week")
        print("  python -m analysis.clv_tracker --season 2025 --week 10 --generate-report")
        print("")
        print("  # Calculate CLV for full season")
        print("  python -m analysis.clv_tracker --season 2025 --generate-report")
        print("")
        print("Note: Requires predictions_log/ with locked predictions and")
        print("      matchup_features.parquet with closing odds.")
        
    except Exception as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# Made with Bob
