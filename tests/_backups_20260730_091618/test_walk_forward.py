#!/usr/bin/env python
"""Quick test of walk-forward validation module."""

from analysis.walk_forward_validation import (
    generate_windows,
    ValidationWindow,
)

def test_window_generation():
    """Test that windows are generated correctly."""
    print("Testing window generation...")

    # Test basic 5-year windows
    windows = generate_windows(
        start_season=2016,
        end_season=2025,
        train_window_years=5,
        min_test_season=2021,
    )

    print(f"\nGenerated {len(windows)} windows:")
    for w in windows:
        print(f"  {w.name}")
        print(f"    Train: {w.train_start}-{w.train_end} ({w.train_years} years)")
        print(f"    Test: {w.test_season}")

    # Verify expected windows
    expected = [
        (2016, 2020, 2021),
        (2017, 2021, 2022),
        (2018, 2022, 2023),
        (2019, 2023, 2024),
        (2020, 2024, 2025),
    ]

    assert len(windows) == len(expected), f"Expected {len(expected)} windows, got {len(windows)}"

    for i, (w, (start, end, test)) in enumerate(zip(windows, expected)):
        assert w.train_start == start, f"Window {i}: expected train_start={start}, got {w.train_start}"
        assert w.train_end == end, f"Window {i}: expected train_end={end}, got {w.train_end}"
        assert w.test_season == test, f"Window {i}: expected test_season={test}, got {w.test_season}"

    print("\n[PASS] All window generation tests passed!")

def test_window_properties():
    """Test ValidationWindow properties."""
    print("\nTesting ValidationWindow properties...")

    w = ValidationWindow(train_start=2016, train_end=2020, test_season=2021)

    assert w.name == "train_2016_2020_test_2021"
    assert w.train_years == 5

    print(f"  Window name: {w.name}")
    print(f"  Train years: {w.train_years}")
    print("\n[PASS] All property tests passed!")
