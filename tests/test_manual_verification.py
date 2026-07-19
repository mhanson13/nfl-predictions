"""
Manual verification script to test core functionality without pytest.
Run this to verify the new modules work correctly.
"""

import sys
import os
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    os.system('chcp 65001 > nul')


def test_teams_module():
    """Test team utilities module."""
    print("\n=== Testing src.utils.teams ===")

    from src.utils.teams import normalize_team_abbr, get_team_abbr_from_name

    # Test normalize_team_abbr
    tests = [
        ("KC", "KC"),
        ("LAR", "LAR"),
        ("LA", "LAR"),  # Should normalize to LAR
        ("INVALID", None),
        (None, None),
        ("", None),
    ]

    passed = 0
    failed = 0

    for input_val, expected in tests:
        result = normalize_team_abbr(input_val)
        if result == expected:
            print(f"  [PASS] normalize_team_abbr('{input_val}') = '{result}'")
            passed += 1
        else:
            print(f"  [FAIL] normalize_team_abbr('{input_val}') = '{result}' (expected '{expected}')")
            failed += 1

    # Test get_team_abbr_from_name
    name_tests = [
        ("Kansas City Chiefs", "KC"),
        ("Kansas City", "KC"),
        ("Los Angeles Rams", "LAR"),
        ("LA Rams", "LAR"),
        ("Invalid Team", None),
    ]

    for input_val, expected in name_tests:
        result = get_team_abbr_from_name(input_val)
        if result == expected:
            print(f"  [PASS] get_team_abbr_from_name('{input_val}') = '{result}'")
            passed += 1
        else:
            print(f"  [FAIL] get_team_abbr_from_name('{input_val}') = '{result}' (expected '{expected}')")
            failed += 1

    print(f"\nTeams Module: {passed} passed, {failed} failed")
    return failed == 0


def test_config_module():
    """Test configuration module."""
    print("\n=== Testing src.config ===")

    from src.config import get_config

    try:
        config = get_config()

        # Check that config loaded
        assert config is not None, "Config should not be None"
        print("  [PASS] Config loaded successfully")

        # Check paths exist
        assert config.paths.data_dir.exists(), "Data directory should exist"
        print(f"  [PASS] Data directory exists: {config.paths.data_dir}")

        # Check pipeline config
        assert hasattr(config.pipeline, 'enable_cache'), "Pipeline should have enable_cache"
        print(f"  [PASS] Pipeline config loaded (cache enabled: {config.pipeline.enable_cache})")

        print("\nConfig Module: All tests passed")
        return True

    except Exception as e:
        print(f"  [FAIL] Config test failed: {e}")
        return False


def test_io_module():
    """Test I/O utilities."""
    print("\n=== Testing src.utils.io ===")

    from src.utils.io import read_df
    import pandas as pd

    test_file = None
    try:
        # Create a test file
        test_file = Path("test_temp.parquet")
        test_df = pd.DataFrame({
            "team": ["KC", "SF"],
            "score": [28, 24]
        })
        test_df.to_parquet(test_file)

        # Test reading
        loaded_df = read_df(test_file)

        assert loaded_df is not None, "DataFrame should not be None"
        assert len(loaded_df) == 2, "Should have 2 rows"
        assert "team" in loaded_df.columns, "Should have 'team' column"

        print("  [PASS] read_df() works correctly")
        print(f"  [PASS] Loaded DataFrame with {len(loaded_df)} rows")

        # Cleanup
        test_file.unlink()

        print("\nI/O Module: All tests passed")
        return True

    except Exception as e:
        print(f"  [FAIL] I/O test failed: {e}")
        if test_file and test_file.exists():
            test_file.unlink()
        return False


def test_cache_module():
    """Test caching utilities."""
    print("\n=== Testing src.utils.cache ===")

    from src.utils.cache import SmartCache, CacheVersion

    try:
        cache = SmartCache()

        # Test cache operations
        test_key = "test_key"
        test_value = {"data": [1, 2, 3]}
        test_version = "v1.0"

        # Set cache
        success = cache.set(test_key, test_value, test_version)
        assert success, "Cache set should succeed"
        print("  [PASS] Cache set() works")

        # Get cache
        retrieved = cache.get(test_key, test_version)
        assert retrieved == test_value, "Retrieved value should match"
        print("  [PASS] Cache get() works")

        # Invalidate cache
        cache.invalidate(test_key)
        retrieved_after = cache.get(test_key, test_version)
        assert retrieved_after is None, "Cache should be invalidated"
        print("  [PASS] Cache invalidate() works")

        # Test version mismatch
        cache.set(test_key, test_value, "v1.0")
        wrong_version = cache.get(test_key, "v2.0")
        assert wrong_version is None, "Wrong version should return None"
        print("  [PASS] Cache version checking works")

        print("\nCache Module: All tests passed")
        return True

    except Exception as e:
        print(f"  [FAIL] Cache test failed: {e}")
        return False


def test_imports():
    """Test that all new modules can be imported."""
    print("\n=== Testing Module Imports ===")

    modules = [
        "src.utils.teams",
        "src.utils.io",
        "src.utils.cache",
        "src.utils.schemas",
        "src.utils.logging_config",
        "src.utils.performance",
        "src.config",
    ]

    passed = 0
    failed = 0

    for module in modules:
        try:
            __import__(module)
            print(f"  [PASS] {module}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {module}: {e}")
            failed += 1

    print(f"\nImports: {passed} passed, {failed} failed")
    return failed == 0
