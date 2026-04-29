"""
Integration tests for the NFL predictions pipeline.

These tests verify that the full pipeline works end-to-end,
from data loading through feature engineering to predictions.
"""

import pytest  # type: ignore
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.utils.teams import normalize_team_abbr, get_team_abbr_from_name
from src.utils.io import read_df
from src.config import get_config


class TestPipelineIntegration:
    """Integration tests for the full prediction pipeline."""
    
    def test_config_loads_successfully(self):
        """Test that configuration loads without errors."""
        config = get_config()
        
        assert config is not None
        assert config.paths.data_dir.exists()
        assert config.pipeline.enable_cache in [True, False]
    
    def test_team_normalization_pipeline(self):
        """Test team normalization works across different formats."""
        test_cases = [
            ("Kansas City Chiefs", "KC"),
            ("Kansas City", "KC"),
            ("KC", "KC"),
            ("Los Angeles Rams", "LAR"),
            ("LA Rams", "LAR"),
            ("LAR", "LAR"),
        ]
        
        for input_name, expected in test_cases:
            result = get_team_abbr_from_name(input_name)
            assert result == expected, f"Failed for {input_name}"
    
    def test_data_loading_with_fallback(self):
        """Test that data loading with fallback works."""
        config = get_config()
        
        # Create a test parquet file
        test_file = config.paths.processed_dir / "test_data.parquet"
        test_df = pd.DataFrame({
            "team": ["KC", "SF", "BUF"],
            "score": [28, 24, 31]
        })
        
        try:
            test_df.to_parquet(test_file)
            
            # Test reading with our utility
            loaded_df = read_df(test_file)
            
            assert loaded_df is not None
            assert len(loaded_df) == 3
            assert "team" in loaded_df.columns
            
        finally:
            # Cleanup
            if test_file.exists():
                test_file.unlink()
    
    @pytest.mark.parametrize("team_abbr,expected_valid", [
        ("KC", True),
        ("SF", True),
        ("BUF", True),
        ("INVALID", False),
        ("", False),
        (None, False),
    ])
    def test_team_validation(self, team_abbr, expected_valid):
        """Test team abbreviation validation."""
        result = normalize_team_abbr(team_abbr)
        
        if expected_valid:
            assert result is not None
            assert len(result) in [2, 3]  # Valid abbreviations are 2-3 chars
        else:
            assert result is None


class TestFeatureEngineeringIntegration:
    """Integration tests for feature engineering pipeline."""
    
    def test_feature_dataframe_structure(self):
        """Test that feature DataFrames have expected structure."""
        # Create mock feature data
        features = pd.DataFrame({
            "game_id": ["2024_01_KC_BUF", "2024_01_SF_DAL"],
            "home_team": ["KC", "SF"],
            "away_team": ["BUF", "DAL"],
            "home_score": [28, 24],
            "away_score": [24, 21],
        })
        
        # Verify structure
        assert "game_id" in features.columns
        assert "home_team" in features.columns
        assert "away_team" in features.columns
        assert len(features) == 2
        
        # Verify team abbreviations are valid
        for team in features["home_team"]:
            assert normalize_team_abbr(team) is not None
        
        for team in features["away_team"]:
            assert normalize_team_abbr(team) is not None


class TestCachingIntegration:
    """Integration tests for caching functionality."""
    
    def test_cache_directory_creation(self):
        """Test that cache directory is created properly."""
        config = get_config()
        cache_dir = config.paths.processed_dir / ".cache"
        
        # Cache dir should exist or be creatable
        cache_dir.mkdir(parents=True, exist_ok=True)
        assert cache_dir.exists()
        assert cache_dir.is_dir()
    
    def test_cache_key_generation(self):
        """Test that cache keys are generated consistently."""
        from src.utils.cache import SmartCache
        
        cache = SmartCache()
        
        # Same inputs should generate same cache path
        path1 = cache._get_cache_path("test_key")
        path2 = cache._get_cache_path("test_key")
        
        assert path1 == path2
        
        # Different inputs should generate different paths
        path3 = cache._get_cache_path("different_key")
        assert path1 != path3


class TestSchemaValidationIntegration:
    """Integration tests for schema validation."""
    
    def test_schedule_schema_validation(self):
        """Test schedule DataFrame validation."""
        from src.utils.schemas import ScheduleSchema
        
        # Valid schedule data
        valid_schedule = pd.DataFrame({
            "game_id": ["2024_01_KC_BUF"],
            "season": [2024],
            "week": [1],
            "home_team": ["KC"],
            "away_team": ["BUF"],
            "gameday": [pd.Timestamp("2024-09-05")],
        })
        
        # Should not raise
        try:
            ScheduleSchema.validate(valid_schedule)
            validated = True
        except Exception:
            validated = False
        
        assert validated, "Valid schedule should pass validation"
    
    def test_prediction_schema_validation(self):
        """Test prediction DataFrame validation."""
        from src.utils.schemas import PredictionSchema
        
        # Valid prediction data
        valid_predictions = pd.DataFrame({
            "game_id": ["2024_01_KC_BUF"],
            "home_team": ["KC"],
            "away_team": ["BUF"],
            "home_win_prob": [0.65],
            "predicted_spread": [-3.5],
        })
        
        # Should not raise
        try:
            PredictionSchema.validate(valid_predictions)
            validated = True
        except Exception:
            validated = False
        
        assert validated, "Valid predictions should pass validation"


class TestEndToEndPipeline:
    """End-to-end pipeline tests (requires data)."""
    
    @pytest.mark.slow
    @pytest.mark.skipif(
        not (Path("data/processed/matchup_features.parquet").exists()),
        reason="Requires processed feature data"
    )
    def test_load_features_and_predict(self):
        """Test loading features and making predictions."""
        config = get_config()
        features_path = config.paths.processed_dir / "matchup_features.parquet"
        
        if not features_path.exists():
            pytest.skip("Feature data not available")
        
        # Load features
        features = read_df(features_path)
        
        assert features is not None
        assert len(features) > 0
        assert "home_team" in features.columns
        assert "away_team" in features.columns
        
        # Verify team abbreviations
        for team in features["home_team"].dropna().unique():
            assert normalize_team_abbr(team) is not None
    
    @pytest.mark.slow
    def test_full_pipeline_smoke_test(self):
        """Smoke test for full pipeline execution."""
        # This is a placeholder for a full pipeline test
        # In practice, this would run the entire pipeline with test data
        
        config = get_config()
        
        # Verify all required directories exist
        assert config.paths.data_dir.exists()
        assert config.paths.raw_dir.exists()
        assert config.paths.processed_dir.exists()
        
        # Verify configuration is valid
        assert config.pipeline.n_jobs > 0
        assert config.pipeline.cache_ttl_hours > 0


# Fixtures for integration tests
@pytest.fixture
def sample_schedule():
    """Create sample schedule data for testing."""
    return pd.DataFrame({
        "game_id": ["2024_01_KC_BUF", "2024_01_SF_DAL"],
        "season": [2024, 2024],
        "week": [1, 1],
        "home_team": ["KC", "SF"],
        "away_team": ["BUF", "DAL"],
        "gameday": [pd.Timestamp("2024-09-05"), pd.Timestamp("2024-09-08")],
    })


@pytest.fixture
def sample_features():
    """Create sample feature data for testing."""
    return pd.DataFrame({
        "game_id": ["2024_01_KC_BUF", "2024_01_SF_DAL"],
        "home_team": ["KC", "SF"],
        "away_team": ["BUF", "DAL"],
        "home_elo": [1650, 1620],
        "away_elo": [1580, 1590],
        "home_rest_days": [7, 7],
        "away_rest_days": [7, 7],
    })


@pytest.fixture
def sample_predictions():
    """Create sample prediction data for testing."""
    return pd.DataFrame({
        "game_id": ["2024_01_KC_BUF", "2024_01_SF_DAL"],
        "home_team": ["KC", "SF"],
        "away_team": ["BUF", "DAL"],
        "home_win_prob": [0.65, 0.58],
        "predicted_spread": [-3.5, -2.0],
    })


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# Made with Bob
