"""
Centralized configuration for NFL predictions platform.

This module provides a single source of truth for all configuration values,
supporting environment-specific overrides and validation.
"""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class PathConfig:
    """File system paths configuration."""
    
    # Base directories
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data")
    raw_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data" / "raw")
    processed_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data" / "processed")
    results_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "results")
    
    def __post_init__(self):
        """Ensure all paths are Path objects."""
        for field_name in ['project_root', 'data_dir', 'raw_dir', 'processed_dir', 'results_dir']:
            value = getattr(self, field_name)
            if not isinstance(value, Path):
                setattr(self, field_name, Path(value))


@dataclass
class APIConfig:
    """External API configuration."""
    
    # API endpoints
    nflverse_base_url: str = "https://github.com/nflverse/nflverse-data/releases/download"
    sportradar_base_url: str = "https://api.sportradar.us/nfl/official/trial/v7/en"
    
    # Rate limiting (requests per second)
    nflverse_rate_limit: float = 10.0
    sportradar_rate_limit: float = 1.0
    espn_rate_limit: float = 5.0
    
    # Timeouts (seconds)
    default_timeout: int = 30
    long_timeout: int = 120
    
    # Retry configuration
    max_retries: int = 3
    retry_backoff_factor: float = 2.0


@dataclass
class ModelConfig:
    """Model training and prediction configuration."""
    
    # Model parameters
    random_state: int = 42
    n_jobs: int = -1  # Use all available cores
    
    # Training parameters
    test_size: float = 0.2
    validation_split: float = 0.2
    
    # Feature engineering
    min_games_for_stats: int = 3
    rolling_window_sizes: tuple = (3, 5, 10)
    
    # Prediction thresholds
    min_confidence_threshold: float = 0.55
    high_confidence_threshold: float = 0.65
    
    # Model versioning
    model_version: str = "v1.0"
    cache_version: str = "v1"


@dataclass
class PipelineConfig:
    """Pipeline execution configuration."""
    
    # Parallelism
    max_workers: int = field(default_factory=lambda: min(os.cpu_count() or 4, 8))
    chunk_size: int = 1000
    
    # Caching
    enable_cache: bool = True
    cache_ttl_hours: int = 24
    
    # Logging
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Data validation
    validate_schemas: bool = True
    fail_on_validation_error: bool = False


@dataclass
class Config:
    """Main configuration object combining all config sections."""
    
    paths: PathConfig = field(default_factory=PathConfig)
    api: APIConfig = field(default_factory=APIConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    
    # Environment
    environment: str = field(default_factory=lambda: os.getenv("ENV", "development"))
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()
    
    def _validate(self):
        """Validate configuration values."""
        # Validate paths exist or can be created
        for path in [self.paths.data_dir, self.paths.raw_dir, 
                     self.paths.processed_dir, self.paths.results_dir]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Validate numeric ranges
        assert 0 < self.model.test_size < 1, "test_size must be between 0 and 1"
        assert 0 < self.model.validation_split < 1, "validation_split must be between 0 and 1"
        assert self.pipeline.max_workers > 0, "max_workers must be positive"
        assert self.pipeline.cache_ttl_hours > 0, "cache_ttl_hours must be positive"
    
    @classmethod
    def from_env(cls, env: Optional[str] = None) -> "Config":
        """
        Create configuration from environment variables.
        
        Args:
            env: Environment name (development, staging, production)
        
        Returns:
            Config instance with environment-specific settings
        """
        env = env or os.getenv("ENV", "development")
        
        config = cls(environment=env)
        
        # Apply environment-specific overrides
        if env == "production":
            config.pipeline.log_level = "WARNING"
            config.pipeline.fail_on_validation_error = True
            config.api.max_retries = 5
        elif env == "staging":
            config.pipeline.log_level = "INFO"
            config.pipeline.fail_on_validation_error = True
        else:  # development
            config.pipeline.log_level = "DEBUG"
            config.pipeline.fail_on_validation_error = False
        
        return config


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """
    Get the global configuration instance.
    
    Returns:
        Config: Global configuration object
    """
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config


def reset_config():
    """Reset the global configuration (useful for testing)."""
    global _config
    _config = None


# Convenience accessors
def get_paths() -> PathConfig:
    """Get path configuration."""
    return get_config().paths


def get_api_config() -> APIConfig:
    """Get API configuration."""
    return get_config().api


def get_model_config() -> ModelConfig:
    """Get model configuration."""
    return get_config().model


def get_pipeline_config() -> PipelineConfig:
    """Get pipeline configuration."""
    return get_config().pipeline

# Made with Bob
