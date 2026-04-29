"""
Data loader factory pattern for consistent data access across the platform.

This module provides a unified interface for loading data from various sources
(parquet files, CSV files, databases) with consistent error handling, caching,
and validation.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol

import pandas as pd

from src.utils.cache import get_cache, CacheVersion
from src.utils.io import read_df
from src.utils.logging_config import get_logger
from src.utils.schemas import validate_dataframe

logger = get_logger(__name__)


class DataSource(Protocol):
    """Protocol for data sources."""
    
    def load(self) -> pd.DataFrame:
        """Load data from the source."""
        ...


@dataclass
class DataLoaderConfig:
    """Configuration for data loaders."""
    
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    validate_schema: bool = True
    required_columns: Optional[list[str]] = None
    column_types: Optional[dict[str, str]] = None


class BaseDataLoader(ABC):
    """Abstract base class for data loaders."""
    
    def __init__(self, config: Optional[DataLoaderConfig] = None):
        """
        Initialize data loader.
        
        Args:
            config: Loader configuration
        """
        self.config = config or DataLoaderConfig()
        self.logger = get_logger(self.__class__.__name__)
    
    @abstractmethod
    def _load_raw(self) -> pd.DataFrame:
        """Load raw data from source. Must be implemented by subclasses."""
        pass
    
    def _get_cache_key(self) -> str:
        """Generate cache key for this loader."""
        # Use class name and config as cache key
        key_data = f"{self.__class__.__name__}_{str(self.config)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate loaded data.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Validated DataFrame
            
        Raises:
            ValueError: If validation fails
        """
        if df.empty:
            self.logger.warning("Loaded DataFrame is empty")
            return df
        
        # Check required columns
        if self.config.required_columns:
            missing = set(self.config.required_columns) - set(df.columns)
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
        
        # Validate column types if schema provided
        if self.config.column_types and self.config.validate_schema:
            try:
                validate_dataframe(df, self.config.column_types)
            except Exception as e:
                self.logger.warning(f"Schema validation failed: {e}")
        
        return df
    
    def load(self, use_cache: Optional[bool] = None) -> pd.DataFrame:
        """
        Load data with caching and validation.
        
        Args:
            use_cache: Override cache setting
            
        Returns:
            Loaded and validated DataFrame
        """
        use_cache = use_cache if use_cache is not None else self.config.cache_enabled
        cache = get_cache()
        
        # Try cache first
        if use_cache:
            cache_key = self._get_cache_key()
            cached = cache.get(cache_key, version=CacheVersion.FEATURES_VERSION)
            if cached is not None:
                self.logger.debug(f"Loaded from cache: {cache_key}")
                return cached
        
        # Load raw data
        self.logger.info(f"Loading data from {self.__class__.__name__}")
        df = self._load_raw()
        
        # Validate
        df = self._validate(df)
        
        # Cache result
        if use_cache:
            cache_key = self._get_cache_key()
            ttl_hours = self.config.cache_ttl_seconds // 3600  # Convert seconds to hours
            cache.set(cache_key, df, version=CacheVersion.FEATURES_VERSION, ttl_hours=ttl_hours)
            self.logger.debug(f"Cached data: {cache_key}")
        
        self.logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        return df


class ParquetDataLoader(BaseDataLoader):
    """Data loader for parquet files."""
    
    def __init__(
        self,
        path: Path,
        columns: Optional[list[str]] = None,
        config: Optional[DataLoaderConfig] = None
    ):
        """
        Initialize parquet loader.
        
        Args:
            path: Path to parquet file
            columns: Optional list of columns to load
            config: Loader configuration
        """
        super().__init__(config)
        self.path = Path(path)
        self.columns = columns
    
    def _load_raw(self) -> pd.DataFrame:
        """Load data from parquet file."""
        return read_df(self.path, columns=self.columns)


class CSVDataLoader(BaseDataLoader):
    """Data loader for CSV files."""
    
    def __init__(
        self,
        path: Path,
        config: Optional[DataLoaderConfig] = None,
        **read_csv_kwargs: Any
    ):
        """
        Initialize CSV loader.
        
        Args:
            path: Path to CSV file
            config: Loader configuration
            **read_csv_kwargs: Additional arguments for pd.read_csv
        """
        super().__init__(config)
        self.path = Path(path)
        self.read_csv_kwargs = read_csv_kwargs
    
    def _load_raw(self) -> pd.DataFrame:
        """Load data from CSV file."""
        if not self.path.exists():
            self.logger.warning(f"CSV file not found: {self.path}")
            return pd.DataFrame()
        
        return pd.read_csv(self.path, **self.read_csv_kwargs)


class MultiFileDataLoader(BaseDataLoader):
    """Data loader that combines multiple files."""
    
    def __init__(
        self,
        paths: list[Path],
        file_type: str = "parquet",
        config: Optional[DataLoaderConfig] = None
    ):
        """
        Initialize multi-file loader.
        
        Args:
            paths: List of file paths
            file_type: Type of files ('parquet' or 'csv')
            config: Loader configuration
        """
        super().__init__(config)
        self.paths = [Path(p) for p in paths]
        self.file_type = file_type
    
    def _load_raw(self) -> pd.DataFrame:
        """Load and combine data from multiple files."""
        dfs = []
        
        for path in self.paths:
            if not path.exists():
                self.logger.warning(f"File not found: {path}")
                continue
            
            try:
                if self.file_type == "parquet":
                    df = read_df(path)
                elif self.file_type == "csv":
                    df = pd.read_csv(path)
                else:
                    raise ValueError(f"Unsupported file type: {self.file_type}")
                
                dfs.append(df)
            except Exception as e:
                self.logger.error(f"Failed to load {path}: {e}")
        
        if not dfs:
            return pd.DataFrame()
        
        return pd.concat(dfs, ignore_index=True)


class DataLoaderFactory:
    """Factory for creating data loaders."""
    
    @staticmethod
    def create_parquet_loader(
        path: Path,
        columns: Optional[list[str]] = None,
        required_columns: Optional[list[str]] = None,
        cache_enabled: bool = True
    ) -> ParquetDataLoader:
        """
        Create a parquet data loader.
        
        Args:
            path: Path to parquet file
            columns: Columns to load
            required_columns: Required columns for validation
            cache_enabled: Enable caching
            
        Returns:
            Configured ParquetDataLoader
        """
        config = DataLoaderConfig(
            cache_enabled=cache_enabled,
            required_columns=required_columns
        )
        return ParquetDataLoader(path, columns=columns, config=config)
    
    @staticmethod
    def create_csv_loader(
        path: Path,
        required_columns: Optional[list[str]] = None,
        cache_enabled: bool = True,
        **read_csv_kwargs: Any
    ) -> CSVDataLoader:
        """
        Create a CSV data loader.
        
        Args:
            path: Path to CSV file
            required_columns: Required columns for validation
            cache_enabled: Enable caching
            **read_csv_kwargs: Additional arguments for pd.read_csv
            
        Returns:
            Configured CSVDataLoader
        """
        config = DataLoaderConfig(
            cache_enabled=cache_enabled,
            required_columns=required_columns
        )
        return CSVDataLoader(path, config=config, **read_csv_kwargs)
    
    @staticmethod
    def create_multi_file_loader(
        paths: list[Path],
        file_type: str = "parquet",
        required_columns: Optional[list[str]] = None,
        cache_enabled: bool = True
    ) -> MultiFileDataLoader:
        """
        Create a multi-file data loader.
        
        Args:
            paths: List of file paths
            file_type: Type of files
            required_columns: Required columns for validation
            cache_enabled: Enable caching
            
        Returns:
            Configured MultiFileDataLoader
        """
        config = DataLoaderConfig(
            cache_enabled=cache_enabled,
            required_columns=required_columns
        )
        return MultiFileDataLoader(paths, file_type=file_type, config=config)


# Convenience functions
def load_parquet(
    path: Path,
    columns: Optional[list[str]] = None,
    required_columns: Optional[list[str]] = None,
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Convenience function to load parquet file.
    
    Args:
        path: Path to parquet file
        columns: Columns to load
        required_columns: Required columns for validation
        use_cache: Use caching
        
    Returns:
        Loaded DataFrame
    """
    loader = DataLoaderFactory.create_parquet_loader(
        path, columns=columns, required_columns=required_columns, cache_enabled=use_cache
    )
    return loader.load()


def load_csv(
    path: Path,
    required_columns: Optional[list[str]] = None,
    use_cache: bool = True,
    **read_csv_kwargs: Any
) -> pd.DataFrame:
    """
    Convenience function to load CSV file.
    
    Args:
        path: Path to CSV file
        required_columns: Required columns for validation
        use_cache: Use caching
        **read_csv_kwargs: Additional arguments for pd.read_csv
        
    Returns:
        Loaded DataFrame
    """
    loader = DataLoaderFactory.create_csv_loader(
        path, required_columns=required_columns, cache_enabled=use_cache, **read_csv_kwargs
    )
    return loader.load()

# Made with Bob
