"""
Abstract base classes for data fetchers.

This module provides a consistent interface for all data fetching operations,
whether from APIs, web scraping, or file systems.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.utils.cache import get_cache, CacheVersion
from src.utils.io import RAW_DIR, write_df
from src.utils.logging_config import get_logger


@dataclass
class FetchResult:
    """Result of a data fetch operation."""
    
    success: bool
    data: Optional[pd.DataFrame] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class BaseDataFetcher(ABC):
    """
    Abstract base class for all data fetchers.
    
    Provides common functionality for:
    - Caching
    - Error handling
    - Logging
    - Data validation
    - File I/O
    """
    
    def __init__(
        self,
        name: str,
        cache_enabled: bool = True,
        cache_ttl_hours: int = 24,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize data fetcher.
        
        Args:
            name: Fetcher name for logging
            cache_enabled: Enable caching
            cache_ttl_hours: Cache time-to-live in hours
            output_dir: Directory for output files
        """
        self.name = name
        self.cache_enabled = cache_enabled
        self.cache_ttl_hours = cache_ttl_hours
        self.output_dir = output_dir or RAW_DIR
        self.logger = get_logger(f"fetcher.{name}")
        self._cache = get_cache()
    
    @abstractmethod
    def _fetch_raw(self, **kwargs) -> FetchResult:
        """
        Fetch raw data from source. Must be implemented by subclasses.
        
        Args:
            **kwargs: Fetcher-specific parameters
            
        Returns:
            FetchResult with data or error
        """
        pass
    
    def _get_cache_key(self, **kwargs) -> str:
        """
        Generate cache key from parameters.
        
        Args:
            **kwargs: Parameters to include in key
            
        Returns:
            Cache key string
        """
        params_str = "|".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        return f"{self.name}:{params_str}"
    
    def _validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate fetched data. Can be overridden by subclasses.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if valid
        """
        if df is None or df.empty:
            self.logger.warning("Fetched data is empty")
            return False
        return True
    
    def _transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform fetched data. Can be overridden by subclasses.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Transformed DataFrame
        """
        return df
    
    def _save_data(self, df: pd.DataFrame, filename: str) -> Path:
        """
        Save data to file.
        
        Args:
            df: DataFrame to save
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename
        self.output_dir.mkdir(parents=True, exist_ok=True)
        write_df(df, output_path)
        self.logger.info(f"Saved {len(df)} rows to {output_path}")
        return output_path
    
    def fetch(
        self,
        use_cache: Optional[bool] = None,
        save_to_file: bool = True,
        output_filename: Optional[str] = None,
        **kwargs
    ) -> FetchResult:
        """
        Fetch data with caching and error handling.
        
        Args:
            use_cache: Override cache setting
            save_to_file: Save result to file
            output_filename: Custom output filename
            **kwargs: Fetcher-specific parameters
            
        Returns:
            FetchResult with data or error
        """
        use_cache = use_cache if use_cache is not None else self.cache_enabled
        
        # Try cache first
        if use_cache:
            cache_key = self._get_cache_key(**kwargs)
            cached = self._cache.get(cache_key, version=CacheVersion.FEATURES_VERSION)
            if cached is not None:
                self.logger.info(f"Loaded from cache: {cache_key}")
                return FetchResult(success=True, data=cached, metadata={"from_cache": True})
        
        # Fetch raw data
        self.logger.info(f"Fetching data from {self.name}")
        result = self._fetch_raw(**kwargs)
        
        if not result.success:
            self.logger.error(f"Fetch failed: {result.error}")
            return result
        
        # Validate
        if result.data is None or not self._validate_data(result.data):
            return FetchResult(
                success=False,
                error="Data validation failed",
                metadata=result.metadata
            )
        
        # Transform
        try:
            result.data = self._transform_data(result.data)
        except Exception as e:
            self.logger.error(f"Transform failed: {e}")
            return FetchResult(success=False, error=f"Transform error: {e}")
        
        # Cache
        if use_cache and result.data is not None:
            cache_key = self._get_cache_key(**kwargs)
            self._cache.set(
                cache_key,
                result.data,
                version=CacheVersion.FEATURES_VERSION,
                ttl_hours=self.cache_ttl_hours
            )
            self.logger.debug(f"Cached data: {cache_key}")
        
        # Save to file
        if save_to_file and result.data is not None:
            if output_filename is None:
                output_filename = f"{self.name}_{datetime.now().strftime('%Y%m%d')}.parquet"
            self._save_data(result.data, output_filename)
        
        self.logger.info(f"Fetched {len(result.data)} rows successfully")
        return result


class APIDataFetcher(BaseDataFetcher):
    """Base class for API-based data fetchers."""
    
    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        """
        Initialize API fetcher.
        
        Args:
            name: Fetcher name
            base_url: API base URL
            api_key: Optional API key
            headers: Optional HTTP headers
            **kwargs: Additional BaseDataFetcher arguments
        """
        super().__init__(name, **kwargs)
        self.base_url = base_url
        self.api_key = api_key
        self.headers = headers or {}
        
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
    
    def _build_url(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> str:
        """
        Build full API URL.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            Full URL
        """
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        if params:
            param_str = "&".join(f"{k}={v}" for k, v in params.items())
            url = f"{url}?{param_str}"
        return url


class WebScraperFetcher(BaseDataFetcher):
    """Base class for web scraping data fetchers."""
    
    def __init__(
        self,
        name: str,
        base_url: str,
        user_agent: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize web scraper.
        
        Args:
            name: Fetcher name
            base_url: Base URL for scraping
            user_agent: Custom user agent
            **kwargs: Additional BaseDataFetcher arguments
        """
        super().__init__(name, **kwargs)
        self.base_url = base_url
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )
        self.headers = {"User-Agent": self.user_agent}


class FileDataFetcher(BaseDataFetcher):
    """Base class for file-based data fetchers."""
    
    def __init__(
        self,
        name: str,
        source_dir: Path,
        file_pattern: str = "*.parquet",
        **kwargs
    ):
        """
        Initialize file fetcher.
        
        Args:
            name: Fetcher name
            source_dir: Source directory
            file_pattern: File pattern to match
            **kwargs: Additional BaseDataFetcher arguments
        """
        super().__init__(name, **kwargs)
        self.source_dir = Path(source_dir)
        self.file_pattern = file_pattern
    
    def _list_files(self) -> List[Path]:
        """
        List matching files in source directory.
        
        Returns:
            List of file paths
        """
        if not self.source_dir.exists():
            self.logger.warning(f"Source directory does not exist: {self.source_dir}")
            return []
        
        return sorted(self.source_dir.glob(self.file_pattern))


class BatchDataFetcher(BaseDataFetcher):
    """Base class for fetchers that process data in batches."""
    
    def __init__(
        self,
        name: str,
        batch_size: int = 100,
        **kwargs
    ):
        """
        Initialize batch fetcher.
        
        Args:
            name: Fetcher name
            batch_size: Number of items per batch
            **kwargs: Additional BaseDataFetcher arguments
        """
        super().__init__(name, **kwargs)
        self.batch_size = batch_size
    
    @abstractmethod
    def _fetch_batch(self, batch_items: List[Any]) -> FetchResult:
        """
        Fetch a single batch of data.
        
        Args:
            batch_items: Items to fetch in this batch
            
        Returns:
            FetchResult for the batch
        """
        pass
    
    def _fetch_raw(self, **kwargs) -> FetchResult:
        """
        Fetch data in batches.
        
        Args:
            **kwargs: Must include 'items' - list of items to fetch
            
        Returns:
            Combined FetchResult
        """
        items = kwargs.get('items', [])
        if not items:
            return FetchResult(success=False, error="No items provided")
        
        all_data = []
        errors = []
        
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            self.logger.info(f"Fetching batch {i // self.batch_size + 1} ({len(batch)} items)")
            
            result = self._fetch_batch(batch)
            
            if result.success and result.data is not None:
                all_data.append(result.data)
            else:
                errors.append(result.error)
        
        if not all_data:
            return FetchResult(
                success=False,
                error=f"All batches failed: {errors}"
            )
        
        combined_df = pd.concat(all_data, ignore_index=True)
        
        return FetchResult(
            success=True,
            data=combined_df,
            metadata={
                "batches": len(all_data),
                "errors": errors if errors else None
            }
        )

# Made with Bob
