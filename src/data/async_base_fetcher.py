"""
Async abstract base classes for data fetchers.

This module provides async versions of data fetchers with connection pooling,
retry logic, and proper async context manager support for improved performance.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import aiofiles
import httpx
import pandas as pd

from src.utils.cache import get_cache, CacheVersion
from src.utils.io import RAW_DIR, write_df
from src.utils.logging_config import get_logger
from src.utils.retry import retry_async, RetryConfig


@dataclass
class AsyncFetchResult:
    """Result of an async data fetch operation."""
    
    success: bool
    data: Optional[pd.DataFrame] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class AsyncBaseDataFetcher(ABC):
    """
    Abstract base class for async data fetchers.
    
    Provides common functionality for:
    - Async operations with connection pooling
    - Caching
    - Error handling with retry logic
    - Logging
    - Data validation
    - File I/O
    - Async context manager support
    """
    
    def __init__(
        self,
        name: str,
        cache_enabled: bool = True,
        cache_ttl_hours: int = 24,
        output_dir: Optional[Path] = None,
        retry_config: Optional[RetryConfig] = None,
        timeout: float = 30.0,
        max_connections: int = 100,
        max_keepalive_connections: int = 20
    ):
        """
        Initialize async data fetcher.
        
        Args:
            name: Fetcher name for logging
            cache_enabled: Enable caching
            cache_ttl_hours: Cache time-to-live in hours
            output_dir: Directory for output files
            retry_config: Retry configuration for transient failures
            timeout: Request timeout in seconds
            max_connections: Maximum concurrent connections
            max_keepalive_connections: Maximum keepalive connections
        """
        self.name = name
        self.cache_enabled = cache_enabled
        self.cache_ttl_hours = cache_ttl_hours
        self.output_dir = output_dir or RAW_DIR
        self.logger = get_logger(f"async_fetcher.{name}")
        self._cache = get_cache()
        
        # Retry configuration
        self.retry_config = retry_config or RetryConfig(
            max_attempts=3,
            initial_delay=1.0,
            max_delay=60.0,
            exponential_base=2.0,
            jitter=True,
            exceptions=(httpx.HTTPError, asyncio.TimeoutError, ConnectionError)
        )
        
        # HTTP client configuration
        self.timeout = timeout
        self.max_connections = max_connections
        self.max_keepalive_connections = max_keepalive_connections
        
        # Client will be initialized in async context
        self._client: Optional[httpx.AsyncClient] = None
        self._client_lock = asyncio.Lock()
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_client()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
        return False
    
    async def _ensure_client(self):
        """Ensure HTTP client is initialized."""
        if self._client is None:
            async with self._client_lock:
                if self._client is None:
                    limits = httpx.Limits(
                        max_connections=self.max_connections,
                        max_keepalive_connections=self.max_keepalive_connections,
                        keepalive_expiry=30.0
                    )
                    
                    timeout_config = httpx.Timeout(
                        connect=10.0,
                        read=self.timeout,
                        write=self.timeout,
                        pool=5.0
                    )
                    
                    self._client = httpx.AsyncClient(
                        limits=limits,
                        timeout=timeout_config,
                        follow_redirects=True
                    )
                    self.logger.debug(f"Initialized async HTTP client for {self.name}")
    
    async def close(self):
        """Close HTTP client and cleanup resources."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            self.logger.debug(f"Closed async HTTP client for {self.name}")
    
    @abstractmethod
    async def _fetch_raw(self, **kwargs) -> AsyncFetchResult:
        """
        Fetch raw data from source. Must be implemented by subclasses.
        
        Args:
            **kwargs: Fetcher-specific parameters
            
        Returns:
            AsyncFetchResult with data or error
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
        return f"{self.name}:async:{params_str}"
    
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
    
    async def _transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform fetched data. Can be overridden by subclasses.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Transformed DataFrame
        """
        return df
    
    async def _save_data(self, df: pd.DataFrame, filename: str) -> Path:
        """
        Save data to file asynchronously.
        
        Args:
            df: DataFrame to save
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run blocking I/O in executor to avoid blocking event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, write_df, df, output_path)
        
        self.logger.info(f"Saved {len(df)} rows to {output_path}")
        return output_path
    
    async def fetch(
        self,
        use_cache: Optional[bool] = None,
        save_to_file: bool = True,
        output_filename: Optional[str] = None,
        **kwargs
    ) -> AsyncFetchResult:
        """
        Fetch data with caching, retry logic, and error handling.
        
        Args:
            use_cache: Override cache setting
            save_to_file: Save result to file
            output_filename: Custom output filename
            **kwargs: Fetcher-specific parameters
            
        Returns:
            AsyncFetchResult with data or error
        """
        use_cache = use_cache if use_cache is not None else self.cache_enabled
        
        # Try cache first (synchronous operation)
        if use_cache:
            cache_key = self._get_cache_key(**kwargs)
            cached = self._cache.get(cache_key, version=CacheVersion.FEATURES_VERSION)
            if cached is not None:
                self.logger.info(f"Loaded from cache: {cache_key}")
                return AsyncFetchResult(
                    success=True,
                    data=cached,
                    metadata={"from_cache": True}
                )
        
        # Ensure client is ready
        await self._ensure_client()
        
        # Fetch raw data with retry logic
        self.logger.info(f"Fetching data from {self.name}")
        
        @retry_async(
            max_attempts=self.retry_config.max_attempts,
            initial_delay=self.retry_config.initial_delay,
            max_delay=self.retry_config.max_delay,
            exponential_base=self.retry_config.exponential_base,
            jitter=self.retry_config.jitter,
            exceptions=self.retry_config.exceptions
        )
        async def _fetch_with_retry():
            return await self._fetch_raw(**kwargs)
        
        try:
            result = await _fetch_with_retry()
        except Exception as e:
            self.logger.error(f"Fetch failed after retries: {e}")
            return AsyncFetchResult(
                success=False,
                error=f"Fetch failed: {str(e)}",
                metadata={"exception_type": type(e).__name__}
            )
        
        if not result.success:
            self.logger.error(f"Fetch failed: {result.error}")
            return result
        
        # Validate
        if result.data is None or not self._validate_data(result.data):
            return AsyncFetchResult(
                success=False,
                error="Data validation failed",
                metadata=result.metadata
            )
        
        # Transform
        try:
            result.data = await self._transform_data(result.data)
        except Exception as e:
            self.logger.error(f"Transform failed: {e}")
            return AsyncFetchResult(
                success=False,
                error=f"Transform error: {e}",
                metadata={"exception_type": type(e).__name__}
            )
        
        # Cache (synchronous operation)
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
            await self._save_data(result.data, output_filename)
        
        self.logger.info(f"Fetched {len(result.data)} rows successfully")
        return result
    
    async def fetch_batch(
        self,
        requests: List[Dict[str, Any]],
        max_concurrent: int = 10,
        **common_kwargs
    ) -> List[AsyncFetchResult]:
        """
        Fetch multiple requests concurrently with rate limiting.
        
        Args:
            requests: List of request parameter dictionaries
            max_concurrent: Maximum concurrent requests
            **common_kwargs: Common kwargs for all requests
            
        Returns:
            List of AsyncFetchResult objects
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def _fetch_with_semaphore(request_kwargs: Dict[str, Any]) -> AsyncFetchResult:
            async with semaphore:
                merged_kwargs = {**common_kwargs, **request_kwargs}
                return await self.fetch(**merged_kwargs)
        
        self.logger.info(f"Fetching {len(requests)} requests with max {max_concurrent} concurrent")
        tasks = [_fetch_with_semaphore(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to failed results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Request {i} failed with exception: {result}")
                processed_results.append(AsyncFetchResult(
                    success=False,
                    error=str(result),
                    metadata={"request_index": i, "exception_type": type(result).__name__}
                ))
            else:
                processed_results.append(result)
        
        success_count = sum(1 for r in processed_results if r.success)
        self.logger.info(f"Batch fetch complete: {success_count}/{len(requests)} successful")
        
        return processed_results


class AsyncAPIDataFetcher(AsyncBaseDataFetcher):
    """Base class for async API-based data fetchers."""
    
    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        """
        Initialize async API fetcher.
        
        Args:
            name: Fetcher name
            base_url: API base URL
            api_key: Optional API key
            headers: Optional HTTP headers
            **kwargs: Additional AsyncBaseDataFetcher arguments
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
            params: Optional query parameters
            
        Returns:
            Full URL string
        """
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        if params:
            param_str = "&".join(f"{k}={v}" for k, v in params.items())
            url = f"{url}?{param_str}"
        return url
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> httpx.Response:
        """
        Make HTTP request to API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            params: Query parameters
            json_data: JSON body data
            **kwargs: Additional request arguments
            
        Returns:
            httpx.Response object
            
        Raises:
            httpx.HTTPError: On request failure
        """
        await self._ensure_client()
        
        if self._client is None:
            raise RuntimeError("HTTP client not initialized")
        
        url = self._build_url(endpoint, params)
        merged_headers = {**self.headers, **kwargs.pop("headers", {})}
        
        self.logger.debug(f"Making {method} request to {url}")
        
        response = await self._client.request(
            method=method,
            url=url,
            headers=merged_headers,
            json=json_data,
            **kwargs
        )
        response.raise_for_status()
        
        return response


class AsyncWebScraperFetcher(AsyncBaseDataFetcher):
    """Base class for async web scraping data fetchers."""
    
    def __init__(
        self,
        name: str,
        user_agent: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize async web scraper.
        
        Args:
            name: Fetcher name
            user_agent: Custom user agent string
            **kwargs: Additional AsyncBaseDataFetcher arguments
        """
        super().__init__(name, **kwargs)
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        )
    
    async def _fetch_html(self, url: str, **kwargs) -> str:
        """
        Fetch HTML content from URL.
        
        Args:
            url: URL to fetch
            **kwargs: Additional request arguments
            
        Returns:
            HTML content as string
            
        Raises:
            httpx.HTTPError: On request failure
        """
        await self._ensure_client()
        
        if self._client is None:
            raise RuntimeError("HTTP client not initialized")
        
        headers = kwargs.pop("headers", {})
        headers["User-Agent"] = self.user_agent
        
        self.logger.debug(f"Fetching HTML from {url}")
        
        response = await self._client.get(url, headers=headers, **kwargs)
        response.raise_for_status()
        
        return response.text


class AsyncFileDataFetcher(AsyncBaseDataFetcher):
    """Base class for async file-based data fetchers."""
    
    async def _read_file(self, file_path: Path) -> str:
        """
        Read file content asynchronously.
        
        Args:
            file_path: Path to file
            
        Returns:
            File content as string
            
        Raises:
            FileNotFoundError: If file doesn't exist
            IOError: On read failure
        """
        self.logger.debug(f"Reading file: {file_path}")
        
        async with aiofiles.open(file_path, mode='r') as f:
            content = await f.read()
        
        return content
    
    async def _read_csv_async(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """
        Read CSV file asynchronously.
        
        Args:
            file_path: Path to CSV file
            **kwargs: Additional pandas read_csv arguments
            
        Returns:
            DataFrame
        """
        self.logger.debug(f"Reading CSV: {file_path}")
        
        # Run blocking pandas operation in executor
        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(None, pd.read_csv, file_path, **kwargs)
        
        return df


class AsyncBatchDataFetcher(AsyncBaseDataFetcher):
    """
    Specialized fetcher for batch operations across multiple sources.
    
    Coordinates multiple async fetchers with intelligent rate limiting
    and error handling.
    """
    
    def __init__(
        self,
        name: str,
        fetchers: List[AsyncBaseDataFetcher],
        **kwargs
    ):
        """
        Initialize batch fetcher.
        
        Args:
            name: Fetcher name
            fetchers: List of async fetchers to coordinate
            **kwargs: Additional AsyncBaseDataFetcher arguments
        """
        super().__init__(name, **kwargs)
        self.fetchers = fetchers
    
    async def __aenter__(self):
        """Enter context for all fetchers."""
        await super().__aenter__()
        for fetcher in self.fetchers:
            await fetcher.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit context for all fetchers."""
        for fetcher in self.fetchers:
            await fetcher.__aexit__(exc_type, exc_val, exc_tb)
        await super().__aexit__(exc_type, exc_val, exc_tb)
        return False
    
    async def _fetch_raw(self, **kwargs) -> AsyncFetchResult:
        """
        Fetch from all sources and combine results.
        
        Args:
            **kwargs: Parameters for all fetchers
            
        Returns:
            Combined AsyncFetchResult
        """
        tasks = [fetcher.fetch(**kwargs) for fetcher in self.fetchers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine successful results
        successful_dfs = []
        errors = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                errors.append(f"Fetcher {i}: {str(result)}")
            elif isinstance(result, AsyncFetchResult):
                if result.success and result.data is not None:
                    successful_dfs.append(result.data)
                elif not result.success:
                    errors.append(f"Fetcher {i}: {result.error}")
        
        if not successful_dfs:
            return AsyncFetchResult(
                success=False,
                error=f"All fetchers failed: {'; '.join(errors)}",
                metadata={"error_count": len(errors)}
            )
        
        # Combine DataFrames
        combined_df = pd.concat(successful_dfs, ignore_index=True)
        
        return AsyncFetchResult(
            success=True,
            data=combined_df,
            metadata={
                "successful_fetchers": len(successful_dfs),
                "failed_fetchers": len(errors),
                "errors": errors if errors else None
            }
        )


# Made with Bob
