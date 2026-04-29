"""
Async data fetching framework for parallel API calls.

Provides async/await support for concurrent data fetching with rate limiting,
retry logic, and connection pooling. Can be used alongside synchronous code.
"""

import asyncio
import aiohttp
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, TypeVar, Generic, Coroutine
from datetime import datetime
import time
from pathlib import Path
import pandas as pd

from src.utils.logging_config import get_logger
from src.utils.retry import RetryConfig

logger = get_logger(__name__)

T = TypeVar('T')


@dataclass
class AsyncFetchConfig:
    """Configuration for async fetching."""
    
    max_concurrent: int = 10  # Maximum concurrent requests
    timeout: float = 30.0  # Request timeout in seconds
    rate_limit: Optional[float] = None  # Requests per second limit
    retry_config: Optional[RetryConfig] = None
    use_connection_pool: bool = True
    pool_size: int = 100
    
    def __post_init__(self):
        """Initialize retry config if not provided."""
        if self.retry_config is None:
            self.retry_config = RetryConfig(max_attempts=3, initial_delay=1.0)


@dataclass
class AsyncFetchResult:
    """Result of an async fetch operation."""
    
    url: str
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    status_code: Optional[int] = None
    fetch_time: float = 0.0
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'url': self.url,
            'success': self.success,
            'status_code': self.status_code,
            'error': self.error,
            'fetch_time': self.fetch_time,
            'retry_count': self.retry_count,
            'metadata': self.metadata,
            'has_data': self.data is not None
        }


class RateLimiter:
    """Rate limiter for async requests."""
    
    def __init__(self, rate: float):
        """
        Initialize rate limiter.
        
        Args:
            rate: Maximum requests per second
        """
        self.rate = rate
        self.min_interval = 1.0 / rate if rate > 0 else 0
        self.last_request = 0.0
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire permission to make a request."""
        async with self._lock:
            now = time.time()
            time_since_last = now - self.last_request
            
            if time_since_last < self.min_interval:
                wait_time = self.min_interval - time_since_last
                await asyncio.sleep(wait_time)
            
            self.last_request = time.time()


class AsyncFetcher(ABC, Generic[T]):
    """Abstract base class for async fetchers."""
    
    def __init__(self, name: str, config: Optional[AsyncFetchConfig] = None):
        """
        Initialize async fetcher.
        
        Args:
            name: Fetcher name
            config: Fetch configuration
        """
        self.name = name
        self.config = config or AsyncFetchConfig()
        self.logger = get_logger(f"{__name__}.{name}")
        self.rate_limiter = RateLimiter(self.config.rate_limit) if self.config.rate_limit else None
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Enter async context."""
        await self.create_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context."""
        await self.close_session()
    
    async def create_session(self):
        """Create aiohttp session."""
        if self.session is None:
            connector = aiohttp.TCPConnector(
                limit=self.config.pool_size,
                limit_per_host=self.config.max_concurrent
            ) if self.config.use_connection_pool else None
            
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout
            )
            
            self.logger.info(f"Created async session for {self.name}")
    
    async def close_session(self):
        """Close aiohttp session."""
        if self.session:
            await self.session.close()
            self.session = None
            self.logger.info(f"Closed async session for {self.name}")
    
    @abstractmethod
    async def fetch_one(self, url: str, **kwargs: Any) -> T:
        """
        Fetch data from a single URL.
        
        Args:
            url: URL to fetch
            **kwargs: Additional arguments
            
        Returns:
            Fetched data
        """
        pass
    
    async def fetch_with_retry(self, url: str, **kwargs: Any) -> AsyncFetchResult:
        """
        Fetch with retry logic.
        
        Args:
            url: URL to fetch
            **kwargs: Additional arguments
            
        Returns:
            Fetch result
        """
        start_time = time.time()
        retry_count = 0
        last_error: Optional[str] = None
        
        # Ensure retry_config is not None
        retry_config = self.config.retry_config
        if retry_config is None:
            retry_config = RetryConfig(max_attempts=3, initial_delay=1.0)
        
        for attempt in range(retry_config.max_attempts):
            try:
                # Apply rate limiting
                if self.rate_limiter:
                    await self.rate_limiter.acquire()
                
                # Fetch data
                data = await self.fetch_one(url, **kwargs)
                
                fetch_time = time.time() - start_time
                
                return AsyncFetchResult(
                    url=url,
                    success=True,
                    data=data,
                    fetch_time=fetch_time,
                    retry_count=retry_count
                )
            
            except asyncio.TimeoutError as e:
                last_error = f"Timeout: {e}"
                retry_count += 1
                self.logger.warning(f"Timeout fetching {url}, attempt {attempt + 1}")
                
            except aiohttp.ClientError as e:
                last_error = f"Client error: {e}"
                retry_count += 1
                self.logger.warning(f"Error fetching {url}, attempt {attempt + 1}: {e}")
            
            except Exception as e:
                last_error = f"Unexpected error: {e}"
                retry_count += 1
                self.logger.error(f"Unexpected error fetching {url}: {e}")
            
            # Wait before retry
            if attempt < retry_config.max_attempts - 1:
                delay = retry_config.initial_delay * (2 ** attempt)
                await asyncio.sleep(delay)
        
        # All retries failed
        fetch_time = time.time() - start_time
        
        return AsyncFetchResult(
            url=url,
            success=False,
            error=last_error,
            fetch_time=fetch_time,
            retry_count=retry_count
        )
    
    async def fetch_many(self, urls: List[str], **kwargs: Any) -> List[AsyncFetchResult]:
        """
        Fetch data from multiple URLs concurrently.
        
        Args:
            urls: List of URLs to fetch
            **kwargs: Additional arguments
            
        Returns:
            List of fetch results
        """
        self.logger.info(f"Fetching {len(urls)} URLs with max {self.config.max_concurrent} concurrent")
        
        # Create session if not exists
        if self.session is None:
            await self.create_session()
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        async def fetch_with_semaphore(url: str) -> AsyncFetchResult:
            async with semaphore:
                return await self.fetch_with_retry(url, **kwargs)
        
        # Fetch all URLs concurrently
        tasks = [fetch_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Task failed for {urls[i]}: {result}")
                final_results.append(AsyncFetchResult(
                    url=urls[i],
                    success=False,
                    error=str(result)
                ))
            else:
                final_results.append(result)
        
        # Log statistics
        successful = sum(1 for r in final_results if r.success)
        total_time = sum(r.fetch_time for r in final_results)
        avg_time = total_time / len(final_results) if final_results else 0
        
        self.logger.info(
            f"Completed: {successful}/{len(urls)} successful, "
            f"avg time: {avg_time:.2f}s"
        )
        
        return final_results


class AsyncAPIFetcher(AsyncFetcher[Dict[str, Any]]):
    """Async fetcher for JSON APIs."""
    
    def __init__(
        self,
        name: str,
        base_url: str,
        headers: Optional[Dict[str, str]] = None,
        config: Optional[AsyncFetchConfig] = None
    ):
        """
        Initialize async API fetcher.
        
        Args:
            name: Fetcher name
            base_url: Base URL for API
            headers: Optional HTTP headers
            config: Fetch configuration
        """
        super().__init__(name, config)
        self.base_url = base_url.rstrip('/')
        self.headers = headers or {}
    
    async def fetch_one(self, url: str, **kwargs: Any) -> Dict[str, Any]:
        """Fetch JSON data from API endpoint."""
        # Support both full URLs and endpoints
        if not url.startswith('http'):
            url = f"{self.base_url}/{url.lstrip('/')}"
        
        if self.session is None:
            raise RuntimeError("Session not initialized. Use async context manager.")
        
        async with self.session.get(url, headers=self.headers, **kwargs) as response:
            response.raise_for_status()
            return await response.json()


class AsyncDataFrameFetcher(AsyncFetcher[pd.DataFrame]):
    """Async fetcher that returns DataFrames."""
    
    def __init__(
        self,
        name: str,
        transform_func: Callable[[Dict[str, Any]], pd.DataFrame],
        config: Optional[AsyncFetchConfig] = None
    ):
        """
        Initialize async DataFrame fetcher.
        
        Args:
            name: Fetcher name
            transform_func: Function to transform response to DataFrame
            config: Fetch configuration
        """
        super().__init__(name, config)
        self.transform_func = transform_func
    
    async def fetch_one(self, url: str, **kwargs: Any) -> pd.DataFrame:
        """Fetch and transform to DataFrame."""
        if self.session is None:
            raise RuntimeError("Session not initialized. Use async context manager.")
        
        async with self.session.get(url, **kwargs) as response:
            response.raise_for_status()
            data = await response.json()
            return self.transform_func(data)
    
    async def fetch_and_combine(self, urls: List[str], **kwargs: Any) -> pd.DataFrame:
        """
        Fetch multiple URLs and combine into single DataFrame.
        
        Args:
            urls: List of URLs to fetch
            **kwargs: Additional arguments
            
        Returns:
            Combined DataFrame
        """
        results = await self.fetch_many(urls, **kwargs)
        
        # Extract successful DataFrames
        dfs = [r.data for r in results if r.success and r.data is not None]
        
        if not dfs:
            self.logger.warning("No successful fetches, returning empty DataFrame")
            return pd.DataFrame()
        
        # Combine DataFrames
        combined = pd.concat(dfs, ignore_index=True)
        self.logger.info(f"Combined {len(dfs)} DataFrames into {len(combined)} rows")
        
        return combined


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """
    Run async coroutine in sync context.
    
    Args:
        coro: Coroutine to run
        
    Returns:
        Result of coroutine
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(coro)


async def fetch_urls_async(
    urls: List[str],
    max_concurrent: int = 10,
    timeout: float = 30.0,
    rate_limit: Optional[float] = None
) -> List[AsyncFetchResult]:
    """
    Convenience function to fetch multiple URLs asynchronously.
    
    Args:
        urls: List of URLs to fetch
        max_concurrent: Maximum concurrent requests
        timeout: Request timeout
        rate_limit: Optional rate limit (requests per second)
        
    Returns:
        List of fetch results
    """
    config = AsyncFetchConfig(
        max_concurrent=max_concurrent,
        timeout=timeout,
        rate_limit=rate_limit
    )
    
    class SimpleFetcher(AsyncFetcher[bytes]):
        async def fetch_one(self, url: str, **kwargs: Any) -> bytes:
            if self.session is None:
                raise RuntimeError("Session not initialized")
            async with self.session.get(url) as response:
                response.raise_for_status()
                return await response.read()
    
    async with SimpleFetcher("simple", config) as fetcher:
        return await fetcher.fetch_many(urls)


def fetch_urls_sync(
    urls: List[str],
    max_concurrent: int = 10,
    timeout: float = 30.0,
    rate_limit: Optional[float] = None
) -> List[AsyncFetchResult]:
    """
    Synchronous wrapper for async URL fetching.
    
    Args:
        urls: List of URLs to fetch
        max_concurrent: Maximum concurrent requests
        timeout: Request timeout
        rate_limit: Optional rate limit (requests per second)
        
    Returns:
        List of fetch results
    """
    return run_async(fetch_urls_async(urls, max_concurrent, timeout, rate_limit))


class AsyncBatchFetcher:
    """Batch fetcher using async for improved performance."""
    
    def __init__(
        self,
        fetcher: AsyncFetcher,
        batch_size: int = 100
    ):
        """
        Initialize async batch fetcher.
        
        Args:
            fetcher: Async fetcher to use
            batch_size: Number of URLs per batch
        """
        self.fetcher = fetcher
        self.batch_size = batch_size
        self.logger = get_logger(__name__)
    
    async def fetch_in_batches(
        self,
        urls: List[str],
        **kwargs: Any
    ) -> List[AsyncFetchResult]:
        """
        Fetch URLs in batches.
        
        Args:
            urls: List of URLs to fetch
            **kwargs: Additional arguments
            
        Returns:
            List of all fetch results
        """
        total_urls = len(urls)
        self.logger.info(f"Fetching {total_urls} URLs in batches of {self.batch_size}")
        
        all_results = []
        
        for i in range(0, total_urls, self.batch_size):
            batch = urls[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (total_urls + self.batch_size - 1) // self.batch_size
            
            self.logger.info(f"Processing batch {batch_num}/{total_batches}")
            
            results = await self.fetcher.fetch_many(batch, **kwargs)
            all_results.extend(results)
            
            # Small delay between batches
            if i + self.batch_size < total_urls:
                await asyncio.sleep(0.1)
        
        successful = sum(1 for r in all_results if r.success)
        self.logger.info(f"Completed all batches: {successful}/{total_urls} successful")
        
        return all_results

# Made with Bob
