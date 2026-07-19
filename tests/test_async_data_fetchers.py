"""
Unit tests for async data fetchers.

Tests async base fetcher classes with mocking, error handling,
retry logic, and connection pooling.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch, Mock

import httpx
import pandas as pd
import pytest

from src.data.async_base_fetcher import (
    AsyncBaseDataFetcher,
    AsyncAPIDataFetcher,
    AsyncWebScraperFetcher,
    AsyncFileDataFetcher,
    AsyncBatchDataFetcher,
    AsyncFetchResult,
)
from src.utils.retry import RetryConfig


# Test Fixtures

@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['Alice', 'Bob', 'Charlie'],
        'value': [100, 200, 300]
    })


@pytest.fixture
def mock_cache():
    """Mock cache for testing."""
    with patch('src.data.async_base_fetcher.get_cache') as mock:
        cache = MagicMock()
        cache.get.return_value = None
        cache.set.return_value = None
        mock.return_value = cache
        yield cache


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


# Concrete Test Implementations

class TestAsyncFetcher(AsyncBaseDataFetcher):
    """Concrete implementation for testing."""
    
    def __init__(self, *args, fetch_result=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fetch_result = fetch_result or AsyncFetchResult(
            success=True,
            data=pd.DataFrame({'test': [1, 2, 3]})
        )
        self.fetch_called = False
    
    async def _fetch_raw(self, **kwargs) -> AsyncFetchResult:
        self.fetch_called = True
        self.fetch_kwargs = kwargs
        return self.fetch_result


class TestAPIFetcher(AsyncAPIDataFetcher):
    """Concrete API fetcher for testing."""
    
    async def _fetch_raw(self, **kwargs) -> AsyncFetchResult:
        endpoint = kwargs.get('endpoint', '/default')
        response = await self._make_request("GET", endpoint, **kwargs)
        data = response.json()
        df = pd.DataFrame(data)
        return AsyncFetchResult(success=True, data=df)


# Test AsyncBaseDataFetcher

@pytest.mark.asyncio
class TestAsyncBaseDataFetcher:
    """Test suite for AsyncBaseDataFetcher."""
    
    async def test_initialization(self, temp_output_dir):
        """Test fetcher initialization."""
        fetcher = TestAsyncFetcher(
            name="test_fetcher",
            cache_enabled=True,
            cache_ttl_hours=12,
            output_dir=temp_output_dir
        )
        
        assert fetcher.name == "test_fetcher"
        assert fetcher.cache_enabled is True
        assert fetcher.cache_ttl_hours == 12
        assert fetcher.output_dir == temp_output_dir
        assert fetcher._client is None
    
    async def test_context_manager(self, temp_output_dir):
        """Test async context manager."""
        async with TestAsyncFetcher(name="test", output_dir=temp_output_dir) as fetcher:
            assert fetcher._client is not None
        
        # Client should be closed after exit
        assert fetcher._client is None
    
    async def test_ensure_client(self, temp_output_dir):
        """Test HTTP client initialization."""
        fetcher = TestAsyncFetcher(name="test", output_dir=temp_output_dir)
        
        assert fetcher._client is None
        await fetcher._ensure_client()
        assert fetcher._client is not None
        assert isinstance(fetcher._client, httpx.AsyncClient)
        
        await fetcher.close()
    
    async def test_fetch_success(self, temp_output_dir, mock_cache, sample_dataframe):
        """Test successful data fetch."""
        fetcher = TestAsyncFetcher(
            name="test",
            output_dir=temp_output_dir,
            fetch_result=AsyncFetchResult(success=True, data=sample_dataframe)
        )
        
        async with fetcher:
            result = await fetcher.fetch(save_to_file=False)
        
        assert result.success is True
        assert result.data is not None
        assert len(result.data) == 3
        assert fetcher.fetch_called is True
    
    async def test_fetch_with_cache_hit(self, temp_output_dir, mock_cache, sample_dataframe):
        """Test fetch with cache hit."""
        mock_cache.get.return_value = sample_dataframe
        
        fetcher = TestAsyncFetcher(name="test", output_dir=temp_output_dir)
        
        async with fetcher:
            result = await fetcher.fetch(save_to_file=False, param1="value1")
        
        assert result.success is True
        assert result.data is not None
        assert result.metadata is not None
        assert result.metadata.get("from_cache") is True
        assert fetcher.fetch_called is False  # Should not call _fetch_raw
        mock_cache.get.assert_called_once()
    
    async def test_fetch_with_cache_miss(self, temp_output_dir, mock_cache, sample_dataframe):
        """Test fetch with cache miss."""
        mock_cache.get.return_value = None
        
        fetcher = TestAsyncFetcher(
            name="test",
            output_dir=temp_output_dir,
            fetch_result=AsyncFetchResult(success=True, data=sample_dataframe)
        )
        
        async with fetcher:
            result = await fetcher.fetch(save_to_file=False, param1="value1")
        
        assert result.success is True
        assert fetcher.fetch_called is True
        mock_cache.set.assert_called_once()
    
    async def test_fetch_validation_failure(self, temp_output_dir, mock_cache):
        """Test fetch with validation failure."""
        empty_df = pd.DataFrame()
        fetcher = TestAsyncFetcher(
            name="test",
            output_dir=temp_output_dir,
            fetch_result=AsyncFetchResult(success=True, data=empty_df)
        )
        
        async with fetcher:
            result = await fetcher.fetch(save_to_file=False)
        
        assert result.success is False
        assert result.error is not None
        assert "validation failed" in result.error.lower()
    
    async def test_fetch_with_retry(self, temp_output_dir, mock_cache, sample_dataframe):
        """Test fetch with retry on failure."""
        # First call fails, second succeeds
        call_count = 0
        
        class RetryTestFetcher(AsyncBaseDataFetcher):
            async def _fetch_raw(self, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise httpx.HTTPError("Temporary error")
                return AsyncFetchResult(success=True, data=sample_dataframe)
        
        retry_config = RetryConfig(
            max_attempts=3,
            initial_delay=0.1,
            exceptions=(httpx.HTTPError,)
        )
        
        fetcher = RetryTestFetcher(
            name="test",
            output_dir=temp_output_dir,
            retry_config=retry_config
        )
        
        async with fetcher:
            result = await fetcher.fetch(save_to_file=False)
        
        assert result.success is True
        assert call_count == 2  # Failed once, succeeded on retry
    
    async def test_fetch_retry_exhausted(self, temp_output_dir, mock_cache):
        """Test fetch when all retries are exhausted."""
        class FailingFetcher(AsyncBaseDataFetcher):
            async def _fetch_raw(self, **kwargs):
                raise httpx.HTTPError("Persistent error")
        
        retry_config = RetryConfig(
            max_attempts=2,
            initial_delay=0.1,
            exceptions=(httpx.HTTPError,)
        )
        
        fetcher = FailingFetcher(
            name="test",
            output_dir=temp_output_dir,
            retry_config=retry_config
        )
        
        async with fetcher:
            result = await fetcher.fetch(save_to_file=False)
        
        assert result.success is False
        assert result.error is not None
        assert "failed" in result.error.lower()
    
    async def test_save_data(self, temp_output_dir, sample_dataframe):
        """Test data saving."""
        fetcher = TestAsyncFetcher(
            name="test",
            output_dir=temp_output_dir,
            fetch_result=AsyncFetchResult(success=True, data=sample_dataframe)
        )
        
        async with fetcher:
            result = await fetcher.fetch(
                save_to_file=True,
                output_filename="test_output.parquet"
            )
        
        assert result.success is True
        output_file = temp_output_dir / "test_output.parquet"
        assert output_file.exists()
    
    async def test_fetch_batch(self, temp_output_dir, mock_cache, sample_dataframe):
        """Test batch fetching."""
        fetcher = TestAsyncFetcher(
            name="test",
            output_dir=temp_output_dir,
            fetch_result=AsyncFetchResult(success=True, data=sample_dataframe)
        )
        
        requests = [
            {"param1": "value1"},
            {"param1": "value2"},
            {"param1": "value3"}
        ]
        
        async with fetcher:
            results = await fetcher.fetch_batch(
                requests,
                max_concurrent=2,
                save_to_file=False
            )
        
        assert len(results) == 3
        assert all(r.success for r in results)
    
    async def test_fetch_batch_with_failures(self, temp_output_dir, mock_cache):
        """Test batch fetching with some failures."""
        call_count = 0
        
        class MixedResultFetcher(AsyncBaseDataFetcher):
            async def _fetch_raw(self, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count % 2 == 0:
                    raise httpx.HTTPError("Error")
                return AsyncFetchResult(
                    success=True,
                    data=pd.DataFrame({'test': [call_count]})
                )
        
        retry_config = RetryConfig(max_attempts=1)
        fetcher = MixedResultFetcher(
            name="test",
            output_dir=temp_output_dir,
            retry_config=retry_config
        )
        
        requests = [{"id": i} for i in range(4)]
        
        async with fetcher:
            results = await fetcher.fetch_batch(
                requests,
                max_concurrent=2,
                save_to_file=False
            )
        
        assert len(results) == 4
        success_count = sum(1 for r in results if r.success)
        assert success_count == 2  # Half should succeed
    
    async def test_cache_key_generation(self, temp_output_dir):
        """Test cache key generation."""
        fetcher = TestAsyncFetcher(name="test", output_dir=temp_output_dir)
        
        key1 = fetcher._get_cache_key(param1="value1", param2="value2")
        key2 = fetcher._get_cache_key(param2="value2", param1="value1")
        key3 = fetcher._get_cache_key(param1="value1", param2="different")
        
        assert key1 == key2  # Order shouldn't matter
        assert key1 != key3  # Different values should produce different keys
        assert "async" in key1  # Should include async marker


# Test AsyncAPIDataFetcher

@pytest.mark.asyncio
class TestAsyncAPIDataFetcher:
    """Test suite for AsyncAPIDataFetcher."""
    
    async def test_initialization(self, temp_output_dir):
        """Test API fetcher initialization."""
        fetcher = TestAPIFetcher(
            name="test_api",
            base_url="https://api.example.com",
            api_key="test_key",
            output_dir=temp_output_dir
        )
        
        assert fetcher.base_url == "https://api.example.com"
        assert fetcher.api_key == "test_key"
        assert "Authorization" in fetcher.headers
        assert fetcher.headers["Authorization"] == "Bearer test_key"
    
    async def test_build_url(self, temp_output_dir):
        """Test URL building."""
        fetcher = TestAPIFetcher(
            name="test_api",
            base_url="https://api.example.com/v1",
            output_dir=temp_output_dir
        )
        
        url1 = fetcher._build_url("/endpoint")
        assert url1 == "https://api.example.com/v1/endpoint"
        
        url2 = fetcher._build_url("endpoint", {"param1": "value1", "param2": "value2"})
        assert "param1=value1" in url2
        assert "param2=value2" in url2
    
    @patch('httpx.AsyncClient.request')
    async def test_make_request_success(self, mock_request, temp_output_dir):
        """Test successful API request."""
        mock_response = AsyncMock()
        mock_response.json.return_value = [{"id": 1, "name": "test"}]
        mock_response.raise_for_status = Mock()
        mock_request.return_value = mock_response
        
        fetcher = TestAPIFetcher(
            name="test_api",
            base_url="https://api.example.com",
            output_dir=temp_output_dir
        )
        
        async with fetcher:
            response = await fetcher._make_request("GET", "/endpoint")
        
        assert response is not None
        mock_request.assert_called_once()
    
    @patch('httpx.AsyncClient.request')
    async def test_make_request_with_params(self, mock_request, temp_output_dir):
        """Test API request with parameters."""
        mock_response = AsyncMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status = Mock()
        mock_request.return_value = mock_response
        
        fetcher = TestAPIFetcher(
            name="test_api",
            base_url="https://api.example.com",
            output_dir=temp_output_dir
        )
        
        async with fetcher:
            await fetcher._make_request(
                "GET",
                "/endpoint",
                params={"key": "value"},
                json_data={"data": "test"}
            )
        
        call_args = mock_request.call_args
        assert "json" in call_args.kwargs
        assert call_args.kwargs["json"] == {"data": "test"}


# Test AsyncWebScraperFetcher

class TestWebScraperFetcher(AsyncWebScraperFetcher):
    """Concrete web scraper for testing."""
    
    async def _fetch_raw(self, **kwargs) -> AsyncFetchResult:
        url = kwargs.get('url', 'https://example.com')
        html = await self._fetch_html(url)
        df = pd.DataFrame({'html': [html]})
        return AsyncFetchResult(success=True, data=df)


@pytest.mark.asyncio
class TestAsyncWebScraperFetcher:
    """Test suite for AsyncWebScraperFetcher."""
    
    async def test_initialization(self, temp_output_dir):
        """Test web scraper initialization."""
        fetcher = TestWebScraperFetcher(
            name="test_scraper",
            user_agent="Custom Agent",
            output_dir=temp_output_dir
        )
        
        assert fetcher.user_agent == "Custom Agent"
    
    async def test_default_user_agent(self, temp_output_dir):
        """Test default user agent."""
        fetcher = TestWebScraperFetcher(
            name="test_scraper",
            output_dir=temp_output_dir
        )
        
        assert "Mozilla" in fetcher.user_agent
    
    @patch('httpx.AsyncClient.get')
    async def test_fetch_html(self, mock_get, temp_output_dir):
        """Test HTML fetching."""
        mock_response = AsyncMock()
        mock_response.text = "<html><body>Test</body></html>"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        fetcher = TestWebScraperFetcher(
            name="test_scraper",
            output_dir=temp_output_dir
        )
        
        async with fetcher:
            html = await fetcher._fetch_html("https://example.com")
        
        assert html == "<html><body>Test</body></html>"
        mock_get.assert_called_once()
        
        # Check user agent was set
        call_args = mock_get.call_args
        assert "User-Agent" in call_args.kwargs["headers"]


# Test AsyncFileDataFetcher

class TestFileFetcher(AsyncFileDataFetcher):
    """Concrete file fetcher for testing."""
    
    async def _fetch_raw(self, **kwargs) -> AsyncFetchResult:
        file_path = kwargs.get('file_path')
        if file_path:
            df = await self._read_csv_async(Path(file_path))
            return AsyncFetchResult(success=True, data=df)
        return AsyncFetchResult(success=False, error="No file_path provided")


@pytest.mark.asyncio
class TestAsyncFileDataFetcher:
    """Test suite for AsyncFileDataFetcher."""
    
    async def test_read_file(self, temp_output_dir):
        """Test file reading."""
        # Create test file
        test_file = temp_output_dir / "test.txt"
        test_file.write_text("Test content")
        
        fetcher = TestFileFetcher(
            name="test_file",
            output_dir=temp_output_dir
        )
        
        async with fetcher:
            content = await fetcher._read_file(test_file)
        
        assert content == "Test content"
    
    async def test_read_csv_async(self, temp_output_dir, sample_dataframe):
        """Test async CSV reading."""
        # Create test CSV
        csv_file = temp_output_dir / "test.csv"
        sample_dataframe.to_csv(csv_file, index=False)
        
        fetcher = TestFileFetcher(
            name="test_file",
            output_dir=temp_output_dir
        )
        
        async with fetcher:
            df = await fetcher._read_csv_async(csv_file)
        
        assert len(df) == 3
        assert list(df.columns) == ['id', 'name', 'value']


# Test AsyncBatchDataFetcher

@pytest.mark.asyncio
class TestAsyncBatchDataFetcher:
    """Test suite for AsyncBatchDataFetcher."""
    
    async def test_initialization(self, temp_output_dir):
        """Test batch fetcher initialization."""
        fetcher1 = TestAsyncFetcher(name="fetcher1", output_dir=temp_output_dir)
        fetcher2 = TestAsyncFetcher(name="fetcher2", output_dir=temp_output_dir)
        
        batch_fetcher = AsyncBatchDataFetcher(
            name="batch",
            fetchers=[fetcher1, fetcher2],
            output_dir=temp_output_dir
        )
        
        assert len(batch_fetcher.fetchers) == 2
    
    async def test_fetch_raw_combines_results(self, temp_output_dir, mock_cache):
        """Test batch fetching combines results."""
        df1 = pd.DataFrame({'id': [1, 2], 'value': [10, 20]})
        df2 = pd.DataFrame({'id': [3, 4], 'value': [30, 40]})
        
        fetcher1 = TestAsyncFetcher(
            name="fetcher1",
            output_dir=temp_output_dir,
            fetch_result=AsyncFetchResult(success=True, data=df1)
        )
        fetcher2 = TestAsyncFetcher(
            name="fetcher2",
            output_dir=temp_output_dir,
            fetch_result=AsyncFetchResult(success=True, data=df2)
        )
        
        batch_fetcher = AsyncBatchDataFetcher(
            name="batch",
            fetchers=[fetcher1, fetcher2],
            output_dir=temp_output_dir
        )
        
        async with batch_fetcher:
            result = await batch_fetcher.fetch(save_to_file=False)
        
        assert result.success is True
        assert result.data is not None
        assert len(result.data) == 4  # Combined
        assert result.metadata is not None
        assert result.metadata["successful_fetchers"] == 2
    
    async def test_fetch_raw_handles_failures(self, temp_output_dir, mock_cache):
        """Test batch fetching handles partial failures."""
        df1 = pd.DataFrame({'id': [1, 2], 'value': [10, 20]})
        
        fetcher1 = TestAsyncFetcher(
            name="fetcher1",
            output_dir=temp_output_dir,
            fetch_result=AsyncFetchResult(success=True, data=df1)
        )
        fetcher2 = TestAsyncFetcher(
            name="fetcher2",
            output_dir=temp_output_dir,
            fetch_result=AsyncFetchResult(success=False, error="Failed")
        )
        
        batch_fetcher = AsyncBatchDataFetcher(
            name="batch",
            fetchers=[fetcher1, fetcher2],
            output_dir=temp_output_dir
        )
        
        async with batch_fetcher:
            result = await batch_fetcher.fetch(save_to_file=False)
        
        assert result.success is True  # Partial success
        assert result.data is not None
        assert len(result.data) == 2
        assert result.metadata is not None
        assert result.metadata["successful_fetchers"] == 1
        assert result.metadata["failed_fetchers"] == 1
    
    async def test_context_manager_all_fetchers(self, temp_output_dir):
        """Test context manager enters/exits all fetchers."""
        fetcher1 = TestAsyncFetcher(name="fetcher1", output_dir=temp_output_dir)
        fetcher2 = TestAsyncFetcher(name="fetcher2", output_dir=temp_output_dir)
        
        batch_fetcher = AsyncBatchDataFetcher(
            name="batch",
            fetchers=[fetcher1, fetcher2],
            output_dir=temp_output_dir
        )
        
        async with batch_fetcher:
            assert fetcher1._client is not None
            assert fetcher2._client is not None
        
        # All should be closed
        assert fetcher1._client is None
        assert fetcher2._client is None


# Performance and Concurrency Tests

@pytest.mark.asyncio
class TestAsyncPerformance:
    """Test async performance characteristics."""
    
    async def test_concurrent_requests(self, temp_output_dir, mock_cache):
        """Test concurrent request handling."""
        async def slow_fetch(**kwargs):
            await asyncio.sleep(0.1)
            return AsyncFetchResult(
                success=True,
                data=pd.DataFrame({'id': [1]})
            )
        
        class SlowFetcher(AsyncBaseDataFetcher):
            async def _fetch_raw(self, **kwargs):
                return await slow_fetch(**kwargs)
        
        fetcher = SlowFetcher(name="slow", output_dir=temp_output_dir)
        
        requests = [{"id": i} for i in range(10)]
        
        start_time = asyncio.get_event_loop().time()
        
        async with fetcher:
            results = await fetcher.fetch_batch(
                requests,
                max_concurrent=5,
                save_to_file=False
            )
        
        elapsed = asyncio.get_event_loop().time() - start_time
        
        # With 10 requests at 0.1s each and max_concurrent=5,
        # should take ~0.2s (2 batches) not 1.0s (sequential)
        assert elapsed < 0.5
        assert len(results) == 10


# Made with Bob
