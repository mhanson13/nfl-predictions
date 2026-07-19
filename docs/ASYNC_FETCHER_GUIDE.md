# Async Data Fetcher Implementation Guide

## Overview

The async data fetcher framework provides high-performance, concurrent data fetching with built-in connection pooling, retry logic, caching, and error handling. This implementation is part of Phase 2A infrastructure improvements targeting 4-6x pipeline speedup.

## Architecture

### Core Components

1. **AsyncBaseDataFetcher** - Abstract base class for all async fetchers
2. **AsyncAPIDataFetcher** - Specialized for REST API calls
3. **AsyncWebScraperFetcher** - Specialized for web scraping
4. **AsyncFileDataFetcher** - Specialized for file operations
5. **AsyncBatchDataFetcher** - Coordinates multiple fetchers

### Key Features

- **Async/Await Pattern**: Non-blocking I/O for concurrent operations
- **Connection Pooling**: HTTP/2 with configurable pool sizes
- **Automatic Retry**: Exponential backoff with jitter
- **Caching**: Transparent caching with TTL support
- **Context Manager**: Proper resource cleanup
- **Batch Operations**: Concurrent fetching with rate limiting
- **Type Safety**: Full type hints and validation

## Quick Start

### Basic Usage

```python
from src.data.async_base_fetcher import AsyncAPIDataFetcher, AsyncFetchResult
import pandas as pd

class MyAPIFetcher(AsyncAPIDataFetcher):
    """Fetch data from my API."""
    
    async def _fetch_raw(self, **kwargs) -> AsyncFetchResult:
        endpoint = kwargs.get('endpoint', '/data')
        response = await self._make_request('GET', endpoint)
        data = response.json()
        df = pd.DataFrame(data)
        return AsyncFetchResult(success=True, data=df)

# Use with context manager
async def fetch_data():
    async with MyAPIFetcher(
        name="my_api",
        base_url="https://api.example.com",
        api_key="your_key"
    ) as fetcher:
        result = await fetcher.fetch(endpoint='/users')
        if result.success:
            print(f"Fetched {len(result.data)} rows")
            return result.data
```

### Batch Fetching

```python
async def fetch_multiple_endpoints():
    async with MyAPIFetcher(
        name="my_api",
        base_url="https://api.example.com"
    ) as fetcher:
        requests = [
            {'endpoint': '/users'},
            {'endpoint': '/posts'},
            {'endpoint': '/comments'}
        ]
        
        results = await fetcher.fetch_batch(
            requests,
            max_concurrent=5  # Limit concurrent requests
        )
        
        successful = [r for r in results if r.success]
        print(f"Fetched {len(successful)}/{len(requests)} successfully")
```

## Configuration

### Retry Configuration

```python
from src.utils.retry import RetryConfig
import httpx

retry_config = RetryConfig(
    max_attempts=5,           # Maximum retry attempts
    initial_delay=1.0,        # Initial delay in seconds
    max_delay=60.0,           # Maximum delay in seconds
    exponential_base=2.0,     # Exponential backoff base
    jitter=True,              # Add random jitter
    exceptions=(              # Exceptions to retry on
        httpx.HTTPError,
        asyncio.TimeoutError,
        ConnectionError
    )
)

fetcher = MyAPIFetcher(
    name="my_api",
    base_url="https://api.example.com",
    retry_config=retry_config
)
```

### Connection Pooling

```python
fetcher = MyAPIFetcher(
    name="my_api",
    base_url="https://api.example.com",
    timeout=30.0,                      # Request timeout
    max_connections=100,               # Max concurrent connections
    max_keepalive_connections=20       # Max keepalive connections
)
```

### Caching

```python
fetcher = MyAPIFetcher(
    name="my_api",
    base_url="https://api.example.com",
    cache_enabled=True,        # Enable caching
    cache_ttl_hours=24         # Cache TTL in hours
)

# Override cache per request
result = await fetcher.fetch(
    use_cache=False,           # Skip cache for this request
    endpoint='/live-data'
)
```

## Advanced Usage

### Custom Validation

```python
class ValidatedAPIFetcher(AsyncAPIDataFetcher):
    """API fetcher with custom validation."""
    
    def _validate_data(self, df: pd.DataFrame) -> bool:
        """Custom validation logic."""
        if df.empty:
            return False
        
        # Check required columns
        required_cols = ['id', 'name', 'timestamp']
        if not all(col in df.columns for col in required_cols):
            self.logger.error(f"Missing required columns")
            return False
        
        # Check data quality
        if df['id'].isna().any():
            self.logger.error("Found null IDs")
            return False
        
        return True
    
    async def _fetch_raw(self, **kwargs) -> AsyncFetchResult:
        # Implementation
        pass
```

### Custom Transformation

```python
class TransformingAPIFetcher(AsyncAPIDataFetcher):
    """API fetcher with data transformation."""
    
    async def _transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform fetched data."""
        # Clean column names
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        
        # Parse timestamps
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Add metadata
        df['fetched_at'] = datetime.now()
        
        return df
    
    async def _fetch_raw(self, **kwargs) -> AsyncFetchResult:
        # Implementation
        pass
```

### Web Scraping

```python
from bs4 import BeautifulSoup

class MyWebScraper(AsyncWebScraperFetcher):
    """Scrape data from websites."""
    
    async def _fetch_raw(self, **kwargs) -> AsyncFetchResult:
        url = kwargs.get('url')
        html = await self._fetch_html(url)
        
        # Parse HTML
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract data
        data = []
        for item in soup.find_all('div', class_='item'):
            data.append({
                'title': item.find('h2').text,
                'price': item.find('span', class_='price').text
            })
        
        df = pd.DataFrame(data)
        return AsyncFetchResult(success=True, data=df)

# Usage
async with MyWebScraper(
    name="my_scraper",
    user_agent="MyBot/1.0"
) as scraper:
    result = await scraper.fetch(url='https://example.com/products')
```

### File Operations

```python
class CSVFileFetcher(AsyncFileDataFetcher):
    """Fetch data from CSV files."""
    
    async def _fetch_raw(self, **kwargs) -> AsyncFetchResult:
        file_path = Path(kwargs.get('file_path'))
        
        # Read CSV asynchronously
        df = await self._read_csv_async(
            file_path,
            parse_dates=['date'],
            dtype={'id': int}
        )
        
        return AsyncFetchResult(success=True, data=df)

# Usage
async with CSVFileFetcher(name="csv_reader") as fetcher:
    result = await fetcher.fetch(file_path='data/input.csv')
```

### Batch Coordination

```python
class MultiSourceFetcher(AsyncBatchDataFetcher):
    """Coordinate multiple data sources."""
    
    def __init__(self, **kwargs):
        api_fetcher = MyAPIFetcher(
            name="api",
            base_url="https://api.example.com"
        )
        
        scraper = MyWebScraper(
            name="scraper"
        )
        
        file_fetcher = CSVFileFetcher(
            name="files"
        )
        
        super().__init__(
            name="multi_source",
            fetchers=[api_fetcher, scraper, file_fetcher],
            **kwargs
        )

# Usage - fetches from all sources concurrently
async with MultiSourceFetcher() as fetcher:
    result = await fetcher.fetch(
        endpoint='/data',
        url='https://example.com',
        file_path='data/input.csv'
    )
    
    # Result contains combined data from all sources
    print(f"Combined {len(result.data)} rows from {len(fetcher.fetchers)} sources")
```

## Error Handling

### Handling Fetch Failures

```python
async def robust_fetch():
    async with MyAPIFetcher(
        name="my_api",
        base_url="https://api.example.com"
    ) as fetcher:
        result = await fetcher.fetch(endpoint='/data')
        
        if not result.success:
            # Log error
            logger.error(f"Fetch failed: {result.error}")
            
            # Check metadata for details
            if result.metadata:
                error_type = result.metadata.get('exception_type')
                logger.error(f"Exception type: {error_type}")
            
            # Fallback to cached data or default
            return get_fallback_data()
        
        return result.data
```

### Partial Batch Failures

```python
async def handle_batch_failures():
    async with MyAPIFetcher(
        name="my_api",
        base_url="https://api.example.com"
    ) as fetcher:
        requests = [{'endpoint': f'/data/{i}'} for i in range(10)]
        results = await fetcher.fetch_batch(requests)
        
        # Separate successful and failed
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        logger.info(f"Success: {len(successful)}, Failed: {len(failed)}")
        
        # Process successful results
        if successful:
            combined_df = pd.concat([r.data for r in successful])
            return combined_df
        
        # Handle all failures
        raise RuntimeError("All batch requests failed")
```

## Performance Optimization

### Concurrent Fetching

```python
async def optimized_fetch():
    """Fetch multiple resources concurrently."""
    async with MyAPIFetcher(
        name="my_api",
        base_url="https://api.example.com",
        max_connections=50  # Allow more concurrent connections
    ) as fetcher:
        # Create 100 requests
        requests = [{'endpoint': f'/data/{i}'} for i in range(100)]
        
        # Fetch with controlled concurrency
        results = await fetcher.fetch_batch(
            requests,
            max_concurrent=20  # Process 20 at a time
        )
        
        return results
```

### Connection Reuse

```python
async def reuse_connections():
    """Reuse connections across multiple fetches."""
    # Context manager keeps connection pool alive
    async with MyAPIFetcher(
        name="my_api",
        base_url="https://api.example.com"
    ) as fetcher:
        # Multiple fetches reuse the same connection pool
        result1 = await fetcher.fetch(endpoint='/users')
        result2 = await fetcher.fetch(endpoint='/posts')
        result3 = await fetcher.fetch(endpoint='/comments')
        
        return result1, result2, result3
```

### Memory Management

```python
async def memory_efficient_fetch():
    """Process large datasets efficiently."""
    async with MyAPIFetcher(
        name="my_api",
        base_url="https://api.example.com"
    ) as fetcher:
        # Fetch in chunks
        chunk_size = 1000
        all_data = []
        
        for offset in range(0, 10000, chunk_size):
            result = await fetcher.fetch(
                endpoint='/data',
                params={'offset': offset, 'limit': chunk_size},
                save_to_file=False  # Don't save intermediate results
            )
            
            if result.success:
                # Process chunk immediately
                processed = process_chunk(result.data)
                all_data.append(processed)
                
                # Clear memory
                del result
        
        return pd.concat(all_data)
```

## Testing

### Unit Testing

```python
import pytest
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_my_fetcher():
    """Test custom fetcher."""
    with patch('httpx.AsyncClient.request') as mock_request:
        # Mock response
        mock_response = AsyncMock()
        mock_response.json.return_value = [{'id': 1, 'name': 'test'}]
        mock_response.raise_for_status = Mock()
        mock_request.return_value = mock_response
        
        # Test fetcher
        async with MyAPIFetcher(
            name="test",
            base_url="https://api.example.com"
        ) as fetcher:
            result = await fetcher.fetch(endpoint='/data')
        
        assert result.success
        assert len(result.data) == 1
```

### Integration Testing

```python
@pytest.mark.asyncio
async def test_real_api():
    """Test against real API (integration test)."""
    async with MyAPIFetcher(
        name="test",
        base_url="https://api.example.com",
        api_key=os.getenv('API_KEY')
    ) as fetcher:
        result = await fetcher.fetch(endpoint='/health')
        
        assert result.success
        assert result.data is not None
```

## Migration Guide

### From Synchronous to Async

**Before (Synchronous):**
```python
from src.data.base_fetcher import APIDataFetcher

class OldFetcher(APIDataFetcher):
    def _fetch_raw(self, **kwargs):
        response = requests.get(self.base_url + '/data')
        return FetchResult(success=True, data=pd.DataFrame(response.json()))

# Usage
fetcher = OldFetcher(name="old", base_url="https://api.example.com")
result = fetcher.fetch()
```

**After (Async):**
```python
from src.data.async_base_fetcher import AsyncAPIDataFetcher

class NewFetcher(AsyncAPIDataFetcher):
    async def _fetch_raw(self, **kwargs):
        response = await self._make_request('GET', '/data')
        return AsyncFetchResult(success=True, data=pd.DataFrame(response.json()))

# Usage
async def main():
    async with NewFetcher(name="new", base_url="https://api.example.com") as fetcher:
        result = await fetcher.fetch()

# Run
asyncio.run(main())
```

## Best Practices

### 1. Always Use Context Managers

```python
# ✅ Good - Proper cleanup
async with MyAPIFetcher(...) as fetcher:
    result = await fetcher.fetch()

# ❌ Bad - Manual cleanup required
fetcher = MyAPIFetcher(...)
await fetcher._ensure_client()
result = await fetcher.fetch()
await fetcher.close()  # Easy to forget!
```

### 2. Configure Appropriate Timeouts

```python
# ✅ Good - Reasonable timeout
fetcher = MyAPIFetcher(
    name="my_api",
    base_url="https://api.example.com",
    timeout=30.0  # 30 seconds
)

# ❌ Bad - Too short or too long
fetcher = MyAPIFetcher(timeout=1.0)   # Too short
fetcher = MyAPIFetcher(timeout=300.0)  # Too long
```

### 3. Limit Concurrent Requests

```python
# ✅ Good - Controlled concurrency
results = await fetcher.fetch_batch(
    requests,
    max_concurrent=10  # Reasonable limit
)

# ❌ Bad - Unlimited concurrency
results = await fetcher.fetch_batch(
    requests,
    max_concurrent=1000  # May overwhelm server
)
```

### 4. Handle Errors Gracefully

```python
# ✅ Good - Comprehensive error handling
async with MyAPIFetcher(...) as fetcher:
    result = await fetcher.fetch()
    if not result.success:
        logger.error(f"Fetch failed: {result.error}")
        return fallback_data()
    return result.data

# ❌ Bad - Assuming success
async with MyAPIFetcher(...) as fetcher:
    result = await fetcher.fetch()
    return result.data  # May be None!
```

### 5. Use Caching Wisely

```python
# ✅ Good - Cache stable data
fetcher = MyAPIFetcher(
    name="reference_data",
    cache_enabled=True,
    cache_ttl_hours=24  # Reference data changes daily
)

# ❌ Bad - Cache real-time data
fetcher = MyAPIFetcher(
    name="live_prices",
    cache_enabled=True,
    cache_ttl_hours=24  # Prices change every second!
)
```

## Performance Metrics

### Expected Improvements

- **Sequential → Async**: 4-6x speedup for I/O-bound operations
- **Connection Pooling**: 10-20x reduction in connection overhead
- **Batch Operations**: Near-linear scaling with concurrency
- **HTTP/2**: 30-50% reduction in latency for multiple requests

### Monitoring

```python
import time

async def measure_performance():
    start = time.time()
    
    async with MyAPIFetcher(...) as fetcher:
        results = await fetcher.fetch_batch(requests, max_concurrent=10)
    
    elapsed = time.time() - start
    success_count = sum(1 for r in results if r.success)
    
    print(f"Fetched {success_count} in {elapsed:.2f}s")
    print(f"Rate: {success_count/elapsed:.2f} requests/sec")
```

## Troubleshooting

### Common Issues

**Issue: "HTTP client not initialized"**
```python
# Solution: Use context manager or call _ensure_client()
async with fetcher:  # Automatically initializes client
    result = await fetcher.fetch()
```

**Issue: "Too many open connections"**
```python
# Solution: Reduce max_connections or max_concurrent
fetcher = MyAPIFetcher(
    max_connections=50,  # Reduce from default 100
    max_keepalive_connections=10  # Reduce from default 20
)
```

**Issue: "Timeout errors"**
```python
# Solution: Increase timeout or implement retry logic
fetcher = MyAPIFetcher(
    timeout=60.0,  # Increase timeout
    retry_config=RetryConfig(max_attempts=5)  # More retries
)
```

## Additional Resources

- [httpx Documentation](https://www.python-httpx.org/)
- [asyncio Documentation](https://docs.python.org/3/library/asyncio.html)
- [Phase 2A Roadmap](./ENHANCEMENT_PHASE_2_ROADMAP.md)
- [GitHub Issue #1](./PHASE_2A_GITHUB_ISSUES.md#issue-1)

## Support

For questions or issues:
1. Check this guide and existing documentation
2. Review test cases in `tests/test_async_data_fetchers.py`
3. Consult the Phase 2A implementation team

---

**Made with Bob** - Part of NFL Predictions Platform Enhancement Phase 2A