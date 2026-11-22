# ADR 0003: Retry Pattern with Exponential Backoff

## Status
Accepted

## Context
The NFL predictions platform makes numerous external API calls to services like ESPN, Sportradar, weather APIs, and web scraping targets. These calls can fail due to:

- Temporary network issues
- Rate limiting (429 errors)
- Service unavailability (503 errors)
- Timeout errors
- Transient server errors (500-502)

Without retry logic, a single transient failure would cause the entire pipeline to fail, requiring manual intervention and re-running.

## Decision
We implemented a comprehensive **Retry Pattern** with exponential backoff and jitter:

### Components

1. **@retry Decorator**: General-purpose retry with exponential backoff
2. **@retry_with_timeout**: Combines retry with timeout enforcement
3. **@retry_on_rate_limit**: Specialized for rate-limited APIs (429 errors)
4. **@retry_async**: For async operations
5. **RetryContext**: Context manager for manual retry logic
6. **RetryConfig**: Configuration dataclass for retry parameters

### Key Features

- **Exponential Backoff**: Delays increase exponentially (1s, 2s, 4s, 8s, ...)
- **Jitter**: Random variation to prevent thundering herd
- **Max Delay Cap**: Prevents delays from growing too large
- **Selective Retries**: Only retry specific exception types
- **Configurable**: All parameters can be customized
- **Logging**: Detailed logging of retry attempts
- **Thread-Safe**: Safe for concurrent use

## Implementation

### Basic Usage
```python
@retry(max_attempts=3, delay=1.0, backoff_factor=2.0)
def fetch_data():
    response = requests.get("https://api.example.com/data")
    response.raise_for_status()
    return response.json()
```

### Rate Limit Handling
```python
@retry_on_rate_limit(max_attempts=5, delay=60.0)
def fetch_with_rate_limit():
    return api_client.get_data()
```

### Manual Retry Control
```python
with RetryContext(max_attempts=3, delay=1.0) as ctx:
    try:
        result = risky_operation()
    except Exception as e:
        if ctx.should_retry(e):
            continue
        raise
```

## Retry Strategy

### Default Configuration
- **Max Attempts**: 3
- **Initial Delay**: 1.0 seconds
- **Backoff Factor**: 2.0 (exponential)
- **Max Delay**: 60.0 seconds
- **Jitter**: Enabled (±25% random variation)

### Delay Calculation
```
delay = min(initial_delay * (backoff_factor ^ attempt), max_delay)
if jitter:
    delay *= random.uniform(0.75, 1.25)
```

### Retryable Exceptions
By default, retries on:
- `requests.exceptions.RequestException`
- `requests.exceptions.Timeout`
- `requests.exceptions.ConnectionError`
- `urllib3.exceptions.HTTPError`

Can be customized per decorator.

## Consequences

### Positive
- **Resilience**: Handles transient failures automatically
- **Reduced Manual Intervention**: Fewer pipeline failures requiring restarts
- **Rate Limit Compliance**: Respects API rate limits with appropriate delays
- **Configurable**: Easy to adjust retry behavior per use case
- **Logging**: Clear visibility into retry attempts
- **Performance**: Exponential backoff prevents overwhelming failing services
- **Thundering Herd Prevention**: Jitter prevents synchronized retries

### Negative
- **Latency**: Retries add latency to operations
- **Resource Usage**: Failed attempts consume resources
- **Complexity**: Adds another layer to debug
- **False Success**: May mask underlying issues if retries always succeed

### Neutral
- **File Location**: `src/utils/retry.py` (368 lines)
- **Dependencies**: Standard library only (no external dependencies)

## Integration Points

### With Circuit Breaker (ADR 0004)
Retry and circuit breaker work together:
1. Retry handles transient failures
2. Circuit breaker prevents cascading failures
3. Use retry inside circuit breaker for best results

```python
@circuit_breaker("api_service")
@retry(max_attempts=3)
def fetch_data():
    return api.get_data()
```

### With Data Fetchers (ADR 0002)
Data fetchers can use retry decorators:

```python
class MyAPIFetcher(APIDataFetcher):
    @retry(max_attempts=3, delay=1.0)
    def _fetch_raw(self, **kwargs):
        return self._make_request()
```

### With Connection Pooling (ADR 0005)
Retry works seamlessly with pooled connections:
- Failed requests return connections to pool
- Retries use fresh connections from pool
- No connection leaks on retry

## Testing Strategy

- Unit tests for retry logic
- Mock failures to test retry behavior
- Verify exponential backoff timing
- Test jitter randomization
- Validate exception filtering
- Test max attempts enforcement

## Monitoring

Retry attempts should be monitored:
- Count of retry attempts per endpoint
- Success rate after retries
- Average retry delay
- Max retries reached (indicates persistent issues)

## Best Practices

1. **Use Appropriate Max Attempts**: 3-5 for most cases
2. **Set Reasonable Delays**: Start with 1-2 seconds
3. **Enable Jitter**: Prevents thundering herd
4. **Log Retry Attempts**: For debugging and monitoring
5. **Combine with Circuit Breaker**: For comprehensive resilience
6. **Don't Retry Non-Idempotent Operations**: Unless safe to do so
7. **Set Timeouts**: Prevent indefinite waits

## Alternatives Considered

### 1. No Retry Logic
**Rejected**: Too fragile for production use

### 2. Fixed Delay Retry
**Rejected**: Can overwhelm failing services

### 3. Linear Backoff
**Rejected**: Not aggressive enough for rate limiting

### 4. Third-Party Library (tenacity, backoff)
**Considered**: Decided to implement custom solution for:
- Full control over behavior
- No external dependencies
- Integration with existing logging/monitoring
- Simpler for team to understand and modify

## Future Enhancements

- Adaptive retry based on error type
- Retry budget to prevent excessive retries
- Distributed retry coordination
- Metrics collection and dashboards

## References
- Implementation: `src/utils/retry.py`
- Tests: `tests/test_retry_circuit_breaker.py`
- Related ADRs:
  - [0002-abstract-base-fetchers.md](0002-abstract-base-fetchers.md)
  - [0004-circuit-breaker-pattern.md](0004-circuit-breaker-pattern.md)
  - [0005-connection-pooling.md](0005-connection-pooling.md)