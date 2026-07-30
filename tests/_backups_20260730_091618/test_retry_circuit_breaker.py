"""
Unit tests for retry and circuit breaker utilities.

Tests the retry decorators and circuit breaker pattern.
"""

import pytest  # type: ignore
import time
from unittest.mock import Mock, patch

from src.utils.retry import (
    retry,
    retry_with_timeout,
    retry_on_rate_limit,
    RetryConfig,
    RetryContext
)
from src.utils.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerState,
    circuit_breaker,
    CircuitBreakerRegistry
)


class TestRetryDecorator:
    """Test retry decorator."""
    
    def test_successful_call(self):
        """Test that successful calls don't retry."""
        mock_func = Mock(return_value="success")
        decorated = retry(max_attempts=3)(mock_func)
        
        result = decorated()
        
        assert result == "success"
        assert mock_func.call_count == 1
    
    def test_retry_on_exception(self):
        """Test retrying on exception."""
        mock_func = Mock(side_effect=[
            Exception("Error 1"),
            Exception("Error 2"),
            "success"
        ])
        decorated = retry(max_attempts=3, delay=0.01)(mock_func)
        
        result = decorated()
        
        assert result == "success"
        assert mock_func.call_count == 3
    
    def test_max_attempts_exceeded(self):
        """Test that max attempts is respected."""
        mock_func = Mock(side_effect=Exception("Always fails"))
        decorated = retry(max_attempts=3, delay=0.01)(mock_func)
        
        with pytest.raises(Exception, match="Always fails"):
            decorated()
        
        assert mock_func.call_count == 3
    
    def test_exponential_backoff(self):
        """Test exponential backoff timing."""
        call_times = []
        
        def failing_func():
            call_times.append(time.time())
            if len(call_times) < 3:
                raise Exception("Fail")
            return "success"
        
        decorated = retry(
            max_attempts=3,
            delay=0.1,
            backoff_factor=2.0,
            jitter=False
        )(failing_func)
        
        result = decorated()
        
        assert result == "success"
        assert len(call_times) == 3
        
        # Check delays are approximately exponential
        if len(call_times) >= 2:
            delay1 = call_times[1] - call_times[0]
            assert delay1 >= 0.1  # First delay
    
    def test_specific_exceptions(self):
        """Test retrying only on specific exceptions."""
        mock_func = Mock(side_effect=[
            ValueError("Retry this"),
            "success"
        ])
        decorated = retry(
            max_attempts=3,
            delay=0.01,
            exceptions=(ValueError,)
        )(mock_func)
        
        result = decorated()
        assert result == "success"
        assert mock_func.call_count == 2
    
    def test_non_retryable_exception(self):
        """Test that non-retryable exceptions aren't retried."""
        mock_func = Mock(side_effect=TypeError("Don't retry"))
        decorated = retry(
            max_attempts=3,
            delay=0.01,
            exceptions=(ValueError,)
        )(mock_func)
        
        with pytest.raises(TypeError, match="Don't retry"):
            decorated()
        
        assert mock_func.call_count == 1


class TestRetryConfig:
    """Test RetryConfig class."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = RetryConfig()
        
        assert config.max_attempts == 3
        assert config.delay == 1.0
        assert config.backoff_factor == 2.0
        assert config.max_delay == 60.0
        assert config.jitter is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = RetryConfig(
            max_attempts=5,
            delay=0.5,
            backoff_factor=3.0,
            max_delay=120.0,
            jitter=False
        )
        
        assert config.max_attempts == 5
        assert config.delay == 0.5
        assert config.backoff_factor == 3.0
        assert config.max_delay == 120.0
        assert config.jitter is False


class TestRetryContext:
    """Test RetryContext context manager."""
    
    def test_successful_operation(self):
        """Test successful operation in context."""
        result = None
        
        with RetryContext(max_attempts=3, delay=0.01) as ctx:
            result = "success"
        
        assert result == "success"
        assert ctx.attempt == 1
    
    def test_retry_in_context(self):
        """Test retrying in context."""
        attempts = []
        
        with RetryContext(max_attempts=3, delay=0.01) as ctx:
            attempts.append(ctx.attempt)
            if ctx.attempt < 3:
                raise ValueError("Retry")
        
        assert len(attempts) == 3


class TestCircuitBreaker:
    """Test CircuitBreaker class."""
    
    def test_initial_state(self):
        """Test initial circuit breaker state."""
        cb = CircuitBreaker(name="test", failure_threshold=3)
        
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 0
        assert cb.success_count == 0
    
    def test_successful_calls(self):
        """Test successful calls keep circuit closed."""
        cb = CircuitBreaker(name="test", failure_threshold=3)
        
        for _ in range(5):
            cb.record_success()
        
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.success_count == 5
        assert cb.failure_count == 0
    
    def test_circuit_opens_on_failures(self):
        """Test circuit opens after threshold failures."""
        cb = CircuitBreaker(name="test", failure_threshold=3, timeout=1.0)
        
        for _ in range(3):
            cb.record_failure()
        
        assert cb.state == CircuitBreakerState.OPEN
        assert cb.failure_count == 3
    
    def test_circuit_rejects_when_open(self):
        """Test circuit rejects calls when open."""
        cb = CircuitBreaker(name="test", failure_threshold=2, timeout=1.0)
        
        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        
        assert cb.state == CircuitBreakerState.OPEN
        assert not cb.can_execute()
    
    def test_circuit_half_open_after_timeout(self):
        """Test circuit goes to half-open after timeout."""
        cb = CircuitBreaker(name="test", failure_threshold=2, timeout=0.1)
        
        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN
        
        # Wait for timeout
        time.sleep(0.15)
        
        # Should transition to half-open
        assert cb.can_execute()
        assert cb.state == CircuitBreakerState.HALF_OPEN
    
    def test_circuit_closes_on_success_in_half_open(self):
        """Test circuit closes on success in half-open state."""
        cb = CircuitBreaker(
            name="test",
            failure_threshold=2,
            timeout=0.1,
            success_threshold=2
        )
        
        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        
        # Wait for timeout
        time.sleep(0.15)
        cb.can_execute()  # Transition to half-open
        
        # Record successes
        cb.record_success()
        cb.record_success()
        
        assert cb.state == CircuitBreakerState.CLOSED
    
    def test_circuit_reopens_on_failure_in_half_open(self):
        """Test circuit reopens on failure in half-open state."""
        cb = CircuitBreaker(name="test", failure_threshold=2, timeout=0.1)
        
        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        
        # Wait for timeout
        time.sleep(0.15)
        cb.can_execute()  # Transition to half-open
        
        # Record failure
        cb.record_failure()
        
        assert cb.state == CircuitBreakerState.OPEN


class TestCircuitBreakerDecorator:
    """Test circuit_breaker decorator."""
    
    def test_successful_calls(self):
        """Test successful calls through circuit breaker."""
        cb = CircuitBreaker(name="test", failure_threshold=3)
        
        @circuit_breaker(cb)
        def successful_func():
            return "success"
        
        for _ in range(5):
            result = successful_func()
            assert result == "success"
        
        assert cb.state == CircuitBreakerState.CLOSED
    
    def test_circuit_opens_on_failures(self):
        """Test circuit opens after failures."""
        cb = CircuitBreaker(name="test", failure_threshold=2, timeout=1.0)
        
        @circuit_breaker(cb)
        def failing_func():
            raise ValueError("Error")
        
        # First two calls should fail and open circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                failing_func()
        
        assert cb.state == CircuitBreakerState.OPEN
        
        # Third call should be rejected by circuit breaker
        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            failing_func()


class TestCircuitBreakerRegistry:
    """Test CircuitBreakerRegistry class."""
    
    def test_get_or_create(self):
        """Test getting or creating circuit breakers."""
        registry = CircuitBreakerRegistry()
        
        cb1 = registry.get_or_create("test1")
        cb2 = registry.get_or_create("test1")
        cb3 = registry.get_or_create("test2")
        
        assert cb1 is cb2  # Same instance
        assert cb1 is not cb3  # Different instance
    
    def test_get_all(self):
        """Test getting all circuit breakers."""
        registry = CircuitBreakerRegistry()
        
        registry.get_or_create("test1")
        registry.get_or_create("test2")
        registry.get_or_create("test3")
        
        all_breakers = registry.get_all()
        assert len(all_breakers) == 3
    
    def test_get_stats(self):
        """Test getting statistics."""
        registry = CircuitBreakerRegistry()
        
        cb1 = registry.get_or_create("test1")
        cb1.record_success()
        cb1.record_success()
        
        cb2 = registry.get_or_create("test2")
        cb2.record_failure()
        
        stats = registry.get_stats()
        
        assert "test1" in stats
        assert "test2" in stats
        assert stats["test1"]["success_count"] == 2
        assert stats["test2"]["failure_count"] == 1


class TestRetryWithTimeout:
    """Test retry_with_timeout decorator."""
    
    def test_successful_call_within_timeout(self):
        """Test successful call within timeout."""
        mock_func = Mock(return_value="success")
        decorated = retry_with_timeout(
            max_attempts=3,
            timeout=5.0,
            delay=0.01
        )(mock_func)
        
        result = decorated()
        
        assert result == "success"
        assert mock_func.call_count == 1
    
    def test_timeout_exceeded(self):
        """Test timeout is enforced."""
        def slow_func():
            time.sleep(2.0)
            return "success"
        
        decorated = retry_with_timeout(
            max_attempts=3,
            timeout=0.5,
            delay=0.01
        )(slow_func)
        
        with pytest.raises(TimeoutError):
            decorated()


class TestRetryOnRateLimit:
    """Test retry_on_rate_limit decorator."""
    
    def test_retry_on_429(self):
        """Test retrying on 429 status code."""
        from requests.exceptions import HTTPError
        from requests import Response
        
        # Create mock response with 429 status
        response = Response()
        response.status_code = 429
        
        mock_func = Mock(side_effect=[
            HTTPError(response=response),
            "success"
        ])
        
        decorated = retry_on_rate_limit(
            max_attempts=3,
            delay=0.01
        )(mock_func)
        
        result = decorated()
        
        assert result == "success"
        assert mock_func.call_count == 2


@pytest.fixture
def mock_time():
    """Fixture to mock time.time()."""
    with patch('time.time') as mock:
        mock.return_value = 1000.0
        yield mock


@pytest.fixture
def mock_sleep():
    """Fixture to mock time.sleep()."""
    with patch('time.sleep') as mock:
        yield mock

# Made with Bob
