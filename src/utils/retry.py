"""
Retry decorators and utilities for handling transient failures.

This module provides decorators and utilities for automatically retrying
operations that may fail due to transient issues like network errors,
rate limiting, or temporary service unavailability.
"""

from __future__ import annotations

import functools
import random
import time
from typing import Any, Callable, Optional, Tuple, Type, TypeVar, Union, Coroutine, cast

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar('T')
P = TypeVar('P')


class RetryError(Exception):
    """Raised when all retry attempts are exhausted."""
    
    def __init__(self, message: str, last_exception: Optional[Exception] = None):
        super().__init__(message)
        self.last_exception = last_exception


class RetryConfig:
    """Configuration for retry behavior."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        exceptions: Tuple[Type[Exception], ...] = (Exception,)
    ):
        """
        Initialize retry configuration.
        
        Args:
            max_attempts: Maximum number of retry attempts
            initial_delay: Initial delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
            exponential_base: Base for exponential backoff
            jitter: Add random jitter to delays
            exceptions: Tuple of exception types to retry on
        """
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.exceptions = exceptions
    
    @property
    def delay(self) -> float:
        """Alias for initial_delay for backward compatibility."""
        return self.initial_delay
    
    @property
    def backoff_factor(self) -> float:
        """Alias for exponential_base for backward compatibility."""
        return self.exponential_base
    
    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for a given attempt number.
        
        Args:
            attempt: Current attempt number (0-indexed)
            
        Returns:
            Delay in seconds
        """
        # Exponential backoff
        delay = min(
            self.initial_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        
        # Add jitter to prevent thundering herd
        if self.jitter:
            delay = delay * (0.5 + random.random())
        
        return delay


def retry(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
    on_retry: Optional[Callable[[Exception, int], None]] = None
):
    """
    Decorator to retry a function on failure with exponential backoff.
    
    Args:
        max_attempts: Maximum number of attempts (including initial)
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential_base: Base for exponential backoff
        jitter: Add random jitter to delays
        exceptions: Exception type(s) to retry on
        on_retry: Optional callback function called on each retry
        
    Returns:
        Decorated function
        
    Example:
        @retry(max_attempts=3, initial_delay=1.0, exceptions=(requests.RequestException,))
        def fetch_data(url: str) -> dict:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
    """
    if not isinstance(exceptions, tuple):
        exceptions = (exceptions,)
    
    config = RetryConfig(
        max_attempts=max_attempts,
        initial_delay=initial_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        jitter=jitter,
        exceptions=exceptions
    )
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)
                
                except config.exceptions as e:
                    last_exception = e
                    
                    if attempt == config.max_attempts - 1:
                        # Last attempt failed
                        logger.error(
                            f"{func.__name__} failed after {config.max_attempts} attempts: {e}"
                        )
                        raise RetryError(
                            f"Failed after {config.max_attempts} attempts",
                            last_exception=e
                        ) from e
                    
                    # Calculate delay and retry
                    delay = config.calculate_delay(attempt)
                    logger.warning(
                        f"{func.__name__} attempt {attempt + 1}/{config.max_attempts} "
                        f"failed: {e}. Retrying in {delay:.2f}s..."
                    )
                    
                    # Call retry callback if provided
                    if on_retry:
                        try:
                            on_retry(e, attempt + 1)
                        except Exception as callback_error:
                            logger.error(f"Retry callback failed: {callback_error}")
                    
                    time.sleep(delay)
            
            # Should never reach here, but just in case
            raise RetryError(
                f"Failed after {config.max_attempts} attempts",
                last_exception=last_exception
            )
        
        return wrapper
    return decorator


def retry_with_timeout(
    max_attempts: int = 3,
    timeout: float = 30.0,
    **retry_kwargs: Any
):
    """
    Decorator combining retry logic with timeout.
    
    Args:
        max_attempts: Maximum number of attempts
        timeout: Timeout for each attempt in seconds
        **retry_kwargs: Additional arguments for retry decorator
        
    Returns:
        Decorated function
        
    Example:
        @retry_with_timeout(max_attempts=3, timeout=10.0)
        def slow_operation():
            # ... operation that might timeout
            pass
    """
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Operation timed out")
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @retry(max_attempts=max_attempts, **retry_kwargs)
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Set timeout alarm
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout))
            
            try:
                result = func(*args, **kwargs)
                signal.alarm(0)  # Cancel alarm
                return result
            except TimeoutError:
                signal.alarm(0)  # Cancel alarm
                raise
        
        return wrapper
    return decorator


class RetryContext:
    """Context manager for retry logic."""
    
    def __init__(self, config: RetryConfig):
        """
        Initialize retry context.
        
        Args:
            config: Retry configuration
        """
        self.config = config
        self.attempt = 0
        self.last_exception: Optional[Exception] = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            return True
        
        if not isinstance(exc_val, self.config.exceptions):
            return False
        
        self.last_exception = exc_val
        self.attempt += 1
        
        if self.attempt >= self.config.max_attempts:
            logger.error(
                f"Operation failed after {self.config.max_attempts} attempts: {exc_val}"
            )
            return False
        
        delay = self.config.calculate_delay(self.attempt - 1)
        logger.warning(
            f"Attempt {self.attempt}/{self.config.max_attempts} failed: {exc_val}. "
            f"Retrying in {delay:.2f}s..."
        )
        time.sleep(delay)
        return True


def retry_on_rate_limit(
    max_attempts: int = 5,
    initial_delay: float = 60.0,
    rate_limit_exceptions: Tuple[Type[Exception], ...] = (Exception,)
):
    """
    Specialized retry decorator for rate-limited APIs.
    
    Uses longer delays suitable for rate limiting scenarios.
    
    Args:
        max_attempts: Maximum number of attempts
        initial_delay: Initial delay (default 60s for rate limits)
        rate_limit_exceptions: Exception types indicating rate limiting
        
    Returns:
        Decorated function
        
    Example:
        @retry_on_rate_limit(max_attempts=5, rate_limit_exceptions=(RateLimitError,))
        def call_rate_limited_api():
            # ... API call
            pass
    """
    return retry(
        max_attempts=max_attempts,
        initial_delay=initial_delay,
        max_delay=300.0,  # 5 minutes max
        exponential_base=2.0,
        jitter=True,
        exceptions=rate_limit_exceptions
    )


def retry_async(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception
) -> Callable[[Callable[..., Coroutine[Any, Any, T]]], Callable[..., Coroutine[Any, Any, T]]]:
    """
    Async version of retry decorator.
    
    Args:
        max_attempts: Maximum number of attempts
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential_base: Base for exponential backoff
        jitter: Add random jitter to delays
        exceptions: Exception type(s) to retry on
        
    Returns:
        Decorated async function
        
    Example:
        @retry_async(max_attempts=3, exceptions=(aiohttp.ClientError,))
        async def fetch_data_async(url: str):
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    return await response.json()
    """
    import asyncio
    
    if not isinstance(exceptions, tuple):
        exceptions = (exceptions,)
    
    config = RetryConfig(
        max_attempts=max_attempts,
        initial_delay=initial_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        jitter=jitter,
        exceptions=exceptions
    )
    
    def decorator(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., Coroutine[Any, Any, T]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Optional[Exception] = None
            
            for attempt in range(config.max_attempts):
                try:
                    return await func(*args, **kwargs)
                
                except config.exceptions as e:
                    last_exception = e
                    
                    if attempt == config.max_attempts - 1:
                        logger.error(
                            f"{func.__name__} failed after {config.max_attempts} attempts: {e}"
                        )
                        raise RetryError(
                            f"Failed after {config.max_attempts} attempts",
                            last_exception=e
                        ) from e
                    
                    delay = config.calculate_delay(attempt)
                    logger.warning(
                        f"{func.__name__} attempt {attempt + 1}/{config.max_attempts} "
                        f"failed: {e}. Retrying in {delay:.2f}s..."
                    )
                    
                    await asyncio.sleep(delay)
            
            raise RetryError(
                f"Failed after {config.max_attempts} attempts",
                last_exception=last_exception
            )
        
        return wrapper
    return decorator

# Made with Bob
