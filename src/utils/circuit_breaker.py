"""
Circuit breaker pattern for handling cascading failures.

This module implements the circuit breaker pattern to prevent cascading
failures when external services are unavailable or slow. The circuit breaker
monitors failures and "opens" to prevent further calls when a threshold is reached.
"""

from __future__ import annotations

import functools
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional, TypeVar, Type, Tuple

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """States of a circuit breaker."""
    
    CLOSED = "closed"      # Normal operation, requests pass through
    OPEN = "open"          # Circuit is open, requests fail immediately
    HALF_OPEN = "half_open"  # Testing if service has recovered


# Alias for backward compatibility
CircuitBreakerState = CircuitState


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""
    
    def __init__(self, message: str, circuit_name: str):
        super().__init__(message)
        self.circuit_name = circuit_name


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    
    failure_threshold: int = 5  # Number of failures before opening
    success_threshold: int = 2  # Number of successes to close from half-open
    timeout: float = 60.0  # Seconds before trying half-open
    expected_exception: Type[Exception] = Exception  # Exception type to count as failure
    
    def __post_init__(self):
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if self.success_threshold < 1:
            raise ValueError("success_threshold must be >= 1")
        if self.timeout <= 0:
            raise ValueError("timeout must be > 0")


class CircuitBreaker:
    """
    Circuit breaker implementation.
    
    Monitors failures and opens circuit when threshold is reached.
    Automatically attempts recovery after timeout period.
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ):
        """
        Initialize circuit breaker.
        
        Args:
            name: Circuit breaker name for logging
            failure_threshold: Failures before opening circuit
            success_threshold: Successes needed to close from half-open
            timeout: Seconds before attempting recovery
            expected_exception: Exception type to monitor
        """
        self.name = name
        self.config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            success_threshold=success_threshold,
            timeout=timeout,
            expected_exception=expected_exception
        )
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._lock = threading.Lock()
        
        self.logger = get_logger(f"circuit_breaker.{name}")
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            return self._state
    
    @property
    def failure_count(self) -> int:
        """Get current failure count."""
        with self._lock:
            return self._failure_count
    
    @property
    def success_count(self) -> int:
        """Get current success count."""
        with self._lock:
            return self._success_count
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self._last_failure_time is None:
            return False
        
        elapsed = (datetime.now() - self._last_failure_time).total_seconds()
        return elapsed >= self.config.timeout
    
    def _record_success(self):
        """Record a successful call."""
        with self._lock:
            self._failure_count = 0
            
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                self.logger.info(
                    f"Success in half-open state "
                    f"({self._success_count}/{self.config.success_threshold})"
                )
                
                if self._success_count >= self.config.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._success_count = 0
                    self.logger.info("Circuit closed - service recovered")
    
    def _record_failure(self):
        """Record a failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = datetime.now()
            
            if self._state == CircuitState.HALF_OPEN:
                # Failed during recovery attempt
                self._state = CircuitState.OPEN
                self._success_count = 0
                self.logger.warning("Circuit re-opened - recovery failed")
            
            elif self._state == CircuitState.CLOSED:
                self.logger.warning(
                    f"Failure recorded "
                    f"({self._failure_count}/{self.config.failure_threshold})"
                )
                
                if self._failure_count >= self.config.failure_threshold:
                    self._state = CircuitState.OPEN
                    self.logger.error(
                        f"Circuit opened after {self._failure_count} failures"
                    )
    
    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Call function through circuit breaker.
        
        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerError: If circuit is open
        """
        with self._lock:
            current_state = self._state
            
            # Check if we should attempt reset
            if current_state == CircuitState.OPEN and self._should_attempt_reset():
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0
                current_state = CircuitState.HALF_OPEN
                self.logger.info("Circuit half-open - attempting recovery")
            
            # Fail fast if circuit is open
            if current_state == CircuitState.OPEN:
                raise CircuitBreakerError(
                    f"Circuit breaker '{self.name}' is open",
                    circuit_name=self.name
                )
        
        # Attempt the call
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        
        except self.config.expected_exception as e:
            self._record_failure()
            raise
    
    def can_execute(self) -> bool:
        """
        Check whether a call can currently be executed.

        Transitions OPEN → HALF_OPEN if the recovery timeout has elapsed.

        Returns
        -------
        bool
            True if the circuit is CLOSED or has just transitioned to HALF_OPEN;
            False if the circuit is still OPEN.
        """
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True
            if self._state == CircuitState.HALF_OPEN:
                return True
            # OPEN: check whether timeout has elapsed
            if self._should_attempt_reset():
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0
                self.logger.info("Circuit half-open - attempting recovery")
                return True
            return False

    def reset(self):
        """Manually reset circuit breaker to closed state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            self.logger.info("Circuit manually reset")
    
    def get_stats(self) -> dict:
        """
        Get circuit breaker statistics.
        
        Returns:
            Dictionary with current stats
        """
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "last_failure": self._last_failure_time.isoformat() if self._last_failure_time else None,
                "config": {
                    "failure_threshold": self.config.failure_threshold,
                    "success_threshold": self.config.success_threshold,
                    "timeout": self.config.timeout,
                }
            }


def circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    success_threshold: int = 2,
    timeout: float = 60.0,
    expected_exception: type = Exception
):
    """
    Decorator to wrap function with circuit breaker.
    
    Args:
        name: Circuit breaker name
        failure_threshold: Failures before opening
        success_threshold: Successes to close from half-open
        timeout: Seconds before recovery attempt
        expected_exception: Exception type to monitor
        
    Returns:
        Decorated function
        
    Example:
        @circuit_breaker(
            name="external_api",
            failure_threshold=3,
            timeout=30.0,
            expected_exception=requests.RequestException
        )
        def call_external_api():
            response = requests.get("https://api.example.com")
            return response.json()
    """
    breaker = CircuitBreaker(
        name=name,
        failure_threshold=failure_threshold,
        success_threshold=success_threshold,
        timeout=timeout,
        expected_exception=expected_exception
    )
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return breaker.call(func, *args, **kwargs)
        
        # Attach breaker to function for inspection/reset
        wrapper.circuit_breaker = breaker  # type: ignore
        return wrapper
    
    return decorator


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""
    
    def __init__(self):
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()
    
    def register(self, breaker: CircuitBreaker):
        """
        Register a circuit breaker.
        
        Args:
            breaker: Circuit breaker to register
        """
        with self._lock:
            self._breakers[breaker.name] = breaker
    
    def get(self, name: str) -> Optional[CircuitBreaker]:
        """
        Get circuit breaker by name.
        
        Args:
            name: Circuit breaker name
            
        Returns:
            Circuit breaker or None
        """
        with self._lock:
            return self._breakers.get(name)
    
    def get_or_create(
        self,
        name: str,
        **config_kwargs
    ) -> CircuitBreaker:
        """
        Get existing circuit breaker or create new one.
        
        Args:
            name: Circuit breaker name
            **config_kwargs: Configuration for new breaker
            
        Returns:
            Circuit breaker
        """
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name, **config_kwargs)
            return self._breakers[name]
    
    def reset_all(self):
        """Reset all registered circuit breakers."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()
    
    def get_all_stats(self) -> dict[str, dict]:
        """
        Get statistics for all circuit breakers.
        
        Returns:
            Dictionary mapping names to stats
        """
        with self._lock:
            return {
                name: breaker.get_stats()
                for name, breaker in self._breakers.items()
            }


# Global registry
_registry = CircuitBreakerRegistry()


def get_circuit_breaker(name: str, **config_kwargs) -> CircuitBreaker:
    """
    Get or create a circuit breaker from global registry.
    
    Args:
        name: Circuit breaker name
        **config_kwargs: Configuration if creating new breaker
        
    Returns:
        Circuit breaker
    """
    return _registry.get_or_create(name, **config_kwargs)


def reset_all_circuit_breakers():
    """Reset all circuit breakers in global registry."""
    _registry.reset_all()


def get_all_circuit_breaker_stats() -> dict[str, dict]:
    """
    Get statistics for all circuit breakers.
    
    Returns:
        Dictionary mapping names to stats
    """
    return _registry.get_all_stats()

# Made with Bob
