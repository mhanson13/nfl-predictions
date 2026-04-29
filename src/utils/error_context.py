"""
Error context management for comprehensive error handling.

Provides context managers and utilities for capturing detailed error information
including stack traces, variable states, and execution context.
"""

import sys
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable, Type
from datetime import datetime
import json
from pathlib import Path

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ErrorContext:
    """Detailed error context information."""
    
    error_type: str
    error_message: str
    timestamp: datetime
    operation: str
    stack_trace: str
    local_vars: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    recovery_attempted: bool = False
    recovery_successful: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'error_type': self.error_type,
            'error_message': self.error_message,
            'timestamp': self.timestamp.isoformat(),
            'operation': self.operation,
            'stack_trace': self.stack_trace,
            'local_vars': {k: str(v) for k, v in self.local_vars.items()},
            'metadata': self.metadata,
            'recovery_attempted': self.recovery_attempted,
            'recovery_successful': self.recovery_successful
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def save(self, path: Path) -> None:
        """Save error context to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            f.write(self.to_json())


class ErrorContextManager:
    """Manager for error contexts."""
    
    def __init__(self):
        self.contexts: List[ErrorContext] = []
        self.max_contexts = 100  # Keep last 100 errors
    
    def add_context(self, context: ErrorContext) -> None:
        """Add error context."""
        self.contexts.append(context)
        
        # Keep only recent contexts
        if len(self.contexts) > self.max_contexts:
            self.contexts = self.contexts[-self.max_contexts:]
    
    def get_recent_contexts(self, n: int = 10) -> List[ErrorContext]:
        """Get n most recent error contexts."""
        return self.contexts[-n:]
    
    def get_contexts_by_operation(self, operation: str) -> List[ErrorContext]:
        """Get all contexts for a specific operation."""
        return [ctx for ctx in self.contexts if ctx.operation == operation]
    
    def clear(self) -> None:
        """Clear all contexts."""
        self.contexts.clear()
    
    def export_to_file(self, path: Path) -> None:
        """Export all contexts to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump([ctx.to_dict() for ctx in self.contexts], f, indent=2)


# Global error context manager
_error_manager = ErrorContextManager()


def get_error_manager() -> ErrorContextManager:
    """Get global error context manager."""
    return _error_manager


@contextmanager
def error_context(
    operation: str,
    capture_locals: bool = True,
    reraise: bool = True,
    recovery_func: Optional[Callable] = None,
    **metadata
):
    """
    Context manager for capturing detailed error information.
    
    Args:
        operation: Name of operation being performed
        capture_locals: Whether to capture local variables
        reraise: Whether to reraise the exception
        recovery_func: Optional function to attempt recovery
        **metadata: Additional metadata to include
        
    Example:
        with error_context("fetch_data", source="ESPN", year=2024):
            data = fetch_espn_data()
    """
    try:
        yield
    except Exception as e:
        # Capture error details
        exc_type, exc_value, exc_tb = sys.exc_info()
        
        # Get stack trace
        stack_trace = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
        
        # Capture local variables if requested
        local_vars = {}
        if capture_locals and exc_tb:
            frame = exc_tb.tb_frame
            local_vars = {
                k: _safe_repr(v)
                for k, v in frame.f_locals.items()
                if not k.startswith('_')
            }
        
        # Create error context
        error_type_name = exc_type.__name__ if exc_type is not None else 'Unknown'
        context = ErrorContext(
            error_type=error_type_name,
            error_message=str(exc_value),
            timestamp=datetime.now(),
            operation=operation,
            stack_trace=stack_trace,
            local_vars=local_vars,
            metadata=metadata
        )
        
        # Attempt recovery if function provided
        if recovery_func:
            context.recovery_attempted = True
            try:
                recovery_func(e, context)
                context.recovery_successful = True
                logger.info(f"Recovery successful for {operation}")
            except Exception as recovery_error:
                logger.error(f"Recovery failed for {operation}: {recovery_error}")
                context.recovery_successful = False
        
        # Add to manager
        _error_manager.add_context(context)
        
        # Log error with context
        error_type_name = exc_type.__name__ if exc_type is not None else 'Unknown'
        logger.error(
            f"Error in {operation}: {error_type_name}: {exc_value}",
            extra={'error_context': context.to_dict()}
        )
        
        # Reraise if requested
        if reraise:
            raise


@contextmanager
def operation_context(
    operation: str,
    log_start: bool = True,
    log_end: bool = True,
    **metadata
):
    """
    Context manager for tracking operations with automatic error handling.
    
    Args:
        operation: Name of operation
        log_start: Whether to log operation start
        log_end: Whether to log operation end
        **metadata: Additional metadata
        
    Example:
        with operation_context("build_features", year=2024):
            features = build_features(year)
    """
    if log_start:
        logger.info(f"Starting operation: {operation}", extra=metadata)
    
    start_time = datetime.now()
    
    try:
        with error_context(operation, **metadata):
            yield
        
        if log_end:
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(
                f"Completed operation: {operation} ({duration:.2f}s)",
                extra={**metadata, 'duration': duration}
            )
    
    except Exception:
        duration = (datetime.now() - start_time).total_seconds()
        logger.error(
            f"Failed operation: {operation} ({duration:.2f}s)",
            extra={**metadata, 'duration': duration}
        )
        raise


def _safe_repr(obj: Any, max_length: int = 100) -> str:
    """
    Get safe string representation of object.
    
    Args:
        obj: Object to represent
        max_length: Maximum length of representation
        
    Returns:
        String representation
    """
    try:
        repr_str = repr(obj)
        if len(repr_str) > max_length:
            repr_str = repr_str[:max_length] + '...'
        return repr_str
    except Exception:
        return f"<{type(obj).__name__} (repr failed)>"


class ErrorHandler:
    """Configurable error handler."""
    
    def __init__(
        self,
        operation: str,
        capture_locals: bool = True,
        log_errors: bool = True,
        save_to_file: bool = False,
        error_dir: Optional[Path] = None
    ):
        """
        Initialize error handler.
        
        Args:
            operation: Operation name
            capture_locals: Whether to capture local variables
            log_errors: Whether to log errors
            save_to_file: Whether to save errors to file
            error_dir: Directory for error files
        """
        self.operation = operation
        self.capture_locals = capture_locals
        self.log_errors = log_errors
        self.save_to_file = save_to_file
        self.error_dir = error_dir or Path("logs/errors")
    
    def __enter__(self):
        """Enter context."""
        return self
    
    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        exc_tb: Optional[Any]
    ) -> bool:
        """Exit context and handle errors."""
        if exc_type is None:
            return False
        
        # Create error context
        stack_trace = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
        
        local_vars: Dict[str, str] = {}
        if self.capture_locals and exc_tb:
            frame = exc_tb.tb_frame
            local_vars = {
                k: _safe_repr(v)
                for k, v in frame.f_locals.items()
                if not k.startswith('_')
            }
        
        error_type_name = exc_type.__name__ if exc_type is not None else 'Unknown'
        context = ErrorContext(
            error_type=error_type_name,
            error_message=str(exc_value),
            timestamp=datetime.now(),
            operation=self.operation,
            stack_trace=stack_trace,
            local_vars=local_vars
        )
        
        # Add to manager
        _error_manager.add_context(context)
        
        # Log if requested
        if self.log_errors:
            error_type_name = exc_type.__name__ if exc_type is not None else 'Unknown'
            logger.error(
                f"Error in {self.operation}: {error_type_name}: {exc_value}",
                extra={'error_context': context.to_dict()}
            )
        
        # Save to file if requested
        if self.save_to_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.operation}_{timestamp}.json"
            filepath = self.error_dir / filename
            context.save(filepath)
            logger.info(f"Error context saved to {filepath}")
        
        # Don't suppress exception
        return False


def handle_errors(
    operation: str,
    capture_locals: bool = True,
    log_errors: bool = True,
    reraise: bool = True,
    default_return: Any = None
):
    """
    Decorator for error handling.
    
    Args:
        operation: Operation name
        capture_locals: Whether to capture local variables
        log_errors: Whether to log errors
        reraise: Whether to reraise exceptions
        default_return: Default value to return on error (if not reraising)
        
    Example:
        @handle_errors("fetch_data", reraise=False, default_return=None)
        def fetch_data():
            return api.get_data()
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                with error_context(
                    operation,
                    capture_locals=capture_locals,
                    reraise=reraise
                ):
                    return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.error(f"Error in {operation}: {e}")
                if not reraise:
                    return default_return
                raise
        
        return wrapper
    return decorator


class ErrorRecoveryStrategy:
    """Base class for error recovery strategies."""
    
    def can_recover(self, error: Exception, context: ErrorContext) -> bool:
        """
        Check if error can be recovered from.
        
        Args:
            error: The exception
            context: Error context
            
        Returns:
            True if recovery is possible
        """
        return False
    
    def recover(self, error: Exception, context: ErrorContext) -> Any:
        """
        Attempt to recover from error.
        
        Args:
            error: The exception
            context: Error context
            
        Returns:
            Recovery result
        """
        raise NotImplementedError


class RetryRecoveryStrategy(ErrorRecoveryStrategy):
    """Recovery strategy that retries the operation."""
    
    def __init__(self, max_retries: int = 3, delay: float = 1.0):
        self.max_retries = max_retries
        self.delay = delay
        self.retry_count = 0
    
    def can_recover(self, error: Exception, context: ErrorContext) -> bool:
        """Check if we can retry."""
        return self.retry_count < self.max_retries
    
    def recover(self, error: Exception, context: ErrorContext) -> Any:
        """Retry the operation."""
        import time
        
        self.retry_count += 1
        logger.info(f"Retry attempt {self.retry_count}/{self.max_retries}")
        time.sleep(self.delay)
        # Caller should retry the operation
        return None


class FallbackRecoveryStrategy(ErrorRecoveryStrategy):
    """Recovery strategy that uses a fallback value."""
    
    def __init__(self, fallback_value: Any):
        self.fallback_value = fallback_value
    
    def can_recover(self, error: Exception, context: ErrorContext) -> bool:
        """Always can recover with fallback."""
        return True
    
    def recover(self, error: Exception, context: ErrorContext) -> Any:
        """Return fallback value."""
        logger.info(f"Using fallback value for {context.operation}")
        return self.fallback_value

# Made with Bob
