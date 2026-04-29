"""
Pipeline recovery strategies for data pipeline failures.

Provides comprehensive recovery mechanisms for different types of pipeline failures,
including data source failures, transformation errors, and validation issues.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable, Type
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from enum import Enum

from src.utils.logging_config import get_logger
from src.utils.error_context import ErrorContext, error_context

logger = get_logger(__name__)


class RecoveryAction(Enum):
    """Types of recovery actions."""
    RETRY = "retry"
    SKIP = "skip"
    USE_CACHE = "use_cache"
    USE_FALLBACK = "use_fallback"
    USE_PARTIAL = "use_partial"
    ABORT = "abort"


@dataclass
class RecoveryResult:
    """Result of a recovery attempt."""
    
    success: bool
    action: RecoveryAction
    data: Optional[Any] = None
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'success': self.success,
            'action': self.action.value,
            'message': self.message,
            'metadata': self.metadata,
            'has_data': self.data is not None
        }


class PipelineRecoveryStrategy(ABC):
    """Base class for pipeline recovery strategies."""
    
    def __init__(self, name: str):
        """
        Initialize recovery strategy.
        
        Args:
            name: Strategy name
        """
        self.name = name
        self.logger = get_logger(f"{__name__}.{name}")
    
    @abstractmethod
    def can_recover(self, error: Exception, context: Dict[str, Any]) -> bool:
        """
        Check if this strategy can recover from the error.
        
        Args:
            error: The exception that occurred
            context: Context information about the failure
            
        Returns:
            True if recovery is possible
        """
        pass
    
    @abstractmethod
    def recover(self, error: Exception, context: Dict[str, Any]) -> RecoveryResult:
        """
        Attempt to recover from the error.
        
        Args:
            error: The exception that occurred
            context: Context information about the failure
            
        Returns:
            Recovery result
        """
        pass


class CacheRecoveryStrategy(PipelineRecoveryStrategy):
    """Recovery strategy using cached data."""
    
    def __init__(self, cache_dir: Path, max_age_hours: int = 24):
        """
        Initialize cache recovery strategy.
        
        Args:
            cache_dir: Directory containing cached data
            max_age_hours: Maximum age of cache to use (hours)
        """
        super().__init__("cache_recovery")
        self.cache_dir = Path(cache_dir)
        self.max_age_hours = max_age_hours
    
    def can_recover(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Check if cached data is available."""
        cache_key = context.get('cache_key')
        if not cache_key:
            return False
        
        cache_file = self.cache_dir / f"{cache_key}.parquet"
        if not cache_file.exists():
            return False
        
        # Check cache age
        file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        return file_age < timedelta(hours=self.max_age_hours)
    
    def recover(self, error: Exception, context: Dict[str, Any]) -> RecoveryResult:
        """Recover using cached data."""
        cache_key = context.get('cache_key')
        cache_file = self.cache_dir / f"{cache_key}.parquet"
        
        try:
            data = pd.read_parquet(cache_file)
            file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            
            self.logger.info(
                f"Recovered from cache: {cache_key} "
                f"(age: {file_age.total_seconds() / 3600:.1f}h)"
            )
            
            return RecoveryResult(
                success=True,
                action=RecoveryAction.USE_CACHE,
                data=data,
                message=f"Used cached data from {file_age.total_seconds() / 3600:.1f}h ago",
                metadata={'cache_file': str(cache_file), 'cache_age_hours': file_age.total_seconds() / 3600}
            )
        
        except Exception as e:
            self.logger.error(f"Failed to load cache: {e}")
            return RecoveryResult(
                success=False,
                action=RecoveryAction.USE_CACHE,
                message=f"Cache load failed: {e}"
            )


class FallbackDataStrategy(PipelineRecoveryStrategy):
    """Recovery strategy using fallback data sources."""
    
    def __init__(self, fallback_sources: Dict[str, Callable]):
        """
        Initialize fallback data strategy.
        
        Args:
            fallback_sources: Map of source names to fallback functions
        """
        super().__init__("fallback_data")
        self.fallback_sources = fallback_sources
    
    def can_recover(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Check if fallback source is available."""
        source = context.get('source')
        return source in self.fallback_sources
    
    def recover(self, error: Exception, context: Dict[str, Any]) -> RecoveryResult:
        """Recover using fallback data source."""
        source = context.get('source')
        
        # Ensure source is not None
        if source is None:
            return RecoveryResult(
                success=False,
                action=RecoveryAction.USE_FALLBACK,
                message="No source specified in context"
            )
        
        # Check if fallback exists for this source
        if source not in self.fallback_sources:
            return RecoveryResult(
                success=False,
                action=RecoveryAction.USE_FALLBACK,
                message=f"No fallback available for source: {source}"
            )
        
        fallback_func = self.fallback_sources[source]
        
        try:
            self.logger.info(f"Attempting fallback for source: {source}")
            data = fallback_func(**context.get('params', {}))
            
            return RecoveryResult(
                success=True,
                action=RecoveryAction.USE_FALLBACK,
                data=data,
                message=f"Used fallback source for {source}",
                metadata={'fallback_source': source}
            )
        
        except Exception as e:
            self.logger.error(f"Fallback failed for {source}: {e}")
            return RecoveryResult(
                success=False,
                action=RecoveryAction.USE_FALLBACK,
                message=f"Fallback failed: {e}"
            )


class PartialDataStrategy(PipelineRecoveryStrategy):
    """Recovery strategy using partial/incomplete data."""
    
    def __init__(self, min_completeness: float = 0.5):
        """
        Initialize partial data strategy.
        
        Args:
            min_completeness: Minimum data completeness ratio (0-1)
        """
        super().__init__("partial_data")
        self.min_completeness = min_completeness
    
    def can_recover(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Check if partial data is available and sufficient."""
        partial_data = context.get('partial_data')
        if partial_data is None:
            return False
        
        if isinstance(partial_data, pd.DataFrame):
            expected_rows = context.get('expected_rows', 0)
            if expected_rows > 0:
                completeness = len(partial_data) / expected_rows
                return completeness >= self.min_completeness
        
        return True
    
    def recover(self, error: Exception, context: Dict[str, Any]) -> RecoveryResult:
        """Recover using partial data."""
        partial_data = context.get('partial_data')
        
        if isinstance(partial_data, pd.DataFrame):
            expected_rows = context.get('expected_rows', len(partial_data))
            completeness = len(partial_data) / expected_rows if expected_rows > 0 else 1.0
            
            self.logger.warning(
                f"Using partial data: {len(partial_data)}/{expected_rows} rows "
                f"({completeness:.1%} complete)"
            )
            
            return RecoveryResult(
                success=True,
                action=RecoveryAction.USE_PARTIAL,
                data=partial_data,
                message=f"Used partial data ({completeness:.1%} complete)",
                metadata={
                    'rows': len(partial_data),
                    'expected_rows': expected_rows,
                    'completeness': completeness
                }
            )
        
        return RecoveryResult(
            success=True,
            action=RecoveryAction.USE_PARTIAL,
            data=partial_data,
            message="Used partial data"
        )


class SkipStepStrategy(PipelineRecoveryStrategy):
    """Recovery strategy that skips non-critical pipeline steps."""
    
    def __init__(self, skippable_steps: List[str]):
        """
        Initialize skip step strategy.
        
        Args:
            skippable_steps: List of step names that can be skipped
        """
        super().__init__("skip_step")
        self.skippable_steps = set(skippable_steps)
    
    def can_recover(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Check if step can be skipped."""
        step = context.get('step')
        return step in self.skippable_steps
    
    def recover(self, error: Exception, context: Dict[str, Any]) -> RecoveryResult:
        """Skip the failed step."""
        step = context.get('step')
        
        self.logger.warning(f"Skipping non-critical step: {step}")
        
        return RecoveryResult(
            success=True,
            action=RecoveryAction.SKIP,
            message=f"Skipped non-critical step: {step}",
            metadata={'skipped_step': step}
        )


class RetryWithBackoffStrategy(PipelineRecoveryStrategy):
    """Recovery strategy with retry and exponential backoff."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        """
        Initialize retry strategy.
        
        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay between retries (seconds)
        """
        super().__init__("retry_backoff")
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.retry_counts: Dict[str, int] = {}
    
    def can_recover(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Check if retries are available."""
        operation = context.get('operation', 'unknown')
        retry_count = self.retry_counts.get(operation, 0)
        
        # Check if error is retryable
        retryable_errors = (
            ConnectionError,
            TimeoutError,
            IOError,
        )
        
        return (
            isinstance(error, retryable_errors) and
            retry_count < self.max_retries
        )
    
    def recover(self, error: Exception, context: Dict[str, Any]) -> RecoveryResult:
        """Retry the operation."""
        import time
        
        operation = context.get('operation', 'unknown')
        retry_count = self.retry_counts.get(operation, 0)
        self.retry_counts[operation] = retry_count + 1
        
        # Calculate delay with exponential backoff
        delay = self.base_delay * (2 ** retry_count)
        
        self.logger.info(
            f"Retry {retry_count + 1}/{self.max_retries} for {operation} "
            f"after {delay}s delay"
        )
        
        time.sleep(delay)
        
        return RecoveryResult(
            success=True,
            action=RecoveryAction.RETRY,
            message=f"Retry attempt {retry_count + 1}/{self.max_retries}",
            metadata={
                'retry_count': retry_count + 1,
                'max_retries': self.max_retries,
                'delay': delay
            }
        )


class PipelineRecoveryManager:
    """Manager for pipeline recovery strategies."""
    
    def __init__(self):
        """Initialize recovery manager."""
        self.strategies: List[PipelineRecoveryStrategy] = []
        self.recovery_history: List[Dict[str, Any]] = []
    
    def add_strategy(self, strategy: PipelineRecoveryStrategy) -> None:
        """
        Add a recovery strategy.
        
        Args:
            strategy: Recovery strategy to add
        """
        self.strategies.append(strategy)
        logger.info(f"Added recovery strategy: {strategy.name}")
    
    def attempt_recovery(
        self,
        error: Exception,
        context: Dict[str, Any]
    ) -> Optional[RecoveryResult]:
        """
        Attempt recovery using available strategies.
        
        Args:
            error: The exception that occurred
            context: Context information about the failure
            
        Returns:
            Recovery result if successful, None otherwise
        """
        logger.info(f"Attempting recovery for error: {type(error).__name__}")
        
        for strategy in self.strategies:
            if strategy.can_recover(error, context):
                logger.info(f"Trying recovery strategy: {strategy.name}")
                
                try:
                    result = strategy.recover(error, context)
                    
                    # Record recovery attempt
                    self.recovery_history.append({
                        'timestamp': datetime.now().isoformat(),
                        'error_type': type(error).__name__,
                        'strategy': strategy.name,
                        'success': result.success,
                        'action': result.action.value,
                        'context': context
                    })
                    
                    if result.success:
                        logger.info(
                            f"Recovery successful using {strategy.name}: "
                            f"{result.message}"
                        )
                        return result
                    else:
                        logger.warning(
                            f"Recovery failed using {strategy.name}: "
                            f"{result.message}"
                        )
                
                except Exception as recovery_error:
                    logger.error(
                        f"Recovery strategy {strategy.name} raised error: "
                        f"{recovery_error}"
                    )
        
        logger.error("All recovery strategies failed")
        return None
    
    def get_recovery_stats(self) -> Dict[str, Any]:
        """Get recovery statistics."""
        if not self.recovery_history:
            return {'total_attempts': 0}
        
        total = len(self.recovery_history)
        successful = sum(1 for r in self.recovery_history if r['success'])
        
        by_strategy = {}
        for record in self.recovery_history:
            strategy = record['strategy']
            if strategy not in by_strategy:
                by_strategy[strategy] = {'attempts': 0, 'successes': 0}
            by_strategy[strategy]['attempts'] += 1
            if record['success']:
                by_strategy[strategy]['successes'] += 1
        
        return {
            'total_attempts': total,
            'successful': successful,
            'success_rate': successful / total if total > 0 else 0,
            'by_strategy': by_strategy
        }


# Global recovery manager
_recovery_manager = PipelineRecoveryManager()


def get_recovery_manager() -> PipelineRecoveryManager:
    """Get global recovery manager."""
    return _recovery_manager


def configure_default_recovery(
    cache_dir: Optional[Path] = None,
    fallback_sources: Optional[Dict[str, Callable]] = None,
    skippable_steps: Optional[List[str]] = None
) -> None:
    """
    Configure default recovery strategies.
    
    Args:
        cache_dir: Directory for cached data
        fallback_sources: Map of fallback data sources
        skippable_steps: List of skippable pipeline steps
    """
    manager = get_recovery_manager()
    
    # Add retry strategy (highest priority)
    manager.add_strategy(RetryWithBackoffStrategy(max_retries=3))
    
    # Add cache recovery
    if cache_dir:
        manager.add_strategy(CacheRecoveryStrategy(cache_dir))
    
    # Add fallback sources
    if fallback_sources:
        manager.add_strategy(FallbackDataStrategy(fallback_sources))
    
    # Add partial data strategy
    manager.add_strategy(PartialDataStrategy(min_completeness=0.7))
    
    # Add skip strategy
    if skippable_steps:
        manager.add_strategy(SkipStepStrategy(skippable_steps))
    
    logger.info(f"Configured {len(manager.strategies)} recovery strategies")

# Made with Bob
