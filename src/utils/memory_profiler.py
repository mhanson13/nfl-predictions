"""
Memory profiling and optimization utilities.

Provides tools for monitoring memory usage, detecting memory leaks,
and optimizing memory consumption in data processing pipelines.
"""

import gc
import sys
import tracemalloc
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable, TypeVar
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
import pandas as pd
import json

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


@dataclass
class MemorySnapshot:
    """Snapshot of memory usage at a point in time."""
    
    timestamp: datetime
    operation: str
    current_mb: float
    peak_mb: float
    allocated_mb: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'operation': self.operation,
            'current_mb': round(self.current_mb, 2),
            'peak_mb': round(self.peak_mb, 2),
            'allocated_mb': round(self.allocated_mb, 2),
            'metadata': self.metadata
        }


@dataclass
class MemoryProfile:
    """Complete memory profile for an operation."""
    
    operation: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    start_memory_mb: float
    end_memory_mb: float
    peak_memory_mb: float
    memory_delta_mb: float
    snapshots: List[MemorySnapshot] = field(default_factory=list)
    top_allocations: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'operation': self.operation,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'duration_seconds': round(self.duration_seconds, 2),
            'start_memory_mb': round(self.start_memory_mb, 2),
            'end_memory_mb': round(self.end_memory_mb, 2),
            'peak_memory_mb': round(self.peak_memory_mb, 2),
            'memory_delta_mb': round(self.memory_delta_mb, 2),
            'snapshots': [s.to_dict() for s in self.snapshots],
            'top_allocations': self.top_allocations
        }
    
    def save(self, path: Path):
        """Save profile to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved memory profile to {path}")


class MemoryProfiler:
    """Memory profiler for tracking memory usage."""
    
    def __init__(self, enabled: bool = True):
        """
        Initialize memory profiler.
        
        Args:
            enabled: Whether profiling is enabled
        """
        self.enabled = enabled
        self.profiles: List[MemoryProfile] = []
        self.current_profile: Optional[MemoryProfile] = None
        self._tracemalloc_started = False
    
    def start_profiling(self):
        """Start memory profiling."""
        if self.enabled and not self._tracemalloc_started:
            tracemalloc.start()
            self._tracemalloc_started = True
            logger.info("Started memory profiling")
    
    def stop_profiling(self):
        """Stop memory profiling."""
        if self._tracemalloc_started:
            tracemalloc.stop()
            self._tracemalloc_started = False
            logger.info("Stopped memory profiling")
    
    def get_current_memory(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            # Fallback to tracemalloc if psutil not available
            if self._tracemalloc_started:
                current, peak = tracemalloc.get_traced_memory()
                return current / 1024 / 1024
            return 0.0
    
    def get_peak_memory(self) -> float:
        """Get peak memory usage in MB."""
        if self._tracemalloc_started:
            current, peak = tracemalloc.get_traced_memory()
            return peak / 1024 / 1024
        return 0.0
    
    def take_snapshot(self, operation: str, **metadata) -> MemorySnapshot:
        """
        Take a memory snapshot.
        
        Args:
            operation: Operation name
            **metadata: Additional metadata
            
        Returns:
            Memory snapshot
        """
        current = self.get_current_memory()
        peak = self.get_peak_memory()
        
        # Get allocated memory from tracemalloc
        allocated = 0.0
        if self._tracemalloc_started:
            snapshot = tracemalloc.take_snapshot()
            allocated = sum(stat.size for stat in snapshot.statistics('lineno')) / 1024 / 1024
        
        return MemorySnapshot(
            timestamp=datetime.now(),
            operation=operation,
            current_mb=current,
            peak_mb=peak,
            allocated_mb=allocated,
            metadata=metadata
        )
    
    def get_top_allocations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get top memory allocations.
        
        Args:
            limit: Number of top allocations to return
            
        Returns:
            List of allocation info
        """
        if not self._tracemalloc_started:
            return []
        
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')[:limit]
        
        allocations = []
        for stat in top_stats:
            allocations.append({
                'file': stat.traceback.format()[0] if stat.traceback else 'unknown',
                'size_mb': stat.size / 1024 / 1024,
                'count': stat.count
            })
        
        return allocations
    
    @contextmanager
    def profile(self, operation: str, save_to: Optional[Path] = None):
        """
        Context manager for profiling an operation.
        
        Args:
            operation: Operation name
            save_to: Optional path to save profile
            
        Example:
            with profiler.profile("load_data"):
                df = pd.read_parquet("large_file.parquet")
        """
        if not self.enabled:
            yield
            return
        
        # Start profiling if not started
        if not self._tracemalloc_started:
            self.start_profiling()
        
        # Take initial snapshot
        start_time = datetime.now()
        start_memory = self.get_current_memory()
        
        # Reset peak
        if self._tracemalloc_started:
            tracemalloc.reset_peak()
        
        logger.info(f"Starting memory profile for: {operation}")
        
        try:
            yield self
        finally:
            # Take final snapshot
            end_time = datetime.now()
            end_memory = self.get_current_memory()
            peak_memory = self.get_peak_memory()
            
            # Get top allocations
            top_allocations = self.get_top_allocations()
            
            # Create profile
            profile = MemoryProfile(
                operation=operation,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=(end_time - start_time).total_seconds(),
                start_memory_mb=start_memory,
                end_memory_mb=end_memory,
                peak_memory_mb=peak_memory,
                memory_delta_mb=end_memory - start_memory,
                top_allocations=top_allocations
            )
            
            self.profiles.append(profile)
            self.current_profile = profile
            
            logger.info(
                f"Completed memory profile for: {operation}\n"
                f"  Duration: {profile.duration_seconds:.2f}s\n"
                f"  Start: {profile.start_memory_mb:.2f} MB\n"
                f"  End: {profile.end_memory_mb:.2f} MB\n"
                f"  Peak: {profile.peak_memory_mb:.2f} MB\n"
                f"  Delta: {profile.memory_delta_mb:+.2f} MB"
            )
            
            # Save if requested
            if save_to:
                profile.save(save_to)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all profiles."""
        if not self.profiles:
            return {}
        
        total_operations = len(self.profiles)
        total_duration = sum(p.duration_seconds for p in self.profiles)
        total_memory_delta = sum(p.memory_delta_mb for p in self.profiles)
        max_peak = max(p.peak_memory_mb for p in self.profiles)
        
        return {
            'total_operations': total_operations,
            'total_duration': total_duration,
            'total_memory_delta_mb': total_memory_delta,
            'max_peak_memory_mb': max_peak,
            'profiles': [p.to_dict() for p in self.profiles]
        }


# Global profiler instance
_profiler = MemoryProfiler()


def get_profiler() -> MemoryProfiler:
    """Get global memory profiler."""
    return _profiler


@contextmanager
def profile_memory(operation: str, save_to: Optional[Path] = None):
    """
    Convenience context manager for memory profiling.
    
    Args:
        operation: Operation name
        save_to: Optional path to save profile
        
    Example:
        with profile_memory("load_large_file"):
            df = pd.read_parquet("large.parquet")
    """
    profiler = get_profiler()
    with profiler.profile(operation, save_to):
        yield


def profile_function(operation: Optional[str] = None, save_to: Optional[Path] = None):
    """
    Decorator for profiling function memory usage.
    
    Args:
        operation: Optional operation name (defaults to function name)
        save_to: Optional path to save profile
        
    Example:
        @profile_function()
        def process_data(df):
            return df.groupby('col').sum()
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args, **kwargs) -> T:
            op_name = operation or func.__name__
            with profile_memory(op_name, save_to):
                return func(*args, **kwargs)
        return wrapper
    return decorator


class MemoryOptimizer:
    """Utilities for optimizing memory usage."""
    
    @staticmethod
    def optimize_dataframe_dtypes(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Optimize DataFrame dtypes to reduce memory usage.
        
        Args:
            df: DataFrame to optimize
            verbose: Whether to log optimization results
            
        Returns:
            Optimized DataFrame
        """
        start_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
        
        # Optimize numeric columns
        for col in df.select_dtypes(include=['int']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        for col in df.select_dtypes(include=['float']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Convert object columns to category if beneficial
        for col in df.select_dtypes(include=['object']).columns:
            num_unique = df[col].nunique()
            num_total = len(df[col])
            
            # Convert to category if less than 50% unique values
            if num_unique / num_total < 0.5:
                df[col] = df[col].astype('category')
        
        end_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
        reduction = (start_memory - end_memory) / start_memory * 100
        
        if verbose:
            logger.info(
                f"DataFrame memory optimization:\n"
                f"  Before: {start_memory:.2f} MB\n"
                f"  After: {end_memory:.2f} MB\n"
                f"  Reduction: {reduction:.1f}%"
            )
        
        return df
    
    @staticmethod
    def chunk_dataframe(df: pd.DataFrame, chunk_size: int = 10000) -> List[pd.DataFrame]:
        """
        Split DataFrame into chunks for memory-efficient processing.
        
        Args:
            df: DataFrame to chunk
            chunk_size: Size of each chunk
            
        Returns:
            List of DataFrame chunks
        """
        chunks = []
        for i in range(0, len(df), chunk_size):
            chunks.append(df.iloc[i:i + chunk_size].copy())
        
        logger.info(f"Split DataFrame into {len(chunks)} chunks of size {chunk_size}")
        return chunks
    
    @staticmethod
    def force_garbage_collection() -> Dict[str, int]:
        """
        Force garbage collection and return statistics.
        
        Returns:
            Garbage collection statistics
        """
        collected = gc.collect()
        stats = {
            'collected': collected,
            'generation_0': gc.get_count()[0],
            'generation_1': gc.get_count()[1],
            'generation_2': gc.get_count()[2]
        }
        
        logger.debug(f"Garbage collection: collected {collected} objects")
        return stats
    
    @staticmethod
    def get_object_size(obj: Any) -> float:
        """
        Get size of Python object in MB.
        
        Args:
            obj: Object to measure
            
        Returns:
            Size in MB
        """
        return sys.getsizeof(obj) / 1024 / 1024
    
    @staticmethod
    def analyze_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze memory usage of DataFrame columns.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            DataFrame with memory analysis
        """
        memory_usage = df.memory_usage(deep=True)
        
        analysis = pd.DataFrame({
            'column': memory_usage.index,
            'dtype': [str(df[col].dtype) if col != 'Index' else 'Index' 
                     for col in memory_usage.index],
            'memory_mb': memory_usage.values / 1024 / 1024,
            'percent': memory_usage.values / memory_usage.sum() * 100
        })
        
        analysis = analysis.sort_values('memory_mb', ascending=False)
        
        total_mb = analysis['memory_mb'].sum()
        logger.info(f"DataFrame memory analysis (total: {total_mb:.2f} MB):")
        logger.info(f"\n{analysis.to_string(index=False)}")
        
        return analysis


def create_memory_report(
    profiler: MemoryProfiler,
    output_path: Path,
    include_recommendations: bool = True
) -> None:
    """
    Create comprehensive memory report.
    
    Args:
        profiler: Memory profiler with collected data
        output_path: Path to save report
        include_recommendations: Whether to include optimization recommendations
    """
    summary = profiler.get_summary()
    
    report_lines = [
        "# Memory Profiling Report",
        f"\nGenerated: {datetime.now().isoformat()}",
        f"\n## Summary",
        f"- Total Operations: {summary.get('total_operations', 0)}",
        f"- Total Duration: {summary.get('total_duration', 0):.2f}s",
        f"- Total Memory Delta: {summary.get('total_memory_delta_mb', 0):+.2f} MB",
        f"- Max Peak Memory: {summary.get('max_peak_memory_mb', 0):.2f} MB",
        "\n## Operations"
    ]
    
    for profile_dict in summary.get('profiles', []):
        report_lines.extend([
            f"\n### {profile_dict['operation']}",
            f"- Duration: {profile_dict['duration_seconds']:.2f}s",
            f"- Start Memory: {profile_dict['start_memory_mb']:.2f} MB",
            f"- End Memory: {profile_dict['end_memory_mb']:.2f} MB",
            f"- Peak Memory: {profile_dict['peak_memory_mb']:.2f} MB",
            f"- Memory Delta: {profile_dict['memory_delta_mb']:+.2f} MB"
        ])
        
        if profile_dict.get('top_allocations'):
            report_lines.append("\n#### Top Allocations")
            for alloc in profile_dict['top_allocations'][:5]:
                report_lines.append(
                    f"- {alloc['file']}: {alloc['size_mb']:.2f} MB ({alloc['count']} objects)"
                )
    
    if include_recommendations:
        report_lines.extend([
            "\n## Recommendations",
            "- Use chunked processing for large DataFrames",
            "- Optimize DataFrame dtypes with `MemoryOptimizer.optimize_dataframe_dtypes()`",
            "- Force garbage collection after large operations",
            "- Consider using generators instead of lists for large datasets",
            "- Use `del` to explicitly delete large objects when done",
            "- Monitor peak memory to identify memory-intensive operations"
        ])
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"Created memory report: {output_path}")

# Made with Bob
