"""
Performance optimization utilities for the NFL predictions platform.

Provides helpers for:
- Vectorized operations
- Parallel processing
- Memory optimization
- Performance profiling
"""

import functools
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any, Callable, Iterable, List, Optional, TypeVar
import numpy as np
import pandas as pd

from src.config import get_config

T = TypeVar('T')
R = TypeVar('R')


def vectorize_operation(func: Callable) -> Callable:
    """
    Decorator to vectorize operations on pandas Series/DataFrames.
    
    Converts element-wise operations to vectorized numpy operations
    for better performance.
    
    Example:
        @vectorize_operation
        def calculate_score(x):
            return x * 2 + 10
        
        df['score'] = calculate_score(df['value'])
    """
    @functools.wraps(func)
    def wrapper(data, *args, **kwargs):
        if isinstance(data, (pd.Series, pd.DataFrame)):
            return data.apply(lambda x: func(x, *args, **kwargs))
        elif isinstance(data, np.ndarray):
            return np.vectorize(func)(data, *args, **kwargs)
        else:
            return func(data, *args, **kwargs)
    return wrapper


def parallel_apply(
    data: pd.DataFrame,
    func: Callable,
    n_jobs: Optional[int] = None,
    chunk_size: Optional[int] = None,
    use_threads: bool = False
) -> pd.DataFrame:
    """
    Apply function to DataFrame in parallel.
    
    Args:
        data: DataFrame to process
        func: Function to apply to each chunk
        n_jobs: Number of parallel jobs (None = use config)
        chunk_size: Size of each chunk (None = auto)
        use_threads: Use threads instead of processes
    
    Returns:
        Processed DataFrame
    
    Example:
        def process_chunk(df):
            df['new_col'] = df['col1'] * df['col2']
            return df
        
        result = parallel_apply(large_df, process_chunk, n_jobs=4)
    """
    config = get_config()
    
    if n_jobs is None:
        n_jobs = config.pipeline.n_workers
    
    if n_jobs == 1:
        return func(data)
    
    # Determine chunk size
    if chunk_size is None:
        chunk_size = max(1, len(data) // (n_jobs * 4))
    
    # Split data into chunks
    chunks = [data.iloc[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    
    # Process in parallel
    executor_class = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    
    with executor_class(max_workers=n_jobs) as executor:
        futures = [executor.submit(func, chunk) for chunk in chunks]
        results = [future.result() for future in as_completed(futures)]
    
    # Combine results
    return pd.concat(results, ignore_index=True)


def parallel_map(
    func: Callable[[T], R],
    items: Iterable[T],
    n_jobs: Optional[int] = None,
    use_threads: bool = False,
    show_progress: bool = False
) -> List[R]:
    """
    Map function over items in parallel.
    
    Args:
        func: Function to apply to each item
        items: Iterable of items to process
        n_jobs: Number of parallel jobs
        use_threads: Use threads instead of processes
        show_progress: Show progress bar (requires tqdm)
    
    Returns:
        List of results
    
    Example:
        def process_game(game_id):
            # ... expensive computation
            return result
        
        results = parallel_map(process_game, game_ids, n_jobs=4)
    """
    config = get_config()
    
    if n_jobs is None:
        n_jobs = config.pipeline.n_workers
    
    items_list = list(items)
    
    if n_jobs == 1:
        if show_progress:
            try:
                from tqdm import tqdm
                return [func(item) for item in tqdm(items_list)]
            except ImportError:
                pass
        return [func(item) for item in items_list]
    
    executor_class = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    
    with executor_class(max_workers=n_jobs) as executor:
        futures = {executor.submit(func, item): item for item in items_list}
        
        if show_progress:
            try:
                from tqdm import tqdm
                futures_iter = tqdm(as_completed(futures), total=len(futures))
            except ImportError:
                futures_iter = as_completed(futures)
        else:
            futures_iter = as_completed(futures)
        
        results = []
        for future in futures_iter:
            results.append(future.result())
    
    return results


def optimize_dtypes(df: pd.DataFrame, aggressive: bool = False) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by downcasting dtypes.
    
    Args:
        df: DataFrame to optimize
        aggressive: Use more aggressive optimization (may lose precision)
    
    Returns:
        Optimized DataFrame
    
    Example:
        df = optimize_dtypes(df)
        print(f"Memory saved: {original_size - df.memory_usage().sum()}")
    """
    df = df.copy()
    
    # Optimize integers
    for col in df.select_dtypes(include=['int']).columns:
        col_min = df[col].min()
        col_max = df[col].max()
        
        if col_min >= 0:
            # Unsigned integers
            if col_max < 255:
                df[col] = df[col].astype(np.uint8)
            elif col_max < 65535:
                df[col] = df[col].astype(np.uint16)
            elif col_max < 4294967295:
                df[col] = df[col].astype(np.uint32)
        else:
            # Signed integers
            if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
    
    # Optimize floats
    for col in df.select_dtypes(include=['float']).columns:
        if aggressive:
            df[col] = df[col].astype(np.float32)
        else:
            # Only downcast if no precision loss
            col_min = df[col].min()
            col_max = df[col].max()
            if (col_min > np.finfo(np.float32).min and 
                col_max < np.finfo(np.float32).max):
                df[col] = df[col].astype(np.float32)
    
    # Convert object columns to category if beneficial
    for col in df.select_dtypes(include=['object']).columns:
        num_unique = df[col].nunique()
        num_total = len(df[col])
        
        # Convert to category if < 50% unique values
        if num_unique / num_total < 0.5:
            df[col] = df[col].astype('category')
    
    return df


def batch_process(
    items: List[T],
    func: Callable[[List[T]], R],
    batch_size: int = 100
) -> List[R]:
    """
    Process items in batches for better memory efficiency.
    
    Args:
        items: List of items to process
        func: Function that processes a batch
        batch_size: Number of items per batch
    
    Returns:
        List of results
    
    Example:
        def process_batch(games):
            return [predict(game) for game in games]
        
        results = batch_process(all_games, process_batch, batch_size=50)
    """
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        result = func(batch)
        results.append(result)
    
    return results


class PerformanceTimer:
    """
    Context manager for timing code execution.
    
    Example:
        with PerformanceTimer("Loading data") as timer:
            df = pd.read_parquet("large_file.parquet")
        print(f"Took {timer.elapsed:.2f}s")
    """
    
    def __init__(self, name: str = "Operation", verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self.start_time = None
        self.end_time = None
        self.elapsed = None
    
    def __enter__(self):
        if self.verbose:
            print(f"[Timer] Starting: {self.name}")
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        if self.verbose:
            print(f"[Timer] Completed: {self.name} (took {self.elapsed:.2f}s)")


def profile_memory(func: Callable) -> Callable:
    """
    Decorator to profile memory usage of a function.
    
    Requires memory_profiler package.
    
    Example:
        @profile_memory
        def load_large_dataset():
            return pd.read_parquet("huge_file.parquet")
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            from memory_profiler import profile
            profiled_func = profile(func)
            return profiled_func(*args, **kwargs)
        except ImportError:
            print("[Warning] memory_profiler not installed, skipping profiling")
            return func(*args, **kwargs)
    return wrapper


def cache_result(maxsize: int = 128):
    """
    Decorator to cache function results (LRU cache).
    
    Args:
        maxsize: Maximum cache size
    
    Example:
        @cache_result(maxsize=256)
        def expensive_calculation(x, y):
            return x ** y + y ** x
    """
    return functools.lru_cache(maxsize=maxsize)


def chunked_read_parquet(
    path: str,
    chunk_size: int = 10000,
    columns: Optional[List[str]] = None
) -> Iterable[pd.DataFrame]:
    """
    Read large parquet file in chunks to save memory.
    
    Args:
        path: Path to parquet file
        chunk_size: Number of rows per chunk
        columns: Columns to read (None = all)
    
    Yields:
        DataFrame chunks
    
    Example:
        for chunk in chunked_read_parquet("large_file.parquet", chunk_size=5000):
            process(chunk)
    """
    import pyarrow.parquet as pq
    
    parquet_file = pq.ParquetFile(path)
    
    for batch in parquet_file.iter_batches(batch_size=chunk_size, columns=columns):
        yield batch.to_pandas()


def reduce_memory_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Reduce DataFrame memory usage by optimizing dtypes.
    
    Args:
        df: DataFrame to optimize
        verbose: Print memory savings
    
    Returns:
        Optimized DataFrame
    """
    start_mem = df.memory_usage().sum() / 1024**2
    
    df = optimize_dtypes(df, aggressive=False)
    
    end_mem = df.memory_usage().sum() / 1024**2
    
    if verbose:
        reduction = 100 * (start_mem - end_mem) / start_mem
        print(f"Memory usage reduced from {start_mem:.2f} MB to {end_mem:.2f} MB "
              f"({reduction:.1f}% reduction)")
    
    return df

# Made with Bob
