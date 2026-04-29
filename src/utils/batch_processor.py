"""
Comprehensive batch processing framework for large datasets.

Provides efficient batch processing with memory management, parallel execution,
progress tracking, and error handling.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Any, Optional, Callable, Iterator, TypeVar, Generic
from pathlib import Path
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from datetime import datetime
import time

from src.utils.logging_config import get_logger
from src.utils.error_context import error_context

logger = get_logger(__name__)

T = TypeVar('T')
R = TypeVar('R')


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    
    batch_size: int = 1000
    max_workers: int = 4
    use_processes: bool = False  # Use processes instead of threads
    show_progress: bool = True
    save_intermediate: bool = False
    intermediate_dir: Optional[Path] = None
    memory_limit_mb: Optional[int] = None
    retry_failed: bool = True
    max_retries: int = 3
    
    def __post_init__(self):
        """Validate configuration."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_workers <= 0:
            raise ValueError("max_workers must be positive")
        if self.save_intermediate and not self.intermediate_dir:
            self.intermediate_dir = Path("data/intermediate")


@dataclass
class BatchResult:
    """Result of batch processing."""
    
    batch_id: int
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    processing_time: float = 0.0
    memory_used_mb: float = 0.0
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'batch_id': self.batch_id,
            'success': self.success,
            'error': self.error,
            'processing_time': self.processing_time,
            'memory_used_mb': self.memory_used_mb,
            'metadata': self.metadata,
            'has_data': self.data is not None
        }


class BatchProcessor(ABC, Generic[T, R]):
    """Abstract base class for batch processors."""
    
    def __init__(self, config: Optional[BatchConfig] = None):
        """
        Initialize batch processor.
        
        Args:
            config: Batch processing configuration
        """
        self.config = config or BatchConfig()
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self.results: List[BatchResult] = []
    
    @abstractmethod
    def process_batch(self, batch: List[T], batch_id: int) -> R:
        """
        Process a single batch.
        
        Args:
            batch: List of items to process
            batch_id: Batch identifier
            
        Returns:
            Processed result
        """
        pass
    
    def create_batches(self, items: List[T]) -> Iterator[List[T]]:
        """
        Create batches from items.
        
        Args:
            items: List of items to batch
            
        Yields:
            Batches of items
        """
        batch_size = self.config.batch_size
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]
    
    def process_single_batch(self, batch: List[T], batch_id: int) -> BatchResult:
        """
        Process a single batch with error handling and metrics.
        
        Args:
            batch: Batch to process
            batch_id: Batch identifier
            
        Returns:
            Batch result
        """
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            with error_context(
                f"batch_{batch_id}",
                batch_size=len(batch),
                batch_id=batch_id
            ):
                result = self.process_batch(batch, batch_id)
                
                processing_time = time.time() - start_time
                memory_used = self._get_memory_usage() - start_memory
                
                batch_result = BatchResult(
                    batch_id=batch_id,
                    success=True,
                    data=result,
                    processing_time=processing_time,
                    memory_used_mb=memory_used,
                    metadata={'batch_size': len(batch)}
                )
                
                # Save intermediate result if configured
                if self.config.save_intermediate:
                    self._save_intermediate(batch_result)
                
                return batch_result
        
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Batch {batch_id} failed: {e}")
            
            return BatchResult(
                batch_id=batch_id,
                success=False,
                error=str(e),
                processing_time=processing_time,
                metadata={'batch_size': len(batch)}
            )
    
    def process_all(self, items: List[T]) -> List[R]:
        """
        Process all items in batches.
        
        Args:
            items: List of items to process
            
        Returns:
            List of processed results
        """
        self.logger.info(
            f"Processing {len(items)} items in batches of {self.config.batch_size} "
            f"using {self.config.max_workers} workers"
        )
        
        batches = list(self.create_batches(items))
        total_batches = len(batches)
        
        self.logger.info(f"Created {total_batches} batches")
        
        # Choose executor based on configuration
        executor_class = ProcessPoolExecutor if self.config.use_processes else ThreadPoolExecutor
        
        results = []
        failed_batches = []
        
        with executor_class(max_workers=self.config.max_workers) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(self.process_single_batch, batch, i): i
                for i, batch in enumerate(batches)
            }
            
            # Process completed batches
            completed = 0
            for future in as_completed(future_to_batch):
                batch_id = future_to_batch[future]
                
                try:
                    batch_result = future.result()
                    self.results.append(batch_result)
                    
                    if batch_result.success:
                        results.append(batch_result.data)
                    else:
                        failed_batches.append((batch_id, batches[batch_id]))
                    
                    completed += 1
                    
                    if self.config.show_progress:
                        self._log_progress(completed, total_batches, batch_result)
                
                except Exception as e:
                    self.logger.error(f"Batch {batch_id} raised exception: {e}")
                    failed_batches.append((batch_id, batches[batch_id]))
        
        # Retry failed batches if configured
        if failed_batches and self.config.retry_failed:
            self.logger.info(f"Retrying {len(failed_batches)} failed batches")
            results.extend(self._retry_failed_batches(failed_batches))
        
        self.logger.info(
            f"Completed processing: {len(results)}/{total_batches} batches successful"
        )
        
        return results
    
    def _retry_failed_batches(self, failed_batches: List[tuple]) -> List[R]:
        """Retry failed batches."""
        results = []
        
        for batch_id, batch in failed_batches:
            for attempt in range(self.config.max_retries):
                self.logger.info(f"Retry {attempt + 1}/{self.config.max_retries} for batch {batch_id}")
                
                batch_result = self.process_single_batch(batch, batch_id)
                
                if batch_result.success:
                    results.append(batch_result.data)
                    break
                
                if attempt < self.config.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        return results
    
    def _log_progress(self, completed: int, total: int, batch_result: BatchResult):
        """Log progress."""
        progress = completed / total * 100
        status = "✓" if batch_result.success else "✗"
        
        self.logger.info(
            f"Progress: {completed}/{total} ({progress:.1f}%) {status} "
            f"Batch {batch_result.batch_id}: {batch_result.processing_time:.2f}s"
        )
    
    def _save_intermediate(self, batch_result: BatchResult):
        """Save intermediate result."""
        if not self.config.intermediate_dir:
            return
        
        self.config.intermediate_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"batch_{batch_result.batch_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        filepath = self.config.intermediate_dir / filename
        
        if isinstance(batch_result.data, pd.DataFrame):
            batch_result.data.to_parquet(filepath)
            self.logger.debug(f"Saved intermediate result to {filepath}")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def get_statistics(self) -> dict:
        """Get processing statistics."""
        if not self.results:
            return {}
        
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]
        
        total_time = sum(r.processing_time for r in self.results)
        avg_time = total_time / len(self.results) if self.results else 0
        
        total_memory = sum(r.memory_used_mb for r in self.results)
        avg_memory = total_memory / len(self.results) if self.results else 0
        
        return {
            'total_batches': len(self.results),
            'successful': len(successful),
            'failed': len(failed),
            'success_rate': len(successful) / len(self.results) if self.results else 0,
            'total_time': total_time,
            'avg_time_per_batch': avg_time,
            'total_memory_mb': total_memory,
            'avg_memory_per_batch_mb': avg_memory
        }


class DataFrameBatchProcessor(BatchProcessor[pd.DataFrame, pd.DataFrame]):
    """Batch processor for DataFrames."""
    
    def __init__(
        self,
        transform_func: Callable[[pd.DataFrame], pd.DataFrame],
        config: Optional[BatchConfig] = None
    ):
        """
        Initialize DataFrame batch processor.
        
        Args:
            transform_func: Function to transform each batch
            config: Batch processing configuration
        """
        super().__init__(config)
        self.transform_func = transform_func
    
    def process_batch(self, batch: List[pd.DataFrame], batch_id: int) -> pd.DataFrame:
        """Process a batch of DataFrames."""
        # Concatenate batch DataFrames
        df = pd.concat(batch, ignore_index=True)
        
        # Apply transformation
        result = self.transform_func(df)
        
        return result
    
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process a large DataFrame in batches.
        
        Args:
            df: DataFrame to process
            
        Returns:
            Processed DataFrame
        """
        # Split DataFrame into batches
        batches = [
            df.iloc[i:i + self.config.batch_size]
            for i in range(0, len(df), self.config.batch_size)
        ]
        
        # Process batches
        results = self.process_all(batches)
        
        # Concatenate results
        if results:
            return pd.concat(results, ignore_index=True)
        else:
            return pd.DataFrame()


class ChunkedFileProcessor:
    """Process large files in chunks."""
    
    def __init__(
        self,
        chunk_size: int = 10000,
        transform_func: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None
    ):
        """
        Initialize chunked file processor.
        
        Args:
            chunk_size: Number of rows per chunk
            transform_func: Optional transformation function
        """
        self.chunk_size = chunk_size
        self.transform_func = transform_func
        self.logger = get_logger(__name__)
    
    def process_csv(
        self,
        input_path: Path,
        output_path: Path,
        **read_kwargs
    ) -> dict:
        """
        Process large CSV file in chunks.
        
        Args:
            input_path: Input CSV file path
            output_path: Output file path
            **read_kwargs: Additional arguments for pd.read_csv
            
        Returns:
            Processing statistics
        """
        self.logger.info(f"Processing CSV file: {input_path}")
        
        chunks_processed = 0
        total_rows = 0
        start_time = time.time()
        
        # Process chunks
        first_chunk = True
        for chunk in pd.read_csv(input_path, chunksize=self.chunk_size, **read_kwargs):
            if self.transform_func:
                chunk = self.transform_func(chunk)
            
            # Write chunk
            mode = 'w' if first_chunk else 'a'
            header = first_chunk
            
            if output_path.suffix == '.parquet':
                # For parquet, need to handle differently
                if first_chunk:
                    chunk.to_parquet(output_path, index=False)
                else:
                    # Append to parquet (requires pyarrow)
                    existing = pd.read_parquet(output_path)
                    combined = pd.concat([existing, chunk], ignore_index=True)
                    combined.to_parquet(output_path, index=False)
            else:
                chunk.to_csv(output_path, mode=mode, header=header, index=False)
            
            chunks_processed += 1
            total_rows += len(chunk)
            first_chunk = False
            
            if chunks_processed % 10 == 0:
                self.logger.info(f"Processed {chunks_processed} chunks ({total_rows} rows)")
        
        processing_time = time.time() - start_time
        
        stats = {
            'chunks_processed': chunks_processed,
            'total_rows': total_rows,
            'processing_time': processing_time,
            'rows_per_second': total_rows / processing_time if processing_time > 0 else 0
        }
        
        self.logger.info(
            f"Completed: {total_rows} rows in {processing_time:.2f}s "
            f"({stats['rows_per_second']:.0f} rows/s)"
        )
        
        return stats
    
    def process_parquet(
        self,
        input_path: Path,
        output_path: Path,
        columns: Optional[List[str]] = None
    ) -> dict:
        """
        Process large Parquet file in chunks.
        
        Args:
            input_path: Input Parquet file path
            output_path: Output file path
            columns: Optional list of columns to read
            
        Returns:
            Processing statistics
        """
        import pyarrow.parquet as pq
        
        self.logger.info(f"Processing Parquet file: {input_path}")
        
        parquet_file = pq.ParquetFile(input_path)
        chunks_processed = 0
        total_rows = 0
        start_time = time.time()
        
        first_chunk = True
        for batch in parquet_file.iter_batches(batch_size=self.chunk_size, columns=columns):
            chunk = batch.to_pandas()
            
            if self.transform_func:
                chunk = self.transform_func(chunk)
            
            # Write chunk
            if first_chunk:
                chunk.to_parquet(output_path, index=False)
            else:
                existing = pd.read_parquet(output_path)
                combined = pd.concat([existing, chunk], ignore_index=True)
                combined.to_parquet(output_path, index=False)
            
            chunks_processed += 1
            total_rows += len(chunk)
            first_chunk = False
            
            if chunks_processed % 10 == 0:
                self.logger.info(f"Processed {chunks_processed} chunks ({total_rows} rows)")
        
        processing_time = time.time() - start_time
        
        stats = {
            'chunks_processed': chunks_processed,
            'total_rows': total_rows,
            'processing_time': processing_time,
            'rows_per_second': total_rows / processing_time if processing_time > 0 else 0
        }
        
        self.logger.info(
            f"Completed: {total_rows} rows in {processing_time:.2f}s "
            f"({stats['rows_per_second']:.0f} rows/s)"
        )
        
        return stats


def batch_process_dataframe(
    df: pd.DataFrame,
    transform_func: Callable[[pd.DataFrame], pd.DataFrame],
    batch_size: int = 1000,
    max_workers: int = 4,
    show_progress: bool = True
) -> pd.DataFrame:
    """
    Convenience function for batch processing DataFrames.
    
    Args:
        df: DataFrame to process
        transform_func: Transformation function
        batch_size: Batch size
        max_workers: Number of parallel workers
        show_progress: Whether to show progress
        
    Returns:
        Processed DataFrame
    """
    config = BatchConfig(
        batch_size=batch_size,
        max_workers=max_workers,
        show_progress=show_progress
    )
    
    processor = DataFrameBatchProcessor(transform_func, config)
    return processor.process_dataframe(df)

# Made with Bob
