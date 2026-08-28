"""
Async Pipeline Orchestrator

Coordinates async execution of data fetching, feature building, and model training
with dependency management, progress tracking, and checkpoint/resume support.

Phase 2A - Week 1-2 Implementation
Issue #2: Refactor Pipeline Orchestrator for Async
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from src.data.async_base_fetcher import AsyncBaseDataFetcher, AsyncFetchResult
from src.utils.checkpoints import load_checkpoint, save_checkpoint
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class StageStatus(Enum):
    """Status of a pipeline stage."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PipelineStage:
    """Represents a single stage in the pipeline."""
    
    name: str
    func: Callable
    dependencies: List[str] = field(default_factory=list)
    status: StageStatus = StageStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    retries: int = 0
    max_retries: int = 3
    
    @property
    def duration(self) -> Optional[float]:
        """Calculate stage duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'status': self.status.value,
            'dependencies': self.dependencies,
            'error': self.error,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'retries': self.retries
        }


@dataclass
class PipelineConfig:
    """Configuration for async pipeline execution."""
    
    # Data fetching
    start_year: int = 2023
    max_parallel_data: int = 4
    data_start_delay: float = 1.5
    
    # Pipeline control
    use_async: bool = True
    enable_checkpoints: bool = True
    checkpoint_path: Path = Path("data/reference/pipeline_checkpoint.json")
    
    # Error handling
    fail_fast: bool = False
    max_stage_retries: int = 3
    
    # Progress tracking
    enable_progress: bool = True
    progress_interval: float = 5.0
    
    # Resource limits
    max_memory_mb: Optional[int] = None
    timeout_seconds: Optional[float] = None


@dataclass
class PipelineResult:
    """Result of pipeline execution."""
    
    success: bool
    stages: Dict[str, PipelineStage]
    total_duration: float
    fetch_results: Optional[List[AsyncFetchResult]] = None
    features: Optional[Any] = None
    models: Optional[Any] = None
    error: Optional[str] = None
    
    @property
    def completed_stages(self) -> List[str]:
        """Get list of completed stage names."""
        return [
            name for name, stage in self.stages.items()
            if stage.status == StageStatus.COMPLETED
        ]
    
    @property
    def failed_stages(self) -> List[str]:
        """Get list of failed stage names."""
        return [
            name for name, stage in self.stages.items()
            if stage.status == StageStatus.FAILED
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'success': self.success,
            'total_duration': self.total_duration,
            'completed_stages': self.completed_stages,
            'failed_stages': self.failed_stages,
            'stages': {name: stage.to_dict() for name, stage in self.stages.items()},
            'error': self.error
        }


class AsyncPipelineOrchestrator:
    """
    Orchestrates async pipeline execution with dependency management.
    
    Features:
    - Dependency graph resolution
    - Concurrent execution where possible
    - Progress tracking and logging
    - Checkpoint/resume support
    - Graceful error handling
    - Resource monitoring
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize pipeline orchestrator.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.stages: Dict[str, PipelineStage] = {}
        self.logger = get_logger(f"{__name__}.AsyncPipelineOrchestrator")
        self._start_time: Optional[float] = None
        self._progress_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
    
    def add_stage(
        self,
        name: str,
        func: Callable,
        dependencies: Optional[List[str]] = None
    ) -> None:
        """
        Add a stage to the pipeline.
        
        Args:
            name: Stage name
            func: Async function to execute
            dependencies: List of stage names this depends on
        """
        if name in self.stages:
            raise ValueError(f"Stage '{name}' already exists")
        
        self.stages[name] = PipelineStage(
            name=name,
            func=func,
            dependencies=dependencies or [],
            max_retries=self.config.max_stage_retries
        )
        self.logger.debug(f"Added stage: {name} (dependencies: {dependencies or []})")
    
    def _validate_dependencies(self) -> None:
        """Validate that all dependencies exist and there are no cycles."""
        # Check all dependencies exist
        for stage in self.stages.values():
            for dep in stage.dependencies:
                if dep not in self.stages:
                    raise ValueError(f"Stage '{stage.name}' depends on unknown stage '{dep}'")
        
        # Check for cycles using DFS
        def has_cycle(node: str, visited: Set[str], rec_stack: Set[str]) -> bool:
            visited.add(node)
            rec_stack.add(node)
            
            for dep in self.stages[node].dependencies:
                if dep not in visited:
                    if has_cycle(dep, visited, rec_stack):
                        return True
                elif dep in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        visited: Set[str] = set()
        for stage_name in self.stages:
            if stage_name not in visited:
                if has_cycle(stage_name, visited, set()):
                    raise ValueError(f"Circular dependency detected involving '{stage_name}'")
    
    def _get_execution_order(self) -> List[List[str]]:
        """
        Get stages grouped by execution level (topological sort).
        
        Returns:
            List of lists, where each inner list contains stages that can run concurrently
        """
        # Calculate in-degree for each stage
        in_degree = {name: len(stage.dependencies) for name, stage in self.stages.items()}
        
        # Find stages with no dependencies
        levels: List[List[str]] = []
        remaining = set(self.stages.keys())
        
        while remaining:
            # Find all stages with no remaining dependencies
            current_level = [
                name for name in remaining
                if in_degree[name] == 0
            ]
            
            if not current_level:
                raise ValueError("Unable to resolve dependencies - possible cycle")
            
            levels.append(current_level)
            
            # Remove current level from remaining
            for name in current_level:
                remaining.remove(name)
                
                # Decrease in-degree for dependent stages
                for other_name in remaining:
                    if name in self.stages[other_name].dependencies:
                        in_degree[other_name] -= 1
        
        return levels
    
    async def _execute_stage(self, stage: PipelineStage) -> None:
        """
        Execute a single stage with retry logic.
        
        Args:
            stage: Stage to execute
        """
        while stage.retries <= stage.max_retries:
            try:
                stage.status = StageStatus.RUNNING
                stage.start_time = time.time()
                
                self.logger.info(f"Starting stage: {stage.name}")
                
                # Execute stage function
                if inspect.iscoroutinefunction(stage.func):
                    stage.result = await stage.func()
                else:
                    # Run sync function in executor
                    loop = asyncio.get_event_loop()
                    stage.result = await loop.run_in_executor(None, stage.func)
                
                stage.end_time = time.time()
                stage.status = StageStatus.COMPLETED
                
                self.logger.info(
                    f"Completed stage: {stage.name} "
                    f"(duration: {stage.duration:.2f}s)"
                )
                
                # Save checkpoint
                if self.config.enable_checkpoints:
                    self._save_checkpoint()
                
                return
                
            except Exception as e:
                stage.retries += 1
                stage.error = str(e)
                
                if stage.retries > stage.max_retries:
                    stage.end_time = time.time()
                    stage.status = StageStatus.FAILED
                    self.logger.error(
                        f"Stage '{stage.name}' failed after {stage.max_retries} retries: {e}"
                    )
                    
                    if self.config.fail_fast:
                        raise
                    return
                
                self.logger.warning(
                    f"Stage '{stage.name}' failed (attempt {stage.retries}/{stage.max_retries}): {e}. "
                    f"Retrying..."
                )
                await asyncio.sleep(2 ** stage.retries)  # Exponential backoff
    
    async def _execute_level(self, stage_names: List[str]) -> None:
        """
        Execute all stages in a level concurrently.
        
        Args:
            stage_names: Names of stages to execute
        """
        tasks = [
            self._execute_stage(self.stages[name])
            for name in stage_names
        ]
        
        await asyncio.gather(*tasks, return_exceptions=not self.config.fail_fast)
    
    async def _progress_tracker(self) -> None:
        """Background task to log progress periodically."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self.config.progress_interval
                )
            except asyncio.TimeoutError:
                self._log_progress()
    
    def _log_progress(self) -> None:
        """Log current pipeline progress."""
        total = len(self.stages)
        completed = sum(1 for s in self.stages.values() if s.status == StageStatus.COMPLETED)
        running = sum(1 for s in self.stages.values() if s.status == StageStatus.RUNNING)
        failed = sum(1 for s in self.stages.values() if s.status == StageStatus.FAILED)
        
        elapsed = time.time() - self._start_time if self._start_time else 0
        
        self.logger.info(
            f"Pipeline progress: {completed}/{total} completed, "
            f"{running} running, {failed} failed "
            f"(elapsed: {elapsed:.1f}s)"
        )
    
    def _save_checkpoint(self) -> None:
        """Save current pipeline state to checkpoint."""
        checkpoint_data = {
            'timestamp': datetime.now().isoformat(),
            'stages': {
                name: stage.to_dict()
                for name, stage in self.stages.items()
            }
        }
        
        self.config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.checkpoint_path.write_text(
            json.dumps(checkpoint_data, indent=2),
            encoding='utf-8'
        )
        
        self.logger.debug(f"Saved checkpoint to {self.config.checkpoint_path}")
    
    def _load_checkpoint(self) -> bool:
        """
        Load pipeline state from checkpoint.
        
        Returns:
            True if checkpoint was loaded successfully
        """
        if not self.config.checkpoint_path.exists():
            return False
        
        try:
            checkpoint_data = json.loads(
                self.config.checkpoint_path.read_text(encoding='utf-8')
            )
            
            # Restore stage statuses
            for name, stage_data in checkpoint_data.get('stages', {}).items():
                if name in self.stages:
                    stage = self.stages[name]
                    status_str = stage_data.get('status', 'pending')
                    stage.status = StageStatus(status_str)
                    
                    # Only restore completed stages
                    if stage.status == StageStatus.COMPLETED:
                        stage.start_time = stage_data.get('start_time')
                        stage.end_time = stage_data.get('end_time')
                        self.logger.info(f"Restored completed stage: {name}")
                    else:
                        # Reset non-completed stages
                        stage.status = StageStatus.PENDING
            
            self.logger.info(f"Loaded checkpoint from {self.config.checkpoint_path}")
            return True
            
        except Exception as e:
            self.logger.warning(f"Failed to load checkpoint: {e}")
            return False
    
    async def run(self) -> PipelineResult:
        """
        Run the complete async pipeline.
        
        Returns:
            PipelineResult with execution details
        """
        self._start_time = time.time()
        
        try:
            # Validate pipeline structure
            self._validate_dependencies()
            
            # Load checkpoint if enabled
            if self.config.enable_checkpoints:
                self._load_checkpoint()
            
            # Get execution order
            levels = self._get_execution_order()
            
            self.logger.info(
                f"Starting async pipeline with {len(self.stages)} stages "
                f"in {len(levels)} levels"
            )
            
            # Start progress tracker
            if self.config.enable_progress:
                self._progress_task = asyncio.create_task(self._progress_tracker())
            
            # Execute stages level by level
            for level_idx, level_stages in enumerate(levels):
                # Skip completed stages
                pending_stages = [
                    name for name in level_stages
                    if self.stages[name].status != StageStatus.COMPLETED
                ]
                
                if not pending_stages:
                    self.logger.info(f"Level {level_idx + 1}: All stages already completed")
                    continue
                
                self.logger.info(
                    f"Level {level_idx + 1}: Executing {len(pending_stages)} stages concurrently"
                )
                
                await self._execute_level(pending_stages)
                
                # Check for failures
                failed = [
                    name for name in level_stages
                    if self.stages[name].status == StageStatus.FAILED
                ]
                
                if failed and self.config.fail_fast:
                    raise RuntimeError(f"Stages failed: {', '.join(failed)}")
            
            # Calculate results
            total_duration = time.time() - self._start_time
            failed_stages = [
                name for name, stage in self.stages.items()
                if stage.status == StageStatus.FAILED
            ]
            
            success = len(failed_stages) == 0
            
            result = PipelineResult(
                success=success,
                stages=self.stages.copy(),
                total_duration=total_duration,
                error=f"Failed stages: {', '.join(failed_stages)}" if failed_stages else None
            )
            
            self.logger.info(
                f"Pipeline {'completed successfully' if success else 'completed with errors'} "
                f"(duration: {total_duration:.2f}s)"
            )
            
            return result
            
        except Exception as e:
            total_duration = time.time() - self._start_time
            self.logger.error(f"Pipeline failed: {e}")
            
            return PipelineResult(
                success=False,
                stages=self.stages.copy(),
                total_duration=total_duration,
                error=str(e)
            )
        
        finally:
            # Shutdown progress tracker
            self._shutdown_event.set()
            if self._progress_task:
                await self._progress_task
    
    async def run_concurrent_fetchers(
        self,
        fetchers: List[Any],  # Accept any fetcher subclass
        max_workers: Optional[int] = None
    ) -> List[AsyncFetchResult]:
        """
        Run data fetchers concurrently with rate limiting.
        
        Args:
            fetchers: List of async fetchers to execute
            max_workers: Maximum concurrent fetchers (default: config.max_parallel_data)
            
        Returns:
            List of fetch results
        """
        max_workers = max_workers or self.config.max_parallel_data
        semaphore = asyncio.Semaphore(max_workers)
        
        async def _fetch_with_semaphore(fetcher: AsyncBaseDataFetcher) -> AsyncFetchResult:
            async with semaphore:
                async with fetcher:
                    result = await fetcher.fetch()
                    
                    # Add delay between fetches to avoid rate limiting
                    if self.config.data_start_delay > 0:
                        await asyncio.sleep(self.config.data_start_delay)
                    
                    return result
        
        self.logger.info(f"Starting {len(fetchers)} fetchers with max {max_workers} concurrent")
        
        tasks = [_fetch_with_semaphore(fetcher) for fetcher in fetchers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to failed results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Fetcher {i} failed: {result}")
                processed_results.append(AsyncFetchResult(
                    success=False,
                    error=str(result)
                ))
            else:
                processed_results.append(result)
        
        success_count = sum(1 for r in processed_results if r.success)
        self.logger.info(f"Fetchers complete: {success_count}/{len(fetchers)} successful")
        
        return processed_results
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        # Cleanup
        self._shutdown_event.set()
        if self._progress_task and not self._progress_task.done():
            self._progress_task.cancel()
            try:
                await self._progress_task
            except asyncio.CancelledError:
                pass
        
        return False


# Made with Bob
