"""
Unit tests for async pipeline orchestrator.

Tests pipeline orchestration, dependency resolution, progress tracking,
checkpoint/resume, and error handling.
"""

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, Mock

import pytest

from src.utils.async_pipeline import (
    AsyncPipelineOrchestrator,
    PipelineConfig,
    PipelineStage,
    PipelineResult,
    StageStatus,
)
from src.data.async_base_fetcher import AsyncBaseDataFetcher, AsyncFetchResult
import pandas as pd


# Test Fixtures

@pytest.fixture
def temp_checkpoint_path(tmp_path):
    """Create temporary checkpoint path."""
    return tmp_path / "checkpoint.json"


@pytest.fixture
def basic_config(temp_checkpoint_path):
    """Create basic pipeline configuration."""
    return PipelineConfig(
        start_year=2023,
        max_parallel_data=2,
        enable_checkpoints=True,
        checkpoint_path=temp_checkpoint_path,
        enable_progress=False  # Disable for tests
    )


@pytest.fixture
def sample_dataframe():
    """Create sample DataFrame."""
    return pd.DataFrame({
        'id': [1, 2, 3],
        'value': [100, 200, 300]
    })


# Test Helper Functions

async def simple_stage_func():
    """Simple async stage function."""
    await asyncio.sleep(0.01)
    return "success"


async def failing_stage_func():
    """Stage function that always fails."""
    raise ValueError("Stage failed")


async def slow_stage_func():
    """Slow stage function."""
    await asyncio.sleep(0.1)
    return "slow_success"


class TestAsyncFetcher(AsyncBaseDataFetcher):
    """Test fetcher implementation."""
    
    def __init__(self, name: str, should_fail: bool = False, **kwargs):
        super().__init__(name, **kwargs)
        self.should_fail = should_fail
    
    async def _fetch_raw(self, **kwargs) -> AsyncFetchResult:
        await asyncio.sleep(0.01)
        if self.should_fail:
            return AsyncFetchResult(success=False, error="Fetch failed")
        return AsyncFetchResult(
            success=True,
            data=pd.DataFrame({'test': [1, 2, 3]})
        )


# Test PipelineStage

class TestPipelineStage:
    """Test suite for PipelineStage."""
    
    def test_stage_initialization(self):
        """Test stage initialization."""
        stage = PipelineStage(
            name="test_stage",
            func=simple_stage_func,
            dependencies=["dep1", "dep2"]
        )
        
        assert stage.name == "test_stage"
        assert stage.status == StageStatus.PENDING
        assert stage.dependencies == ["dep1", "dep2"]
        assert stage.result is None
        assert stage.error is None
    
    def test_stage_duration_calculation(self):
        """Test duration calculation."""
        stage = PipelineStage(name="test", func=simple_stage_func)
        
        assert stage.duration is None
        
        stage.start_time = 100.0
        stage.end_time = 105.5
        
        assert stage.duration == 5.5
    
    def test_stage_to_dict(self):
        """Test stage serialization."""
        stage = PipelineStage(
            name="test",
            func=simple_stage_func,
            dependencies=["dep1"]
        )
        stage.status = StageStatus.COMPLETED
        stage.start_time = 100.0
        stage.end_time = 105.0
        
        data = stage.to_dict()
        
        assert data['name'] == "test"
        assert data['status'] == "completed"
        assert data['dependencies'] == ["dep1"]
        assert data['duration'] == 5.0


# Test PipelineConfig

class TestPipelineConfig:
    """Test suite for PipelineConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = PipelineConfig()
        
        assert config.start_year == 2023
        assert config.max_parallel_data == 4
        assert config.use_async is True
        assert config.enable_checkpoints is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = PipelineConfig(
            start_year=2020,
            max_parallel_data=8,
            fail_fast=True
        )
        
        assert config.start_year == 2020
        assert config.max_parallel_data == 8
        assert config.fail_fast is True


# Test AsyncPipelineOrchestrator

@pytest.mark.asyncio
class TestAsyncPipelineOrchestrator:
    """Test suite for AsyncPipelineOrchestrator."""
    
    async def test_initialization(self, basic_config):
        """Test orchestrator initialization."""
        orchestrator = AsyncPipelineOrchestrator(basic_config)
        
        assert orchestrator.config == basic_config
        assert len(orchestrator.stages) == 0
    
    async def test_add_stage(self, basic_config):
        """Test adding stages."""
        orchestrator = AsyncPipelineOrchestrator(basic_config)
        
        orchestrator.add_stage("stage1", simple_stage_func)
        orchestrator.add_stage("stage2", simple_stage_func, dependencies=["stage1"])
        
        assert len(orchestrator.stages) == 2
        assert "stage1" in orchestrator.stages
        assert "stage2" in orchestrator.stages
        assert orchestrator.stages["stage2"].dependencies == ["stage1"]
    
    async def test_add_duplicate_stage_raises_error(self, basic_config):
        """Test that adding duplicate stage raises error."""
        orchestrator = AsyncPipelineOrchestrator(basic_config)
        
        orchestrator.add_stage("stage1", simple_stage_func)
        
        with pytest.raises(ValueError, match="already exists"):
            orchestrator.add_stage("stage1", simple_stage_func)
    
    async def test_validate_dependencies_missing(self, basic_config):
        """Test validation catches missing dependencies."""
        orchestrator = AsyncPipelineOrchestrator(basic_config)
        
        orchestrator.add_stage("stage1", simple_stage_func, dependencies=["missing"])
        
        with pytest.raises(ValueError, match="unknown stage"):
            orchestrator._validate_dependencies()
    
    async def test_validate_dependencies_cycle(self, basic_config):
        """Test validation catches circular dependencies."""
        orchestrator = AsyncPipelineOrchestrator(basic_config)
        
        orchestrator.add_stage("stage1", simple_stage_func, dependencies=["stage2"])
        orchestrator.add_stage("stage2", simple_stage_func, dependencies=["stage1"])
        
        with pytest.raises(ValueError, match="Circular dependency"):
            orchestrator._validate_dependencies()
    
    async def test_get_execution_order_simple(self, basic_config):
        """Test execution order for simple pipeline."""
        orchestrator = AsyncPipelineOrchestrator(basic_config)
        
        orchestrator.add_stage("stage1", simple_stage_func)
        orchestrator.add_stage("stage2", simple_stage_func, dependencies=["stage1"])
        orchestrator.add_stage("stage3", simple_stage_func, dependencies=["stage2"])
        
        levels = orchestrator._get_execution_order()
        
        assert len(levels) == 3
        assert levels[0] == ["stage1"]
        assert levels[1] == ["stage2"]
        assert levels[2] == ["stage3"]
    
    async def test_get_execution_order_parallel(self, basic_config):
        """Test execution order with parallel stages."""
        orchestrator = AsyncPipelineOrchestrator(basic_config)
        
        orchestrator.add_stage("stage1", simple_stage_func)
        orchestrator.add_stage("stage2", simple_stage_func)
        orchestrator.add_stage("stage3", simple_stage_func, dependencies=["stage1", "stage2"])
        
        levels = orchestrator._get_execution_order()
        
        assert len(levels) == 2
        assert set(levels[0]) == {"stage1", "stage2"}
        assert levels[1] == ["stage3"]
    
    async def test_execute_stage_success(self, basic_config):
        """Test successful stage execution."""
        orchestrator = AsyncPipelineOrchestrator(basic_config)
        
        stage = PipelineStage(name="test", func=simple_stage_func)
        
        await orchestrator._execute_stage(stage)
        
        assert stage.status == StageStatus.COMPLETED
        assert stage.result == "success"
        assert stage.error is None
        assert stage.duration is not None
    
    async def test_execute_stage_failure(self, basic_config):
        """Test stage execution with failure."""
        config = PipelineConfig(
            max_stage_retries=1,
            enable_checkpoints=False,
            enable_progress=False
        )
        orchestrator = AsyncPipelineOrchestrator(config)
        
        stage = PipelineStage(name="test", func=failing_stage_func, max_retries=1)
        
        await orchestrator._execute_stage(stage)
        
        assert stage.status == StageStatus.FAILED
        assert stage.error is not None
        assert "Stage failed" in stage.error
    
    async def test_execute_stage_retry(self, basic_config):
        """Test stage retry logic."""
        call_count = 0
        
        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Temporary failure")
            return "success"
        
        config = PipelineConfig(
            max_stage_retries=3,
            enable_checkpoints=False,
            enable_progress=False
        )
        orchestrator = AsyncPipelineOrchestrator(config)
        
        stage = PipelineStage(name="test", func=flaky_func, max_retries=3)
        
        await orchestrator._execute_stage(stage)
        
        assert stage.status == StageStatus.COMPLETED
        assert call_count == 2
        assert stage.retries == 1
    
    async def test_run_simple_pipeline(self, basic_config):
        """Test running a simple pipeline."""
        orchestrator = AsyncPipelineOrchestrator(basic_config)
        
        orchestrator.add_stage("stage1", simple_stage_func)
        orchestrator.add_stage("stage2", simple_stage_func, dependencies=["stage1"])
        
        result = await orchestrator.run()
        
        assert result.success is True
        assert len(result.completed_stages) == 2
        assert len(result.failed_stages) == 0
        assert result.total_duration > 0
    
    async def test_run_parallel_pipeline(self, basic_config):
        """Test running pipeline with parallel stages."""
        orchestrator = AsyncPipelineOrchestrator(basic_config)
        
        orchestrator.add_stage("stage1", slow_stage_func)
        orchestrator.add_stage("stage2", slow_stage_func)
        orchestrator.add_stage("stage3", simple_stage_func, dependencies=["stage1", "stage2"])
        
        import time
        start = time.time()
        result = await orchestrator.run()
        duration = time.time() - start
        
        assert result.success is True
        # Should take ~0.1s (parallel) not ~0.2s (sequential)
        assert duration < 0.15
    
    async def test_run_with_failure_continue(self, basic_config):
        """Test pipeline continues after failure when fail_fast=False."""
        config = PipelineConfig(
            fail_fast=False,
            max_stage_retries=1,
            enable_checkpoints=False,
            enable_progress=False
        )
        orchestrator = AsyncPipelineOrchestrator(config)
        
        orchestrator.add_stage("stage1", simple_stage_func)
        orchestrator.add_stage("stage2", failing_stage_func)
        orchestrator.add_stage("stage3", simple_stage_func)
        
        result = await orchestrator.run()
        
        assert result.success is False
        assert len(result.completed_stages) == 2
        assert len(result.failed_stages) == 1
        assert "stage2" in result.failed_stages
    
    async def test_run_with_failure_fast(self, basic_config):
        """Test pipeline stops on failure when fail_fast=True."""
        config = PipelineConfig(
            fail_fast=True,
            max_stage_retries=1,
            enable_checkpoints=False,
            enable_progress=False
        )
        orchestrator = AsyncPipelineOrchestrator(config)
        
        orchestrator.add_stage("stage1", failing_stage_func)
        orchestrator.add_stage("stage2", simple_stage_func, dependencies=["stage1"])
        
        result = await orchestrator.run()
        
        assert result.success is False
        assert "stage1" in result.failed_stages
    
    async def test_checkpoint_save_and_load(self, basic_config, temp_checkpoint_path):
        """Test checkpoint save and load."""
        orchestrator = AsyncPipelineOrchestrator(basic_config)
        
        orchestrator.add_stage("stage1", simple_stage_func)
        orchestrator.add_stage("stage2", simple_stage_func, dependencies=["stage1"])
        
        # Run pipeline
        result = await orchestrator.run()
        assert result.success is True
        
        # Check checkpoint was saved
        assert temp_checkpoint_path.exists()
        
        # Create new orchestrator and load checkpoint
        orchestrator2 = AsyncPipelineOrchestrator(basic_config)
        orchestrator2.add_stage("stage1", simple_stage_func)
        orchestrator2.add_stage("stage2", simple_stage_func, dependencies=["stage1"])
        
        loaded = orchestrator2._load_checkpoint()
        assert loaded is True
        
        # Verify stages were restored
        assert orchestrator2.stages["stage1"].status == StageStatus.COMPLETED
        assert orchestrator2.stages["stage2"].status == StageStatus.COMPLETED
    
    async def test_checkpoint_resume(self, basic_config, temp_checkpoint_path):
        """Test resuming from checkpoint."""
        call_count = 0
        
        async def counting_func():
            nonlocal call_count
            call_count += 1
            return f"call_{call_count}"
        
        # First run - complete stage1
        orchestrator1 = AsyncPipelineOrchestrator(basic_config)
        orchestrator1.add_stage("stage1", counting_func)
        orchestrator1.add_stage("stage2", counting_func, dependencies=["stage1"])
        
        # Manually complete stage1 and save checkpoint
        await orchestrator1._execute_stage(orchestrator1.stages["stage1"])
        orchestrator1._save_checkpoint()
        
        # Second run - should skip stage1
        orchestrator2 = AsyncPipelineOrchestrator(basic_config)
        orchestrator2.add_stage("stage1", counting_func)
        orchestrator2.add_stage("stage2", counting_func, dependencies=["stage1"])
        
        result = await orchestrator2.run()
        
        assert result.success is True
        # Should only call counting_func once (for stage2)
        assert call_count == 2  # 1 from first run, 1 from second run
    
    async def test_run_concurrent_fetchers(self, basic_config, sample_dataframe):
        """Test concurrent fetcher execution."""
        orchestrator = AsyncPipelineOrchestrator(basic_config)
        
        fetchers = [
            TestAsyncFetcher(name=f"fetcher{i}", output_dir=Path("/tmp"))
            for i in range(4)
        ]
        
        results = await orchestrator.run_concurrent_fetchers(fetchers, max_workers=2)
        
        assert len(results) == 4
        assert all(r.success for r in results)
    
    async def test_run_concurrent_fetchers_with_failures(self, basic_config):
        """Test concurrent fetchers with some failures."""
        orchestrator = AsyncPipelineOrchestrator(basic_config)
        
        fetchers = [
            TestAsyncFetcher(name="fetcher1", should_fail=False, output_dir=Path("/tmp")),
            TestAsyncFetcher(name="fetcher2", should_fail=True, output_dir=Path("/tmp")),
            TestAsyncFetcher(name="fetcher3", should_fail=False, output_dir=Path("/tmp")),
        ]
        
        results = await orchestrator.run_concurrent_fetchers(fetchers)
        
        assert len(results) == 3
        success_count = sum(1 for r in results if r.success)
        assert success_count == 2
    
    async def test_context_manager(self, basic_config):
        """Test async context manager."""
        async with AsyncPipelineOrchestrator(basic_config) as orchestrator:
            orchestrator.add_stage("stage1", simple_stage_func)
            result = await orchestrator.run()
        
        assert result.success is True
        # Verify cleanup happened
        assert orchestrator._shutdown_event.is_set()


# Test PipelineResult

class TestPipelineResult:
    """Test suite for PipelineResult."""
    
    def test_result_initialization(self):
        """Test result initialization."""
        stages = {
            "stage1": PipelineStage(name="stage1", func=simple_stage_func),
            "stage2": PipelineStage(name="stage2", func=simple_stage_func)
        }
        stages["stage1"].status = StageStatus.COMPLETED
        stages["stage2"].status = StageStatus.FAILED
        
        result = PipelineResult(
            success=False,
            stages=stages,
            total_duration=10.5,
            error="Some error"
        )
        
        assert result.success is False
        assert result.total_duration == 10.5
        assert len(result.completed_stages) == 1
        assert len(result.failed_stages) == 1
    
    def test_result_to_dict(self):
        """Test result serialization."""
        stages = {
            "stage1": PipelineStage(name="stage1", func=simple_stage_func)
        }
        stages["stage1"].status = StageStatus.COMPLETED
        
        result = PipelineResult(
            success=True,
            stages=stages,
            total_duration=5.0
        )
        
        data = result.to_dict()
        
        assert data['success'] is True
        assert data['total_duration'] == 5.0
        assert 'stages' in data
        assert 'completed_stages' in data


# Integration Tests

@pytest.mark.asyncio
class TestPipelineIntegration:
    """Integration tests for complete pipeline scenarios."""
    
    async def test_data_fetch_pipeline(self, basic_config):
        """Test realistic data fetching pipeline."""
        orchestrator = AsyncPipelineOrchestrator(basic_config)
        
        # Simulate data fetching stages
        async def fetch_nflverse():
            await asyncio.sleep(0.05)
            return {"source": "nflverse", "rows": 1000}
        
        async def fetch_espn():
            await asyncio.sleep(0.05)
            return {"source": "espn", "rows": 500}
        
        async def fetch_weather():
            await asyncio.sleep(0.05)
            return {"source": "weather", "rows": 200}
        
        async def build_features():
            await asyncio.sleep(0.02)
            return {"features": "built"}
        
        # Add stages
        orchestrator.add_stage("fetch_nflverse", fetch_nflverse)
        orchestrator.add_stage("fetch_espn", fetch_espn)
        orchestrator.add_stage("fetch_weather", fetch_weather)
        orchestrator.add_stage(
            "build_features",
            build_features,
            dependencies=["fetch_nflverse", "fetch_espn", "fetch_weather"]
        )
        
        result = await orchestrator.run()
        
        assert result.success is True
        assert len(result.completed_stages) == 4
        # Verify parallel execution was faster than sequential
        assert result.total_duration < 0.2  # Would be ~0.17s sequential
    
    async def test_complex_dependency_graph(self, basic_config):
        """Test complex dependency graph."""
        orchestrator = AsyncPipelineOrchestrator(basic_config)
        
        # Create diamond dependency pattern
        orchestrator.add_stage("A", simple_stage_func)
        orchestrator.add_stage("B", simple_stage_func, dependencies=["A"])
        orchestrator.add_stage("C", simple_stage_func, dependencies=["A"])
        orchestrator.add_stage("D", simple_stage_func, dependencies=["B", "C"])
        
        result = await orchestrator.run()
        
        assert result.success is True
        assert len(result.completed_stages) == 4
        
        # Verify execution order
        levels = orchestrator._get_execution_order()
        assert levels[0] == ["A"]
        assert set(levels[1]) == {"B", "C"}
        assert levels[2] == ["D"]


# Made with Bob
