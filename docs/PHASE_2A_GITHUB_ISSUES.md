# Phase 2A GitHub Issues - Foundation (Weeks 1-4)

This document contains detailed GitHub issue templates for Phase 2A tasks. Copy each issue into GitHub with the appropriate labels and assignments.

---

## Week 1: Async Pipeline Architecture

### Issue #1: Create Async Base Data Fetcher Classes

**Labels:** `enhancement`, `infrastructure`, `week-1`, `phase-2a`  
**Assignee:** Backend Engineer  
**Estimate:** 3 days  
**Priority:** P0 (Critical)

#### Description
Create async base classes for all data fetchers to enable concurrent data ingestion and improve pipeline performance.

#### Acceptance Criteria
- [ ] Create `AsyncBaseDataFetcher` abstract class in `src/data/async_base_fetcher.py`
- [ ] Implement async versions of all base fetcher methods (`fetch()`, `validate()`, `cache()`)
- [ ] Add async context manager support (`__aenter__`, `__aexit__`)
- [ ] Implement connection pooling for HTTP requests (reuse existing `http_pool.py`)
- [ ] Add proper error handling with async retry logic
- [ ] Create async batch processing capabilities
- [ ] Write comprehensive docstrings and type hints
- [ ] Add unit tests with >90% coverage

#### Technical Details
```python
# Expected class structure
class AsyncBaseDataFetcher(ABC):
    async def fetch(self, **kwargs) -> FetchResult:
        """Async fetch implementation"""
        pass
    
    async def validate(self, data: Any) -> bool:
        """Async validation"""
        pass
    
    async def cache_result(self, key: str, data: Any) -> None:
        """Async caching"""
        pass
```

#### Dependencies
- Existing `BaseDataFetcher` class in `src/data/base_fetcher.py`
- `HTTPSessionPool` from `src/utils/http_pool.py`
- `retry` decorator from `src/utils/retry.py`

#### Files to Create
- `src/data/async_base_fetcher.py` (~400 lines)

#### Files to Modify
- None (new functionality)

#### Testing Requirements
- Unit tests in `tests/test_async_data_fetchers.py`
- Mock async HTTP requests
- Test connection pooling
- Test error handling and retries
- Test batch processing

#### Documentation
- Add docstrings to all public methods
- Update `DEVELOPER_ONBOARDING.md` with async patterns
- Create ADR for async architecture decision

---

### Issue #2: Refactor Pipeline Orchestrator for Async

**Labels:** `enhancement`, `infrastructure`, `week-1`, `phase-2a`  
**Assignee:** Backend Engineer  
**Estimate:** 4 days  
**Priority:** P0 (Critical)

#### Description
Refactor `tools/run_pipeline.py` to use asyncio for concurrent execution of data fetching and feature building stages.

#### Acceptance Criteria
- [ ] Create `src/utils/async_pipeline.py` with async orchestration logic
- [ ] Implement async job scheduler with configurable concurrency limits
- [ ] Add async progress tracking and logging
- [ ] Implement graceful shutdown on errors
- [ ] Add checkpoint support for resuming failed runs
- [ ] Maintain backward compatibility with sync mode (feature flag)
- [ ] Update CLI arguments to support async options
- [ ] Achieve 4-6x speedup on data fetching stage

#### Technical Details
```python
# Expected async pipeline structure
async def run_async_pipeline(config: PipelineConfig) -> PipelineResult:
    async with AsyncPipelineOrchestrator(config) as orchestrator:
        # Stage 1: Concurrent data fetching
        fetch_results = await orchestrator.run_concurrent_fetchers(
            max_workers=config.max_parallel_data
        )
        
        # Stage 2: Sequential feature building
        features = await orchestrator.build_features(fetch_results)
        
        # Stage 3: Model training
        models = await orchestrator.train_models(features)
        
        return PipelineResult(fetch_results, features, models)
```

#### Dependencies
- Issue #1 (Async base fetcher classes)
- Existing `tools/run_pipeline.py`
- `src/utils/checkpoints.py`

#### Files to Create
- `src/utils/async_pipeline.py` (~700 lines)

#### Files to Modify
- `tools/run_pipeline.py` (add async mode support)

#### Testing Requirements
- Integration tests in `tests/test_async_pipeline.py`
- Test concurrent execution
- Test error handling and recovery
- Test checkpoint/resume functionality
- Benchmark performance improvement

#### Performance Targets
- Data fetching: 2-3 hours → 20-30 minutes (4-6x improvement)
- Memory usage: No increase from sync version
- CPU utilization: 60-80% during concurrent phase

#### Documentation
- Update `README.md` with async pipeline usage
- Add async troubleshooting guide
- Document performance benchmarks

---

## Week 2: Schema Validation Framework

### Issue #3: Implement Pydantic Schema Models for All Data Sources

**Labels:** `enhancement`, `data-quality`, `week-2`, `phase-2a`  
**Assignee:** Backend Engineer  
**Estimate:** 3 days  
**Priority:** P0 (Critical)

#### Description
Create comprehensive Pydantic models for all data sources to enable automatic schema validation and catch data issues early.

#### Acceptance Criteria
- [ ] Extend `src/utils/schemas.py` with Pydantic models for all data sources
- [ ] Create schemas for: NFLverse, ESPN, Sportradar, Weather APIs, Yahoo
- [ ] Add field validators for data types, ranges, and formats
- [ ] Implement custom validators for NFL-specific fields (team abbr, dates, etc.)
- [ ] Add schema versioning support
- [ ] Create schema documentation generator
- [ ] Write comprehensive tests for all schemas

#### Technical Details
```python
# Example schema structure
from pydantic import BaseModel, Field, validator
from datetime import datetime
from typing import Optional

class NFLversePlayByPlaySchema(BaseModel):
    game_id: str = Field(..., regex=r"^\d{4}_\d{2}_[A-Z]{2,3}_[A-Z]{2,3}$")
    play_id: int = Field(..., ge=1)
    posteam: str = Field(..., min_length=2, max_length=3)
    defteam: str = Field(..., min_length=2, max_length=3)
    game_date: datetime
    epa: Optional[float] = Field(None, ge=-10, le=10)
    
    @validator('posteam', 'defteam')
    def validate_team_abbr(cls, v):
        from src.utils.teams import is_valid_team_abbr
        if not is_valid_team_abbr(v):
            raise ValueError(f"Invalid team abbreviation: {v}")
        return v
    
    class Config:
        schema_extra = {
            "version": "1.0.0",
            "source": "nflverse"
        }
```

#### Data Sources to Schema
1. **NFLverse** (5 schemas)
   - Play-by-play data
   - Roster data
   - Injuries data
   - Team stats
   - Schedule data

2. **ESPN** (4 schemas)
   - Team defense stats
   - Player news
   - Team news
   - Schedule data

3. **Sportradar** (3 schemas)
   - Game summaries
   - Player stats
   - Team stats

4. **Weather APIs** (3 schemas)
   - Visual Crossing
   - NOAA
   - Tomorrow.io

5. **Yahoo** (1 schema)
   - Odds data

#### Dependencies
- Existing `src/utils/schemas.py`
- `src/utils/teams.py` for team validation

#### Files to Modify
- `src/utils/schemas.py` (expand from ~200 to ~800 lines)

#### Files to Create
- `docs/schemas/` (auto-generated schema documentation)

#### Testing Requirements
- Unit tests in `tests/test_schemas.py`
- Test valid data passes validation
- Test invalid data raises appropriate errors
- Test custom validators
- Test schema versioning

#### Documentation
- Auto-generate schema docs from Pydantic models
- Add schema evolution guide
- Document validation error handling

---

### Issue #4: Build Schema Validation Framework

**Labels:** `enhancement`, `data-quality`, `week-2`, `phase-2a`  
**Assignee:** Backend Engineer  
**Estimate:** 2 days  
**Priority:** P1 (High)

#### Description
Create a framework to automatically validate all data against schemas during ingestion and provide detailed error reporting.

#### Acceptance Criteria
- [ ] Create `src/data_quality/schema_validator.py` with validation logic
- [ ] Implement automatic schema detection based on data source
- [ ] Add detailed validation error reporting with field-level errors
- [ ] Create schema registry for managing multiple schema versions
- [ ] Add validation hooks to all data fetchers
- [ ] Implement validation metrics collection (pass/fail rates)
- [ ] Add validation error alerting
- [ ] Write comprehensive tests

#### Technical Details
```python
# Expected validator structure
class SchemaValidator:
    def __init__(self, registry: SchemaRegistry):
        self.registry = registry
    
    def validate(self, data: pd.DataFrame, source: str, version: str = "latest") -> ValidationResult:
        schema = self.registry.get_schema(source, version)
        errors = []
        
        for idx, row in data.iterrows():
            try:
                schema.parse_obj(row.to_dict())
            except ValidationError as e:
                errors.append(ValidationError(idx, e.errors()))
        
        return ValidationResult(
            passed=len(errors) == 0,
            total_rows=len(data),
            error_count=len(errors),
            errors=errors
        )
```

#### Dependencies
- Issue #3 (Pydantic schemas)
- All data fetchers in `src/data/`

#### Files to Create
- `src/data_quality/schema_validator.py` (~450 lines)
- `src/data_quality/schema_registry.py` (~350 lines)

#### Files to Modify
- All data fetchers to add validation hooks

#### Testing Requirements
- Unit tests in `tests/test_schema_validation.py`
- Test validation with valid data
- Test validation with invalid data
- Test error reporting
- Test schema registry

#### Metrics to Track
- Validation pass rate (target: 100%)
- Average validation time per dataset
- Most common validation errors
- Schema version usage

---

## Week 3: Monitoring & Observability

### Issue #5: Implement Prometheus Metrics Collection

**Labels:** `enhancement`, `production`, `week-3`, `phase-2a`  
**Assignee:** DevOps Engineer  
**Estimate:** 2 days  
**Priority:** P0 (Critical)

#### Description
Add Prometheus metrics collection throughout the pipeline to enable real-time monitoring and alerting.

#### Acceptance Criteria
- [ ] Create `src/monitoring/metrics_collector.py` with Prometheus client
- [ ] Add metrics for pipeline stages (duration, success/failure, throughput)
- [ ] Add metrics for data fetchers (API calls, cache hits, errors)
- [ ] Add metrics for model performance (AUC, Brier, prediction count)
- [ ] Add system metrics (memory, CPU, disk usage)
- [ ] Create Prometheus configuration file
- [ ] Add metrics endpoint to expose metrics
- [ ] Write comprehensive tests

#### Technical Details
```python
# Expected metrics structure
from prometheus_client import Counter, Histogram, Gauge, Summary

# Pipeline metrics
pipeline_duration = Histogram(
    'pipeline_stage_duration_seconds',
    'Duration of pipeline stages',
    ['stage', 'status']
)

pipeline_runs = Counter(
    'pipeline_runs_total',
    'Total pipeline runs',
    ['status']
)

# Data fetcher metrics
api_calls = Counter(
    'api_calls_total',
    'Total API calls',
    ['source', 'status']
)

cache_hits = Counter(
    'cache_hits_total',
    'Cache hit/miss count',
    ['source', 'result']
)

# Model metrics
model_auc = Gauge(
    'model_auc',
    'Current model AUC',
    ['model_type']
)

predictions_generated = Counter(
    'predictions_generated_total',
    'Total predictions generated',
    ['model_type', 'season', 'week']
)
```

#### Metrics to Implement

**Pipeline Metrics:**
- `pipeline_stage_duration_seconds` (Histogram)
- `pipeline_runs_total` (Counter)
- `pipeline_errors_total` (Counter)
- `pipeline_memory_usage_bytes` (Gauge)

**Data Fetcher Metrics:**
- `api_calls_total` (Counter)
- `api_call_duration_seconds` (Histogram)
- `cache_hits_total` (Counter)
- `data_validation_errors_total` (Counter)

**Model Metrics:**
- `model_auc` (Gauge)
- `model_brier_score` (Gauge)
- `model_training_duration_seconds` (Histogram)
- `predictions_generated_total` (Counter)

#### Dependencies
- `prometheus_client` library
- All pipeline stages

#### Files to Create
- `src/monitoring/metrics_collector.py` (~400 lines)
- `docker/prometheus/prometheus.yml`

#### Files to Modify
- `tools/run_pipeline.py` (add metrics collection)
- All data fetchers (add metrics)
- `src/models/train.py` (add metrics)

#### Testing Requirements
- Unit tests in `tests/test_metrics.py`
- Test metric registration
- Test metric updates
- Test metrics endpoint

#### Documentation
- Document all metrics and their meanings
- Add metrics collection guide
- Create troubleshooting guide

---

### Issue #6: Create Grafana Dashboards

**Labels:** `enhancement`, `production`, `week-3`, `phase-2a`  
**Assignee:** DevOps Engineer  
**Estimate:** 2 days  
**Priority:** P1 (High)

#### Description
Create comprehensive Grafana dashboards to visualize pipeline performance, model metrics, and system health.

#### Acceptance Criteria
- [ ] Create main pipeline dashboard with stage durations and success rates
- [ ] Create data quality dashboard with validation metrics
- [ ] Create model performance dashboard with AUC/Brier trends
- [ ] Create system health dashboard with resource usage
- [ ] Add alerting rules for critical metrics
- [ ] Export dashboards as JSON for version control
- [ ] Write dashboard documentation

#### Dashboard Specifications

**1. Pipeline Overview Dashboard**
- Pipeline run history (success/failure timeline)
- Stage duration breakdown (stacked bar chart)
- Current pipeline status (running/idle)
- Average pipeline duration (last 7 days)
- Error rate by stage
- Memory usage over time

**2. Data Quality Dashboard**
- Schema validation pass rate
- Data freshness by source
- API call success rate
- Cache hit rate
- Top validation errors
- Data volume by source

**3. Model Performance Dashboard**
- AUC trend over time (line chart)
- Brier score trend over time
- Accuracy by week
- Prediction volume
- Feature importance changes
- Model training duration

**4. System Health Dashboard**
- CPU usage
- Memory usage
- Disk usage
- Network I/O
- Active connections
- Error rate

#### Dependencies
- Issue #5 (Prometheus metrics)
- Grafana installation

#### Files to Create
- `docker/grafana/dashboards/pipeline_overview.json`
- `docker/grafana/dashboards/data_quality.json`
- `docker/grafana/dashboards/model_performance.json`
- `docker/grafana/dashboards/system_health.json`
- `docker/grafana/provisioning/dashboards.yml`
- `docker/grafana/provisioning/datasources.yml`

#### Testing Requirements
- Manual testing of all dashboards
- Verify all panels display data correctly
- Test alerting rules
- Test dashboard refresh rates

#### Documentation
- Create dashboard user guide
- Document alerting thresholds
- Add troubleshooting section

---

### Issue #7: Set Up Health Check Endpoints

**Labels:** `enhancement`, `production`, `week-3`, `phase-2a`  
**Assignee:** DevOps Engineer  
**Estimate:** 1 day  
**Priority:** P1 (High)

#### Description
Create health check endpoints to monitor system status and enable automated health monitoring.

#### Acceptance Criteria
- [ ] Create `src/monitoring/health_checks.py` with health check logic
- [ ] Implement `/health` endpoint (basic liveness check)
- [ ] Implement `/health/ready` endpoint (readiness check)
- [ ] Implement `/health/detailed` endpoint (component-level health)
- [ ] Add checks for: database, cache, API connectivity, disk space
- [ ] Add health check metrics to Prometheus
- [ ] Write comprehensive tests

#### Technical Details
```python
# Expected health check structure
from fastapi import FastAPI
from enum import Enum

class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

class HealthCheck:
    def __init__(self):
        self.checks = {
            "database": self.check_database,
            "cache": self.check_cache,
            "disk": self.check_disk_space,
            "apis": self.check_api_connectivity
        }
    
    async def check_health(self) -> dict:
        results = {}
        overall_status = HealthStatus.HEALTHY
        
        for name, check_func in self.checks.items():
            result = await check_func()
            results[name] = result
            if result["status"] == HealthStatus.UNHEALTHY:
                overall_status = HealthStatus.UNHEALTHY
            elif result["status"] == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                overall_status = HealthStatus.DEGRADED
        
        return {
            "status": overall_status,
            "checks": results,
            "timestamp": datetime.utcnow().isoformat()
        }
```

#### Health Checks to Implement
1. **Database Check** - Verify SQLite/tuning.db is accessible
2. **Cache Check** - Verify cache directory is writable
3. **Disk Space Check** - Ensure >10GB free space
4. **API Connectivity** - Test connection to key APIs
5. **Model Files Check** - Verify model files exist and are valid
6. **Data Freshness Check** - Ensure data is <24 hours old

#### Dependencies
- FastAPI or Flask for HTTP endpoints
- Existing utility modules

#### Files to Create
- `src/monitoring/health_checks.py` (~300 lines)
- `src/monitoring/health_server.py` (~150 lines)

#### Testing Requirements
- Unit tests in `tests/test_health_checks.py`
- Test each health check individually
- Test overall health aggregation
- Test endpoint responses

#### Documentation
- Document health check endpoints
- Add health check integration guide
- Document expected response formats

---

## Week 4: Expand Unit Test Coverage

### Issue #8: Add Unit Tests for Feature Engineering Modules

**Labels:** `testing`, `quality`, `week-4`, `phase-2a`  
**Assignee:** QA Engineer  
**Estimate:** 3 days  
**Priority:** P1 (High)

#### Description
Create comprehensive unit tests for all feature engineering modules to increase code coverage from 55% to 70%+.

#### Acceptance Criteria
- [ ] Add tests for `src/features/build_features.py` (core logic)
- [ ] Add tests for `src/features/adjusted_efficiency.py`
- [ ] Add tests for `src/features/passing_epa_features.py`
- [ ] Add tests for `src/features/pressure_features.py`
- [ ] Add tests for `src/features/qb_health.py`
- [ ] Add tests for `src/features/redzone_features.py`
- [ ] Add tests for `src/features/volatility.py`
- [ ] Achieve >85% coverage for feature modules

#### Technical Details
```python
# Example test structure
import pytest
import pandas as pd
from src.features.qb_health import build_qb_health_features

class TestQBHealthFeatures:
    @pytest.fixture
    def sample_injury_data(self):
        return pd.DataFrame({
            'team': ['KC', 'KC', 'BUF'],
            'player': ['P.Mahomes', 'P.Mahomes', 'J.Allen'],
            'position': ['QB', 'QB', 'QB'],
            'status': ['OUT', 'QUESTIONABLE', 'ACTIVE'],
            'game_date': ['2024-01-01', '2024-01-08', '2024-01-01']
        })
    
    def test_qb_status_flag_calculation(self, sample_injury_data):
        result = build_qb_health_features(sample_injury_data)
        assert 'qb_status_flag' in result.columns
        assert result.loc[0, 'qb_status_flag'] == 1  # OUT
        assert result.loc[1, 'qb_status_flag'] == 0.5  # QUESTIONABLE
    
    def test_handles_missing_data(self):
        empty_df = pd.DataFrame()
        result = build_qb_health_features(empty_df)
        assert len(result) == 0
    
    def test_rolling_calculations(self, sample_injury_data):
        result = build_qb_health_features(sample_injury_data)
        assert 'qb_status_delta_rolling3' in result.columns
```

#### Modules to Test
1. **build_features.py** - Core feature building logic
   - Test data loading
   - Test feature merging
   - Test error handling
   - Test output format

2. **adjusted_efficiency.py** - Opponent-adjusted metrics
   - Test efficiency calculations
   - Test rolling windows
   - Test opponent adjustments

3. **passing_epa_features.py** - Passing EPA metrics
   - Test EPA calculations
   - Test league adjustments
   - Test rolling averages

4. **pressure_features.py** - Pressure metrics
   - Test pressure rate calculations
   - Test normalization
   - Test rolling windows

5. **qb_health.py** - QB injury features
   - Test status flag calculations
   - Test rolling deltas
   - Test games started tracking

6. **redzone_features.py** - Red zone efficiency
   - Test trip/TD calculations
   - Test conversion rates
   - Test zero-trip handling

7. **volatility.py** - Volatility features
   - Test volatility scoring
   - Test shrinkage application
   - Test classifier integration

#### Dependencies
- pytest
- pandas
- numpy
- Existing feature modules

#### Files to Create
- `tests/test_features.py` (~800 lines)

#### Testing Strategy
- Use fixtures for sample data
- Test happy path and edge cases
- Test error handling
- Test data type validation
- Use parametrize for multiple scenarios

#### Coverage Targets
- Overall feature module coverage: >85%
- Critical functions: 100%
- Error handling paths: >80%

#### Documentation
- Add docstrings to test functions
- Document test data fixtures
- Create testing guide for features

---

### Issue #9: Add Unit Tests for Data Fetchers

**Labels:** `testing`, `quality`, `week-4`, `phase-2a`  
**Assignee:** QA Engineer  
**Estimate:** 2 days  
**Priority:** P1 (High)

#### Description
Create comprehensive unit tests for all data fetcher modules to ensure reliable data ingestion.

#### Acceptance Criteria
- [ ] Add tests for all fetchers in `src/data/` directory
- [ ] Mock external API calls
- [ ] Test error handling and retries
- [ ] Test caching behavior
- [ ] Test data validation
- [ ] Achieve >80% coverage for data modules

#### Fetchers to Test
1. **nflverse.py** - NFLverse data fetching
2. **espn_players.py** - ESPN player data
3. **espn_team_defense.py** - ESPN team defense
4. **espn_player_news.py** - ESPN player news
5. **espn_team_news.py** - ESPN team news
6. **sportradar.py** - Sportradar data
7. **weather.py** - Weather data aggregation
8. **visualcrossing.py** - Visual Crossing weather
9. **noaa.py** - NOAA weather
10. **yahoo.py** - Yahoo odds

#### Technical Details
```python
# Example test structure
import pytest
from unittest.mock import Mock, patch
from src.data.nflverse import fetch_nflverse_pbp

class TestNFLverseFetcher:
    @pytest.fixture
    def mock_response(self):
        return Mock(
            status_code=200,
            json=lambda: {'data': [{'game_id': '2024_01_KC_BUF'}]}
        )
    
    @patch('requests.get')
    def test_successful_fetch(self, mock_get, mock_response):
        mock_get.return_value = mock_response
        result = fetch_nflverse_pbp(season=2024)
        assert len(result) > 0
        assert 'game_id' in result.columns
    
    @patch('requests.get')
    def test_handles_api_error(self, mock_get):
        mock_get.side_effect = requests.RequestException("API Error")
        with pytest.raises(requests.RequestException):
            fetch_nflverse_pbp(season=2024)
    
    @patch('requests.get')
    def test_uses_cache(self, mock_get, mock_response):
        mock_get.return_value = mock_response
        # First call
        fetch_nflverse_pbp(season=2024)
        # Second call should use cache
        fetch_nflverse_pbp(season=2024)
        assert mock_get.call_count == 1  # Only called once
```

#### Dependencies
- pytest
- unittest.mock
- requests-mock
- Existing data fetchers

#### Files to Modify
- `tests/test_data_fetchers.py` (expand existing tests)

#### Testing Strategy
- Mock all external API calls
- Test successful responses
- Test error responses (4xx, 5xx)
- Test timeout handling
- Test retry logic
- Test caching behavior

#### Coverage Targets
- Data fetcher coverage: >80%
- Error handling: 100%
- Cache logic: 100%

---

### Issue #10: Create Integration Tests for Async Pipeline

**Labels:** `testing`, `integration`, `week-4`, `phase-2a`  
**Assignee:** QA Engineer  
**Estimate:** 2 days  
**Priority:** P1 (High)

#### Description
Create end-to-end integration tests for the new async pipeline to ensure all components work together correctly.

#### Acceptance Criteria
- [ ] Create integration test suite in `tests/test_async_pipeline.py`
- [ ] Test full pipeline execution with async mode
- [ ] Test concurrent data fetching
- [ ] Test error recovery and checkpointing
- [ ] Test performance benchmarks
- [ ] All tests pass in CI/CD

#### Technical Details
```python
# Example integration test structure
import pytest
import asyncio
from src.utils.async_pipeline import AsyncPipelineOrchestrator

@pytest.mark.asyncio
class TestAsyncPipeline:
    async def test_full_pipeline_execution(self):
        config = PipelineConfig(
            start_year=2024,
            max_parallel_data=4,
            use_async=True
        )
        
        async with AsyncPipelineOrchestrator(config) as orchestrator:
            result = await orchestrator.run()
            
            assert result.success
            assert len(result.fetch_results) > 0
            assert result.features is not None
    
    async def test_concurrent_fetching(self):
        config = PipelineConfig(max_parallel_data=4)
        orchestrator = AsyncPipelineOrchestrator(config)
        
        start_time = time.time()
        results = await orchestrator.run_concurrent_fetchers()
        duration = time.time() - start_time
        
        # Should be faster than sequential
        assert duration < 60  # Less than 1 minute for test data
        assert len(results) > 0
    
    async def test_error_recovery(self):
        config = PipelineConfig(enable_checkpoints=True)
        orchestrator = AsyncPipelineOrchestrator(config)
        
        # Simulate failure
        with pytest.raises(Exception):
            await orchestrator.run_with_failure()
        
        # Resume from checkpoint
        result = await orchestrator.resume()
        assert result.success
```

#### Test Scenarios
1. **Happy Path** - Full pipeline execution succeeds
2. **Concurrent Fetching** - Multiple fetchers run in parallel
3. **Error Recovery** - Pipeline recovers from failures
4. **Checkpoint Resume** - Pipeline resumes from checkpoint
5. **Performance** - Async is faster than sync
6. **Resource Usage** - Memory stays within limits

#### Dependencies
- Issue #1 (Async base fetchers)
- Issue #2 (Async pipeline)
- pytest-asyncio

#### Files to Create
- `tests/integration/test_async_pipeline.py` (~500 lines)

#### Testing Requirements
- Use test data (small subset)
- Mock external APIs
- Test in isolated environment
- Measure performance metrics

#### Performance Benchmarks
- Async pipeline should be 4-6x faster than sync
- Memory usage should not exceed 4GB
- CPU utilization should be 60-80%

---

## Phase 2A Summary

### Week-by-Week Breakdown

**Week 1: Async Pipeline Architecture**
- Issues #1-2
- Deliverables: Async base classes, refactored pipeline
- Team: Backend Engineer (100%)

**Week 2: Schema Validation Framework**
- Issues #3-4
- Deliverables: Pydantic schemas, validation framework
- Team: Backend Engineer (100%)

**Week 3: Monitoring & Observability**
- Issues #5-7
- Deliverables: Prometheus metrics, Grafana dashboards, health checks
- Team: DevOps Engineer (100%)

**Week 4: Expand Unit Test Coverage**
- Issues #8-10
- Deliverables: Feature tests, fetcher tests, integration tests
- Team: QA Engineer (100%)

### Success Criteria for Phase 2A

- [ ] Async pipeline operational and 4-6x faster
- [ ] 100% schema validation coverage
- [ ] Monitoring dashboards live in Grafana
- [ ] Code coverage increased from 55% to 70%+
- [ ] All tests passing in CI/CD
- [ ] Zero critical bugs

### Milestone: Foundation Complete ✓

Upon completion of Phase 2A, the platform will have:
- ✅ Async architecture for 4-6x performance improvement
- ✅ Comprehensive schema validation
- ✅ Production-grade monitoring
- ✅ Significantly improved test coverage
- ✅ Solid foundation for Phase 2B enhancements

---

## GitHub Project Setup

### Labels to Create
- `phase-2a` - Phase 2A tasks
- `week-1`, `week-2`, `week-3`, `week-4` - Week tracking
- `infrastructure` - Infrastructure improvements
- `data-quality` - Data quality enhancements
- `production` - Production readiness
- `testing` - Testing and quality
- `P0`, `P1`, `P2` - Priority levels

### Milestones to Create
1. **Week 1 Complete** - Async pipeline operational
2. **Week 2 Complete** - Schema validation live
3. **Week 3 Complete** - Monitoring dashboards live
4. **Week 4 Complete** - Test coverage >70%
5. **Phase 2A Complete** - Foundation milestone achieved

### Project Board Columns
1. **Backlog** - Not started
2. **In Progress** - Currently working
3. **In Review** - Code review
4. **Testing** - QA testing
5. **Done** - Completed

---

## Next Steps

1. **Create GitHub Issues** - Copy each issue template into GitHub
2. **Assign Team Members** - Assign issues to appropriate team members
3. **Set Up Project Board** - Create project board with columns
4. **Schedule Kickoff** - Schedule Week 1 kickoff meeting
5. **Begin Development** - Start with Issue #1 on Monday

**Ready to start Phase 2A! 🚀**