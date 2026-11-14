# GitHub Issues Creation Guide - Phase 2A

This guide provides step-by-step instructions for creating all 10 Phase 2A GitHub issues.

## Quick Setup

### Step 1: Create Labels

Go to your repository → Issues → Labels → New Label and create these labels:

```
phase-2a          | Color: #0E8A16 | Phase 2A Foundation tasks
week-1            | Color: #1D76DB | Week 1 tasks
week-2            | Color: #0052CC | Week 2 tasks
week-3            | Color: #5319E7 | Week 3 tasks
week-4            | Color: #E99695 | Week 4 tasks
infrastructure    | Color: #D4C5F9 | Infrastructure improvements
data-quality      | Color: #C5DEF5 | Data quality enhancements
production        | Color: #FBCA04 | Production readiness
testing           | Color: #BFD4F2 | Testing and quality
P0                | Color: #B60205 | Critical priority
P1                | Color: #D93F0B | High priority
P2                | Color: #FBCA04 | Medium priority
```

### Step 2: Create Milestones

Go to Issues → Milestones → New Milestone:

1. **Week 1 Complete** - Due: 1 week from start - "Async pipeline operational"
2. **Week 2 Complete** - Due: 2 weeks from start - "Schema validation live"
3. **Week 3 Complete** - Due: 3 weeks from start - "Monitoring dashboards live"
4. **Week 4 Complete** - Due: 4 weeks from start - "Test coverage >70%"
5. **Phase 2A Complete** - Due: 4 weeks from start - "Foundation milestone achieved"

### Step 3: Create Project Board

Go to Projects → New Project → Board:
- Name: "Phase 2A - Foundation"
- Columns: Backlog, In Progress, In Review, Testing, Done

---

## Issue Templates

Copy each section below into a new GitHub issue.

---

## Issue #1: Create Async Base Data Fetcher Classes

**Title:** `[Phase 2A] Create Async Base Data Fetcher Classes`

**Labels:** `enhancement`, `infrastructure`, `week-1`, `phase-2a`, `P0`

**Milestone:** Week 1 Complete

**Assignee:** Backend Engineer

**Description:**

```markdown
## Description
Create async base classes for all data fetchers to enable concurrent data ingestion and improve pipeline performance.

## Acceptance Criteria
- [ ] Create `AsyncBaseDataFetcher` abstract class in `src/data/async_base_fetcher.py`
- [ ] Implement async versions of all base fetcher methods (`fetch()`, `validate()`, `cache()`)
- [ ] Add async context manager support (`__aenter__`, `__aexit__`)
- [ ] Implement connection pooling for HTTP requests (reuse existing `http_pool.py`)
- [ ] Add proper error handling with async retry logic
- [ ] Create async batch processing capabilities
- [ ] Write comprehensive docstrings and type hints
- [ ] Add unit tests with >90% coverage

## Technical Details
Expected class structure:
- `AsyncBaseDataFetcher` with async fetch/validate/cache methods
- Connection pooling integration
- Async retry logic
- Batch processing support

## Dependencies
- Existing `BaseDataFetcher` class in `src/data/base_fetcher.py`
- `HTTPSessionPool` from `src/utils/http_pool.py`
- `retry` decorator from `src/utils/retry.py`

## Files to Create
- `src/data/async_base_fetcher.py` (~400 lines)

## Testing Requirements
- Unit tests in `tests/test_async_data_fetchers.py`
- Mock async HTTP requests
- Test connection pooling, error handling, retries, batch processing

## Documentation
- Add docstrings to all public methods
- Update `DEVELOPER_ONBOARDING.md` with async patterns
- Create ADR for async architecture decision

## Estimate
3 days
```

---

## Issue #2: Refactor Pipeline Orchestrator for Async

**Title:** `[Phase 2A] Refactor Pipeline Orchestrator for Async`

**Labels:** `enhancement`, `infrastructure`, `week-1`, `phase-2a`, `P0`

**Milestone:** Week 1 Complete

**Assignee:** Backend Engineer

**Description:**

```markdown
## Description
Refactor `tools/run_pipeline.py` to use asyncio for concurrent execution of data fetching and feature building stages.

## Acceptance Criteria
- [ ] Create `src/utils/async_pipeline.py` with async orchestration logic
- [ ] Implement async job scheduler with configurable concurrency limits
- [ ] Add async progress tracking and logging
- [ ] Implement graceful shutdown on errors
- [ ] Add checkpoint support for resuming failed runs
- [ ] Maintain backward compatibility with sync mode (feature flag)
- [ ] Update CLI arguments to support async options
- [ ] Achieve 4-6x speedup on data fetching stage

## Technical Details
Create async pipeline orchestrator that:
- Runs data fetchers concurrently (configurable max workers)
- Handles errors gracefully with checkpointing
- Maintains progress tracking
- Supports both async and sync modes

## Dependencies
- Issue #1 (Async base fetcher classes)
- Existing `tools/run_pipeline.py`
- `src/utils/checkpoints.py`

## Files to Create
- `src/utils/async_pipeline.py` (~700 lines)

## Files to Modify
- `tools/run_pipeline.py` (add async mode support)

## Testing Requirements
- Integration tests in `tests/test_async_pipeline.py`
- Test concurrent execution, error handling, checkpoint/resume
- Benchmark performance improvement

## Performance Targets
- Data fetching: 2-3 hours → 20-30 minutes (4-6x improvement)
- Memory usage: No increase from sync version
- CPU utilization: 60-80% during concurrent phase

## Estimate
4 days
```

---

## Issue #3: Implement Pydantic Schema Models for All Data Sources

**Title:** `[Phase 2A] Implement Pydantic Schema Models for All Data Sources`

**Labels:** `enhancement`, `data-quality`, `week-2`, `phase-2a`, `P0`

**Milestone:** Week 2 Complete

**Assignee:** Backend Engineer

**Description:**

```markdown
## Description
Create comprehensive Pydantic models for all data sources to enable automatic schema validation and catch data issues early.

## Acceptance Criteria
- [ ] Extend `src/utils/schemas.py` with Pydantic models for all data sources
- [ ] Create schemas for: NFLverse (5), ESPN (4), Sportradar (3), Weather APIs (3), Yahoo (1)
- [ ] Add field validators for data types, ranges, and formats
- [ ] Implement custom validators for NFL-specific fields (team abbr, dates, etc.)
- [ ] Add schema versioning support
- [ ] Create schema documentation generator
- [ ] Write comprehensive tests for all schemas

## Data Sources to Schema
1. **NFLverse** (5 schemas): Play-by-play, Roster, Injuries, Team stats, Schedule
2. **ESPN** (4 schemas): Team defense, Player news, Team news, Schedule
3. **Sportradar** (3 schemas): Game summaries, Player stats, Team stats
4. **Weather APIs** (3 schemas): Visual Crossing, NOAA, Tomorrow.io
5. **Yahoo** (1 schema): Odds data

## Dependencies
- Existing `src/utils/schemas.py`
- `src/utils/teams.py` for team validation

## Files to Modify
- `src/utils/schemas.py` (expand from ~200 to ~800 lines)

## Files to Create
- `docs/schemas/` (auto-generated schema documentation)

## Testing Requirements
- Unit tests in `tests/test_schemas.py`
- Test valid data passes validation
- Test invalid data raises appropriate errors
- Test custom validators and schema versioning

## Estimate
3 days
```

---

## Issue #4: Build Schema Validation Framework

**Title:** `[Phase 2A] Build Schema Validation Framework`

**Labels:** `enhancement`, `data-quality`, `week-2`, `phase-2a`, `P1`

**Milestone:** Week 2 Complete

**Assignee:** Backend Engineer

**Description:**

```markdown
## Description
Create a framework to automatically validate all data against schemas during ingestion and provide detailed error reporting.

## Acceptance Criteria
- [ ] Create `src/data_quality/schema_validator.py` with validation logic
- [ ] Implement automatic schema detection based on data source
- [ ] Add detailed validation error reporting with field-level errors
- [ ] Create schema registry for managing multiple schema versions
- [ ] Add validation hooks to all data fetchers
- [ ] Implement validation metrics collection (pass/fail rates)
- [ ] Add validation error alerting
- [ ] Write comprehensive tests

## Technical Details
Create SchemaValidator class that:
- Validates DataFrames against Pydantic schemas
- Provides detailed error reporting
- Tracks validation metrics
- Integrates with data fetchers

## Dependencies
- Issue #3 (Pydantic schemas)
- All data fetchers in `src/data/`

## Files to Create
- `src/data_quality/schema_validator.py` (~450 lines)
- `src/data_quality/schema_registry.py` (~350 lines)

## Files to Modify
- All data fetchers to add validation hooks

## Testing Requirements
- Unit tests in `tests/test_schema_validation.py`
- Test validation with valid/invalid data
- Test error reporting and schema registry

## Metrics to Track
- Validation pass rate (target: 100%)
- Average validation time per dataset
- Most common validation errors

## Estimate
2 days
```

---

## Issue #5: Implement Prometheus Metrics Collection

**Title:** `[Phase 2A] Implement Prometheus Metrics Collection`

**Labels:** `enhancement`, `production`, `week-3`, `phase-2a`, `P0`

**Milestone:** Week 3 Complete

**Assignee:** DevOps Engineer

**Description:**

```markdown
## Description
Add Prometheus metrics collection throughout the pipeline to enable real-time monitoring and alerting.

## Acceptance Criteria
- [ ] Create `src/monitoring/metrics_collector.py` with Prometheus client
- [ ] Add metrics for pipeline stages (duration, success/failure, throughput)
- [ ] Add metrics for data fetchers (API calls, cache hits, errors)
- [ ] Add metrics for model performance (AUC, Brier, prediction count)
- [ ] Add system metrics (memory, CPU, disk usage)
- [ ] Create Prometheus configuration file
- [ ] Add metrics endpoint to expose metrics
- [ ] Write comprehensive tests

## Metrics to Implement

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

## Dependencies
- `prometheus_client` library
- All pipeline stages

## Files to Create
- `src/monitoring/metrics_collector.py` (~400 lines)
- `docker/prometheus/prometheus.yml`

## Files to Modify
- `tools/run_pipeline.py` (add metrics collection)
- All data fetchers (add metrics)
- `src/models/train.py` (add metrics)

## Testing Requirements
- Unit tests in `tests/test_metrics.py`
- Test metric registration, updates, and endpoint

## Estimate
2 days
```

---

## Issue #6: Create Grafana Dashboards

**Title:** `[Phase 2A] Create Grafana Dashboards`

**Labels:** `enhancement`, `production`, `week-3`, `phase-2a`, `P1`

**Milestone:** Week 3 Complete

**Assignee:** DevOps Engineer

**Description:**

```markdown
## Description
Create comprehensive Grafana dashboards to visualize pipeline performance, model metrics, and system health.

## Acceptance Criteria
- [ ] Create main pipeline dashboard with stage durations and success rates
- [ ] Create data quality dashboard with validation metrics
- [ ] Create model performance dashboard with AUC/Brier trends
- [ ] Create system health dashboard with resource usage
- [ ] Add alerting rules for critical metrics
- [ ] Export dashboards as JSON for version control
- [ ] Write dashboard documentation

## Dashboard Specifications

**1. Pipeline Overview Dashboard**
- Pipeline run history (success/failure timeline)
- Stage duration breakdown (stacked bar chart)
- Current pipeline status, average duration (last 7 days)
- Error rate by stage, memory usage over time

**2. Data Quality Dashboard**
- Schema validation pass rate, data freshness by source
- API call success rate, cache hit rate
- Top validation errors, data volume by source

**3. Model Performance Dashboard**
- AUC/Brier trend over time (line charts)
- Accuracy by week, prediction volume
- Feature importance changes, model training duration

**4. System Health Dashboard**
- CPU/Memory/Disk usage, Network I/O
- Active connections, error rate

## Dependencies
- Issue #5 (Prometheus metrics)
- Grafana installation

## Files to Create
- `docker/grafana/dashboards/pipeline_overview.json`
- `docker/grafana/dashboards/data_quality.json`
- `docker/grafana/dashboards/model_performance.json`
- `docker/grafana/dashboards/system_health.json`
- `docker/grafana/provisioning/dashboards.yml`
- `docker/grafana/provisioning/datasources.yml`

## Testing Requirements
- Manual testing of all dashboards
- Verify all panels display data correctly
- Test alerting rules and refresh rates

## Estimate
2 days
```

---

## Issue #7: Set Up Health Check Endpoints

**Title:** `[Phase 2A] Set Up Health Check Endpoints`

**Labels:** `enhancement`, `production`, `week-3`, `phase-2a`, `P1`

**Milestone:** Week 3 Complete

**Assignee:** DevOps Engineer

**Description:**

```markdown
## Description
Create health check endpoints to monitor system status and enable automated health monitoring.

## Acceptance Criteria
- [ ] Create `src/monitoring/health_checks.py` with health check logic
- [ ] Implement `/health` endpoint (basic liveness check)
- [ ] Implement `/health/ready` endpoint (readiness check)
- [ ] Implement `/health/detailed` endpoint (component-level health)
- [ ] Add checks for: database, cache, API connectivity, disk space
- [ ] Add health check metrics to Prometheus
- [ ] Write comprehensive tests

## Health Checks to Implement
1. **Database Check** - Verify SQLite/tuning.db is accessible
2. **Cache Check** - Verify cache directory is writable
3. **Disk Space Check** - Ensure >10GB free space
4. **API Connectivity** - Test connection to key APIs
5. **Model Files Check** - Verify model files exist and are valid
6. **Data Freshness Check** - Ensure data is <24 hours old

## Dependencies
- FastAPI or Flask for HTTP endpoints
- Existing utility modules

## Files to Create
- `src/monitoring/health_checks.py` (~300 lines)
- `src/monitoring/health_server.py` (~150 lines)

## Testing Requirements
- Unit tests in `tests/test_health_checks.py`
- Test each health check individually
- Test overall health aggregation and endpoint responses

## Estimate
1 day
```

---

## Issue #8: Add Unit Tests for Feature Engineering Modules

**Title:** `[Phase 2A] Add Unit Tests for Feature Engineering Modules`

**Labels:** `testing`, `quality`, `week-4`, `phase-2a`, `P1`

**Milestone:** Week 4 Complete

**Assignee:** QA Engineer

**Description:**

```markdown
## Description
Create comprehensive unit tests for all feature engineering modules to increase code coverage from 55% to 70%+.

## Acceptance Criteria
- [ ] Add tests for `src/features/build_features.py` (core logic)
- [ ] Add tests for `src/features/adjusted_efficiency.py`
- [ ] Add tests for `src/features/passing_epa_features.py`
- [ ] Add tests for `src/features/pressure_features.py`
- [ ] Add tests for `src/features/qb_health.py`
- [ ] Add tests for `src/features/redzone_features.py`
- [ ] Add tests for `src/features/volatility.py`
- [ ] Achieve >85% coverage for feature modules

## Modules to Test
1. **build_features.py** - Core feature building logic
2. **adjusted_efficiency.py** - Opponent-adjusted metrics
3. **passing_epa_features.py** - Passing EPA metrics
4. **pressure_features.py** - Pressure metrics
5. **qb_health.py** - QB injury features
6. **redzone_features.py** - Red zone efficiency
7. **volatility.py** - Volatility features

## Dependencies
- pytest, pandas, numpy
- Existing feature modules

## Files to Create
- `tests/test_features.py` (~800 lines)

## Testing Strategy
- Use fixtures for sample data
- Test happy path and edge cases
- Test error handling and data type validation
- Use parametrize for multiple scenarios

## Coverage Targets
- Overall feature module coverage: >85%
- Critical functions: 100%
- Error handling paths: >80%

## Estimate
3 days
```

---

## Issue #9: Add Unit Tests for Data Fetchers

**Title:** `[Phase 2A] Add Unit Tests for Data Fetchers`

**Labels:** `testing`, `quality`, `week-4`, `phase-2a`, `P1`

**Milestone:** Week 4 Complete

**Assignee:** QA Engineer

**Description:**

```markdown
## Description
Create comprehensive unit tests for all data fetcher modules to ensure reliable data ingestion.

## Acceptance Criteria
- [ ] Add tests for all fetchers in `src/data/` directory
- [ ] Mock external API calls
- [ ] Test error handling and retries
- [ ] Test caching behavior
- [ ] Test data validation
- [ ] Achieve >80% coverage for data modules

## Fetchers to Test
1. nflverse.py, espn_players.py, espn_team_defense.py
2. espn_player_news.py, espn_team_news.py
3. sportradar.py, weather.py
4. visualcrossing.py, noaa.py, yahoo.py

## Dependencies
- pytest, unittest.mock, requests-mock
- Existing data fetchers

## Files to Modify
- `tests/test_data_fetchers.py` (expand existing tests)

## Testing Strategy
- Mock all external API calls
- Test successful responses, error responses (4xx, 5xx)
- Test timeout handling, retry logic, caching behavior

## Coverage Targets
- Data fetcher coverage: >80%
- Error handling: 100%
- Cache logic: 100%

## Estimate
2 days
```

---

## Issue #10: Create Integration Tests for Async Pipeline

**Title:** `[Phase 2A] Create Integration Tests for Async Pipeline`

**Labels:** `testing`, `integration`, `week-4`, `phase-2a`, `P1`

**Milestone:** Week 4 Complete

**Assignee:** QA Engineer

**Description:**

```markdown
## Description
Create end-to-end integration tests for the new async pipeline to ensure all components work together correctly.

## Acceptance Criteria
- [ ] Create integration test suite in `tests/test_async_pipeline.py`
- [ ] Test full pipeline execution with async mode
- [ ] Test concurrent data fetching
- [ ] Test error recovery and checkpointing
- [ ] Test performance benchmarks
- [ ] All tests pass in CI/CD

## Test Scenarios
1. **Happy Path** - Full pipeline execution succeeds
2. **Concurrent Fetching** - Multiple fetchers run in parallel
3. **Error Recovery** - Pipeline recovers from failures
4. **Checkpoint Resume** - Pipeline resumes from checkpoint
5. **Performance** - Async is faster than sync
6. **Resource Usage** - Memory stays within limits

## Dependencies
- Issue #1 (Async base fetchers)
- Issue #2 (Async pipeline)
- pytest-asyncio

## Files to Create
- `tests/integration/test_async_pipeline.py` (~500 lines)

## Testing Requirements
- Use test data (small subset)
- Mock external APIs
- Test in isolated environment
- Measure performance metrics

## Performance Benchmarks
- Async pipeline should be 4-6x faster than sync
- Memory usage should not exceed 4GB
- CPU utilization should be 60-80%

## Estimate
2 days
```

---

## Summary Checklist

After creating all issues:

- [ ] All 10 issues created with proper labels
- [ ] All issues assigned to appropriate team members
- [ ] All issues added to Phase 2A project board
- [ ] All issues linked to appropriate milestones
- [ ] Dependencies documented in issue descriptions
- [ ] Estimates added to all issues

## Next Steps

1. ✅ Create all 10 issues using templates above
2. Schedule Week 1 kickoff meeting
3. Set up development branches
4. Begin implementation with Issue #1

**Ready to start Phase 2A! 🚀**