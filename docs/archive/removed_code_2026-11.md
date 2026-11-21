# Removed code and files (2026-11 cleanup)

Files deleted because no entry point uses them. An import graph was traced from `tools.run_pipeline` (including every
`python -m` job it launches), `tools.weekly_run`, `tools.dispatcher`, `tools.publish_mtb_csvs`, `scripts/doc_agent.py`
and `conftest.py`. The pr17-baseline compare scripts and MTB `scripts/` were also checked. Pipeline outputs are unchanged.

Restore any file from the commit listed for it:

```
git show <commit>:<path>            # view
git checkout <commit> -- <path>     # restore
```

## PR A: tracked junk (restore from 4c72cc1)

| Path | What it was | Why removed |
|---|---|---|
| `tmp/` (20 files) | September 2026 Week 2 debugging CSVs (`week2_*`) and Yahoo OAuth auth-URL scratch files | Scratch output, never read by code; `tmp/` is now gitignored |
| `tests/_backups_20260730_091618/` (20 files) | Copies of tests taken 2026-07-30 | Already excluded from pytest (`norecursedirs`); live tests are in `tests/` |
| `tuning.db` (repo root) | 2025-10 Optuna study | The pipeline uses `tuning/tuning.db` |
| `docker-compose.yml` | Milvus / etcd / MinIO stack | Not part of this project |
| `.github/ISSUE_TEMPLATE/phase2a-week1-issue1.md` | Phase 2A issue draft | Phase 2A plans are archived |
| `test_years.bat` | Echoed the list of years from 2002 to now | Did nothing else |

## PR B: unused framework layer and Sportradar (restore from 4c72cc1)

The April 2026 "enhancement phase 2" layer: generic fetchers, loaders and resilience helpers that the pipeline never adopted. The live fetchers in `src/data/` use `requests` directly. Only their own tests used these modules.

| Path | What it did |
|---|---|
| `src/utils/async_pipeline.py`, `src/utils/async_fetcher.py`, `src/data/async_base_fetcher.py` | Async (httpx/aiofiles) fetch pipeline. `tools.run_pipeline --use-async` uses its own asyncio subprocess runner, not this |
| `src/data/base_fetcher.py` | Abstract API/file fetcher base classes (ADR 0002) |
| `src/utils/data_loader.py`, `src/utils/cache.py`, `src/utils/schemas.py` | Data loader factory (ADR 0001), versioned file cache, pandera schemas |
| `src/utils/retry.py`, `src/utils/circuit_breaker.py`, `src/utils/rate_limit.py`, `src/utils/http.py`, `src/utils/http_pool.py` | Retry decorator (ADR 0003), circuit breaker, token-bucket limiter, Sportradar HTTP client, pooled HTTP session |
| `src/utils/batch_processor.py`, `src/utils/pipeline_recovery.py`, `src/utils/error_context.py`, `src/utils/memory_profiler.py`, `src/utils/performance.py`, `src/utils/checkpoints.py` | Batch runner, checkpoint/recovery, error context, memory and timing helpers |
| `src/utils/logging.py` | Deprecated shim over `src/utils/logging_config.py` (kept) |
| `src/utils/column_docs.py` | Column documentation helper |
| `src/predict/model_strategy.py` | Strategy-pattern model wrapper |
| `src/data/nflsdv.py` | SportsDataverse ESPN schedules/PBP fetcher. `src.data.nflverse` supplies both; cached `espn_schedule*.parquet` files are still read as a fallback |
| `src/data/espn_team_defense.py` | ESPN team-defense scraper |
| `src/data/sportradar.py`, `src/data/sportradar_transform.py`, `tools/sportradar_report.py` | Sportradar fetcher, transform and report. `build_features` still reads previously saved `data/processed/sportradar/*` files as a legacy fallback |
| `tools/enrich_weather_csv.py` | Hand-run tool that added weather columns (kickoff, roof, venue) to a `predictions_full.csv` from `weather_games.parquet`; the pipeline now writes them itself |
| Tests: `test_async_data_fetchers`, `test_async_pipeline`, `test_data_fetchers`, `test_data_loaders`, `test_retry_circuit_breaker`, `test_schemas`, `test_sportradar`, `test_sportradar_report`, `test_sportradar_transform`; the cache, schema, checkpoint, rate-limit, performance and HTTP sections of `test_utils_coverage`, `test_api_contracts`, `test_integration_pipeline`, `test_manual_verification`, `test_property_based` | Tests of the modules above |
| `docs/ASYNC_FETCHER_GUIDE.md` | Guide to the async layer |

Moved to `docs/archive/`: `DEVELOPER_ONBOARDING.md` and ADRs 0001-0003, which describe this layer.
Dropped dependencies (used only by the removed code): `sportsdataverse`, `httpx`, `aiofiles`, `pandera`.
