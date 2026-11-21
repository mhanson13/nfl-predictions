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
