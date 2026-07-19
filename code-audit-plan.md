# NFL Predictions Repository — Code Audit & Cleanup Plan

## Overview

A full audit of the `nfl-predictions` repository was performed. The goal is to remove clutter, dead code, unused dependencies, misplaced files, and security risks — leaving a clean, maintainable codebase without altering any production logic.

Findings are grouped into 6 sub-tasks ordered by priority. Each sub-task is scoped to be independent and reviewable.

## Decisions Made

- **Credentials:** Leave `yahoo_temp.py` and `video_from_picture.py` credentials as-is (no rotation required by this plan)
- **Inspection scripts:** Move `inspect_weather.py`, `inspect_features.py`, `inspect_schedule.py` to a new `tools/` folder rather than deleting
- **`.code-workspace` file:** Add to `.gitignore` (contains hard-coded user-specific Anaconda paths)
- **FastAPI / uvicorn:** Confirmed completely unused — zero imports, no API directory, no decorators anywhere in the codebase. Remove from `requirements.txt` only (they are already absent from `pyproject.toml`)

---

## Sub-Task 1 — Remove Security Risks & Exposed Credentials

**Status:** `[x] done`

### Intent
`yahoo_temp.py` contains Yahoo OAuth credentials. `video_from_picture.py` is a completely unrelated personal project file with no relevance to NFL predictions. Both should be removed from the repository. Credentials do not need to be rotated as part of this cleanup.

### Expected Outcomes
- `yahoo_temp.py` deleted from working tree
- `video_from_picture.py` deleted from working tree
- Both files confirmed absent from the repository

### Todo List
1. Delete `yahoo_temp.py` (root level)
2. Delete `video_from_picture.py` (root level)

### Relevant Context
- `yahoo_temp.py` — root level, in `.gitignore` but may have been committed previously
- `video_from_picture.py` — root level, entirely unrelated to the project

---

## Sub-Task 2 — Delete Backup & Temp Junk Files

**Status:** `[x] done`

### Intent
Over 114 numbered backup files exist in `predictions/` (e.g. `predictions.csv.backup_1` through `predictions.csv.backup_114`, and matching `predictions_full.csv.backup_*` variants, plus migration files). These are rotation artifacts from the backup system, not useful for development, and bloat the repository significantly. Root-level temp and log files are similarly disposable.

### Expected Outcomes
- All `.backup_*` and `.migration_*` files deleted from `predictions/`
- Temp files at root deleted
- Log files at root deleted
- Empty placeholder files deleted
- No production code affected

### Todo List
1. Delete all files in `predictions/` matching `*.backup_*` (114+ files)
2. Delete all files in `predictions/` matching `*.migration_*`
3. Delete root-level temp files: `temp_block.txt`, `temp_out.txt`, `temp_script.py`, `tmp_output.txt`, `all_cols.txt`, `inj_cols.txt`
4. Delete root-level log files: `bf_debug.log`, `bf_run.log`, `build_features.log`, `bf_test_output.txt`
5. Delete `run_pipeline.bak` (backup of old batch script)
6. Verify that `.gitignore` is updated to prevent future backup files from being tracked (add patterns like `predictions/*.backup_*`, `predictions/*.migration_*`, `*.log`, `temp_*.txt`, `tmp_*.txt`)

### Relevant Context
- `predictions/` directory
- Backup rotation logic likely in a pipeline script — do not touch the rotation logic itself, only delete the already-generated files
- `.gitignore` at root level

---

## Sub-Task 3 — Delete Redundant, Unrelated & Misplaced Files

**Status:** `[x] done`

### Intent
Several files exist at the root or inside source directories that don't belong: development inspection scripts, a personal audio/photo, timestamped results, IDE workspace settings with hard-coded user paths, and a committed Docker volumes directory.

### Expected Outcomes
- Root-level inspection/debug scripts removed or relocated
- Personal media files removed from `inputs/`
- Stale timestamped results directory cleaned up
- `volumes/` directory removed from git tracking
- `nfl-predictions.code-workspace` either cleaned of hard-coded paths or moved to `.gitignore`
- `STRUTTURA_PROGETTO.md` deleted (Italian-language file with no useful project content)
- `INSTALLATION_GUIDE.md` deleted (duplicate of `INSTALL.md`)

### Todo List
1. Create a `tools/` directory at the project root if it doesn't already exist
2. Move root-level inspection scripts into `tools/`: `inspect_weather.py`, `inspect_features.py`, `inspect_schedule.py`
3. Delete `src/data/test_build_features.py` (debug/inspection script misplaced in source directory — not a real test)
3. Delete `inputs/matty_pimp_daddy.png` and `inputs/scholar_line.mp3` (personal media files)
4. Delete `results/2025_10_31_16.28.34/` (stale timestamped results output)
5. Delete `STRUTTURA_PROGETTO.md` (empty Italian project structure file, only contains timestamps)
6. Delete `INSTALLATION_GUIDE.md` (duplicate of `INSTALL.md`; verify `INSTALL.md` is the more complete version before deleting)
7. Add `volumes/` to `.gitignore` and remove the `volumes/` directory from git tracking (`git rm -r --cached volumes/`)
8. Add `nfl-predictions.code-workspace` to `.gitignore` (contains hard-coded user-specific Anaconda paths that should not be shared)
9. Delete `nfl_predictions.egg-info/` build artifact and add it to `.gitignore` if not already present
10. Delete empty `build/` directory if present; ensure it is in `.gitignore`

### Relevant Context
- `inputs/` directory
- `results/` directory
- `volumes/` directory (Docker volumes for milvus/etcd/minio)
- `nfl-predictions.code-workspace`
- `STRUTTURA_PROGETTO.md`, `INSTALLATION_GUIDE.md` at root
- `nfl_predictions.egg-info/` at root
- `src/data/test_build_features.py`

---

## Sub-Task 4 — Reorganize Test Files

**Status:** `[x] done`

### Intent
Test files exist in three locations: the proper `tests/` directory, the root level, and inside `src/`. Root-level test files are not discovered by pytest and create maintenance confusion. All tests should live under `tests/`.

### Expected Outcomes
- All test files consolidated under `tests/`
- Root-level test files either moved to `tests/` or deleted if they duplicate existing tests
- No test logic is lost
- `pytest` discovers and runs all tests correctly

### Todo List
1. Audit root-level test files and compare against `tests/` to identify duplicates vs. unique tests:
   - `test_benchmark_comparison.py`
   - `test_calibration_monitor.py`
   - `test_clv_tracker.py`
   - `test_live_tracking.py`
   - `test_manual_verification.py`
   - `test_paper_trading.py`
   - `test_walk_forward.py`
2. For each root-level test file:
   - If a matching test already exists in `tests/`, delete the root duplicate
   - If it contains unique tests, move it into the appropriate location under `tests/`
3. Verify `pyproject.toml` `[tool.pytest.ini_options]` has `testpaths = ["tests"]` set correctly
4. Run `pytest --collect-only` to confirm all tests are discovered after reorganization

### Relevant Context
- `tests/` directory (proper pytest structure)
- Root-level `test_*.py` files (7 files)
- `pyproject.toml` — `[tool.pytest.ini_options]` section

---

## Sub-Task 5 — Remove Unused Dependencies

**Status:** `[x] done`

### Intent
Six dependencies in `requirements.txt` have no imports anywhere in the codebase. `fastapi` and `uvicorn` were confirmed completely unused (zero imports, no API directory, no decorators — the app uses Streamlit). `gpt4all`, `einops`, `sentence-transformers`, and `pymilvus` are also absent from all Python files. Removing them reduces install size and eliminates confusion.

### Expected Outcomes
- `gpt4all`, `einops`, `sentence-transformers`, `pymilvus`, `fastapi`, and `uvicorn` removed from `requirements.txt`
- `gpt4all`, `einops`, `sentence-transformers`, and `pymilvus` removed from `pyproject.toml` optional dependency groups (note: `fastapi`/`uvicorn` were already absent from `pyproject.toml`)
- No import errors introduced

### Todo List
1. Remove from `requirements.txt`: `fastapi`, `uvicorn`, `gpt4all`, `einops`, `sentence-transformers`, `pymilvus`
2. Remove from `pyproject.toml` optional dependency groups: `gpt4all`, `einops`, `sentence-transformers`, `pymilvus`
3. Add a note in `docker-compose.yml` (or in this plan) that the milvus Docker service is unused since `pymilvus` has been removed; consider removing the milvus service from `docker-compose.yml` as well
4. Confirm remaining dependencies install cleanly: `pip install -e .` (no errors expected)

### Relevant Context
- `requirements.txt` at root
- `pyproject.toml` — `[project.dependencies]` and `[project.optional-dependencies]` sections
- `docker-compose.yml` — milvus service references `pymilvus` indirectly

---

## Sub-Task 6 — Configuration & Documentation Cleanup

**Status:** `[x] done`

### Intent
Several low-priority but meaningful configuration inconsistencies and missing documentation artifacts can be resolved: pre-commit hook version drift, `.gitignore` gaps, missing `secrets.env.example`, and outdated batch script documentation.

### Expected Outcomes
- `.gitignore` gaps filled (volumes, egg-info, backup files, logs)
- `secrets.env.example` created to document required environment variables (no real values)
- Pre-commit hook versions bumped to latest stable (black 24.x, ruff 0.3+, mypy 1.8+)
- `run_pipeline.bak` confirmed deleted (covered in Sub-Task 2)
- README updated to reference key docs in `docs/` directory

### Todo List
1. Audit `.gitignore` and add missing patterns:
   - `predictions/*.backup_*`
   - `predictions/*.migration_*`
   - `volumes/`
   - `nfl_predictions.egg-info/`
   - `build/`
   - `*.log`
   - `temp_*.txt`
   - `tmp_*.txt`
   - `*.bak`
2. Create `secrets.env.example` at root with placeholder keys:
   - `TOMORROW_API_KEY=your_key_here`
   - `VISUAL_CROSSING_API_KEY=your_key_here`
   - `ODDS_API_KEY=your_key_here`
   - `SPORTSDATAIO_API_KEY=your_key_here`
3. Update pre-commit hook versions in `.pre-commit-config.yaml`:
   - black: bump to `24.x` latest stable
   - ruff: bump to `0.3+` latest stable
   - mypy: bump to `1.8+` latest stable
4. Add a "Documentation" section to `README.md` linking to key files in `docs/`
5. Verify `pyproject.toml` and `requirements.txt` version constraints are consistent — if they diverge, align them and note `pyproject.toml` as the source of truth

### Relevant Context
- `.gitignore` at root
- `.pre-commit-config.yaml` at root
- `pyproject.toml`
- `requirements.txt`
- `README.md`

---

## Notes for Implementor

- Sub-Tasks 1 and 2 are the highest-priority and should be completed first
- Sub-Task 3 and 4 can be run in parallel after Sub-Tasks 1–2
- Sub-Task 5 should only be done after verifying with grep — do not remove a dependency without confirming zero usage
- Sub-Task 6 is non-breaking configuration work; safe to do last
- **Credential rotation** (Yahoo OAuth, weather APIs, odds APIs) must be done by the repository owner — it is outside the scope of code changes in this plan
