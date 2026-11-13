# Weekly run (`tools.weekly_run`)

`tools.weekly_run` is the week-aware front end for `tools.run_pipeline`. It
works out which slate to predict and where we are in the week, runs the live
pipeline for that slate, and always writes a debug log.

```powershell
conda activate nflgpu
python -m tools.weekly_run                     # resolve, run, log
python -m tools.weekly_run --dry-run           # print the plan only (also logged)
python -m tools.weekly_run -- --skip-tomorrow-weather   # pass extra run_pipeline flags after --
python -m tools.weekly_run --week 11 --phase early_week # override what was resolved
python -m tools.weekly_run --no-publish        # skip MattyTheBookie publishing
```

## Running `run_pipeline` directly

`python -m tools.run_pipeline ... --debug` now writes the same debug log by
itself, so no `*>` redirect is needed. The file is `output_wk<N>_debug.txt`,
where `<N>` is the NFL week being predicted:
- regular season: 1-18;
- postseason, as numbered by nflverse: 19 = Wild Card, 20 = Divisional,
  21 = Conference, 22 = Super Bowl.

A timestamped copy is kept under `logs/pipeline/<season>/`.

- `--prediction-week auto --live-run` resolves the season and week from the
  schedule. Before, this combination raised an error.
- `--debug-log <path>` writes the log to a path of your choice.
- `--debug-log none` turns the log off.
- `tools.weekly_run` uses the same logger, so the pipeline it starts does not
  log twice.

## How the week is resolved

`src/utils/nfl_calendar.py` reads `data/raw/nfl_schedules.parquet`. Kickoff
times are `gameday` + `gametime` in US/Eastern. A game counts as final when it
has a score or is more than 4 hours past kickoff. The local schedule can be a
day stale, so the clock is the fallback.

The **target slate** is the earliest week that has a game not yet final. The
**season** comes from the schedule, not the calendar year, so January playoff
runs predict the right season (`run_pipeline --season`).

| Phase | When | Slate | MTB week archive |
|---|---|---|---|
| `early_week` | No game started, first kickoff more than 36h away (Tue/Wed) | Full | Overwritten: this run is the picks of record |
| `pre_kickoff` | No game started, first kickoff within 36h (Thu before TNF) | Full | Kept (the Tue/Wed record stands) |
| `in_progress` | At least one game has kicked off | Partial (started games dropped) | Kept |
| `offseason` | Nothing left on the schedule | — | The runner exits without running |

Every phase runs the standard live recipe: `--start-year 2017
--train-start-year 2017 --max-parallel-data 8 --data-start-delay 1 --use-async
--use-gpu --skip-logit --live-run --debug --season <S> --prediction-week <W>`,
plus `--publish-mtb --mtb-repo-dir C:\Code\mtb`.

## The debug log (always written)

The log goes to `logs/pipeline/<season>/output_wk<NN>_<phase>_<YYYYmmdd-HHMM>_debug.txt`.
It is also written live to `output_wk<N>_debug.txt` in the repo root, where `<N>` is the NFL week. Both
locations are gitignored.

- **Header:**
  - the resolved slate, phase and kickoff window;
  - games started and final, and whether the previous week is scored;
  - the archive policy;
  - the interpreter and key package versions, with a warning when the env is not `nflgpu`;
  - the git branch, commit and dirty state;
  - which API keys are present (names only);
  - the schedule's age and the exact command.
- **Body:** every line of pipeline output, unbuffered so stages appear in order.
- **Footer:**
  - the exit code and duration;
  - per-stage timings;
  - deduplicated issues found in the output: tracebacks, warnings, failures, rate limits (HTTP 429), zero-row fetches and "unavailable" feeds.

Start with the footer after each run.
