# Unattended game-day runs

The pipeline runs itself on every NFL game day (Thu, Sat, Sun, Mon, plus Wed/Fri
holiday games), records leak-free picks of record, publishes the site, reviews the
run, and sends an alert.

## Pieces

| Piece | What it does |
|---|---|
| `tools/install_dispatcher_task.ps1` | Registers Windows task **NFL Pipeline Dispatcher**: every 15 min, hidden, wakes the PC, never overlaps. |
| `tools/dispatcher.py` | Decides whether a game-day slot is due; runs it; keeps `logs/dispatcher/state.json`. |
| `tools/weekly_run.py` | Runs `tools.run_pipeline` for the slot's season/week with the debug log. |
| `src/evaluation/picks_of_record.py` | Pipeline job: per-game ledger + picks of record (`predictions/picks_of_record/`). |
| MTB `scripts/publish_site.py` | Import, lint/typecheck/test/build, commit, push, GKE deploy. Only after a successful run. |
| `tools/review_run.py` + `tools/review_prompt.md` | Facts from the log/ledger + read-only Claude review → `logs/reviews/<season>/<slot>_review.md`. |
| `src/utils/alerts.py` | Alert to `logs/alerts/` always; email via Bird SMTP (or SendGrid) when configured. |

## Timing

- One slot per game day (US/Eastern date), e.g. `2026_wk10_Sun_2026-11-15`. A day with
  morning (international, before noon ET) games plus later games gets a second, earlier
  slot, e.g. `2026_wk10_Sun-early_2026-11-15`, for the morning games.
- Window opens **3h30m before the slot's first kickoff** (06:00 ET for a 09:30 London game,
  09:30 ET for the 1 pm slate).
- New launches are allowed while any of that day's games is **≥ 2h45m away** (catch-up
  if the PC was asleep). A failed slot gets up to 3 attempts; one that fails within
  10 minutes (a crash) is retried a minute later in the same tick.
- A run takes ~30 min, so picks are published ~2.5–3h before kickoff.
- Each game's **pick of record** = latest ledger row recorded ≥ 2h before *its* kickoff.
  Thursday's game is locked by the Thursday run, Sunday's games by the Sunday run, etc.

## Grading results

- From **8h after a week's last kickoff** (Tue ~04:15 ET after Monday night), each
  dispatcher tick with no game-day slot due checks the week. Until every game is final
  (nflverse schedule, fetched in memory) it re-checks hourly, for up to 60h.
- Then it runs `picks_of_record --resolve`, grades the week
  (`src.evaluation.live_results --require-complete --refresh-scores`), copies results to
  MTB (`tools.publish_mtb_csvs --results-only`: every graded week's
  `live_prediction_vs_actual[_summary|_segments]_week_NN.csv`, the cumulative files and
  picks of record) and runs MTB `scripts/publish_site.py`. The alert carries the week's
  record and the season record; game-day alerts repeat the latest grading line.
- **Which pick is graded:** each game's pick of record (locked, else the first `late`
  row -- the same rule MTB's week archive uses); games without one fall back to the MTB
  archive CSV (the week's first published snapshot). Weeks 2-9 have no ledger, so they
  grade the archive CSV.
- Cumulative `live_prediction_vs_actual.csv` / `_summary.csv` hold every graded week;
  re-grading a week replaces that week's rows.
- Backfill / regrade by hand:
  `python -m src.evaluation.live_results --season 2026 --completed-weeks` then
  `python -m tools.publish_mtb_csvs --season 2026 --results-only`. State key:
  `grade_<season>_wkNN` in `logs/dispatcher/state.json` (delete it to retry).

## When something dies

- **A data job** (nflverse, ESPN, etc.) that exits non-zero logs
  `[pipeline] ERROR: data job <name> died: <exit status> (attempt 1/2)` with Windows crash codes
  decoded (e.g. `0xC0000409: stack buffer overrun / fail-fast`), waits 30s and runs once more
  (`DATA_JOB_RETRIES` / `DATA_JOB_RETRY_DELAY` in `tools/run_pipeline.py`). If it still fails, one
  line names every failed job (`[pipeline] ERROR: N required data job(s) failed; stopping the run`)
  and the run exits non-zero. Optional jobs (`yahoo_static`, `noaa_weather`) only warn. A job that
  hangs is not timed out.
- **The run-log wrapper itself** (`src/utils/run_log.py`) failing writes
  `[run_log] ERROR: run_log wrapper failed (...); pipeline stopped` plus the traceback into the
  debug log, stops the pipeline it launched (so nothing publishes unwatched), and still writes the
  run summary footer. The 2026-11-15 Sunday-early failures were this case (a cp1252 console and a
  tqdm progress bar), which PR #12 fixed; before, the traceback only reached
  `logs/dispatcher/dispatcher.log`.

## Leakage guards

- Training excludes the whole current week (`--live-run`), so later-in-week runs never
  learn from that week's earlier results.
- Games already kicked off are never recorded; ledger rows are never rewritten.
- The ledger refuses a `predictions.csv` older than the current run.
- Each row fingerprints the game's slow-moving features; a change between a game's
  picks (other than odds/injuries/news/weather) logs a warning the review escalates.

## Operating

```powershell
pwsh -File tools\install_dispatcher_task.ps1               # install / update
python -m tools.dispatcher --dry-run                       # upcoming slots + what is due
python -m tools.dispatcher --dry-run --now 2026-11-15T11:10:00+00:00
Get-Content logs\dispatcher\dispatcher.log -Tail 50
pwsh -File tools\install_dispatcher_task.ps1 -Uninstall    # stop automation
```

- Re-run a failed slot by hand: `python -m tools.weekly_run --season 2026 --week 10`,
  or delete the slot from `logs/dispatcher/state.json` to let the dispatcher retry.
- The task runs only while you are logged on (keeps git/gh/npm credentials working).
  Leave the PC signed in (locking the screen is fine); it may sleep — the task wakes it.
- Email: set `BIRD_ACCESS_KEY` (Bird API key with the emails scope; it is the SMTP password for
  us1.smtp.bird.com:587, STARTTLS), `ALERT_EMAIL_FROM`, `ALERT_EMAIL_TO` in `secrets.env`.
  `SENDGRID_API_KEY` works as a fallback provider. Until then alerts are only written to `logs/alerts/`.
- Review agent: uses the Claude Code CLI (`$CLAUDE_BIN`, PATH, or the VS Code extension
  binary). Without it the review contains the extracted facts only.
