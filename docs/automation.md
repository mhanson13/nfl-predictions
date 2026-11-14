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
  if the PC was asleep). A failed slot is retried once.
- A run takes ~30 min, so picks are published ~2.5–3h before kickoff.
- Each game's **pick of record** = latest ledger row recorded ≥ 2h before *its* kickoff.
  Thursday's game is locked by the Thursday run, Sunday's games by the Sunday run, etc.

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
