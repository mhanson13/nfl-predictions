"""Schedule-driven dispatcher for unattended live runs (Task Scheduler entry point).

Run this every 15 minutes (see tools/install_dispatcher_task.ps1). Each call is
cheap: it reads the local schedule, finds today's game-day slot, and only starts
work when the slot's launch window is open and the slot has not succeeded yet.

Slots: one per game day (US/Eastern date) -- Thu, Sat, Sun, Mon, plus any
Wed/Fri/holiday games. A day with morning (international) games plus later games
gets two slots: ``<Day>-early`` for the morning games and ``<Day>`` for the rest. The launch window opens ``--start-lead`` before the day's
first kickoff (default 3h30m) and stays open while at least one of that day's
games is still ``--last-start-lead`` or more away (default 2h45m), so a missed
window (PC asleep) still catches the later games. A failed slot gets up to
``MAX_ATTEMPTS`` tries while the window is open; an attempt that fails within
``QUICK_FAIL`` (a crash, not a slow run) is retried in the same tick after
``QUICK_RETRY_DELAY`` instead of waiting 15 minutes for the next one.

For a slot it:
  1. runs ``tools.weekly_run`` for the slot's season/week (debug log + picks of
     record ledger; ``NFL_RUN_SLOT`` is set so ledger rows carry the slot id),
  2. on success publishes the site (MTB ``scripts/publish_site.py``: import,
     lint/typecheck/test/build, commit, push, deploy) -- never after a failure,
  3. runs the review agent (``tools.review_run``),
  4. sends one alert (``src.utils.alerts``) with the outcome and the review.

Grading: once a week's last game is final (from ``GRADE_DELAY`` after its last
kickoff -- Tuesday early morning for a Monday night game), it resolves the picks
of record, grades the week against final scores (``src.evaluation.live_results``,
the pick MTB showed), copies the results to MTB (``tools.publish_mtb_csvs
--results-only``) and publishes the site, then alerts with the record. Until the
scores are final it re-checks hourly; game-day slots always run first. Game-day
alerts also carry the latest grading outcome.

State lives in ``logs/dispatcher/state.json``; a lock file prevents overlap.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence

import pandas as pd

from src.utils.nfl_calendar import DEFAULT_SCHEDULE_PATH, EASTERN, _kickoffs

STATE_DIR = Path("logs/dispatcher")
STATE_PATH = STATE_DIR / "state.json"
LOCK_PATH = STATE_DIR / "running.lock"
DEFAULT_MTB_REPO = Path(r"C:\Code\mtb")
START_LEAD = timedelta(hours=3, minutes=30)
LAST_START_LEAD = timedelta(hours=2, minutes=45)
MAX_ATTEMPTS = 3
QUICK_FAIL = timedelta(minutes=10)      # a failure this fast is a crash; retry right away
QUICK_RETRY_DELAY = 60.0                # seconds before that in-tick retry
STALE_LOCK = timedelta(hours=4)
SLOT_ENV = "NFL_RUN_SLOT"
EARLY_CUTOFF_HOUR_ET = 12  # games before noon ET (London/Dublin/Berlin) get their own slot
GRADE_DELAY = timedelta(hours=8)        # after a week's last kickoff (MNF 8:15 pm ET -> Tue ~4 am ET)
GRADE_RECHECK = timedelta(hours=1)      # scores not final yet / retry after an error
GRADE_WINDOW = timedelta(hours=60)      # stop trying after this; older weeks are a manual backfill
GRADE_MAX_FAILURES = 4
GRADE_EXIT_NOT_FINAL = 3                # src.evaluation.live_results EXIT_NOT_FINAL


@dataclass(frozen=True)
class Slot:
    slot_id: str
    season: int
    week: int
    day_et: str
    first_kickoff: datetime
    last_kickoff: datetime
    kickoffs: tuple[datetime, ...]
    game_ids: tuple[str, ...]

    def window_open(self, now: datetime, start_lead: timedelta, last_start_lead: timedelta) -> bool:
        if now < self.first_kickoff - start_lead:
            return False
        return any(k - now >= last_start_lead for k in self.kickoffs)


def build_slots(schedule: pd.DataFrame) -> list[Slot]:
    sched = schedule.copy()
    sched["kickoff"] = _kickoffs(sched)
    sched = sched.dropna(subset=["kickoff"])
    sched["day_et"] = sched["kickoff"].dt.strftime("%Y-%m-%d")
    sched["early"] = sched["kickoff"].dt.hour < EARLY_CUTOFF_HOUR_ET
    slots = []
    for (season, day), day_games in sched.groupby(["season", "day_et"]):
        # Morning (international) games get their own earlier slot when the day also
        # has later games, so the 1 pm / 4 pm / night games are picked closer to kickoff.
        groups = [day_games]
        if day_games["early"].any() and not day_games["early"].all():
            groups = [day_games[day_games["early"]], day_games[~day_games["early"]]]
        for games in groups:
            slots.append(_slot(int(season), day, games, early=len(groups) > 1 and bool(games["early"].all())))
    return sorted(slots, key=lambda s: s.first_kickoff)


def _slot(season: int, day: str, games: pd.DataFrame, *, early: bool) -> Slot:
    games = games.sort_values("kickoff")
    kicks = tuple(k.to_pydatetime().astimezone(timezone.utc) for k in games["kickoff"])
    week = int(games["week"].iloc[0])
    dow = games["kickoff"].iloc[0].strftime("%a") + ("-early" if early else "")
    return Slot(
        slot_id=f"{season}_wk{week:02d}_{dow}_{day}",
        season=season,
        week=week,
        day_et=day,
        first_kickoff=kicks[0],
        last_kickoff=kicks[-1],
        kickoffs=kicks,
        game_ids=tuple(games["game_id"].astype(str)),
    )


def load_state(path: Path = STATE_PATH) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(state: dict, path: Path = STATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def due_slot(
    slots: Sequence[Slot],
    state: dict,
    now: datetime,
    *,
    start_lead: timedelta = START_LEAD,
    last_start_lead: timedelta = LAST_START_LEAD,
    max_attempts: int = MAX_ATTEMPTS,
) -> Optional[Slot]:
    """The slot to run now, or None (not in a window, already succeeded, or out of retries)."""
    for slot in slots:
        if not slot.window_open(now, start_lead, last_start_lead):
            continue
        rec = state.get(slot.slot_id, {})
        if rec.get("status") == "success":
            continue
        if int(rec.get("attempts", 0)) >= max_attempts:
            continue
        return slot
    return None


def grade_key(season: int, week: int) -> str:
    return f"grade_{season}_wk{week:02d}"


def week_last_kickoffs(slots: Sequence[Slot]) -> dict[tuple[int, int], datetime]:
    ends: dict[tuple[int, int], datetime] = {}
    for s in slots:
        key = (s.season, s.week)
        ends[key] = max(ends.get(key, s.last_kickoff), s.last_kickoff)
    return ends


def due_grading(
    slots: Sequence[Slot],
    state: dict,
    now: datetime,
    *,
    delay: timedelta = GRADE_DELAY,
    recheck: timedelta = GRADE_RECHECK,
    window: timedelta = GRADE_WINDOW,
    max_failures: int = GRADE_MAX_FAILURES,
) -> Optional[tuple[int, int]]:
    """The (season, week) to grade now, or None.

    A week becomes due ``delay`` after its last kickoff and stays due for ``window``
    (older weeks are left to a manual backfill), re-checking every ``recheck`` until
    it is graded or has failed ``max_failures`` times.
    """
    for (season, week), last in sorted(week_last_kickoffs(slots).items(), key=lambda kv: kv[1]):
        if not (last + delay <= now <= last + window):
            continue
        rec = state.get(grade_key(season, week), {})
        if rec.get("status") == "success" or int(rec.get("failures", 0)) >= max_failures:
            continue
        last_try = rec.get("last_attempt")
        if last_try and now - datetime.fromisoformat(last_try) < recheck:
            continue
        return season, week
    return None


def run_grading(season: int, week: int, args: argparse.Namespace, *, runner: Optional[Callable] = None) -> dict:
    """Resolve picks of record, grade the week, copy results to MTB, publish the site."""
    runner = runner or _run
    py = sys.executable
    report_path = STATE_DIR / f"{grade_key(season, week)}.json"
    report_path.unlink(missing_ok=True)
    log = (STATE_DIR / f"{grade_key(season, week)}.txt").resolve()
    result: dict = {"season": season, "week": week}

    runner([py, "-m", "src.evaluation.picks_of_record", "--resolve", "--season", str(season)],
           timeout=10 * 60, log=log)
    rc = runner([py, "-m", "src.evaluation.live_results", "--season", str(season), "--week", str(week),
                 "--require-complete", "--refresh-scores", "--report-json", str(report_path),
                 "--mtb-csv-root", str(Path(args.mtb_repo_dir) / "data" / "prediction-csvs")],
                timeout=20 * 60, log=log)
    result["grade_rc"] = rc
    try:
        result["report"] = json.loads(report_path.read_text(encoding="utf-8")).get("report", "")
    except Exception:
        result["report"] = ""
    if rc == GRADE_EXIT_NOT_FINAL:
        result["status"] = "pending"
        return result
    if rc != 0:
        result["status"] = "failed"
        return result

    publish_rc = None
    if not args.no_publish:
        publish_rc = runner([py, "-m", "tools.publish_mtb_csvs", "--repo-dir", str(args.mtb_repo_dir),
                             "--season", str(season), "--results-only"], timeout=10 * 60, log=log)
        if publish_rc == 0 and not args.no_site_deploy:
            site_log = (STATE_DIR / f"{grade_key(season, week)}_publish.txt").resolve()
            publish_rc = runner(
                [py, "scripts/publish_site.py", "--commit-message",
                 f"Publish graded results {season} week {week}"],
                cwd=args.mtb_repo_dir, timeout=args.publish_timeout_min * 60, log=site_log,
            )
            result["publish_log"] = str(site_log)
    result["publish_rc"] = publish_rc
    result["status"] = "success" if publish_rc in (None, 0) else "failed"
    return result


def grading_summary(state: dict) -> Optional[str]:
    """One line on the most recent grading attempt, for game-day alerts."""
    recs = [(k, v) for k, v in state.items() if k.startswith("grade_") and isinstance(v, dict)]
    if not recs:
        return None
    key, rec = max(recs, key=lambda kv: kv[1].get("last_attempt", ""))
    last = (rec.get("report") or "").strip().splitlines()
    tail = f" -- {last[-1]}" if last else ""
    return f"Results grading {key}: {rec.get('status')} (at {rec.get('last_attempt')}){tail}"


def send_grading_alert(season: int, week: int, result: dict, *, final: bool) -> None:
    from src.utils.alerts import send_alert

    from tools.review_run import blog_failure_lines, blog_failures, with_blog_note

    status = result["status"]
    blog = blog_failures(result.get("publish_log"))
    base = "RESULTS" if status == "success" else "RESULTS FAILED"
    if "ALERT:" in (result.get("report") or ""):
        base += " ALERT"  # e.g. a final game with no pick of record that is not on the known list
    label = with_blog_note(base, blog)
    lines = [
        f"Week: {season} week {week}",
        f"Grade exit: {result.get('grade_rc')}   MTB publish exit: {result.get('publish_rc')}",
        f"Log: {STATE_DIR / (grade_key(season, week) + '.txt')}",
    ]
    if result.get("publish_log"):
        lines.append(f"Publish log: {result['publish_log']}")
    if status != "success":
        lines.append("Results were not published. " + (
            "Giving up; grade by hand (docs/automation.md)." if final else "Will retry within the hour."))
    lines += blog_failure_lines(blog)
    body = "\n".join(lines)
    if result.get("report"):
        body += "\n\n" + result["report"]
    send_alert(f"[NFL pipeline] {label}: {season} week {week}", body)


def handle_grading(season: int, week: int, args: argparse.Namespace, state: dict, now: datetime, *,
                   runner: Optional[Callable] = None, alert: Optional[Callable] = None) -> dict:
    """Run one grading attempt, update its state record, alert on success / first and last failure."""
    rec = state.setdefault(grade_key(season, week), {})
    rec["last_attempt"] = now.isoformat(timespec="seconds")
    rec["attempts"] = int(rec.get("attempts", 0)) + 1
    result = run_grading(season, week, args, runner=runner)
    rec.update({k: v for k, v in result.items() if k not in ("season", "week")})
    if result["status"] == "failed":
        rec["failures"] = int(rec.get("failures", 0)) + 1
    final = int(rec.get("failures", 0)) >= GRADE_MAX_FAILURES
    if result["status"] == "pending":
        print(f"[dispatcher] {grade_key(season, week)}: scores not final yet; re-checking later")
    elif alert is not None and (result["status"] == "success" or rec.get("failures") == 1 or final):
        alert(season, week, result, final=final)
    return rec


def acquire_lock(now: datetime, path: Path = LOCK_PATH) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            started = datetime.fromisoformat(json.loads(path.read_text())["started"])
            if now - started < STALE_LOCK:
                return False
        except Exception:
            pass
        path.unlink(missing_ok=True)
    path.write_text(json.dumps({"pid": os.getpid(), "started": now.isoformat()}))
    return True


def release_lock(path: Path = LOCK_PATH) -> None:
    path.unlink(missing_ok=True)


def keep_awake(on: bool) -> None:
    """Stop Windows from sleeping mid-run (no-op elsewhere)."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0)
        ctypes.windll.kernel32.SetThreadExecutionState(flags)
    except Exception:
        pass


def newest_debug_log(season: int, since: float, log_dir: Path = Path("logs/pipeline")) -> Optional[Path]:
    logs = [p for p in (log_dir / str(season)).glob("output_wk*_debug.txt") if p.stat().st_mtime >= since]
    return max(logs, key=lambda p: p.stat().st_mtime) if logs else None


def _run(cmd: Sequence[str], *, cwd: Optional[Path] = None, env: Optional[dict] = None,
         timeout: Optional[float] = None, log: Optional[Path] = None) -> int:
    print(f"[dispatcher] $ {' '.join(map(str, cmd))}" + (f"  (cwd={cwd})" if cwd else ""), flush=True)
    out = open(log, "a", encoding="utf-8") if log else None
    try:
        return subprocess.run(list(map(str, cmd)), cwd=cwd, env=env, timeout=timeout,
                              stdout=out, stderr=subprocess.STDOUT if out else None, check=False).returncode
    except subprocess.TimeoutExpired:
        print(f"[dispatcher] timed out after {timeout}s: {cmd[0]}")
        return 124
    finally:
        if out:
            out.close()


def run_slot(slot: Slot, args: argparse.Namespace, *, runner: Callable = _run,
             alert: Optional[Callable] = None) -> dict:
    started = time.time()
    env = dict(os.environ, **{SLOT_ENV: slot.slot_id})
    py = sys.executable
    result = {"slot": slot.slot_id, "started": datetime.now(timezone.utc).isoformat(timespec="seconds")}

    weekly = [py, "-m", "tools.weekly_run", "--season", str(slot.season), "--week", str(slot.week),
              "--mtb-repo-dir", str(args.mtb_repo_dir)]
    if args.no_publish:
        weekly.append("--no-publish")
    rc = runner(weekly, env=env, timeout=args.pipeline_timeout_min * 60)
    log = newest_debug_log(slot.season, started)
    result.update(pipeline_rc=rc, debug_log=str(log) if log else None)

    publish_rc = None
    if rc == 0 and not args.no_publish and not args.no_site_deploy:
        publish_log = STATE_DIR / f"{slot.slot_id}_publish.txt"
        publish_rc = runner(
            [py, "scripts/publish_site.py", "--commit-message",
             f"Publish NFL predictions {slot.season} week {slot.week} ({slot.slot_id})"],
            cwd=args.mtb_repo_dir, timeout=args.publish_timeout_min * 60, log=publish_log.resolve(),
        )
        result.update(publish_rc=publish_rc, publish_log=str(publish_log))
    ok = rc == 0 and publish_rc in (None, 0)
    result["status"] = "success" if ok else "failed"
    result["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    review_path = None
    if not args.no_review:
        review_path = STATE_DIR.parent / "reviews" / str(slot.season) / f"{slot.slot_id}_review.md"
        review_cmd = [py, "-m", "tools.review_run", "--slot", slot.slot_id, "--status", result["status"],
                      "--out", str(review_path)]
        if log:
            review_cmd += ["--debug-log", str(log)]
        if result.get("publish_log"):
            review_cmd += ["--publish-log", result["publish_log"]]
        runner(review_cmd, timeout=args.review_timeout_min * 60)
        result["review"] = str(review_path)

    if alert is not None:
        alert(slot, result, review_path)
    return result


def send_slot_alert(slot: Slot, result: dict, review_path: Optional[Path]) -> None:
    from src.utils.alerts import send_alert

    from tools.review_run import blog_failure_lines, blog_failures, with_blog_note

    blog = blog_failures(result.get("publish_log"))
    status = with_blog_note(result["status"].upper(), blog)
    lines = [
        f"Slot: {slot.slot_id}  (season {slot.season} week {slot.week}, {len(slot.game_ids)} games that day)",
        f"First kickoff: {slot.first_kickoff.astimezone(EASTERN):%a %Y-%m-%d %H:%M} ET",
        f"Pipeline exit: {result.get('pipeline_rc')}   Site publish exit: {result.get('publish_rc')}",
        f"Debug log: {result.get('debug_log')}",
    ]
    if result.get("publish_log"):
        lines.append(f"Publish log: {result['publish_log']}")
    if result["status"] != "success":
        lines.append("Nothing was published after the failure; the last good picks stay live.")
    lines += blog_failure_lines(blog)
    grading = grading_summary(load_state())
    if grading:
        lines.append(grading)
    body = "\n".join(lines)
    if review_path and Path(review_path).exists():
        body += "\n\n---- Review ----\n\n" + Path(review_path).read_text(encoding="utf-8")
    send_alert(f"[NFL pipeline] {status}: {slot.slot_id}", body)


def run_slot_attempts(
    slot: Slot,
    args: argparse.Namespace,
    state: dict,
    *,
    start_lead: timedelta = START_LEAD,
    last_start_lead: timedelta = LAST_START_LEAD,
    max_attempts: int = MAX_ATTEMPTS,
    runner: Callable = None,
    alert: Optional[Callable] = None,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """Run ``slot``; retry at once (same tick) while attempts fail fast and the window is open."""
    while True:
        rec = state.setdefault(slot.slot_id, {})
        rec["attempts"] = int(rec.get("attempts", 0)) + 1
        rec["status"] = "running"
        save_state(state)
        began = clock()
        result = run_slot(slot, args, alert=alert, **({"runner": runner} if runner else {}))
        rec.update({k: v for k, v in result.items() if k != "slot"})
        save_state(state)
        print(f"[dispatcher] {slot.slot_id}: {result['status']} (attempt {rec['attempts']}/{max_attempts})")
        if result["status"] == "success" or rec["attempts"] >= max_attempts:
            return result
        if clock() - began >= QUICK_FAIL:
            return result  # a slow failure; the next tick retries if the window is still open
        sleep(QUICK_RETRY_DELAY)
        if not slot.window_open(clock(), start_lead, last_start_lead):
            print(f"[dispatcher] {slot.slot_id}: launch window closed; no more retries")
            return result
        print(f"[dispatcher] {slot.slot_id}: failed after {(clock() - began).total_seconds():.0f}s; retrying now")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE_PATH)
    p.add_argument("--mtb-repo-dir", type=Path, default=DEFAULT_MTB_REPO)
    p.add_argument("--start-lead-min", type=int, default=int(START_LEAD.total_seconds() // 60))
    p.add_argument("--last-start-lead-min", type=int, default=int(LAST_START_LEAD.total_seconds() // 60))
    p.add_argument("--pipeline-timeout-min", type=int, default=120)
    p.add_argument("--publish-timeout-min", type=int, default=30)
    p.add_argument("--review-timeout-min", type=int, default=15)
    p.add_argument("--no-publish", action="store_true", help="Run and record picks, but publish nothing.")
    p.add_argument("--no-site-deploy", action="store_true", help="Copy CSVs to MTB but do not commit/push/deploy.")
    p.add_argument("--no-review", action="store_true")
    p.add_argument("--no-alert", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Show slots and what would run now.")
    p.add_argument("--now", help="Override current time (ISO, for testing).")
    p.add_argument("--no-grading", action="store_true", help="Skip grading completed weeks.")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    now = datetime.fromisoformat(args.now) if args.now else datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    start_lead = timedelta(minutes=args.start_lead_min)
    last_lead = timedelta(minutes=args.last_start_lead_min)
    slots = build_slots(pd.read_parquet(args.schedule))
    state = load_state()
    slot = due_slot(slots, state, now, start_lead=start_lead, last_start_lead=last_lead)

    if args.dry_run:
        upcoming = [s for s in slots if s.last_kickoff > now][:6]
        for s in upcoming:
            open_at = (s.first_kickoff - start_lead).astimezone(EASTERN)
            print(f"{s.slot_id:28s} first kick {s.first_kickoff.astimezone(EASTERN):%a %m-%d %H:%M} ET  "
                  f"window opens {open_at:%a %H:%M} ET  games={len(s.game_ids)}  state={state.get(s.slot_id, {})}")
        print(f"[dispatcher] due now: {slot.slot_id if slot else 'nothing'}")
        grade = None if args.no_grading else due_grading(slots, state, now)
        print(f"[dispatcher] grading due: {grade_key(*grade) if grade else 'nothing'}")
        return 0
    grade = None if (slot is not None or args.no_grading) else due_grading(slots, state, now)
    if slot is None and grade is None:
        return 0
    if not acquire_lock(now):
        print("[dispatcher] another run holds the lock; skipping")
        return 0
    keep_awake(True)
    try:
        if slot is None:  # game-day slots always take priority over grading
            rec = handle_grading(*grade, args, state, now,
                                 alert=None if args.no_alert else send_grading_alert)
            save_state(state)
            print(f"[dispatcher] {grade_key(*grade)}: {rec['status']}")
            return 0 if rec["status"] in ("success", "pending") else 1
        result = run_slot_attempts(slot, args, state, start_lead=start_lead, last_start_lead=last_lead,
                                   alert=None if args.no_alert else send_slot_alert)
        return 0 if result["status"] == "success" else 1
    finally:
        keep_awake(False)
        release_lock()


if __name__ == "__main__":
    sys.exit(main())
