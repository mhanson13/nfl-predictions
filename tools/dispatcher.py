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
  2. on success takes MTB's shared publish lock (``<mtb>/.publish.lock``, see
     ``src/utils/publish_lock.py``), copies the CSVs into MTB and runs its JSON
     import (the commands run_pipeline left in the slot's publish plan), publishes
     the site (MTB ``scripts/publish_site.py``: import, lint/typecheck/test/build,
     commit, push, deploy) under that same lock, and releases it -- never after a
     failure. If another publish holds the lock (or publish_site.py exits 75), the
     slot stays ``publish_pending``: no attempt is used, no alert is sent, and the
     next tick publishes the same run (or re-runs the pipeline if a game has
     kicked off since),
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

from src.utils import publish_lock
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
# MTB's shared publish lock (src/utils/publish_lock.py, MTB docs/publish-lock.md). Held from
# the first CSV copy into the MTB checkout until scripts/publish_site.py finishes.
PUBLISH_LOCK_OWNER = "nfl-predictions dispatcher"
PUBLISH_LOCK_SPORT = "nfl"
PUBLISH_LOCK_STALE_MIN = 90             # must cover copy + import + the whole site publish
PUBLISH_LOCK_TOKEN_ENV = "MTB_PUBLISH_LOCK_TOKEN"
PUBLISH_LOCK_PATH_ENV = "MTB_PUBLISH_LOCK_PATH"  # tests only, same as MTB's publish_site.py
PUBLISH_COPY_TIMEOUT = 10 * 60          # each MTB copy/import step
PUBLISH_PENDING = "publish_pending"     # lock busy / exit 75: retried next tick, no attempt used


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


def kicked_off_since(slot: Slot, since_iso: Optional[str], now: datetime) -> bool:
    """True when one of the slot's games kicked off after ``since_iso`` (or that time is unknown)."""
    try:
        since = datetime.fromisoformat(since_iso)
    except (TypeError, ValueError):
        return True
    return any(since < k <= now for k in slot.kickoffs)


def expired_deferred_publishes(
    slots: Sequence[Slot],
    state: dict,
    now: datetime,
    *,
    start_lead: timedelta = START_LEAD,
    last_start_lead: timedelta = LAST_START_LEAD,
) -> list[Slot]:
    """Slots still waiting on a deferred publish after their launch window closed."""
    return [
        s for s in slots
        if state.get(s.slot_id, {}).get("status") == PUBLISH_PENDING
        and now >= s.first_kickoff - start_lead
        and not s.window_open(now, start_lead, last_start_lead)
    ]


def fail_expired_publishes(expired: Sequence[Slot], state: dict, *, alert: Optional[Callable] = None) -> None:
    for slot in expired:
        rec = state[slot.slot_id]
        rec["status"] = "failed"
        rec["publish_note"] = (f"{rec.get('publish_note') or 'publish deferred'}; the launch window closed "
                               "before MTB's publish lock came free, so this run was never published")
        save_state(state)
        print(f"[dispatcher] {slot.slot_id}: {rec['publish_note']}")
        if alert is not None:
            alert(slot, dict(rec), None)


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
        if rec.get("status") == PUBLISH_PENDING:
            return season, week  # graded, but the publish lock was busy: retry on this tick
        if last_try and now - datetime.fromisoformat(last_try) < recheck:
            continue
        return season, week
    return None


def publish_lock_path(repo_dir: Path) -> Path:
    """MTB's ``<repo>/.publish.lock`` (``MTB_PUBLISH_LOCK_PATH`` overrides it, for tests only)."""
    return Path(os.environ.get(PUBLISH_LOCK_PATH_ENV) or Path(repo_dir) / publish_lock.LOCK_FILE_NAME)


def run_locked_publish(steps: Sequence[dict], repo_dir: Path, *, runner: Callable) -> dict:
    """Run ``steps`` (copy into MTB, then publish_site.py) inside MTB's shared publish lock.

    Each step is ``{"cmd": [...], **runner_kwargs}``. The lock is taken before the first
    step and released after the last, on every path (a failed step, a timeout, an
    exception). Every step gets the lock token in ``MTB_PUBLISH_LOCK_TOKEN``, so
    publish_site.py runs inside this hold instead of waiting on it.

    Returns ``{"outcome": "success" | "failed" | PUBLISH_PENDING, "rc": ..., "note": ...}``.
    PUBLISH_PENDING means nothing could publish now (the lock is busy, or publish_site.py
    exited 75); the caller retries on its next tick. Exit 77 (token rejected) is a failure.
    """
    path = publish_lock_path(repo_dir)
    log = lambda msg: print(f"[dispatcher] {msg}", flush=True)  # noqa: E731
    try:
        record = publish_lock.acquire(path, owner=PUBLISH_LOCK_OWNER, sport=PUBLISH_LOCK_SPORT,
                                      stale_minutes=PUBLISH_LOCK_STALE_MIN, wait_seconds=0, log=log)
    except publish_lock.LockHeld as exc:
        log(f"{exc} Nothing copied or published; retrying on the next tick.")
        return {"outcome": PUBLISH_PENDING, "rc": None, "note": f"publish lock busy: {exc}"}
    except OSError as exc:
        log(f"could not take the publish lock {path}: {exc}")
        return {"outcome": "failed", "rc": None, "note": f"publish lock error: {exc}"}

    log(f"took publish lock {path} (sport={PUBLISH_LOCK_SPORT}, stale after {PUBLISH_LOCK_STALE_MIN} min)")
    rc: Optional[int] = None
    failed_step = None
    try:
        for step in steps:
            kwargs = {k: v for k, v in step.items() if k not in ("cmd", "env")}
            env = dict(step.get("env") or os.environ, **{PUBLISH_LOCK_TOKEN_ENV: record["token"]})
            rc = runner(step["cmd"], env=env, **kwargs)
            if rc != 0:
                failed_step = _step_name(step["cmd"])
                break
    finally:
        publish_lock.release(path, record)
        log(f"released publish lock {path}")

    if rc == publish_lock.LOCK_HELD_EXIT_CODE:
        log("publish_site.py exited 75 (publish lock held elsewhere); retrying on the next tick.")
        return {"outcome": PUBLISH_PENDING, "rc": rc, "note": "publish_site.py exit 75 (publish lock held)"}
    if rc == publish_lock.LOCK_TOKEN_MISMATCH_EXIT_CODE:
        return {"outcome": "failed", "rc": rc,
                "note": "publish_site.py exit 77: it rejected the dispatcher's publish lock token"}
    if rc in (None, 0):
        return {"outcome": "success", "rc": rc, "note": None}
    return {"outcome": "failed", "rc": rc, "note": f"{failed_step} exited {rc}; later publish steps were skipped"}


def _step_name(cmd: Sequence) -> str:
    """``tools.publish_mtb_csvs`` / ``scripts/publish_site.py`` for a step's command."""
    parts = [str(p) for p in cmd]
    if "-m" in parts and parts.index("-m") + 1 < len(parts):
        return parts[parts.index("-m") + 1]
    return next((p for p in parts[1:] if p.endswith(".py")), parts[0] if parts else "?")


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

    result["publish_rc"] = None
    result["status"] = "success"
    if not args.no_publish:
        steps = [{"cmd": [py, "-m", "tools.publish_mtb_csvs", "--repo-dir", str(args.mtb_repo_dir),
                          "--season", str(season), "--results-only"],
                  "timeout": PUBLISH_COPY_TIMEOUT, "log": log}]
        if not args.no_site_deploy:
            site_log = (STATE_DIR / f"{grade_key(season, week)}_publish.txt").resolve()
            steps.append({"cmd": [py, "scripts/publish_site.py", "--commit-message",
                                  f"Publish graded results {season} week {week}"],
                          "cwd": args.mtb_repo_dir, "timeout": args.publish_timeout_min * 60, "log": site_log})
            result["publish_log"] = str(site_log)
        pub = run_locked_publish(steps, args.mtb_repo_dir, runner=runner)
        result.update(publish_rc=pub["rc"], status=pub["outcome"])
        if pub["note"]:
            result["publish_note"] = pub["note"]
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
    if result.get("publish_note"):
        lines.append(f"Publish: {result['publish_note']}")
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
    rec.pop("publish_note", None)
    rec.update({k: v for k, v in result.items() if k not in ("season", "week")})
    if result["status"] == "failed":
        rec["failures"] = int(rec.get("failures", 0)) + 1
    final = int(rec.get("failures", 0)) >= GRADE_MAX_FAILURES
    if result["status"] == "pending":
        print(f"[dispatcher] {grade_key(season, week)}: scores not final yet; re-checking later")
    elif result["status"] == PUBLISH_PENDING:
        print(f"[dispatcher] {grade_key(season, week)}: graded, but not published "
              f"({result.get('publish_note')}); retrying on the next tick, no failure counted")
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


def publish_plan_path(slot: Slot) -> Path:
    return (STATE_DIR / f"{slot.slot_id}_mtb_publish_plan.json").resolve()


def load_publish_plan(path: Path) -> Optional[list[list[str]]]:
    """The MTB copy/import commands run_pipeline deferred (``--mtb-publish-plan``), or None."""
    try:
        jobs = json.loads(Path(path).read_text(encoding="utf-8"))["jobs"]
        return [[str(part) for part in job["command"]] for job in jobs]
    except Exception:
        return None


def run_slot(slot: Slot, args: argparse.Namespace, *, runner: Callable = _run,
             alert: Optional[Callable] = None, publish_only: Optional[dict] = None) -> dict:
    """Run the pipeline, then copy into MTB and publish the site under MTB's publish lock.

    The pipeline (weekly_run -> run_pipeline) does not copy into MTB itself: it writes
    the copy/import commands to the slot's publish plan, and they run here inside the
    lock, with publish_site.py. ``publish_only`` is the slot's state record from a run
    whose pipeline succeeded but whose publish was deferred (PUBLISH_PENDING): the
    pipeline is not re-run and that run's plan is published.
    """
    started = time.time()
    env = dict(os.environ, **{SLOT_ENV: slot.slot_id})
    py = sys.executable
    result = {"slot": slot.slot_id, "started": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    plan_path = publish_plan_path(slot)
    publishing = not args.no_publish

    if publish_only is not None:
        rc = 0
        log = Path(publish_only["debug_log"]) if publish_only.get("debug_log") else None
        result.update(pipeline_rc=0, debug_log=publish_only.get("debug_log"),
                      pipeline_started=publish_only.get("pipeline_started"), publish_retry=True)
        print(f"[dispatcher] {slot.slot_id}: publishing the pipeline run of "
              f"{publish_only.get('pipeline_started')} (deferred: {publish_only.get('publish_note')})")
    else:
        plan_path.unlink(missing_ok=True)
        weekly = [py, "-m", "tools.weekly_run", "--season", str(slot.season), "--week", str(slot.week),
                  "--mtb-repo-dir", str(args.mtb_repo_dir)]
        if publishing:
            weekly += ["--", "--mtb-publish-plan", str(plan_path)]
        else:
            weekly.append("--no-publish")
        rc = runner(weekly, env=env, timeout=args.pipeline_timeout_min * 60)
        log = newest_debug_log(slot.season, started)
        result.update(pipeline_rc=rc, debug_log=str(log) if log else None, pipeline_started=result["started"])

    status = "success" if rc == 0 else "failed"
    if rc == 0 and publishing:
        jobs = load_publish_plan(plan_path)
        if jobs is None:
            status = "failed"
            result["publish_note"] = f"no MTB publish plan at {plan_path}; nothing was copied or published"
            print(f"[dispatcher] {slot.slot_id}: {result['publish_note']}")
        else:
            # Copy/import output goes to the run's debug log, where it was before the split.
            copy_log = (log or STATE_DIR / f"{slot.slot_id}_publish.txt").resolve()
            steps = [{"cmd": cmd, "env": env, "timeout": PUBLISH_COPY_TIMEOUT, "log": copy_log} for cmd in jobs]
            if not args.no_site_deploy:
                publish_log = STATE_DIR / f"{slot.slot_id}_publish.txt"
                steps.append({
                    "cmd": [py, "scripts/publish_site.py", "--commit-message",
                            f"Publish NFL predictions {slot.season} week {slot.week} ({slot.slot_id})"],
                    "env": env, "cwd": args.mtb_repo_dir, "timeout": args.publish_timeout_min * 60,
                    "log": publish_log.resolve(),
                })
                result["publish_log"] = str(publish_log)
            pub = run_locked_publish(steps, args.mtb_repo_dir, runner=runner)
            status = pub["outcome"]
            result["publish_rc"] = pub["rc"]
            if pub["note"]:
                result["publish_note"] = pub["note"]
    result["status"] = status
    result["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if status == PUBLISH_PENDING:
        return result  # no review or alert: the next tick publishes this run

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
    if result.get("publish_note"):
        lines.append(f"Publish: {result['publish_note']}")
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
    """Run ``slot``; retry at once (same tick) while attempts fail fast and the window is open.

    A run whose publish was deferred (PUBLISH_PENDING: MTB's publish lock was busy) does
    not use an attempt. The next call publishes that run without re-running the
    pipeline, unless one of the slot's games has kicked off since it ran; then the
    pipeline runs again so started games are left out as usual.
    """
    while True:
        rec = state.setdefault(slot.slot_id, {})
        publish_only = None
        if rec.get("status") == PUBLISH_PENDING:
            if kicked_off_since(slot, rec.get("pipeline_started"), clock()):
                print(f"[dispatcher] {slot.slot_id}: a game kicked off since the deferred run; re-running the pipeline")
            else:
                publish_only = dict(rec)
        if publish_only is None:
            rec["attempts"] = int(rec.get("attempts", 0)) + 1
        rec["status"] = "running"
        save_state(state)
        began = clock()
        result = run_slot(slot, args, alert=alert, publish_only=publish_only,
                          **({"runner": runner} if runner else {}))
        for key in ("publish_note", "publish_retry", "publish_rc", "publish_log"):
            rec.pop(key, None)
        rec.update({k: v for k, v in result.items() if k != "slot"})
        if publish_only is None:
            rec["pipeline_started"] = began.isoformat(timespec="seconds")  # same clock as kicked_off_since
        if result["status"] == PUBLISH_PENDING:
            if publish_only is None:
                rec["attempts"] -= 1  # the pipeline ran, but only the publish has to be retried
            save_state(state)
            print(f"[dispatcher] {slot.slot_id}: publish deferred ({result.get('publish_note')}); "
                  f"no attempt used ({rec['attempts']}/{max_attempts}), retrying on the next tick")
            return result
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
        print(f"[dispatcher] due now: {slot.slot_id if slot else 'nothing'}"
              + (" (publish only: deferred run)" if slot and state.get(slot.slot_id, {}).get("status")
                 == PUBLISH_PENDING else ""))
        lock_path = publish_lock_path(args.mtb_repo_dir)
        holder = publish_lock.read_lock(lock_path) if lock_path.exists() else None
        print(f"[dispatcher] publish lock {lock_path}: "
              + (f"held by {holder}" if lock_path.exists() else "free"))
        for s in expired_deferred_publishes(slots, state, now, start_lead=start_lead, last_start_lead=last_lead):
            print(f"[dispatcher] would mark {s.slot_id} failed: deferred publish outlived its window")
        grade = None if args.no_grading else due_grading(slots, state, now)
        print(f"[dispatcher] grading due: {grade_key(*grade) if grade else 'nothing'}")
        return 0
    grade = None if (slot is not None or args.no_grading) else due_grading(slots, state, now)
    expired = expired_deferred_publishes(slots, state, now, start_lead=start_lead, last_start_lead=last_lead)
    if slot is None and grade is None and not expired:
        return 0
    if not acquire_lock(now):
        print("[dispatcher] another run holds the lock; skipping")
        return 0
    keep_awake(True)
    try:
        if expired:
            fail_expired_publishes(expired, state, alert=None if args.no_alert else send_slot_alert)
        if slot is None and grade is None:
            return 0
        if slot is None:  # game-day slots always take priority over grading
            rec = handle_grading(*grade, args, state, now,
                                 alert=None if args.no_alert else send_grading_alert)
            save_state(state)
            print(f"[dispatcher] {grade_key(*grade)}: {rec['status']}")
            return 0 if rec["status"] in ("success", "pending", PUBLISH_PENDING) else 1
        result = run_slot_attempts(slot, args, state, start_lead=start_lead, last_start_lead=last_lead,
                                   alert=None if args.no_alert else send_slot_alert)
        return 0 if result["status"] in ("success", PUBLISH_PENDING) else 1
    finally:
        keep_awake(False)
        release_lock()


if __name__ == "__main__":
    sys.exit(main())
