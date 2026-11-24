"""Week-aware entry point for the live pipeline, with a mandatory debug log.

Usage (from the nflgpu env):

    python -m tools.weekly_run              # resolve week/phase, run, log
    python -m tools.weekly_run --dry-run    # show the plan only
    python -m tools.weekly_run -- --skip-tomorrow-weather   # extra run_pipeline flags

The runner reads the local schedule to decide which slate to predict and where
we are in the week (see ``src.utils.nfl_calendar``), builds the standard live
``tools.run_pipeline`` command for that slate, and tees every line of output to
``logs/pipeline/<season>/output_wk<NN>_<phase>_<stamp>_debug.txt``. A copy is
left at ``output_wk<N>_debug.txt`` in the repo root. The log starts with a
header (resolved week, phase, command, environment, git state) and ends with a
footer (exit code, stage timings, warnings/errors found in the output).

Phase policy:
* early_week (Tue/Wed, picks of record): full slate; overwrite the MTB week archive.
* pre_kickoff (within 36h of the first game): full slate; keep the existing archive.
* in_progress (a game has kicked off): partial slate; keep the existing archive.
* offseason: nothing to predict; exits without running.

MTB publish lock: NFL and MLB both write into ``C:\\Code\\mtb``. A run that publishes
passes ``--mtb-publish-plan`` to the pipeline, so the pipeline defers its MTB CSV copy
and JSON import; after a successful run they run here inside MTB's shared publish lock
(``tools.dispatcher.run_locked_publish``, same 300 s wait). If the lock stays busy,
nothing is copied, the run exits 75, and ``--publish-only <plan>`` publishes that run
later without re-running the pipeline. The site deploy (``scripts/publish_site.py``)
stays a separate manual step, as before. When the caller already passes
``--mtb-publish-plan`` (the dispatcher), the caller owns the publish and nothing changes.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

from src.utils import publish_lock
from src.utils.nfl_calendar import (
    DEFAULT_SCHEDULE_PATH,
    PHASE_EARLY_WEEK,
    PHASE_OFFSEASON,
    WeekContext,
    load_week_context,
)
from src.utils.run_log import (
    DEFAULT_LOG_DIR,
    RULE,
    debug_log_paths,
    is_issue_line,  # noqa: F401  (re-exported for callers/tests)
    preflight,
    run_logged,
)

DEFAULT_MTB_REPO = Path(r"C:\Code\mtb")
PUBLISH_PLAN_FLAG = "--mtb-publish-plan"
STANDARD_FLAGS = [
    "--start-year", "2017",
    "--train-start-year", "2017",
    "--max-parallel-data", "8",
    "--data-start-delay", "1",
    "--use-async",
    "--use-gpu",
    "--skip-logit",
    "--live-run",
    "--debug",
]


@dataclass
class RunPlan:
    context: WeekContext
    season: int
    week: int
    phase: str
    command: list[str]
    log_path: Path
    latest_copy: Path
    overwrite_archive: bool
    notes: list[str] = field(default_factory=list)
    publish_plan: Path | None = None  # set when this run copies into MTB itself, under the lock


def parse_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season", type=int, help="Override the resolved season.")
    parser.add_argument("--week", type=int, help="Override the resolved week.")
    parser.add_argument("--phase", choices=["early_week", "pre_kickoff", "in_progress"], help="Override the phase.")
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE_PATH)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--mtb-repo-dir", type=Path, default=DEFAULT_MTB_REPO)
    parser.add_argument("--no-publish", action="store_true", help="Do not publish to MattyTheBookie.")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan (and write it to the log) without running.")
    parser.add_argument("--publish-only", type=Path, metavar="PLAN",
                        help="Skip the pipeline; copy/import a finished run's MTB publish plan under the publish lock.")
    parser.add_argument("--publish-lock-wait-sec", type=float, default=None,
                        help="Seconds to wait for a busy MTB publish lock (default: the dispatcher's).")
    argv = list(sys.argv[1:] if argv is None else argv)
    extra: list[str] = []
    if "--" in argv:
        idx = argv.index("--")
        argv, extra = argv[:idx], argv[idx + 1:]
    return parser.parse_args(argv), extra


def build_plan(args: argparse.Namespace, extra: list[str], now: Optional[datetime] = None) -> RunPlan:
    ctx = load_week_context(args.schedule, now=now)
    notes = list(ctx.notes)
    season = args.season or ctx.season
    week = args.week or ctx.week
    phase = args.phase or ctx.phase
    if args.season or args.week or args.phase:
        notes.append(f"Overrides applied: season={args.season} week={args.week} phase={args.phase}")
    if phase == PHASE_OFFSEASON or season is None or week is None:
        log_path, latest = debug_log_paths(None, None, PHASE_OFFSEASON, log_dir=args.log_dir)
        return RunPlan(ctx, season or 0, week or 0, PHASE_OFFSEASON, [], log_path, latest, False, notes)

    overwrite = phase == PHASE_EARLY_WEEK
    command = [sys.executable, "-u", "-m", "tools.run_pipeline", *STANDARD_FLAGS,
               "--season", str(season), "--prediction-week", str(week)]
    if not args.no_publish:
        command += ["--publish-mtb", "--mtb-repo-dir", str(args.mtb_repo_dir)]
        if overwrite:
            command.append("--mtb-overwrite-week-archive")
    log_path, latest = debug_log_paths(season, week, phase, log_dir=args.log_dir)
    publish_plan = None
    if not args.no_publish and PUBLISH_PLAN_FLAG not in extra:
        publish_plan = log_path.with_name(log_path.name.replace("_debug.txt", "_mtb_publish_plan.json")).resolve()
        command += [PUBLISH_PLAN_FLAG, str(publish_plan)]
    command += extra
    return RunPlan(ctx, season, week, phase, command, log_path, latest, overwrite, notes, publish_plan)


def publish_locked(plan_path: Path, mtb_repo_dir: Path, *, log_path: Path | None = None,
                   wait_seconds: float | None = None, runner=None) -> int:
    """Run a deferred MTB copy/import plan inside MTB's publish lock.

    Returns 0 on success, 75 when the lock stayed busy (nothing copied), otherwise the failing
    step's exit code (1 when the plan is missing or the lock could not be taken).
    """
    from tools import dispatcher

    jobs = dispatcher.load_publish_plan(plan_path)
    if jobs is None:
        print(f"[weekly_run] no MTB publish plan at {plan_path}; nothing was copied.")
        return 1
    steps = [{"cmd": cmd, "timeout": dispatcher.PUBLISH_COPY_TIMEOUT, "log": log_path} for cmd in jobs]
    wait = dispatcher.PUBLISH_LOCK_WAIT_SEC if wait_seconds is None else float(wait_seconds)
    pub = dispatcher.run_locked_publish(steps, mtb_repo_dir, runner=runner or dispatcher._run, wait_seconds=wait)
    if pub["outcome"] == dispatcher.PUBLISH_PENDING:
        print(f"[weekly_run] MTB publish lock busy ({pub['note']}); nothing was copied into MTB. "
              f"Publish this run later with: python -m tools.weekly_run --publish-only {plan_path}")
        return publish_lock.LOCK_HELD_EXIT_CODE
    if pub["outcome"] != "success":
        print(f"[weekly_run] MTB copy/import failed: {pub['note']}")
        return int(pub["rc"]) if pub["rc"] not in (None, 0) else 1
    print(f"[weekly_run] MTB copy/import done under the publish lock ({len(jobs)} steps).")
    return 0


def _header(plan: RunPlan, env_lines: list[str]) -> str:
    ctx = plan.context
    out = [RULE, "NFL weekly pipeline run", RULE,
           f"started     : {datetime.now().isoformat(timespec='seconds')} local  ({ctx.now_et} ET)",
           f"slate       : season {plan.season} week {plan.week} ({ctx.game_type or '?'})",
           f"phase       : {plan.phase}  (full slate: {plan.phase != 'in_progress'})",
           f"kickoffs    : first {ctx.first_kickoff_et}  last {ctx.last_kickoff_et}  "
           f"(hours to first: {ctx.hours_to_first_kickoff})",
           f"games       : total {ctx.games_total}  started {ctx.games_started}  final {ctx.games_final}",
           f"prev week   : {ctx.previous_week} scored={ctx.previous_week_scored}",
           f"mtb archive : {'overwrite (picks of record)' if plan.overwrite_archive else 'keep existing'}",
           *[f"note        : {n}" for n in plan.notes],
           *env_lines,
           "command     : " + " ".join(plan.command),
           RULE, ""]
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    args, extra = parse_args(argv)
    if args.publish_only is not None:
        return publish_locked(args.publish_only, args.mtb_repo_dir, wait_seconds=args.publish_lock_wait_sec)
    plan = build_plan(args, extra)
    header = _header(plan, preflight(args.schedule))
    context = "[weekly_run] context: " + json.dumps(plan.context.as_dict()) + "\n"
    if plan.phase == PHASE_OFFSEASON or args.dry_run:
        reason = "nothing to predict (offseason or schedule exhausted)" if plan.phase == PHASE_OFFSEASON else "dry run"
        plan.log_path.parent.mkdir(parents=True, exist_ok=True)
        text = header + context + f"[weekly_run] {reason}: pipeline not started.\n"
        plan.log_path.write_text(text, encoding="utf-8")
        print(text)
        return 0
    if plan.publish_plan is not None:
        plan.publish_plan.unlink(missing_ok=True)
    rc = run_logged(plan.command, header + context, [plan.log_path, plan.latest_copy])
    print(f"[weekly_run] debug log: {plan.log_path} (latest: {plan.latest_copy})")
    if rc == 0 and plan.publish_plan is not None:
        rc = publish_locked(plan.publish_plan, args.mtb_repo_dir, log_path=plan.log_path.resolve(),
                            wait_seconds=args.publish_lock_wait_sec)
    return rc


if __name__ == "__main__":
    sys.exit(main())
