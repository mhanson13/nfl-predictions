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
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence, TextIO

from src.utils.nfl_calendar import (
    DEFAULT_SCHEDULE_PATH,
    PHASE_EARLY_WEEK,
    PHASE_OFFSEASON,
    WeekContext,
    load_week_context,
)

DEFAULT_LOG_DIR = Path("logs/pipeline")
DEFAULT_MTB_REPO = Path(r"C:\Code\mtb")
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
REQUIRED_MODULES = ("xgboost", "sklearn", "optuna", "nfl_data_py", "pandas")
SECRET_NAMES = (
    "ODDS_API_FREE_KEY", "ODDS_API_PAID_KEY", "BALLDONTLIE_API_KEY", "PROPLINE_API_KEY",
    "TOMORROW_API_KEY", "VISUAL_CROSSING_API_KEY", "XWEATHER_CLIENT_ID",
)
_STAGE_RE = re.compile(r"^\[pipeline\] (Starting|Completed) ([A-Za-z0-9_]+)")
# Only log-style lines (``[tag] ...``, logging prefixes, tracebacks, exceptions) are
# scanned; printed summary tables are too noisy to flag.
_LOG_LINE_RE = re.compile(r"^(\[[^\]]+\]|(WARNING|ERROR|CRITICAL)[:\s]|Traceback|[A-Za-z_.]*(Error|Exception|Warning)\b)")
_ISSUE_RE = re.compile(
    r"Traceback|\bERROR\b|Error\b|Exception\b|\bfailed\b|Warning|\b429\b|Too Many|unavailable|\bmissing\b|rows=0\b",
    re.IGNORECASE,
)
_ISSUE_IGNORE = re.compile(
    r"^\[pipeline\] Starting |missing venue\(lat,lon\)=\(0,0\)|unexpected_null>20%: 0|Skipping missing optional CSV"
    r"|\b0 missing\b|missing\(lat,lon,dt\)=\(0,0,0\)",
    re.IGNORECASE,
)
# Flagged anywhere, including inside tables (rate limits, crashes).
_ALWAYS_RE = re.compile(r"Too Many (Requests|Calls)|HTTP(Error)? 429|status(_code)?[=: ]+429|Traceback \(most recent call last\)")


def is_issue_line(text: str) -> bool:
    if _ALWAYS_RE.search(text):
        return True
    return bool(_LOG_LINE_RE.match(text) and _ISSUE_RE.search(text) and not _ISSUE_IGNORE.search(text))


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
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    if phase == PHASE_OFFSEASON or season is None or week is None:
        log_path = args.log_dir / f"output_offseason_{stamp}_debug.txt"
        return RunPlan(ctx, season or 0, week or 0, PHASE_OFFSEASON, [], log_path, Path("output_offseason_debug.txt"), False, notes)

    overwrite = phase == PHASE_EARLY_WEEK
    command = [sys.executable, "-u", "-m", "tools.run_pipeline", *STANDARD_FLAGS,
               "--season", str(season), "--prediction-week", str(week)]
    if not args.no_publish:
        command += ["--publish-mtb", "--mtb-repo-dir", str(args.mtb_repo_dir)]
        if overwrite:
            command.append("--mtb-overwrite-week-archive")
    command += extra
    log_path = args.log_dir / str(season) / f"output_wk{week:02d}_{phase}_{stamp}_debug.txt"
    return RunPlan(ctx, season, week, phase, command, log_path, Path(f"output_wk{week}_debug.txt"), overwrite, notes)


def _git(*cmd: str) -> str:
    try:
        return subprocess.run(["git", *cmd], capture_output=True, text=True, check=False).stdout.strip()
    except Exception:
        return "?"


def preflight() -> list[str]:
    """Environment facts for the log header; problems are prefixed with WARN."""
    lines = [f"python      : {sys.executable} ({sys.version.split()[0]})"]
    if "nflgpu" not in sys.executable.lower():
        lines.append("WARN python : not the nflgpu conda env; model artifacts will be written by this interpreter")
    for mod in REQUIRED_MODULES:
        spec = importlib.util.find_spec(mod)
        if spec is None:
            lines.append(f"WARN module : {mod} not importable in this interpreter")
            continue
        version = "?"
        try:
            version = getattr(__import__(mod), "__version__", "?")
        except Exception:
            pass
        lines.append(f"module      : {mod} {version}")
    dirty = _git("status", "--porcelain")
    lines.append(f"git         : {_git('rev-parse', '--abbrev-ref', 'HEAD')} @ {_git('rev-parse', '--short', 'HEAD')}"
                 f" ({'dirty: ' + str(len(dirty.splitlines())) + ' files' if dirty else 'clean'})")
    try:
        from src.utils.secrets import get_secret

        present = [n for n in SECRET_NAMES if get_secret(n)]
        absent = [n for n in SECRET_NAMES if n not in present]
        lines.append(f"secrets     : present={present} absent={absent}")
    except Exception as exc:  # pragma: no cover - best effort
        lines.append(f"WARN secrets: could not check ({exc})")
    if DEFAULT_SCHEDULE_PATH.exists():
        age_h = (time.time() - DEFAULT_SCHEDULE_PATH.stat().st_mtime) / 3600.0
        lines.append(f"schedule    : {DEFAULT_SCHEDULE_PATH} (updated {age_h:.1f}h ago)")
        if age_h > 7 * 24:
            lines.append("WARN schedule: local schedule older than a week; week resolution may be stale")
    return lines


def _header(plan: RunPlan, env_lines: list[str]) -> str:
    ctx = plan.context
    rule = "=" * 100
    out = [rule, "NFL weekly pipeline run", rule,
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
           rule, ""]
    return "\n".join(out)


def _footer(rc: int, started: float, stages: dict[str, list[float]], issues: list[str], plan: RunPlan) -> str:
    rule = "=" * 100
    out = ["", rule, "Run summary", rule,
           f"finished    : {datetime.now().isoformat(timespec='seconds')}",
           f"exit code   : {rc} ({'OK' if rc == 0 else 'FAILED'})",
           f"duration    : {(time.time() - started) / 60.0:.1f} min",
           "stage timings (min):"]
    for name, marks in stages.items():
        dur = f"{(marks[1] - marks[0]) / 60.0:6.1f}" if len(marks) == 2 else "  open"
        out.append(f"  {dur}  {name}")
    out.append(f"issues found in output: {len(issues)} (first 40 unique shown)")
    out += [f"  - {line[:220]}" for line in issues[:40]]
    out += [f"log         : {plan.log_path}", rule, ""]
    return "\n".join(out)


def _tee(text: str, *streams: TextIO) -> None:
    for stream in streams:
        stream.write(text)
        stream.flush()


def execute(plan: RunPlan, log: TextIO) -> int:
    env = os.environ.copy()
    env.update({"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"})
    stages: dict[str, list[float]] = {}
    issues: list[str] = []
    seen: set[str] = set()
    started = time.time()
    proc = subprocess.Popen(plan.command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    assert proc.stdout is not None
    try:
        for raw in iter(proc.stdout.readline, b""):
            line = raw.decode("utf-8", errors="replace").replace("\r\n", "\n")
            _tee(line, sys.stdout, log)
            text = line.strip()
            match = _STAGE_RE.match(text)
            if match:
                stages.setdefault(match.group(2), []).append(time.time())
            if is_issue_line(text):
                key = re.sub(r"\d+", "#", text)[:160]
                if key not in seen:
                    seen.add(key)
                    issues.append(text)
        rc = proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        rc = proc.wait()
        issues.insert(0, "Interrupted by user (Ctrl+C)")
    _tee(_footer(rc, started, stages, issues, plan), sys.stdout, log)
    return rc


def main(argv: Sequence[str] | None = None) -> int:
    args, extra = parse_args(argv)
    plan = build_plan(args, extra)
    plan.log_path.parent.mkdir(parents=True, exist_ok=True)
    with plan.log_path.open("w", encoding="utf-8") as log:
        _tee(_header(plan, preflight()), sys.stdout, log)
        if plan.phase == PHASE_OFFSEASON:
            _tee("[weekly_run] Nothing to predict (offseason or schedule exhausted).\n", sys.stdout, log)
            return 0
        _tee("[weekly_run] context: " + json.dumps(plan.context.as_dict()) + "\n", sys.stdout, log)
        if args.dry_run:
            _tee("[weekly_run] dry run: pipeline not started.\n", sys.stdout, log)
            rc = 0
        else:
            rc = execute(plan, log)
    if not args.dry_run:
        shutil.copyfile(plan.log_path, plan.latest_copy)
        print(f"[weekly_run] debug log: {plan.log_path} (copy: {plan.latest_copy})")
    return rc


if __name__ == "__main__":
    sys.exit(main())
