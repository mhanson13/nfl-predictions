"""Debug logging for pipeline runs: tee a child process to log files and summarize it.

Used by ``tools.weekly_run`` and by ``tools.run_pipeline --debug`` (which re-runs
itself under this tee so subprocess output is captured too). Each run writes:

* ``logs/pipeline/<season>/output_wk<NN>_<phase>_<stamp>_debug.txt`` (kept per run)
* ``output_wk<N>_debug.txt`` in the repo root (latest run for that week)

where ``<N>`` is the NFL week being predicted (regular season 1-18, postseason
19-22 as numbered by nflverse). Both files are written live, line by line.
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence, TextIO

DEFAULT_LOG_DIR = Path("logs/pipeline")
CHILD_ENV_FLAG = "NFL_PIPELINE_LOGGED"  # set for children already being teed
REQUIRED_MODULES = ("xgboost", "sklearn", "optuna", "nfl_data_py", "pandas")
SECRET_NAMES = (
    "ODDS_API_FREE_KEY", "ODDS_API_PAID_KEY", "BALLDONTLIE_API_KEY", "PROPLINE_API_KEY",
    "TOMORROW_API_KEY", "VISUAL_CROSSING_API_KEY", "XWEATHER_CLIENT_API_KEY", "XWEATHER_CLIENT_API_SECRET",
)
RULE = "=" * 100

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


def debug_log_paths(
    season: Optional[int],
    week: Optional[int],
    phase: str,
    *,
    log_dir: Path = DEFAULT_LOG_DIR,
    stamp: Optional[str] = None,
) -> tuple[Path, Path]:
    """Return (per-run archive path, repo-root latest path) named by the NFL week."""
    stamp = stamp or datetime.now().strftime("%Y%m%d-%H%M")
    if season is None or week is None:
        return log_dir / f"output_{phase}_{stamp}_debug.txt", Path(f"output_{phase}_debug.txt")
    return (
        log_dir / str(season) / f"output_wk{int(week):02d}_{phase}_{stamp}_debug.txt",
        Path(f"output_wk{int(week)}_debug.txt"),
    )


def _git(*cmd: str) -> str:
    try:
        return subprocess.run(["git", *cmd], capture_output=True, text=True, check=False).stdout.strip()
    except Exception:
        return "?"


def preflight(schedule_path: Optional[Path] = None) -> list[str]:
    """Environment facts for the log header; problems are prefixed with WARN."""
    lines = [f"python      : {sys.executable} ({sys.version.split()[0]})"]
    if "nflgpu" not in sys.executable.lower():
        lines.append("WARN python : not the nflgpu conda env; model artifacts will be written by this interpreter")
    for mod in REQUIRED_MODULES:
        if importlib.util.find_spec(mod) is None:
            lines.append(f"WARN module : {mod} not importable in this interpreter")
            continue
        try:
            version = getattr(__import__(mod), "__version__", "?")
        except Exception:
            version = "?"
        lines.append(f"module      : {mod} {version}")
    dirty = _git("status", "--porcelain")
    lines.append(
        f"git         : {_git('rev-parse', '--abbrev-ref', 'HEAD')} @ {_git('rev-parse', '--short', 'HEAD')}"
        f" ({'dirty: ' + str(len(dirty.splitlines())) + ' files' if dirty else 'clean'})"
    )
    try:
        from src.utils.secrets import get_secret

        present = [n for n in SECRET_NAMES if get_secret(n)]
        absent = [n for n in SECRET_NAMES if n not in present]
        lines.append(f"secrets     : present={present} absent={absent}")
    except Exception as exc:  # pragma: no cover - best effort
        lines.append(f"WARN secrets: could not check ({exc})")
    if schedule_path is not None and schedule_path.exists():
        age_h = (time.time() - schedule_path.stat().st_mtime) / 3600.0
        lines.append(f"schedule    : {schedule_path} (updated {age_h:.1f}h ago)")
        if age_h > 7 * 24:
            lines.append("WARN schedule: local schedule older than a week; week resolution may be stale")
    return lines


def write(text: str, streams: Iterable[TextIO]) -> None:
    for stream in streams:
        stream.write(text)
        stream.flush()


def footer(rc: int, started: float, stages: dict[str, list[float]], issues: list[str], log_paths: Sequence[Path]) -> str:
    out = ["", RULE, "Run summary", RULE,
           f"finished    : {datetime.now().isoformat(timespec='seconds')}",
           f"exit code   : {rc} ({'OK' if rc == 0 else 'FAILED'})",
           f"duration    : {(time.time() - started) / 60.0:.1f} min",
           "stage timings (min):"]
    for name, marks in stages.items():
        dur = f"{(marks[1] - marks[0]) / 60.0:6.1f}" if len(marks) >= 2 else "  open"
        out.append(f"  {dur}  {name}")
    out.append(f"issues found in output: {len(issues)} (first 40 unique shown)")
    out += [f"  - {line[:220]}" for line in issues[:40]]
    out += [f"log         : {p}" for p in log_paths]
    out += [RULE, ""]
    return "\n".join(out)


def run_logged(command: Sequence[str], header: str, log_paths: Sequence[Path]) -> int:
    """Run ``command``; stream its output to the console and every path in ``log_paths``."""
    for path in log_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    logs = [path.open("w", encoding="utf-8") for path in log_paths]
    streams = [sys.stdout, *logs]
    env = os.environ.copy()
    env.update({"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", CHILD_ENV_FLAG: "1"})
    stages: dict[str, list[float]] = {}
    issues: list[str] = []
    seen: set[str] = set()
    started = time.time()
    rc = 1
    try:
        write(header, streams)
        proc = subprocess.Popen(list(command), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        assert proc.stdout is not None
        try:
            for raw in iter(proc.stdout.readline, b""):
                line = raw.decode("utf-8", errors="replace").replace("\r\n", "\n")
                write(line, streams)
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
        write(footer(rc, started, stages, issues, log_paths), streams)
    finally:
        for log in logs:
            log.close()
    return rc
