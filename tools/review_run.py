"""Post-run review: deterministic facts + a read-only Claude review of the run.

    python -m tools.review_run --slot 2026_wk10_Sun_2026-11-15 --status success \
        --debug-log logs/pipeline/2026/output_wk10_..._debug.txt --out logs/reviews/2026/x.md

Part 1 (always): facts pulled from the debug log and the picks-of-record ledger
-- exit code, duration, issue lines, picks-of-record status, home-vs-market
watch, weather trial, probability moves since the previous run, leakage-guard
flags. Part 2 (when the Claude CLI is available): ``claude -p`` with read-only
tools (Read/Grep/Glob) follows tools/review_prompt.md and writes the verdict.
The review never edits files, publishes, or reruns anything.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

PROMPT_PATH = Path(__file__).with_name("review_prompt.md")
LEDGER_GLOB = "predictions/picks_of_record/picks_ledger_{season}.csv"
MOVE_THRESHOLD = 0.05
KEY_PREFIXES = ("[picks_of_record]", "[market_bias]", "[weather_trial]", "[calibrate] selected", "[pipeline] Live run")


def find_claude() -> Optional[str]:
    """Claude Code CLI: $CLAUDE_BIN, PATH, or the newest VS Code extension binary."""
    if os.environ.get("CLAUDE_BIN") and Path(os.environ["CLAUDE_BIN"]).exists():
        return os.environ["CLAUDE_BIN"]
    on_path = shutil.which("claude")
    if on_path:
        return on_path
    pattern = str(Path.home() / ".vscode/extensions/anthropic.claude-code-*/resources/native-binary/claude.exe")
    found = glob.glob(pattern)

    def version(p: str) -> tuple:
        m = re.search(r"claude-code-(\d+)\.(\d+)\.(\d+)", p)
        return tuple(map(int, m.groups())) if m else (0,)

    return max(found, key=version) if found else None


def log_facts(text: str) -> list[str]:
    lines = text.splitlines()
    facts = [ln.strip() for ln in lines if ln.strip().startswith(KEY_PREFIXES)]
    tail_start = next((i for i in range(len(lines) - 1, -1, -1) if lines[i].startswith("Run summary")), None)
    if tail_start is not None:
        for ln in lines[tail_start:]:
            s = ln.strip()
            if s.startswith(("finished", "exit code", "duration", "issues found", "- ")):
                facts.append(s)
    if "Traceback (most recent call last)" in text:
        facts.append("Traceback present in log")
    return facts


def probability_moves(ledger: pd.DataFrame, slot: str, threshold: float = MOVE_THRESHOLD) -> list[str]:
    """Games whose probability moved >= threshold vs the previous run for that game."""
    if ledger.empty or "slot" not in ledger:
        return []
    led = ledger.sort_values("recorded_at")
    out = []
    for gid, rows in led.groupby("game_id"):
        cur = rows[rows["slot"] == slot]
        prev = rows[rows["recorded_at"] < cur["recorded_at"].min()] if not cur.empty else rows.iloc[0:0]
        if cur.empty or prev.empty:
            continue
        a, b = float(prev["home_win_prob"].iloc[-1]), float(cur["home_win_prob"].iloc[-1])
        if abs(b - a) >= threshold:
            ma = prev["market_home_implied_prob_novig"].iloc[-1]
            mb = cur["market_home_implied_prob_novig"].iloc[-1]
            out.append(f"{gid}: home win prob {a:.3f} -> {b:.3f} (market {ma:.3f} -> {mb:.3f})")
    return out


def build_facts(slot: str, status: str, debug_log: Optional[Path], publish_log: Optional[Path]) -> str:
    parts = [f"# Run review: {slot}", "", f"**Status:** {status.upper()}", ""]
    if debug_log and debug_log.exists():
        text = debug_log.read_text(encoding="utf-8", errors="replace")
        parts += ["## Facts from the debug log", f"`{debug_log}`", ""]
        parts += [f if f.startswith("- ") else f"- {f}" for f in log_facts(text)] or ["- (no summary lines found)"]
    else:
        parts += ["## Debug log", "- not found (pipeline may not have started)"]
    if publish_log and publish_log.exists():
        tail = publish_log.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-8:]
        parts += ["", "## Site publish (last lines)", "```", *tail, "```"]
    season = slot.split("_", 1)[0]
    ledger_path = Path(LEDGER_GLOB.format(season=season))
    if ledger_path.exists():
        moves = probability_moves(pd.read_csv(ledger_path), slot)
        parts += ["", f"## Probability moves >= {MOVE_THRESHOLD:.0%} since the previous run"]
        parts += [f"- {m}" for m in moves] or ["- none"]
    return "\n".join(parts) + "\n"


def run_agent(facts_path: Path, debug_log: Optional[Path], claude: str, timeout: int = 900) -> str:
    prompt = PROMPT_PATH.read_text(encoding="utf-8").format(
        facts=facts_path.as_posix(), debug_log=debug_log.as_posix() if debug_log else "(none)"
    )
    cmd = [claude, "-p", prompt, "--allowedTools", "Read,Grep,Glob",
           "--disallowedTools", "Bash,Edit,Write,NotebookEdit,WebFetch,WebSearch",
           "--max-turns", "25", "--output-format", "text"]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                             timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return "_Review agent timed out._"
    if res.returncode != 0:
        return f"_Review agent failed (exit {res.returncode}): {(res.stderr or res.stdout)[-400:]}_"
    return res.stdout.strip()


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--slot", required=True)
    p.add_argument("--status", default="unknown")
    p.add_argument("--debug-log", type=Path)
    p.add_argument("--publish-log", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--no-agent", action="store_true")
    args = p.parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    facts = build_facts(args.slot, args.status, args.debug_log, args.publish_log)
    args.out.write_text(facts, encoding="utf-8")
    claude = None if args.no_agent else find_claude()
    if claude:
        review = run_agent(args.out, args.debug_log, claude)
    else:
        review = "_Review agent not run (Claude CLI not found or disabled)._"
    args.out.write_text(facts + "\n## Review agent\n\n" + review + "\n", encoding="utf-8")
    print(f"[review] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
