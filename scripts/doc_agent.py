#!/usr/bin/env python3
"""
Lightweight documentation agent for detecting README/docstring drift.

Example warning output:

[DOC-AGENT] WARNING: Code changes detected without README.md updates.
  - What changed: src/predict/predict_upcoming.py
  - Missing: README updates covering sections: Usage, Training

[DOC-AGENT] WARNING: Missing docstring in src/models/train.py (FunctionDef: train_models)
  - Suggested README sections to mention: Training, Architecture
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence


WARNING_PREFIX = "[DOC-AGENT] WARNING"
DEFAULT_BASE = "origin/main"
SECTION_SUGGESTIONS = {
    "data": ["Data sources"],
    "feature": ["Features"],
    "predict": ["Usage"],
    "model": ["Training"],
    "train": ["Training"],
    "analysis": ["Evaluation"],
    "eval": ["Evaluation"],
    "streamlit": ["Usage"],
}
DEFAULT_SECTIONS = ["Usage", "Architecture"]


@dataclass
class WarningMessage:
    title: str
    changed: Sequence[str]
    missing: str
    sections: Sequence[str]


def run_git(args: Sequence[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def git_ref_exists(ref: str) -> bool:
    result = run_git(["rev-parse", "--verify", f"{ref}^{{commit}}"])
    return result.returncode == 0


def detect_origin_default_branch() -> str | None:
    result = run_git(["symbolic-ref", "--quiet", "refs/remotes/origin/HEAD"])
    if result.returncode != 0:
        return None
    target = result.stdout.strip()
    if not target:
        return None
    if target.startswith("refs/remotes/"):
        return target.replace("refs/remotes/", "")
    return target


def resolve_base_ref(base: str | None, head: str) -> str:
    desired = base or DEFAULT_BASE
    candidates: List[str] = [desired]
    if desired.startswith("origin/"):
        default_branch = detect_origin_default_branch()
        if default_branch and default_branch not in candidates:
            candidates.append(default_branch)
    if "main" not in candidates:
        candidates.append("main")
    if "master" not in candidates:
        candidates.append("master")
    if head not in candidates:
        candidates.append(head)
    for candidate in candidates:
        if git_ref_exists(candidate):
            return candidate
    print(
        f"{WARNING_PREFIX}: Unable to resolve base reference from {candidates}, "
        f"defaulting to {head}.",
        file=sys.stderr,
    )
    return head


def detect_code_dirs() -> List[str]:
    candidates = ["src", "app", "lib"]
    existing = [c for c in candidates if Path(c).is_dir()]
    return existing if existing else ["."]


def gather_changed_files(base: str, head: str) -> List[str]:
    result = run_git(["diff", "--name-only", f"{base}...{head}"])
    if result.returncode != 0:
        print(
            f"{WARNING_PREFIX}: git diff failed: {result.stderr.strip()}",
            file=sys.stderr,
        )
        return []
    files = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return files


def is_in_code_dirs(path: str, code_dirs: Sequence[str]) -> bool:
    if "." in code_dirs:
        return True
    return any(path.startswith(f"{d.rstrip('/')}/") for d in code_dirs)


def suggest_sections(path: str) -> List[str]:
    sections: List[str] = []
    lowered = path.lower()
    for key, targets in SECTION_SUGGESTIONS.items():
        if key in lowered:
            for target in targets:
                if target not in sections:
                    sections.append(target)
    if not sections:
        sections.extend(DEFAULT_SECTIONS)
    return sections


def analyze_docstrings(py_path: Path) -> List[tuple[int, str, str]]:
    try:
        text = py_path.read_text(encoding="utf-8")
    except OSError as exc:
        print(
            f"{WARNING_PREFIX}: Could not read {py_path}: {exc}",
            file=sys.stderr,
        )
        return []
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        print(
            f"{WARNING_PREFIX}: Skipping {py_path} due to syntax error: {exc}",
            file=sys.stderr,
        )
        return []

    missing: List[tuple[int, str, str]] = []

    def visit(node: ast.AST) -> None:
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            name = getattr(node, "name", "")
            if name and not name.startswith("_"):
                if ast.get_docstring(node, clean=False) is None:
                    missing.append((node.lineno, node.__class__.__name__, name))
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    return missing


def diff_contains_docstring(base: str, head: str, path: str) -> bool:
    result = run_git(["diff", f"{base}...{head}", "--", path])
    if result.returncode != 0:
        return False
    diff_text = result.stdout
    return '"""' in diff_text or "'''" in diff_text


def emit_warning(message: WarningMessage) -> None:
    changed = ", ".join(message.changed)
    sections = ", ".join(message.sections)
    print(f"{WARNING_PREFIX}: {message.title}")
    print(f"  - What changed: {changed}")
    print(f"  - Missing: {message.missing}")
    if sections:
        print(f"  - Suggested README sections to update: {sections}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Documentation hygiene agent.")
    parser.add_argument("--base", default=None, help="Base git ref (default: origin/main).")
    parser.add_argument("--head", default="HEAD", help="Head git ref (default: HEAD).")
    args = parser.parse_args()

    base_ref = resolve_base_ref(args.base, args.head)
    head_ref = args.head

    changed_files = gather_changed_files(base_ref, head_ref)
    if not changed_files:
        print(f"{WARNING_PREFIX}: No changes detected between {base_ref} and {head_ref}.")
        return 0

    code_dirs = detect_code_dirs()
    readme_changed = "README.md" in changed_files
    code_changes = [
        path for path in changed_files if is_in_code_dirs(path, code_dirs)
    ]

    if code_changes and not readme_changed:
        sections = sorted(
            {section for path in code_changes for section in suggest_sections(path)}
        )
        emit_warning(
            WarningMessage(
                title="Code changes detected without README.md updates.",
                changed=sorted(code_changes)[:10],
                missing="README.md did not change alongside code updates.",
                sections=sections,
            )
        )

    for path in changed_files:
        if not path.endswith(".py"):
            continue
        py_path = Path(path)
        if not py_path.is_file():
            continue
        missing = analyze_docstrings(py_path)
        for lineno, node_type, name in missing:
            emit_warning(
                WarningMessage(
                    title=f"Missing docstring in {path} ({node_type}: {name})",
                    changed=[f"{path}:{lineno}"],
                    missing="Add an immediate triple-quoted docstring describing the symbol.",
                    sections=suggest_sections(path),
                )
            )
        if not diff_contains_docstring(base_ref, head_ref, path):
            emit_warning(
                WarningMessage(
                    title=f"No docstring updates detected in {path}",
                    changed=[path],
                    missing="Consider updating docstrings for modified functions/classes.",
                    sections=suggest_sections(path),
                )
            )

    if not code_changes:
        print(f"{WARNING_PREFIX}: No files under {code_dirs} changed.")
    else:
        print(f"{WARNING_PREFIX}: Documentation check completed for {len(code_changes)} code files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
