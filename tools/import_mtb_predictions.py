# Copyright (c) 2025 Matt Hanson
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Run the MattyTheBookie prediction importer after CSV publishing."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Sequence

from tools.publish_mtb_csvs import DEFAULT_MTB_REPO_DIR


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the local MattyTheBookie npm prediction importer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=DEFAULT_MTB_REPO_DIR,
        help="Local MattyTheBookie repository directory.",
    )
    parser.add_argument(
        "--script",
        default="import:predictions",
        help="npm script to run in the MattyTheBookie repo.",
    )
    return parser.parse_args(argv)


def import_predictions(repo_dir: Path, script: str = "import:predictions") -> None:
    repo_root = repo_dir.resolve()
    if not repo_root.exists():
        raise FileNotFoundError(f"MattyTheBookie repo not found: {repo_root}")
    if not repo_root.is_dir():
        raise NotADirectoryError(f"MattyTheBookie repo path is not a directory: {repo_root}")

    package_json = repo_root / "package.json"
    if not package_json.exists():
        raise FileNotFoundError(f"package.json not found in MattyTheBookie repo: {package_json}")

    npm = shutil.which("npm.cmd") or shutil.which("npm")
    if not npm:
        raise FileNotFoundError("npm was not found on PATH; cannot run MattyTheBookie importer.")

    print(f"[import_mtb] Running npm run {script} in {repo_root}")
    result = subprocess.run([npm, "run", script], cwd=repo_root, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"MattyTheBookie importer failed with exit code {result.returncode}")
    print("[import_mtb] MattyTheBookie prediction JSON updated.")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    import_predictions(args.repo_dir, script=args.script)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
