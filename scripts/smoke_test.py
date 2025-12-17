#!/usr/bin/env python3
# Copyright (c) 2025 Matt Hanson
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Minimal smoke test to confirm the NFL predictions stack can import its
dependencies and key entry points. Run after installing dependencies:

    python scripts/smoke_test.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


CORE_MODULES: tuple[str, ...] = (
    "pandas",
    "numpy",
    "sklearn",
    "xgboost",
    "pyarrow",
)

ENTRYPOINTS: tuple[str, ...] = (
    "tools.run_pipeline",
    "src.features.build_features",
    "src.models.train",
    "src.predict.predict_upcoming",
    "src.evaluation.evaluate_predictions",
)

REQUIRED_DIRS: tuple[str, ...] = (
    "data",
    "data/raw",
    "data/processed",
    "models",
    "predictions",
    "results",
)


def check_imports(modules: Iterable[str]) -> list[str]:
    failures: list[str] = []
    for module in modules:
        try:
            importlib.import_module(module)
        except Exception as exc:  # pylint: disable=broad-except
            failures.append(f"Import failed for '{module}': {exc}")
    return failures


def check_directories(directories: Iterable[str]) -> list[str]:
    missing = [path for path in directories if not Path(path).exists()]
    if missing:
        return [f"Missing required directories: {', '.join(missing)}"]
    return []


def main() -> int:
    failures = []
    failures.extend(check_imports(CORE_MODULES))
    failures.extend(check_imports(ENTRYPOINTS))
    failures.extend(check_directories(REQUIRED_DIRS))

    if failures:
        print("Smoke test failed:\n- " + "\n- ".join(failures))
        return 1

    print("Smoke test passed: imports and project layout look healthy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
