"""Root conftest.py — adds project root to sys.path for analysis.* imports."""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so that top-level packages like
# `analysis` (which are not installed via setuptools) can be imported in tests.
_project_root = str(Path(__file__).parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
