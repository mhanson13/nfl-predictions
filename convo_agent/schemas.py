from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class Document:
    """Lightweight container for retrievable text and its metadata."""

    text: str
    metadata: Dict[str, Any]

