from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from src.utils.io import REF_DIR

CHECKPOINT_PATH = REF_DIR / "sportradar_checkpoint.json"


def load_checkpoint() -> Dict[str, Any]:
    if not CHECKPOINT_PATH.exists():
        return {}
    try:
        return json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_checkpoint(data: Dict[str, Any]) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def get_last_timestamp(key: str) -> datetime | None:
    data = load_checkpoint()
    value = data.get(key)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def update_timestamp(key: str, timestamp: datetime) -> None:
    data = load_checkpoint()
    data[key] = timestamp.isoformat()
    save_checkpoint(data)
