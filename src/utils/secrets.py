from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

_SECRETS_CACHE: dict[str, Optional[str]] = {}


def _load_env_file() -> dict[str, str]:
    secrets_path = Path("secrets.env")
    if not secrets_path.exists():
        return {}
    data: dict[str, str] = {}
    for line in secrets_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip()
    return data


_FILE_SECRETS = _load_env_file()


def get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    if name in _SECRETS_CACHE:
        return _SECRETS_CACHE[name]
    value = os.getenv(name)
    if value is None:
        value = _FILE_SECRETS.get(name, default)
    _SECRETS_CACHE[name] = value
    return value

