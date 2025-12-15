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

