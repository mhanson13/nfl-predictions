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

import builtins
import os
from typing import Callable

_ORIGINAL_PRINT: Callable[..., None] = builtins.print


def configure(debug: bool) -> None:
    """
    Configure lightweight stdout logging for pipeline scripts.
    When debug is False, suppresses print statements for quieter runs.
    """
    os.environ["PIPELINE_DEBUG"] = "1" if debug else "0"
    if debug:
        builtins.print = _ORIGINAL_PRINT
        return

    def _noop(*_args, **_kwargs) -> None:
        pass

    builtins.print = _noop
