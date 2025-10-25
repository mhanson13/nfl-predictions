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
