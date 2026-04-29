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
DEPRECATED: This module is deprecated. Use src.utils.logging_config instead.

This module now re-exports from logging_config for backward compatibility.
All new code should import directly from src.utils.logging_config.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

# Re-export everything from the new logging_config module
from src.utils.logging_config import (
    setup_logging,
    get_logger,
    log_execution_time,
    log_progress,
    LoggerAdapter,
    configure_root_logger,
    ColoredFormatter,
    LOG_FORMAT,
    DATE_FORMAT,
)

# Backward-compatible configure function
def configure(debug: bool = False, log_file: Optional[str] = None) -> None:
    """
    DEPRECATED: Use setup_logging() or configure_root_logger() instead.
    
    Configure logging for pipeline scripts.
    
    Args:
        debug: If True, sets log level to DEBUG, otherwise INFO
        log_file: Optional path to log file
    """
    warnings.warn(
        "src.utils.logging.configure() is deprecated. "
        "Use src.utils.logging_config.setup_logging() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    level = "DEBUG" if debug else "INFO"
    
    if log_file:
        setup_logging("nfl_predictions", level=level, log_file=Path(log_file))
    else:
        setup_logging("nfl_predictions", level=level)


__all__ = [
    "configure",
    "setup_logging",
    "get_logger",
    "log_execution_time",
    "log_progress",
    "LoggerAdapter",
    "configure_root_logger",
    "ColoredFormatter",
    "LOG_FORMAT",
    "DATE_FORMAT",
]
