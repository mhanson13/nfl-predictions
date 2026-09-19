"""
Standardized logging configuration for the NFL predictions platform.

This module provides consistent logging setup across all modules with:
- Structured logging format
- Configurable log levels
- File and console handlers
- Performance tracking
- Context managers for timed operations
"""

import logging
import re
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.config import get_config


# Standard log format with timestamp, level, module, and message
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

SENSITIVE_QUERY_RE = re.compile(
    r"(?i)([?&](?:key|api_key|apikey|token|access_token|refresh_token|client_secret)=)([^&\s]+)"
)


def redact_secrets(text: str) -> str:
    """Mask common query-string credentials before log output."""
    return SENSITIVE_QUERY_RE.sub(r"\1***REDACTED***", text)


class RedactingFormatter(logging.Formatter):
    """Formatter that redacts secret-looking query params."""

    def format(self, record):
        original_msg = record.msg
        original_args = record.args
        try:
            record.msg = redact_secrets(record.getMessage())
            record.args = ()
            return super().format(record)
        finally:
            record.msg = original_msg
            record.args = original_args


class ColoredFormatter(RedactingFormatter):
    """Formatter that adds colors to console output."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        """Format log record with color."""
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(
    name: str = "nfl_predictions",
    level: Optional[str] = None,
    log_file: Optional[Path] = None,
    console: bool = True,
    colored: bool = True
) -> logging.Logger:
    """
    Set up standardized logging for a module.
    
    Args:
        name: Logger name (typically __name__)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
        console: Whether to log to console
        colored: Whether to use colored output (console only)
    
    Returns:
        Configured logger instance
    
    Example:
        logger = setup_logging(__name__)
        logger.info("Starting prediction pipeline")
    """
    config = get_config()
    
    # Determine log level
    if level is None:
        level = config.pipeline.log_level
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        
        if colored and sys.stdout.isatty():
            formatter = ColoredFormatter(LOG_FORMAT, DATE_FORMAT)
        else:
            formatter = RedactingFormatter(LOG_FORMAT, DATE_FORMAT)
        
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(RedactingFormatter(LOG_FORMAT, DATE_FORMAT))
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with standard configuration.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # If logger has no handlers, set it up
    if not logger.handlers:
        return setup_logging(name)
    
    return logger


@contextmanager
def log_execution_time(logger: logging.Logger, operation: str, level: str = "INFO"):
    """
    Context manager to log execution time of an operation.
    
    Args:
        logger: Logger instance
        operation: Description of the operation
        level: Log level for the timing message
    
    Example:
        with log_execution_time(logger, "Loading features"):
            df = pd.read_parquet("features.parquet")
    """
    start_time = time.time()
    log_func = getattr(logger, level.lower())
    
    log_func(f"Starting: {operation}")
    
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        log_func(f"Completed: {operation} (took {elapsed:.2f}s)")


@contextmanager
def log_progress(logger: logging.Logger, total: int, operation: str, interval: int = 100):
    """
    Context manager for logging progress of iterative operations.
    
    Args:
        logger: Logger instance
        total: Total number of items to process
        operation: Description of the operation
        interval: Log progress every N items
    
    Example:
        with log_progress(logger, len(games), "Processing games", interval=50) as progress:
            for i, game in enumerate(games):
                # ... process game
                progress(i + 1)
    """
    start_time = time.time()
    logger.info(f"Starting: {operation} ({total} items)")
    
    def update_progress(current: int):
        """Update progress log."""
        if current % interval == 0 or current == total:
            elapsed = time.time() - start_time
            rate = current / elapsed if elapsed > 0 else 0
            pct = (current / total * 100) if total > 0 else 0
            logger.info(
                f"Progress: {current}/{total} ({pct:.1f}%) | "
                f"Rate: {rate:.1f} items/s | Elapsed: {elapsed:.1f}s"
            )
    
    try:
        yield update_progress
    finally:
        elapsed = time.time() - start_time
        rate = total / elapsed if elapsed > 0 else 0
        logger.info(
            f"Completed: {operation} | "
            f"Total: {total} items | Rate: {rate:.1f} items/s | Time: {elapsed:.1f}s"
        )


class LoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that adds context to all log messages.
    
    Example:
        logger = get_logger(__name__)
        context_logger = LoggerAdapter(logger, {"season": 2024, "week": 10})
        context_logger.info("Processing games")
        # Output: ... | season=2024 week=10 | Processing games
    """
    
    def process(self, msg, kwargs):
        """Add context to log message."""
        if self.extra:
            context = " ".join(f"{k}={v}" for k, v in self.extra.items())
            msg = f"{context} | {msg}"
        return msg, kwargs


def configure_root_logger(level: str = "INFO", log_dir: Optional[Path] = None):
    """
    Configure the root logger for the entire application.
    
    Args:
        level: Log level
        log_dir: Directory for log files
    
    Example:
        configure_root_logger("DEBUG", Path("logs"))
    """
    config = get_config()
    
    if log_dir is None:
        log_dir = config.paths.results_dir / "logs"
    
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"nfl_predictions_{timestamp}.log"
    
    # Set up root logger
    setup_logging(
        name="nfl_predictions",
        level=level,
        log_file=log_file,
        console=True,
        colored=True
    )


# Convenience function for backward compatibility
def configure(level: str = "INFO", log_file: Optional[str] = None):
    """
    Legacy configure function for backward compatibility.
    
    Args:
        level: Log level
        log_file: Optional log file path
    """
    if log_file:
        setup_logging("nfl_predictions", level=level, log_file=Path(log_file))
    else:
        setup_logging("nfl_predictions", level=level)

# Made with Bob
