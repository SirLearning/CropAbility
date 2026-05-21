"""
Logging utilities
=================
Structured logging setup with colored console output, file rotation, and optional JSON format.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

# ANSI color codes (enabled on TTY)
_COLORS = {
    "DEBUG": "\033[36m",     # cyan
    "INFO": "\033[32m",      # green
    "WARNING": "\033[33m",   # yellow
    "ERROR": "\033[31m",     # red
    "CRITICAL": "\033[35m",  # magenta
    "RESET": "\033[0m",
}


class _ColorFormatter(logging.Formatter):
    """Add colors to log levels in the terminal."""

    def __init__(self, fmt: str, datefmt: str, use_color: bool = True) -> None:
        super().__init__(fmt, datefmt=datefmt)
        self._use_color = use_color and sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if self._use_color:
            color = _COLORS.get(record.levelname, "")
            reset = _COLORS["RESET"]
            msg = f"{color}{msg}{reset}"
        return msg


_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_initialized = False


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    fmt: str = _LOG_FORMAT,
    datefmt: str = _DATE_FORMAT,
    max_bytes: int = 50 * 1024 * 1024,
    backup_count: int = 5,
) -> None:
    """Initialize global logging. Call once at program entry."""
    global _initialized
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_ColorFormatter(fmt, datefmt))
        root.addHandler(handler)

    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        root.addHandler(file_handler)

    # Suppress noisy third-party loggers
    for noisy in ("PIL", "matplotlib", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger; applies default setup if logging is not initialized."""
    if not _initialized:
        setup_logging()
    return logging.getLogger(name)
