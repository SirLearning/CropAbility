"""
日志工具模块
============
提供结构化日志配置，支持彩色控制台输出、文件轮转以及可选的 JSON 格式。
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

# ANSI 颜色码（在 TTY 中启用）
_COLORS = {
    "DEBUG": "\033[36m",     # cyan
    "INFO": "\033[32m",      # green
    "WARNING": "\033[33m",   # yellow
    "ERROR": "\033[31m",     # red
    "CRITICAL": "\033[35m",  # magenta
    "RESET": "\033[0m",
}


class _ColorFormatter(logging.Formatter):
    """在终端中为日志级别添加颜色。"""

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
    """初始化全局日志配置。应在程序入口处调用一次。"""
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

    # 抑制第三方库的冗余日志
    for noisy in ("PIL", "matplotlib", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """获取具名 logger，若全局日志尚未初始化则先进行默认初始化。"""
    if not _initialized:
        setup_logging()
    return logging.getLogger(name)
