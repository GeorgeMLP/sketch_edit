"""Logging utilities with Pacific timezone support."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

from pytz import timezone

# Pacific timezone
TZ = timezone("US/Pacific")


def _timetz(*args) -> tuple:
    """Return timetuple in Pacific timezone for logging formatter."""
    return datetime.now(TZ).timetuple()


def setup_logging(
    log_file: str | Path | None = None,
    level: int = logging.INFO,
    name: str | None = None,
) -> logging.Logger:
    """
    Set up logging with Pacific timezone.

    Args:
        log_file: Optional path to log file. If None, logs only to stderr.
        level: Logging level (default: INFO).
        name: Logger name. If None, configures root logger.

    Returns:
        Configured logger instance.
    """
    # Create formatter with Pacific timezone
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    formatter.converter = _timetz

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))

    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(level)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []  # Clear existing handlers
    for handler in handlers:
        logger.addHandler(handler)

    return logger


def get_logger(name: str, log_file: str | Path | None = None) -> logging.Logger:
    """
    Get a logger with Pacific timezone formatting.

    Args:
        name: Logger name (typically __name__).
        log_file: Optional log file path.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        formatter.converter = _timetz

        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(formatter)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        if log_file is not None:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.INFO)
            logger.addHandler(file_handler)

    return logger
