"""Structured logging for Pixel v2.

Uses structlog for structured, context-rich logging.
Supports console (dev) and JSON (production) output formats.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


_configured = False


def setup_logging(level: str = "INFO", fmt: str = "console", log_file: str = "") -> None:
    """Configure structured logging for the entire application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        fmt: Output format — "console" for human-readable, "json" for structured.
        log_file: Optional file path for log output. Empty = stdout only.
    """
    global _configured
    if _configured:
        return

    log_level = getattr(logging, level.upper(), logging.INFO)

    # Shared processors
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if fmt == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    # Configure root logger
    root = logging.getLogger()
    root.setLevel(log_level)
    root.handlers.clear()

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    _configured = True


def get_logger(name: str = "pixel", **initial_context: Any) -> structlog.stdlib.BoundLogger:
    """Get a bound structured logger.

    Args:
        name: Logger name (usually module name).
        **initial_context: Key-value pairs bound to every log entry.

    Returns:
        A structlog BoundLogger instance.
    """
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    if initial_context:
        logger = logger.bind(**initial_context)
    return logger


def bind_request_context(request_id: str, **extra: Any) -> None:
    """Bind request context to all loggers in the current async context.

    Args:
        request_id: Unique request identifier.
        **extra: Additional context to bind.
    """
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id, **extra)


def clear_request_context() -> None:
    """Clear per-request logging context."""
    structlog.contextvars.clear_contextvars()
