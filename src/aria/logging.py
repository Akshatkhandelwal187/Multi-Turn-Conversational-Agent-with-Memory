"""Structured logging via :mod:`structlog`.

A single :func:`configure_logging` call sets up either human-friendly console
output (development) or line-delimited JSON (production / when ``ARIA_LOG_JSON`` is
set). Loggers returned by :func:`get_logger` support contextual binding, so nodes
can attach ``thread_id`` and ``turn`` once and have them appear on every log line.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

_CONFIGURED = False


def configure_logging(level: str = "INFO", json: bool = False) -> None:
    """Configure standard library + structlog logging.

    Idempotent: safe to call more than once (e.g. on every Streamlit rerun).

    Args:
        level: Root log level name (``"DEBUG"``, ``"INFO"``, ...).
        json: Emit JSON lines instead of a colourised console renderer.
    """
    global _CONFIGURED

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(format="%(message)s", stream=sys.stderr, level=numeric_level)

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]
    renderer: Any = (
        structlog.processors.JSONRenderer()
        if json
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True


def get_logger(name: str | None = None) -> Any:
    """Return a bound structlog logger, configuring logging on first use."""
    if not _CONFIGURED:
        configure_logging()
    return structlog.get_logger(name)


__all__ = ["configure_logging", "get_logger"]
