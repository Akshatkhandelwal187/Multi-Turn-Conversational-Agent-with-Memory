"""A clock tool so the agent can answer time/date questions accurately."""

from __future__ import annotations

from datetime import datetime

from langchain_core.tools import tool


@tool
def current_datetime() -> str:
    """Return the current local date and time in ISO 8601 format.

    Use this whenever the user asks about the current date, time, or day of week.
    """
    return datetime.now().isoformat(timespec="seconds")


__all__ = ["current_datetime"]
