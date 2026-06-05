"""User interface (Streamlit). Kept import-light: importing this package does not
import Streamlit. Use ``from aria.ui.streamlit_app import main`` for the app.
"""

from __future__ import annotations

from .conversations import ConversationRegistry

__all__ = ["ConversationRegistry"]
