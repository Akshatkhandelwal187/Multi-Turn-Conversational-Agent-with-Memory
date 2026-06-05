"""Streamlit entrypoint for Aria.

Run with::

    streamlit run app.py

The implementation lives in :mod:`aria.ui.streamlit_app`; this file is a thin shim so
the familiar ``streamlit run app.py`` command keeps working.
"""

from __future__ import annotations

from aria.ui.streamlit_app import main

if __name__ == "__main__":
    main()
