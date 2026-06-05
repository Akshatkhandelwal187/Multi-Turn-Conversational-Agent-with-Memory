"""Document loaders for RAG ingestion.

Plain-text/markdown are read directly; PDFs go through :mod:`pypdf` (lazily imported,
so the dependency is only needed when a PDF is actually loaded). Both a path-based and
a bytes-based entry point are provided — the latter for Streamlit file uploads.
"""

from __future__ import annotations

import io
from pathlib import Path

from ..exceptions import AriaError


def _load_pdf_stream(stream: io.BytesIO) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - only without the rag extra
        raise AriaError(
            "Reading PDFs requires pypdf. Install the rag extra: pip install -e '.[rag]'."
        ) from exc
    reader = PdfReader(stream)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_path(path: str | Path) -> str:
    """Load text from a file path (.pdf via pypdf, otherwise read as UTF-8 text)."""
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        with p.open("rb") as fh:
            return _load_pdf_stream(io.BytesIO(fh.read()))
    return p.read_text(encoding="utf-8", errors="ignore")


def load_bytes(filename: str, data: bytes) -> str:
    """Load text from in-memory bytes (e.g. a Streamlit upload), keyed by filename."""
    if filename.lower().endswith(".pdf"):
        return _load_pdf_stream(io.BytesIO(data))
    return data.decode("utf-8", errors="ignore")


__all__ = ["load_bytes", "load_path"]
