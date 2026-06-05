"""Fixed-size text chunking with overlap.

Splits a document into overlapping character windows, preferring to break at the
nearest whitespace so words are not cut in half. Overlap preserves context across
chunk boundaries for better retrieval.
"""

from __future__ import annotations


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    """Split ``text`` into overlapping chunks of roughly ``chunk_size`` characters."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    overlap = max(0, min(overlap, chunk_size - 1))

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            # Back up to the last whitespace in the window for a cleaner break.
            window = text.rfind(" ", start + overlap, end)
            if window != -1 and window > start:
                end = window
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


__all__ = ["chunk_text"]
