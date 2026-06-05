"""The document index for retrieval-augmented generation.

Documents are chunked, embedded with the same embedder as memory, and stored in a
*separate* vector collection (namespace ``docs``) so document retrieval never mixes
with conversational memory. Retrieval returns excerpts annotated with
``[source #chunk]`` citations the agent can quote.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ..embeddings.base import Embedder
from ..vectorstore.base import VectorHit, VectorStore
from .chunker import chunk_text
from .loaders import load_bytes, load_path

if TYPE_CHECKING:
    from ..config import Settings


class DocumentIndex:
    """A chunked, embedded, citable document store for RAG."""

    def __init__(self, embedder: Embedder, store: VectorStore, settings: Settings) -> None:
        self.embedder = embedder
        self.store = store
        self.settings = settings

    def add_document(self, text: str, source: str) -> int:
        """Chunk, embed, and index a document. Returns the number of chunks added."""
        chunks = chunk_text(text, self.settings.chunk_size, self.settings.chunk_overlap)
        for i, chunk in enumerate(chunks):
            self.store.add(
                f"{source}::{i}",
                self.embedder.embed_one(chunk),
                {"text": chunk, "source": source, "chunk_id": i},
            )
        return len(chunks)

    def add_path(self, path: str | Path) -> int:
        return self.add_document(load_path(path), Path(path).name)

    def add_bytes(self, filename: str, data: bytes) -> int:
        return self.add_document(load_bytes(filename, data), filename)

    def search(self, query: str, k: int | None = None) -> list[VectorHit]:
        k = self.settings.rag_top_k if k is None else k
        if k <= 0 or len(self.store) == 0:
            return []
        return self.store.search(self.embedder.embed_one(query), k=k)

    def retrieve_with_citations(self, query: str, k: int | None = None) -> str:
        hits = self.search(query, k)
        if not hits:
            return "No relevant documents found."
        blocks = []
        for hit in hits:
            source = hit.metadata.get("source", "?")
            chunk_id = hit.metadata.get("chunk_id", "?")
            blocks.append(f"[{source} #{chunk_id}] {hit.text}")
        return "\n\n".join(blocks)

    def sources(self) -> list[str]:
        seen: dict[str, None] = {}
        for mid in self.store.all_ids():
            meta = self.store.get(mid)
            if meta:
                seen.setdefault(str(meta.get("source", "?")), None)
        return list(seen)

    def persist(self) -> None:
        self.store.persist()

    def __len__(self) -> int:
        return len(self.store)


__all__ = ["DocumentIndex"]
