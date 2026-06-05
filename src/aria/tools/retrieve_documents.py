"""The document-retrieval (RAG) tool."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.tools import tool

if TYPE_CHECKING:
    from ..rag.index import DocumentIndex


def make_retrieve_documents_tool(doc_index: DocumentIndex, settings: Any):
    """Build a ``retrieve_documents`` tool bound to a document index."""

    @tool
    def retrieve_documents(query: str) -> str:
        """Search the user's uploaded documents for passages relevant to the query.

        Returns excerpts with [source #chunk] citations. Use this whenever the user asks
        about the content of a document they provided, and cite the sources in your answer.
        """
        return doc_index.retrieve_with_citations(query, k=settings.rag_top_k)

    return retrieve_documents


__all__ = ["make_retrieve_documents_tool"]
