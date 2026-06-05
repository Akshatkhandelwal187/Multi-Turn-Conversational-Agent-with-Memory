"""Tests for document RAG: chunking, indexing, citations, and the tool."""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from aria.config import Settings
from aria.embeddings.hashing import HashingEmbedder
from aria.graph.cognitive import build_cognitive_agent
from aria.models.fakes import ScriptedModel
from aria.rag.chunker import chunk_text
from aria.rag.index import DocumentIndex
from aria.rag.loaders import load_bytes
from aria.tools.retrieve_documents import make_retrieve_documents_tool
from aria.vectorstore.numpy_store import NumpyVectorStore

_DOC = (
    "The Aria project implements a cognitive memory architecture. "
    "Episodic memory uses a Generative Agents retrieval score. "
    "Photosynthesis converts sunlight into chemical energy in plants. "
    "The capital of France is Paris and it sits on the Seine."
)


def _index() -> DocumentIndex:
    settings = Settings(persist=False, embedder="hashing", hashing_dim=256, chunk_size=80, chunk_overlap=20)
    emb = HashingEmbedder(dim=256)
    return DocumentIndex(emb, NumpyVectorStore(dim=256), settings)


def test_chunk_text_basic():
    assert chunk_text("") == []
    assert chunk_text("short text") == ["short text"]
    chunks = chunk_text("word " * 200, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)


def test_load_bytes_text():
    assert load_bytes("a.txt", b"hello world") == "hello world"


def test_index_and_retrieve_with_citations():
    idx = _index()
    n = idx.add_document(_DOC, "notes.txt")
    assert n >= 1
    assert idx.sources() == ["notes.txt"]
    out = idx.retrieve_with_citations("what is the capital of France?", k=1)
    assert "Paris" in out
    assert "[notes.txt #" in out


def test_retrieve_documents_tool():
    idx = _index()
    idx.add_document(_DOC, "notes.txt")
    tool = make_retrieve_documents_tool(idx, idx.settings)
    out = tool.invoke({"query": "Generative Agents retrieval"})
    assert "Episodic" in out or "Generative" in out


def test_empty_index_returns_message():
    idx = _index()
    assert "No relevant documents" in idx.retrieve_with_citations("anything")


def test_rag_end_to_end_through_graph():
    settings = Settings(
        persist=False, embedder="hashing", hashing_dim=256,
        reflection_every_k_turns=0, summary_token_budget=100_000,
        enable_tools=True, enabled_tools=["retrieve_documents"],
        prefer_native_tool_calls=False, max_tool_iters=4, chunk_size=80, chunk_overlap=20,
    )
    model = ScriptedModel(
        [
            'Action: {"tool": "retrieve_documents", "args": {"query": "capital of France"}}',
            "Final: According to your notes, it is Paris.",
        ]
    )
    agent = build_cognitive_agent(model=model, settings=settings)
    agent.manager.document_index.add_document(_DOC, "notes.txt")

    result = agent.invoke(
        {"messages": [HumanMessage(content="What does my document say about France?")]},
        {"configurable": {"thread_id": "rag"}},
    )
    assert "Paris" in result["messages"][-1].content
    assert result["usage"]["tool_calls"] >= 1
