"""Streamlit chat UI for Aria.

Surfaces the whole system: streaming replies, resumable named conversations (durable
via the SQLite checkpointer), document upload for RAG, and live panels for the memory
state (profile facts, reflections, store sizes) and per-turn / session usage metrics.
The compiled agent is cached with ``@st.cache_resource`` so the durable stores and the
memory manager persist across Streamlit reruns.
"""

from __future__ import annotations

import time
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from ..config import Settings, get_settings
from ..graph.cognitive import build_cognitive_agent
from ..logging import configure_logging
from ..observability.tokens import TurnUsage, UsageTracker
from ..utils.messages import message_text
from .conversations import ConversationRegistry

load_dotenv()


def _ui_settings() -> Settings:
    settings = get_settings()
    if "retrieve_documents" not in settings.enabled_tools:
        settings.enabled_tools = [*settings.enabled_tools, "retrieve_documents"]
    settings.ensure_dirs()
    configure_logging(settings.log_level, settings.log_json)
    return settings


@st.cache_resource(show_spinner="Building the cognitive agent…")
def get_agent() -> Any:
    return build_cognitive_agent(settings=_ui_settings())


@st.cache_resource(show_spinner=False)
def get_registry(_settings: Settings) -> ConversationRegistry:
    path = _settings.data_dir / "conversations.json" if _settings.persist else None
    return ConversationRegistry(path)


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _typewriter(text: str):
    """Yield a final answer in small chunks for a streaming feel (model-agnostic)."""
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.012)


def _history(agent: Any, thread_id: str) -> list:
    try:
        state = agent.get_state(_config(thread_id))
        return state.values.get("messages", []) if state and state.values else []
    except Exception:
        return []


def _run_turn(agent: Any, prompt: str, thread_id: str) -> tuple[str, dict]:
    result = agent.invoke({"messages": [HumanMessage(content=prompt)]}, _config(thread_id))
    answer = ""
    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage) and message_text(msg).strip():
            answer = message_text(msg)
            break
    return answer or "(no response)", result.get("usage", {})


def _sidebar(agent: Any, settings: Settings, registry: ConversationRegistry) -> None:
    with st.sidebar:
        st.header("🧠 Aria")
        st.caption("Cognitive memory · ReAct tools · RAG · LangGraph")

        token_present = bool(settings.hf_token)
        st.markdown(f"**Model:** `{settings.hf_model}`")
        st.markdown(f"**Embedder:** `{settings.embedder}`")
        if token_present:
            st.success("HF token detected", icon="✅")
        else:
            st.warning("No HUGGINGFACEHUB_API_TOKEN set (see .env)", icon="⚠️")

        # --- conversations ---
        st.subheader("Conversations")
        conversations = registry.list()
        labels = {c["id"]: c["name"] for c in conversations}
        if conversations:
            current = st.session_state.thread_id
            options = [c["id"] for c in conversations]
            index = options.index(current) if current in options else 0
            chosen = st.selectbox(
                "Active conversation", options, index=index, format_func=lambda i: labels[i]
            )
            if chosen != st.session_state.thread_id:
                st.session_state.thread_id = chosen
                st.rerun()
        if st.button("➕ New conversation", use_container_width=True):
            st.session_state.thread_id = registry.create()
            st.rerun()

        # --- documents (RAG) ---
        st.subheader("Documents (RAG)")
        uploads = st.file_uploader(
            "Upload .txt / .md / .pdf", type=["txt", "md", "pdf"], accept_multiple_files=True
        )
        if uploads:
            for file in uploads:
                key = f"indexed::{file.name}"
                if not st.session_state.get(key):
                    n = agent.manager.document_index.add_bytes(file.name, file.getvalue())
                    agent.manager.persist()
                    st.session_state[key] = True
                    st.toast(f"Indexed {file.name} ({n} chunks)")
        sources = agent.manager.document_index.sources()
        if sources:
            st.caption("Indexed: " + ", ".join(sources))

        _memory_panel(agent)
        _metrics_panel()

        if st.button("🔄 Reset everything", use_container_width=True):
            st.session_state.thread_id = registry.create()
            st.session_state.tracker = UsageTracker()
            st.rerun()


def _memory_panel(agent: Any) -> None:
    st.subheader("Memory")
    st.caption(
        f"Episodic: {len(agent.manager.episodic)} · "
        f"Docs: {len(agent.manager.document_index)}"
    )
    facts = agent.manager.semantic.facts()
    with st.expander(f"User profile ({len(facts)} facts)"):
        if facts:
            for key, value in facts.items():
                st.markdown(f"- **{key.replace('_', ' ')}**: {value}")
        else:
            st.caption("Nothing learned yet.")
    reflections = agent.manager.reflections(limit=10)
    if reflections:
        with st.expander(f"Reflections ({len(reflections)})"):
            for insight in reflections:
                st.markdown(f"- {insight}")


def _metrics_panel() -> None:
    tracker: UsageTracker = st.session_state.tracker
    last = tracker.last()
    totals = tracker.totals()
    st.subheader("Metrics")
    if last is not None:
        col1, col2 = st.columns(2)
        col1.metric("Last tokens", last.total_tokens)
        col2.metric("Last latency", f"{last.latency_ms:.0f} ms")
        st.caption(
            f"Retrieved {last.retrieved_memories} memories · {last.tool_calls} tool calls"
            + (" · reflected" if last.reflected else "")
            + (" · summarized" if last.summarized else "")
        )
    st.caption(
        f"Session: {totals['turns']} turns · {totals['total_tokens']} tokens · "
        f"{totals['tool_calls']} tool calls"
    )


def main() -> None:
    st.set_page_config(page_title="Aria — Memory Agent", page_icon="🧠", layout="wide")
    settings = _ui_settings()
    agent = get_agent()
    registry = get_registry(settings)

    if "thread_id" not in st.session_state:
        existing = registry.list()
        st.session_state.thread_id = existing[0]["id"] if existing else registry.create()
    if "tracker" not in st.session_state:
        st.session_state.tracker = UsageTracker()

    _sidebar(agent, settings, registry)

    st.title("🧠 Aria — Multi-Turn Agent with Cognitive Memory")
    st.caption(
        "Ask something, then follow up — Aria recalls earlier turns, learns durable facts, "
        "uses tools, and can read documents you upload. Memory persists across conversations."
    )

    for msg in _history(agent, st.session_state.thread_id):
        if isinstance(msg, HumanMessage):
            st.chat_message("user").markdown(message_text(msg))
        elif isinstance(msg, AIMessage) and message_text(msg).strip():
            st.chat_message("assistant").markdown(message_text(msg))

    if prompt := st.chat_input("Ask me anything…"):
        st.chat_message("user").markdown(prompt)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking…"):
                    answer, usage = _run_turn(agent, prompt, st.session_state.thread_id)
                st.write_stream(_typewriter(answer))
            except Exception as exc:  # surface config/network errors instead of crashing
                st.error(
                    "The model call failed. Check that HUGGINGFACEHUB_API_TOKEN is set and "
                    f"the model `{settings.hf_model}` is reachable.\n\n```\n{exc}\n```"
                )
                usage = {}
        if usage:
            st.session_state.tracker.record(
                TurnUsage(
                    tokens_in=int(usage.get("tokens_in", 0)),
                    tokens_out=int(usage.get("tokens_out", 0)),
                    tool_calls=int(usage.get("tool_calls", 0)),
                    retrieved_memories=int(usage.get("retrieved_memories", 0)),
                    latency_ms=float(usage.get("latency_ms", 0.0)),
                    reflected=bool(usage.get("reflected", False)),
                    summarized=bool(usage.get("summarized", False)),
                )
            )
        registry.touch(st.session_state.thread_id)
        st.rerun()


if __name__ == "__main__":
    main()
