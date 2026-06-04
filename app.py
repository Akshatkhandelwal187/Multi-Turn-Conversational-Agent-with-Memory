"""Streamlit chat UI for the multi-turn conversational agent with memory."""

from __future__ import annotations

import os
import uuid

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from agent import DEFAULT_MODEL, SYSTEM_PERSONA, build_agent

load_dotenv()


@st.cache_resource(show_spinner=False)
def get_agent():
    """Build the agent once and reuse it across Streamlit reruns.

    Caching is essential: the in-memory ``MemorySaver`` checkpointer lives inside
    the compiled graph, so a single cached instance is what keeps conversation
    memory alive between Streamlit reruns.
    """
    return build_agent()


def main() -> None:
    st.set_page_config(page_title="Memory Chatbot", page_icon="🧠")

    active_model = os.environ.get("HF_MODEL", DEFAULT_MODEL)
    token_present = bool(os.environ.get("HUGGINGFACEHUB_API_TOKEN"))

    # --- per-session state ---
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- sidebar ---
    with st.sidebar:
        st.header("🧠 Memory Chatbot")
        st.caption("LangGraph state graph · MemorySaver · Hugging Face")
        st.markdown(f"**Model:** `{active_model}`")
        if token_present:
            st.success("HUGGINGFACEHUB_API_TOKEN detected", icon="✅")
        else:
            st.warning(
                "No HUGGINGFACEHUB_API_TOKEN found. Copy `.env.example` to `.env` "
                "and set your token to talk to the model.",
                icon="⚠️",
            )
        with st.expander("System persona"):
            st.write(SYSTEM_PERSONA)
        st.caption(f"Thread: `{st.session_state.thread_id[:8]}`")
        if st.button("🔄 Reset conversation", use_container_width=True):
            # A new thread_id means the checkpointer starts with a fresh memory.
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.rerun()

    st.title("🧠 Multi-Turn Chatbot with Memory")
    st.caption(
        "Ask something, then ask a follow-up like *“what did I just ask you?”* — "
        "the agent remembers the whole conversation."
    )

    # --- render prior history ---
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # --- handle new input ---
    if prompt := st.chat_input("Ask me anything..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Build the agent lazily on first use so the page still
                    # renders (and shows the token warning) without a token.
                    agent = get_agent()
                    result = agent.invoke(
                        {"messages": [HumanMessage(content=prompt)]}, config
                    )
                    answer = result["messages"][-1].content
                except Exception as exc:  # surface config/network errors in the UI
                    answer = (
                        "⚠️ The model call failed. Check that "
                        "`HUGGINGFACEHUB_API_TOKEN` is set and that the model "
                        f"`{active_model}` is available via the Inference API.\n\n"
                        f"```\n{exc}\n```"
                    )
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
