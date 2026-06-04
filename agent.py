"""LangGraph conversational agent with memory and a system persona.

The agent is a minimal LangGraph state graph::

    START --> model

State is the built-in :class:`MessagesState` (a growing list of messages). A
:class:`MemorySaver` checkpointer persists that list per ``thread_id``, so every
turn the model is re-invoked with the *entire* prior conversation. That
full-history replay is what lets the bot answer follow-ups such as "what did I
just ask you?".

The model is an open-source LLM served through the Hugging Face Inference API via
``langchain-huggingface``. :func:`build_agent` accepts an injectable ``model`` so
tests can run fully offline with a fake chat model.
"""

from __future__ import annotations

import os
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

# Default open-source instruct model. Qwen2.5-7B-Instruct is openly accessible
# (no gating) and served by Hugging Face Inference Providers. Override with the
# HF_MODEL environment variable.
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Fixed system persona. Prepended to every model call (never stored in state) so
# the personality stays constant and is not duplicated in the message history.
SYSTEM_PERSONA = (
    "You are Aria, a warm, concise, and helpful assistant. "
    "You remember everything the user has said earlier in this conversation and "
    "use that context to answer follow-up questions accurately. "
    "When the user refers to something they mentioned before, recall it precisely. "
    "Keep answers clear and to the point, and admit when you don't know something."
)


def build_model() -> BaseChatModel:
    """Create the Hugging Face-backed chat model from environment configuration.

    Reads ``HF_MODEL`` (default :data:`DEFAULT_MODEL`) and
    ``HUGGINGFACEHUB_API_TOKEN``. The import is local so the rest of the module
    (and the offline tests) does not require ``langchain-huggingface``.
    """
    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

    llm = HuggingFaceEndpoint(
        repo_id=os.environ.get("HF_MODEL", DEFAULT_MODEL),
        task="text-generation",
        max_new_tokens=512,
        temperature=0.7,
        provider="auto",  # let Hugging Face route to an available inference provider
        huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
    )
    return ChatHuggingFace(llm=llm)


def build_agent(model: Optional[BaseChatModel] = None, persona: str = SYSTEM_PERSONA):
    """Build and compile the LangGraph agent with conversation memory.

    Args:
        model: Chat model to use. Defaults to the Hugging Face model from
            :func:`build_model`. Inject a fake model here to run offline.
        persona: System persona prepended to every model invocation.

    Returns:
        A compiled LangGraph application. Invoke it with a ``thread_id`` so the
        checkpointer can keep memory for that conversation::

            app.invoke(
                {"messages": [HumanMessage(content="hi")]},
                {"configurable": {"thread_id": "some-stable-id"}},
            )
    """
    if model is None:
        model = build_model()

    def call_model(state: MessagesState) -> dict:
        # Prepend the persona to the full running history for this thread, so the
        # model always sees who it is plus everything said so far.
        messages = [SystemMessage(content=persona)] + state["messages"]
        response = model.invoke(messages)
        return {"messages": response}

    graph = StateGraph(MessagesState)
    graph.add_node("model", call_model)
    graph.add_edge(START, "model")

    # MemorySaver keeps each thread's message list in memory between invocations.
    return graph.compile(checkpointer=MemorySaver())
