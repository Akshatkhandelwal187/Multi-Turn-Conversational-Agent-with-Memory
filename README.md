# 🧠 Multi-Turn Conversational Agent with Memory

A Streamlit chatbot that **remembers your conversation**. It keeps full multi-turn
history, has a fixed system persona, and correctly answers follow-up questions such as
*"what did I just ask you?"* and *"based on my earlier question, now tell me X"* using
context from earlier in the chat.

Built with **LangGraph** (state graph + checkpointer for memory), an **open-source LLM
served via the Hugging Face Inference API**, and **Streamlit** for the UI.

[![CI](https://github.com/Akshatkhandelwal187/Multi-Turn-Conversational-Agent-with-Memory/actions/workflows/ci.yml/badge.svg)](https://github.com/Akshatkhandelwal187/Multi-Turn-Conversational-Agent-with-Memory/actions/workflows/ci.yml)

## How memory works

```
Streamlit UI
    │  HumanMessage + thread_id
    ▼
LangGraph state graph   (START ──▶ model node)
    │   state = MessagesState   (the running list of messages)
    ▼
MemorySaver checkpointer ── persists the message list per thread_id
    │   full prior history replayed every turn
    ▼
ChatHuggingFace ──▶ Hugging Face Inference API (open-source model)
```

- The graph's state is a growing **list of messages** (`MessagesState`).
- A **`MemorySaver` checkpointer** stores that list keyed by a **`thread_id`**.
- On every turn LangGraph loads the saved messages, appends the new user message, and
  the node re-invokes the model with the **entire conversation** (plus the persona).
  Replaying the full history is exactly what lets the model resolve follow-up questions.
- The **system persona** is prepended to each call but never stored in history, so it
  stays constant and is never duplicated.

This is the modern replacement for LangChain's deprecated `ConversationBufferMemory`.

## Project structure

| File | Purpose |
|------|---------|
| `agent.py` | LangGraph graph, the system persona, and the Hugging Face model factory |
| `app.py` | Streamlit chat UI (cached agent, per-session thread, reset button) |
| `tests/test_agent.py` | Offline tests proving persona injection + multi-turn memory |
| `requirements.txt` | Dependencies |
| `.env.example` | Template for `HUGGINGFACEHUB_API_TOKEN` and `HF_MODEL` |
| `.github/workflows/ci.yml` | Runs the test suite on push / PR |

## Setup

1. **Clone & install**
   ```bash
   git clone https://github.com/Akshatkhandelwal187/Multi-Turn-Conversational-Agent-with-Memory.git
   cd Multi-Turn-Conversational-Agent-with-Memory
   python -m venv .venv && source .venv/bin/activate   # optional
   pip install -r requirements.txt
   ```

2. **Add your Hugging Face token**

   Create a token at <https://huggingface.co/settings/tokens> (a *Read* token is enough),
   then copy the example env file and paste it in:
   ```bash
   cp .env.example .env
   # edit .env and set HUGGINGFACEHUB_API_TOKEN=hf_...
   ```

   The default model `Qwen/Qwen2.5-7B-Instruct` is openly accessible. Some models
   (e.g. `meta-llama/Llama-3.1-8B-Instruct`, `mistralai/Mistral-7B-Instruct-v0.3`) are
   **gated** — visit the model page and accept the terms first, then set `HF_MODEL`.

3. **Run the app**
   ```bash
   streamlit run app.py
   ```
   Open the URL Streamlit prints (usually <http://localhost:8501>).

## Demo: prove it remembers

Try this conversation in order:

| # | You type | What the agent does |
|---|----------|---------------------|
| 1 | `My favorite language is Python and I'm building a recommender system.` | Acknowledges and stores it in context. |
| 2 | `What did I just ask you?` | Recalls turn 1 — summarizes that you mentioned Python and a recommender system. |
| 3 | `Based on my earlier question, suggest a good library.` | Uses the earlier context (Python + recommender) to suggest e.g. `surprise`, `implicit`, or `LightFM`. |

Click **🔄 Reset conversation** in the sidebar to start a fresh thread with no memory.

## Testing

The offline tests inject a fake model, so they need **no token and no network**:

```bash
pytest -q
```

They verify that (1) the system persona is injected, (2) earlier turns are replayed to
the model on later turns (memory), and (3) separate threads stay isolated. An optional
live test runs only when `HUGGINGFACEHUB_API_TOKEN` is set.

## Tech stack

- **LangGraph** — `StateGraph`, `MessagesState`, `MemorySaver` checkpointer
- **langchain-huggingface** — `ChatHuggingFace` + `HuggingFaceEndpoint` (`provider="auto"`)
- **Streamlit** — chat UI
- **python-dotenv** — loads `.env`
