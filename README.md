<div align="center">

# ✨ SynthMind — GenAI Multi-Agent Research Assistant

**A production-quality, LangGraph-orchestrated multi-agent AI system that routes research queries across local RAG, arXiv, Wikipedia, and live web search — then synthesizes coherent, cited answers.**

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph-FF6B6B?style=flat-square)](https://github.com/langchain-ai/langgraph)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?style=flat-square&logo=chainlink)](https://langchain.com)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![FAISS](https://img.shields.io/badge/VectorStore-FAISS-4A90E2?style=flat-square)](https://faiss.ai)
[![Google AI](https://img.shields.io/badge/LLM-Google%20AI%20Studio-4285F4?style=flat-square&logo=google&logoColor=white)](https://aistudio.google.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

</div>

---

## 🎯 What It Does

SynthMind answers complex research questions by **automatically selecting and combining multiple knowledge sources**. Upload your own PDFs for private RAG retrieval; let the system fall back to arXiv preprints, Wikipedia, or live Tavily web search when local knowledge is insufficient. Every answer is synthesized by an LLM and attributed to its source(s).

This project demonstrates **real-world AI engineering** skills including multi-agent orchestration, retrieval-augmented generation, clean software architecture, and production-quality error handling.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                   Streamlit UI  (ui/app.py)                   │
│    Glassmorphism dark theme · source badges · chat export     │
└───────────────────────────┬──────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────┐
│                    Orchestrator Layer                          │
│                                                               │
│  ┌──────────────┐    ┌─────────────────────┐                 │
│  │  QueryRouter │───▶│  LangGraph          │                 │
│  │  (classify)  │    │  StateGraph (DAG)   │                 │
│  └──────────────┘    └──────────┬──────────┘                 │
│                                 │  conditional fan-out        │
│            ┌────────────────────┼─────────────────┐          │
│            ▼                   ▼                   ▼          │
│     ┌─────────────┐   ┌──────────────┐   ┌────────────────┐  │
│     │  RAG Agent  │   │ Arxiv Agent  │   │  Wiki / Tavily │  │
│     │  (FAISS)    │   │  (live API)  │   │  Agents        │  │
│     └──────┬──────┘   └──────┬───────┘   └───────┬────────┘  │
│            └────────────────▼─────────────────────┘          │
│                       ┌────────────┐                          │
│                       │ Synthesizer│  ← merges all results    │
│                       └────────────┘                          │
└──────────────────────────────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────┐
│                      Core Layer                               │
│   Settings (Pydantic) · LLMFactory · Types · Exceptions       │
└──────────────────────────────────────────────────────────────┘
```

**Data flow:**
1. User submits a query in the Streamlit UI.
2. `QueryRouter` classifies the query using structured LLM output.
3. `StateGraph` fans out to the relevant agent(s) in parallel via `add_conditional_edges`.
4. Each agent retrieves information from its knowledge source.
5. `ResultSynthesizer` merges all results with citations and returns a final answer.
6. The UI streams the response word-by-word and displays color-coded source badges.

---

## ✨ Key Features

| Feature | Detail |
|---|---|
| 🔀 **LangGraph Orchestration** | `StateGraph` with `add_conditional_edges` for runtime-determined multi-agent fan-out |
| 📚 **RAG Pipeline** | FAISS-backed ANN search over user-uploaded PDFs with deduplication |
| 📰 **arXiv Integration** | Real-time academic paper search and summarization |
| 🌐 **Wikipedia + Tavily** | General knowledge and live web search fallback |
| 🔄 **Multi-Source Synthesis** | LLM merges answers from all active agents with source attribution |
| ♻️ **Exponential Backoff** | Retry decorator on all agent calls to handle transient API failures |
| 💉 **Dependency Injection** | Agents receive LLM/vectorstore instances via constructor — fully testable |
| 🧪 **Per-Agent Health Checks** | Sidebar health dashboard shows live status of every agent |
| 📥 **Chat Export** | Download full conversation history as JSON |
| 🎨 **Polished Dark UI** | Glassmorphism chat cards, neon source badges, animated transitions |

---

## 🧠 Engineering Highlights

### SOLID Principles

| Principle | Implementation |
|---|---|
| **Single Responsibility** | Each module owns exactly one concern — `faiss_store.py` manages vectors; `router.py` classifies queries |
| **Open / Closed** | Add new agents by subclassing `BaseAgent` and calling `.register_agent()` — zero changes to the orchestrator |
| **Liskov Substitution** | All agents extend `BaseAgent`; they are interchangeable in the graph without altering behavior |
| **Interface Segregation** | `BaseAgent` exposes only `execute()` and `health_check()` — no bloated interfaces |
| **Dependency Inversion** | Agents receive LLM/FAISS instances via constructors; the core never imports concrete implementations |

### DSA Concepts Applied

- **FAISS IVF ANN Search** — O(√n) average retrieval vs. O(n) brute-force, enabling sub-second semantic search over large document sets.
- **DAG-based Workflow** — LangGraph `StateGraph` forms a Directed Acyclic Graph; conditional edges allow parallel fan-out determined at runtime.
- **Exponential Backoff** — Retry decorator doubles delay per attempt (`delay = base_delay × 2^(attempt-1)`) to prevent thundering-herd effects on rate-limited APIs.
- **Deduplication** — FAISS manager tracks ingested document hashes to prevent redundant embedding on re-upload.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- A [Google AI Studio](https://aistudio.google.com/) API key (free tier works)
- _(Optional)_ A [Tavily](https://tavily.com/) API key for live web search

### 1. Clone & Install

```bash
git clone https://github.com/ApurbaCodes/synthmind.git
cd synthmind/AI_assistant
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Open .env and set your GOOGLE_API_KEY
```

### 3. Add Documents _(optional)_

Drop any PDF files into the `documents/` directory. They are automatically chunked, embedded, and indexed into FAISS on first run.

### 4. Launch

```bash
streamlit run run.py
```

Navigate to **http://localhost:8501**.

---

## 📁 Project Structure

```
AI_assistant/
├── .env.example              # Config template (never commit .env)
├── requirements.txt          # Pinned dependencies
├── run.py                    # Single entry point
├── documents/                # Drop PDFs here for RAG indexing
│
├── config/
│   └── settings.py           # Pydantic BaseSettings — fail-fast validation
│
├── core/
│   ├── exceptions.py         # Custom exception hierarchy (5 typed errors)
│   ├── llm_factory.py        # LLM + embeddings factory (Google AI Studio)
│   └── types.py              # Shared TypedDicts, enums, Pydantic models
│
├── agents/
│   ├── base.py               # Abstract BaseAgent + exponential-backoff retry
│   ├── rag_agent.py          # FAISS semantic retrieval over uploaded PDFs
│   ├── arxiv_agent.py        # Live arXiv paper search & summarization
│   ├── wiki_agent.py         # Wikipedia knowledge lookup
│   └── tavily_agent.py       # Tavily real-time web search
│
├── vectorstore/
│   └── faiss_store.py        # FAISS lifecycle: load, ingest, dedup, search
│
├── orchestrator/
│   ├── router.py             # LLM-powered query classifier (structured output)
│   ├── graph.py              # LangGraph WorkflowBuilder (DAG construction)
│   └── synthesizer.py        # Multi-source result merger with citations
│
└── ui/
    └── app.py                # Streamlit UI — dark theme, badges, streaming, export
```

---

## ⚙️ Configuration

All settings are managed via `.env` — no hardcoded secrets anywhere.

| Variable | Required | Default | Description |
|---|---|---|---|
| `GOOGLE_API_KEY` | ✅ | — | Google AI Studio API key |
| `LLM_CHAT_MODEL` | ❌ | `gemma-4-31b-it` | Chat model identifier |
| `LLM_EMBEDDING_MODEL` | ❌ | `gemini-embedding-2-preview` | Embedding model |
| `TAVILY_API_KEY` | ❌ | — | For live web search (gracefully disabled if absent) |
| `LOG_LEVEL` | ❌ | `INFO` | Logging verbosity |

---

## 🛡️ Error Handling

A typed exception hierarchy ensures every failure carries structured context for debugging:

```
AIAssistantError (base)
├── ConfigurationError    — missing or invalid environment variables
├── VectorStoreError      — FAISS index I/O failures
├── LLMError              — API timeouts, quota errors, rate limits
├── AgentError            — agent execution failures (source, query, retry info)
└── DocumentLoadError     — PDF parsing and chunking failures
```

Every `AgentError` stores `agent_name`, `query`, and retry metadata so failures are fully traceable.

---

## 🔭 Extending the System

Adding a new knowledge source takes **three steps** — no changes to the orchestrator:

```python
# 1. Subclass BaseAgent
class MyNewAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "MySource"

    def execute(self, query: str) -> AgentOutput:
        ...

# 2. Register with the builder
builder.register_agent(MyNewAgent(llm))

# 3. Done — the router and synthesizer handle the rest automatically
```

---

## 🗺️ Roadmap

- [ ] Streaming LLM responses via Server-Sent Events
- [ ] Redis-backed LangGraph checkpointer for persistent multi-turn memory
- [ ] Docker Compose deployment (FastAPI backend + Streamlit frontend)
- [ ] OpenAI / Anthropic / local Ollama LLM provider toggle
- [ ] `pytest` suite with mocked agents and integration graph tests

---

## 📄 License

Released under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/my-feature`.
3. Commit your changes with clear messages.
4. Open a pull request — all PRs should include relevant tests.

---

<div align="center">

**Built to demonstrate mastery of LLM systems, multi-agent orchestration, and clean Python architecture.**

*If you found this useful, please ⭐ the repo — it helps others discover it!*

</div>
