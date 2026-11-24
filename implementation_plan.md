# AI Assistant — Professional Rewrite Plan

## Goal

Rewrite the existing AI Assistant project from a collection of flat scripts into a **production-quality, portfolio-worthy Python application** that demonstrates mastery of:

- **SOLID Principles** — Single Responsibility, Open/Closed, Dependency Inversion
- **DSA** — Efficient vector retrieval (FAISS), graph-based agent orchestration (LangGraph)
- **Error Handling** — Custom exception hierarchy, retry logic, graceful degradation
- **Clean Architecture** — Modular packages, type safety, configuration management

This project will serve as a showcase for job switching into AI/ML engineering roles.

---

## Current State Analysis

| File | Issues |
|---|---|
| `rag_agent.py` | Hardcoded Linux paths, SSL disabled globally, no error recovery, tight coupling to OpenAI |
| `other_agents.py` | Duplicate LLM initialization, no separation of concerns |
| `test.py` | Acts as main orchestrator (misleading name), state classes mixed with logic, graph built at import time |
| `ui.py` | Minimal UI, no error display, no loading states |
| `.env` | API keys exposed, missing config validation |

---

## Proposed Architecture

```
e:\AI_assistant\
├── .env                          # Secrets only (API keys)
├── .env.example                  # Template for required env vars
├── requirements.txt              # Pinned dependencies
├── README.md                     # Project documentation
├── run.py                        # Single entry point
├── documents/                    # PDF knowledge base (unchanged)
│
├── config/
│   ├── __init__.py
│   └── settings.py               # [NEW] Pydantic settings, all config in one place
│
├── core/
│   ├── __init__.py
│   ├── exceptions.py             # [NEW] Custom exception hierarchy
│   ├── llm_factory.py            # [NEW] LLM provider abstraction (DI / Open-Closed)
│   └── types.py                  # [NEW] Shared type definitions (TypedDict, enums)
│
├── agents/
│   ├── __init__.py
│   ├── base.py                   # [NEW] Abstract base agent (Interface Segregation)
│   ├── rag_agent.py              # [REWRITE] Clean RAG agent
│   ├── arxiv_agent.py            # [NEW] Extracted from other_agents.py
│   └── wiki_agent.py             # [NEW] Extracted from other_agents.py
│
├── vectorstore/
│   ├── __init__.py
│   └── faiss_store.py            # [NEW] FAISS manager with proper DSA
│
├── orchestrator/
│   ├── __init__.py
│   ├── router.py                 # [NEW] Query classification logic
│   ├── graph.py                  # [NEW] LangGraph workflow builder
│   └── synthesizer.py            # [NEW] Result synthesis logic
│
└── ui/
    ├── __init__.py
    └── app.py                    # [REWRITE] Polished Streamlit UI
```

---

## User Review Required

> [!IMPORTANT]
> **LLM Provider**: The current code uses TCS GenAI Lab (`genailab.tcs.in`) with Azure-hosted OpenAI models. The rewrite will keep this as the default but make it configurable via `.env` so you can easily switch to OpenAI direct, Azure OpenAI, or any compatible provider.

> [!WARNING]
> **Breaking Change — File Structure**: All existing `.py` files in the project root will be replaced by the new package structure. The `documents/` folder and `.env` file will be preserved.

> [!IMPORTANT]
> **SSL Verification**: The current code disables SSL (`verify=False`). The rewrite will make this configurable (`SSL_VERIFY=false` in `.env`) rather than hardcoded, with a logged warning when disabled.

---

## Proposed Changes

### 1. Configuration Layer (`config/`)

#### [NEW] [settings.py](file:///e:/AI_assistant/config/settings.py)
- Pydantic `BaseSettings` class for type-safe configuration
- Validates all required env vars at startup (fail-fast)
- Resolves paths dynamically (no more hardcoded Linux paths)
- Groups: `LLMSettings`, `VectorStoreSettings`, `AppSettings`

---

### 2. Core Layer (`core/`)

#### [NEW] [exceptions.py](file:///e:/AI_assistant/core/exceptions.py)
- Custom exception hierarchy:
  - `AIAssistantError` (base)
  - `ConfigurationError` — missing env vars, invalid config
  - `VectorStoreError` — FAISS load/save failures
  - `LLMError` — API call failures, rate limits
  - `AgentError` — agent execution failures
  - `DocumentLoadError` — PDF parsing failures
- Each exception carries structured context (source, query, retry info)

#### [NEW] [llm_factory.py](file:///e:/AI_assistant/core/llm_factory.py)
- Factory pattern for LLM instantiation (eliminates duplicate `ChatOpenAI` creation)
- Single point to configure HTTP client, timeouts, retries
- **Open/Closed**: Add new providers without modifying existing code
- **Dependency Inversion**: Agents depend on abstraction, not concrete `ChatOpenAI`

#### [NEW] [types.py](file:///e:/AI_assistant/core/types.py)
- `AgentInput`, `AgentOutput`, `Classification`, `RouterState` — all TypedDict/Pydantic models
- `AgentSource` enum (`RAG`, `ARXIV`, `WIKIPEDIA`)
- Centralizes all type definitions used across modules

---

### 3. Vector Store Layer (`vectorstore/`)

#### [NEW] [faiss_store.py](file:///e:/AI_assistant/vectorstore/faiss_store.py)
- `FAISSManager` class with proper lifecycle:
  - `load_or_create()` — idempotent initialization
  - `add_documents(paths)` — incremental indexing with deduplication
  - `search(query, k)` — retrieval with scoring
  - `save()` / `delete()` — persistence management
- Tracks processed files via a JSON manifest (not a flat `.txt` file)
- Proper chunking strategy with configurable `chunk_size` and `chunk_overlap`
- Thread-safe operations with file locking
- **DSA highlight**: Efficient approximate nearest neighbor search via FAISS IVF indexing

---

### 4. Agents Layer (`agents/`)

#### [NEW] [base.py](file:///e:/AI_assistant/agents/base.py)
- `BaseAgent` abstract class:
  - `execute(query: str) -> AgentOutput` — standard interface
  - `health_check() -> bool` — verify agent readiness
- Enforces **Interface Segregation** — each agent has a clean contract
- Built-in retry decorator with exponential backoff

#### [REWRITE] [rag_agent.py](file:///e:/AI_assistant/agents/rag_agent.py)
- Extends `BaseAgent`
- Depends on `FAISSManager` (injected, not created internally)
- Clean separation: retrieval tool → agent executor
- Configurable retrieval parameters (`k`, `score_threshold`)

#### [NEW] [arxiv_agent.py](file:///e:/AI_assistant/agents/arxiv_agent.py)
- Extends `BaseAgent`
- Extracted from `other_agents.py`
- Wraps `ArxivQueryRun` with error handling and timeout

#### [NEW] [wiki_agent.py](file:///e:/AI_assistant/agents/wiki_agent.py)
- Extends `BaseAgent`
- Extracted from `other_agents.py`
- Wraps `WikipediaQueryRun` with error handling and timeout

---

### 5. Orchestrator Layer (`orchestrator/`)

#### [NEW] [router.py](file:///e:/AI_assistant/orchestrator/router.py)
- `QueryRouter` class — uses structured output to classify queries
- Maps queries to one or more agent sources with optimized sub-questions
- Validates classification output (prevents unknown sources)
- **DSA highlight**: Graph-based routing with LangGraph's conditional edges

#### [NEW] [graph.py](file:///e:/AI_assistant/orchestrator/graph.py)
- `WorkflowBuilder` class — constructs the LangGraph `StateGraph`
- Registers agents dynamically (Open/Closed — add agents without modifying builder)
- Compiles with `InMemorySaver` for thread persistence
- Exposes `ask(query, config) -> str` as the single public API

#### [NEW] [synthesizer.py](file:///e:/AI_assistant/orchestrator/synthesizer.py)
- `ResultSynthesizer` class — merges multi-agent results
- Handles edge cases: single result, empty results, conflicting sources
- Uses a dedicated synthesis prompt with source attribution

---

### 6. UI Layer (`ui/`)

#### [REWRITE] [app.py](file:///e:/AI_assistant/ui/app.py)
- Polished Streamlit interface with:
  - Professional dark theme styling
  - Animated loading states with progress indicators
  - Error messages displayed as dismissible alerts (not raw tracebacks)
  - Source attribution badges (shows which agents contributed)
  - Expandable "Sources" section per response
  - Chat export functionality
  - Sidebar with system health indicators

---

### 7. Root Files

#### [NEW] [run.py](file:///e:/AI_assistant/run.py)
- Single entry point: `python run.py` or `streamlit run run.py`
- Validates environment before starting
- Initializes all components in correct dependency order

#### [NEW] [requirements.txt](file:///e:/AI_assistant/requirements.txt)
- Pinned versions for all dependencies

#### [NEW] [.env.example](file:///e:/AI_assistant/.env.example)
- Template showing all required/optional env vars with descriptions

#### [NEW] [README.md](file:///e:/AI_assistant/README.md)
- Professional README with architecture diagram, setup instructions, usage examples

#### [DELETE] [test.py](file:///e:/AI_assistant/test.py)
- Logic moved to `orchestrator/graph.py` and `core/types.py`

#### [DELETE] [other_agents.py](file:///e:/AI_assistant/other_agents.py)
- Split into `agents/arxiv_agent.py` and `agents/wiki_agent.py`

#### [DELETE] [ui.py](file:///e:/AI_assistant/ui.py)
- Replaced by `ui/app.py`

---

## SOLID Principles Applied

| Principle | Implementation |
|---|---|
| **Single Responsibility** | Each module has exactly one reason to change (e.g., `faiss_store.py` only manages vectors) |
| **Open/Closed** | `BaseAgent` allows new agents without modifying orchestrator; `LLMFactory` supports new providers |
| **Liskov Substitution** | All agents extend `BaseAgent` and are interchangeable in the graph |
| **Interface Segregation** | `BaseAgent` exposes only `execute()` and `health_check()` — no bloated interfaces |
| **Dependency Inversion** | Agents receive LLM/vectorstore instances via constructor injection, not internal creation |

---

## Open Questions

> [!IMPORTANT]
> 1. **LLM Provider**: Should I keep `genailab.tcs.in` as the default base URL, or switch to standard OpenAI (`api.openai.com`)? The rewrite supports both — just need to know the default.

> [!IMPORTANT]
> 2. **Resume/Portfolio Documents**: Should the `documents/` folder continue to hold ML textbook PDFs, or would you like to also add your resume and job descriptions for a more job-search-focused RAG? I can add a "resume analyzer" agent that compares your skills against job postings.

> [!IMPORTANT]
> 3. **Additional Job-Switch Features**: Would you like me to add any of these features?
>    - **Mock Interview Agent** — generates technical interview questions based on your resume and target role
>    - **Job Description Analyzer** — extracts key skills from JDs and maps them to your experience
>    - **Cover Letter Generator** — drafts tailored cover letters using RAG over your resume
>    - **DSA Practice Helper** — generates and explains coding problems for interview prep

---

## Verification Plan

### Automated Tests
1. **Config validation**: Run with missing env vars → expect `ConfigurationError`
2. **Vector store**: Create, add docs, search, verify results with `pytest`
3. **Agent execution**: Mock LLM responses, verify each agent returns `AgentOutput`
4. **Graph execution**: End-to-end test with a sample query through the full pipeline
5. **Error handling**: Simulate API failures, verify graceful degradation

### Manual Verification
1. Run `streamlit run run.py` and verify the UI renders correctly
2. Ask sample queries across all three sources (RAG, Arxiv, Wikipedia)
3. Verify source attribution appears in responses
4. Test error scenarios (e.g., disconnect network, invalid API key)
