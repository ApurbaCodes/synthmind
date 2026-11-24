

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import datetime

import streamlit as st

# Suppress harmless FAISS AVX512 fallback warning
logging.getLogger("faiss.loader").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SynthMind: GenAI Multi-Agent Assistant",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS - The Aesthetic Overhaul
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ---- Import Google Fonts ---- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400&display=swap');

/* ---- Global Typography & Theme ---- */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #e2e8f0;
}

/* Background tweaks for deeper dark mode */
.stApp, [data-testid="stAppViewContainer"] {
    background-color: #0b0f19 !important;
    background-image: radial-gradient(circle at 50% 0%, #1e293b 0%, transparent 70%) !important;
}

/* ---- Custom Scrollbar ---- */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: #334155;
    border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover {
    background: #475569;
}

/* ---- Header Styling ---- */
.main-header {
    background: linear-gradient(135deg, #a855f7 0%, #3b82f6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.75rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
    letter-spacing: -1px;
    text-align: center;
}

.sub-header {
    color: #94a3b8;
    font-size: 1rem;
    font-weight: 400;
    text-align: center;
    margin-bottom: 3rem;
    letter-spacing: 0.5px;
}

/* ---- Chat Messages (Glassmorphism) ---- */
.stChatMessage {
    background: rgba(30, 41, 59, 0.4) !important;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 1rem !important;
    padding: 1.5rem !important;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    animation: slideUp 0.4s cubic-bezier(0.16, 1, 0.3, 1);
}

@keyframes slideUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ---- Source Badges (Neon/Glow Effect) ---- */
.source-badge {
    display: inline-flex;
    align-items: center;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 8px;
    margin-top: 8px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    transition: all 0.2s ease;
}
.source-badge:hover {
    transform: translateY(-1px);
    filter: brightness(1.2);
}

.badge-rag        { background: rgba(59, 130, 246, 0.1); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); box-shadow: 0 0 10px rgba(59, 130, 246, 0.1); }
.badge-arxiv      { background: rgba(236, 72, 153, 0.1); color: #f472b6; border: 1px solid rgba(236, 72, 153, 0.3); box-shadow: 0 0 10px rgba(236, 72, 153, 0.1); }
.badge-wikipedia  { background: rgba(16, 185, 129, 0.1); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); box-shadow: 0 0 10px rgba(16, 185, 129, 0.1); }
.badge-tavily     { background: rgba(245, 158, 11, 0.1); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); box-shadow: 0 0 10px rgba(245, 158, 11, 0.1); }

/* ---- Health Indicators ---- */
.health-ok   { color: #34d399; font-weight: 500; }
.health-fail { color: #f87171; font-weight: 500; }

/* ---- Sidebar Styling ---- */
[data-testid="stSidebar"] {
    background-color: rgba(15, 23, 42, 0.8) !important;
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}

/* ---- Chat Input Container tweaks ---- */
[data-testid="stChatInput"] {
    border-radius: 1rem !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    background: rgba(30, 41, 59, 0.7) !important;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1) !important;
}

/* ---- Buttons ---- */
div[data-testid="stButton"] > button {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    color: #e2e8f0;
    border-radius: 0.5rem;
    transition: all 0.2s ease;
}
div[data-testid="stButton"] > button:hover {
    background: rgba(255, 255, 255, 0.1);
    border-color: rgba(255, 255, 255, 0.2);
    color: #fff;
    transform: scale(1.02);
}

/* Special styling for the export button if we want to target it (using Streamlit types) */
div[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, #a855f7 0%, #3b82f6 100%);
    border: none;
    color: white;
}
div[data-testid="stDownloadButton"] > button:hover {
    opacity: 0.9;
    transform: scale(1.02);
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
def _init_session() -> None:
    defaults = {
        "messages": [],
        "thread_id": str(uuid.uuid4()),
        "workflow": None,
        "init_error": None,
        "sources_history": [],   # list of lists of source names per message
        "uploaded_files": [],    # list of file paths uploaded in this session
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ---------------------------------------------------------------------------
# Initialize the workflow (cached in session state)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _create_workflow():
    """Build the complete workflow pipeline — cached across reruns."""
    try:
        from config.settings import get_settings
        from core.llm_factory import LLMFactory
        from vectorstore.faiss_store import FAISSManager
        from agents.rag_agent import RAGAgent
        from agents.arxiv_agent import ArxivAgent
        from agents.wiki_agent import WikiAgent
        from agents.tavily_agent import TavilyAgent
        from orchestrator.router import QueryRouter
        from orchestrator.synthesizer import ResultSynthesizer
        from orchestrator.graph import WorkflowBuilder

        settings = get_settings()
        factory = LLMFactory(settings)
        chat_llm = factory.create_chat_llm()
        embeddings = factory.create_embeddings()

        # Vector store
        faiss_mgr = FAISSManager(embeddings, settings).load_or_create()

        # Agents
        rag = RAGAgent(faiss_mgr, chat_llm, k=settings.vectorstore.search_k)
        arxiv = ArxivAgent(chat_llm)
        wiki = WikiAgent(chat_llm)
        tavily_agent = TavilyAgent(chat_llm)

        # Router + Synthesizer
        router = QueryRouter(chat_llm)
        synthesizer = ResultSynthesizer(chat_llm)

        # Build workflow
        workflow = (
            WorkflowBuilder(router, synthesizer)
            .register_agent(rag)
            .register_agent(arxiv)
            .register_agent(wiki)
            .register_agent(tavily_agent)
            .build()
        )
        workflow.faiss_mgr = faiss_mgr  # Attach manager to access in UI
        return workflow, None
    except Exception as exc:
        logger.exception("Failed to initialize workflow")
        return None, str(exc)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def _render_sidebar(workflow) -> None:
    with st.sidebar:
        st.markdown("### 🎛️ Control Panel")
        st.markdown("<br>", unsafe_allow_html=True) # Adding a bit of breathing room

        # 1. File Uploader
        st.markdown("#### 📄 Upload Document")
        uploaded_file = st.file_uploader("Upload a file to the knowledge base", type=["pdf", "txt", "md"], label_visibility="collapsed")
        if uploaded_file:
            from config.settings import get_settings
            settings = get_settings()
            docs_dir = settings.vectorstore.documents_dir
            os.makedirs(docs_dir, exist_ok=True)
            
            file_path = docs_dir / uploaded_file.name
            file_path_str = str(file_path)
            
            if file_path_str not in st.session_state.get("uploaded_files", []):
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.session_state.uploaded_files.append(file_path_str)
                
                # Optionally add to workflow if initialized
                if workflow and hasattr(workflow, "faiss_mgr"):
                    try:
                        workflow.faiss_mgr.add_documents([file_path])
                    except Exception as e:
                        logger.error(f"Failed to add document to vector store: {e}")
                        
                st.success(f"Uploaded: {uploaded_file.name}")
        
        st.divider()

        # 2. Health report (System Status)
        if workflow is not None:
            st.markdown("#### System Status")
            health = workflow.health_report()
            for name, ok in health.items():
                icon = "🟢" if ok else "🔴"
                css = "health-ok" if ok else "health-fail"
                st.markdown(
                    f'{icon} &nbsp;<span class="{css}">{name}</span>',
                    unsafe_allow_html=True,
                )
            st.divider()

        # 3. Session ID and Actions
        thread_id = st.session_state.get("thread_id", "unknown")
        st.markdown(f"**Session ID:** `{thread_id[:8]}`")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear", use_container_width=True, key="clear_chat"):
                # Delete uploaded files
                for file_path in st.session_state.get("uploaded_files", []):
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                    except Exception as e:
                        logger.error(f"Failed to delete uploaded file {file_path}: {e}")
                
                if workflow and hasattr(workflow, "faiss_mgr"):
                    workflow.faiss_mgr.clear()
                
                st.session_state.messages = []
                st.session_state.sources_history = []
                st.session_state.uploaded_files = []
                st.session_state.thread_id = str(uuid.uuid4())
                st.rerun()
        with col2:
            _export_chat()

        st.divider()

        st.caption(
            f"Messages: {len(st.session_state.messages)} &nbsp;•&nbsp; "
            f"Model: Gemma 4"
        )


def _export_chat() -> None:
    """Export chat history as JSON."""
    if not st.session_state.messages:
        # Replaced sidebar warning with a disabled-looking button for better UX
        st.button("📥 Export", disabled=True, use_container_width=True)
        return
    
    export_data = {
        "thread_id": st.session_state.thread_id,
        "exported_at": datetime.now().isoformat(),
        "messages": st.session_state.messages,
    }
    st.download_button(
        label="📥 Export",
        data=json.dumps(export_data, indent=2),
        file_name=f"chat_export_{st.session_state.thread_id[:8]}.json",
        mime="application/json",
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Source badges
# ---------------------------------------------------------------------------
def _render_source_badges(sources: list[str]) -> None:
    if not sources:
        return
    badges_html = ""
    for src in sources:
        css_cls = {
            "RAG": "badge-rag",
            "Arxiv": "badge-arxiv",
            "Wikipedia": "badge-wikipedia",
            "Tavily": "badge-tavily",
        }.get(src, "badge-rag")
        badges_html += f'<span class="source-badge {css_cls}">{src}</span>'
    st.markdown(badges_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Streaming text generator
# ---------------------------------------------------------------------------
def _stream_text(text: str):
    """Yield words one at a time with a small delay for streaming effect."""
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02) # Slightly faster generation feels more responsive


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
def main() -> None:
    _init_session()

    # Initialize workflow
    with st.spinner("Initializing AI pipeline..."):
        workflow, init_error = _create_workflow()

    if init_error:
        st.error(
            f"⚠️ **Initialization Error**\n\n{init_error}\n\n"
            "Please check your `.env` file and ensure all dependencies are installed."
        )
        st.stop()

    st.session_state.workflow = workflow
    _render_sidebar(workflow)

    # Header
    st.markdown('<div class="main-header">SynthMind</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">'
        'Powered by RAG • arXiv • Wikipedia • Tavily'
        '</div>',
        unsafe_allow_html=True,
    )

    # Display chat history
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Show source badges for assistant messages
            if msg["role"] == "assistant" and i // 2 < len(st.session_state.sources_history):
                _render_source_badges(st.session_state.sources_history[i // 2])

    # Chat input
    if prompt := st.chat_input("Ask about NLP, latest papers, or AI history..."):
        # User message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Assistant response
        with st.chat_message("assistant"):
            with st.spinner("Consulting knowledge bases..."):
                config = {
                    "configurable": {"thread_id": st.session_state.get("thread_id", "default")}
                }
                # Fallback for UI testing if backend isn't running
                if workflow:
                    response = workflow.ask(prompt, config)
                else:
                    time.sleep(1)
                    response = "System is running in UI-only mode for aesthetic testing."

            # Normalize response to plain text
            try:
                from core.types import extract_text_content
                if isinstance(response, list):
                    response = extract_text_content(
                        response[-1].content if hasattr(response[-1], "content") else response[-1]
                    )
                else:
                    response = extract_text_content(response)
            except ImportError:
                # Safe fallback if core.types isn't available
                response = str(response)

            # Stream the response
            st.write_stream(_stream_text(response))

            # Detect sources
            detected_sources = []
            for src in ["RAG", "Arxiv", "Wikipedia", "Tavily"]:
                if f"[{src}]" in response or src.lower() in response.lower():
                    detected_sources.append(src)
            if not detected_sources:
                detected_sources = ["RAG"]

            _render_source_badges(detected_sources)
            st.session_state.sources_history.append(detected_sources)
            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )


if __name__ == "__main__":
    main()