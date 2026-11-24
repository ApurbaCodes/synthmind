

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
# Custom CSS - Material 3 (M3) Design System
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ---- Import Google Fonts & Material Symbols ---- */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

:root {
    /* M3 Dark Baseline Color Palette */
    --md-surface: #141218;
    --md-surface-dim: #110f14;
    --md-surface-container-low: #1d1b20;
    --md-surface-container: #211f26;
    --md-surface-container-high: #2b2930;
    --md-surface-container-highest: #36343b;
    --md-on-surface: #e6e0e9;
    --md-on-surface-variant: #cac4d0;
    --md-outline: #79747e;
    --md-outline-variant: rgba(121, 116, 126, 0.2);

    /* M3 Primary (Electric Violet / Lavender) */
    --md-primary: #d0bcff;
    --md-on-primary: #381e72;
    --md-primary-container: #3b2a59;
    --md-on-primary-container: #eaddff;

    /* M3 Secondary & Tertiary */
    --md-secondary: #ccc2dc;
    --md-secondary-container: #332d41;
    --md-tertiary: #a8eff0;
    --md-tertiary-container: #004f58;

    /* M3 Shapes & Motion */
    --md-shape-sm: 8px;
    --md-shape-md: 16px;
    --md-shape-lg: 24px;
    --md-shape-full: 9999px;
    --md-easing: cubic-bezier(0.2, 0.0, 0.0, 1.0);
}

/* ---- Global Typography & Background ---- */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--md-on-surface);
}

.stApp, [data-testid="stAppViewContainer"] {
    background-color: var(--md-surface) !important;
    background-image: 
        radial-gradient(ellipse 80% 50% at 50% -20%, rgba(79, 55, 139, 0.18), transparent 70%),
        radial-gradient(circle at 10% 20%, rgba(0, 79, 88, 0.08), transparent 40%) !important;
}

/* ---- Smooth Scrollbar ---- */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: var(--md-surface-container-highest);
    border-radius: var(--md-shape-full);
}
::-webkit-scrollbar-thumb:hover {
    background: var(--md-outline);
}

/* ---- Material 3 Header Hero ---- */
.m3-hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    background: var(--md-surface-container-high);
    border: 1px solid var(--md-outline-variant);
    border-radius: var(--md-shape-full);
    font-size: 0.75rem;
    font-weight: 700;
    color: var(--md-primary);
    letter-spacing: 0.8px;
    text-transform: uppercase;
    margin-bottom: 12px;
}

.m3-hero-badge .pulse-dot {
    width: 6px;
    height: 6px;
    background: #34d399;
    border-radius: 50%;
    box-shadow: 0 0 8px #34d399;
}

.m3-main-title {
    font-size: 2.85rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.03em;
    line-height: 1.1;
    margin-bottom: 8px;
    text-align: center;
    background: linear-gradient(135deg, #ffffff 40%, var(--md-primary) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.m3-sub-title {
    color: var(--md-on-surface-variant);
    font-size: 1rem;
    font-weight: 400;
    text-align: center;
    max-width: 580px;
    margin: 0 auto 2.2rem auto;
    line-height: 1.5;
}

.m3-chip-row {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 2.5rem;
}

.m3-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: var(--md-shape-sm);
    font-size: 0.78rem;
    font-weight: 500;
    background: var(--md-surface-container);
    border: 1px solid var(--md-outline-variant);
    color: var(--md-on-surface);
}

/* ---- Material 3 Chat Cards ---- */
[data-testid="stChatMessage"] {
    background: var(--md-surface-container) !important;
    border: 1px solid var(--md-outline-variant) !important;
    border-radius: var(--md-shape-lg) !important;
    padding: 1.25rem 1.5rem !important;
    margin-bottom: 1.25rem;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25) !important;
    transition: transform 0.2s var(--md-easing), box-shadow 0.2s var(--md-easing);
}

[data-testid="stChatMessage"]:hover {
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35) !important;
}

/* Target User Message bubble for distinct M3 Container styling */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: var(--md-primary-container) !important;
    border: 1px solid rgba(208, 188, 255, 0.2) !important;
    border-radius: 24px 24px 6px 24px !important;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    border-radius: 24px 24px 24px 6px !important;
}

/* ---- M3 Filter & Assist Chips (Source Badges) ---- */
.m3-source-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: var(--md-shape-sm);
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.3px;
    margin-right: 6px;
    margin-top: 10px;
    text-transform: uppercase;
    transition: all 0.2s var(--md-easing);
}
.m3-source-badge:hover {
    transform: translateY(-1px);
    filter: brightness(1.15);
}

.m3-badge-rag {
    background: rgba(59, 130, 246, 0.15);
    color: #93c5fd;
    border: 1px solid rgba(147, 197, 253, 0.3);
}
.m3-badge-arxiv {
    background: rgba(244, 114, 182, 0.15);
    color: #f472b6;
    border: 1px solid rgba(244, 114, 182, 0.3);
}
.m3-badge-wikipedia {
    background: rgba(52, 211, 153, 0.15);
    color: #6ee7b7;
    border: 1px solid rgba(110, 231, 183, 0.3);
}
.m3-badge-tavily {
    background: rgba(251, 191, 36, 0.15);
    color: #fcd34d;
    border: 1px solid rgba(252, 211, 77, 0.3);
}

/* ---- M3 Sidebar / Navigation Drawer ---- */
[data-testid="stSidebar"] {
    background-color: var(--md-surface-container-low) !important;
    border-right: 1px solid var(--md-outline-variant) !important;
}

.m3-sidebar-card {
    background: var(--md-surface-container);
    border: 1px solid var(--md-outline-variant);
    border-radius: var(--md-shape-md);
    padding: 14px 16px;
    margin-bottom: 12px;
}

.m3-card-title {
    font-size: 0.75rem;
    font-weight: 700;
    color: var(--md-on-surface-variant);
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 8px;
}

/* Status indicator rows in sidebar */
.m3-status-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 6px 0;
    font-size: 0.84rem;
    color: var(--md-on-surface);
    border-bottom: 1px solid rgba(255, 255, 255, 0.04);
}
.m3-status-item:last-child {
    border-bottom: none;
}
.m3-status-pill {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-size: 0.72rem;
    padding: 2px 8px;
    border-radius: var(--md-shape-full);
}
.m3-status-ok {
    background: rgba(52, 211, 153, 0.12);
    color: #34d399;
}
.m3-status-err {
    background: rgba(248, 113, 113, 0.12);
    color: #f87171;
}

/* ---- M3 Input Field ---- */
[data-testid="stChatInput"] {
    border-radius: var(--md-shape-full) !important;
    border: 1px solid var(--md-outline-variant) !important;
    background: var(--md-surface-container-high) !important;
    padding: 4px 8px !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.28) !important;
    transition: all 0.2s var(--md-easing);
}

[data-testid="stChatInput"]:focus-within {
    border-color: var(--md-primary) !important;
    box-shadow: 0 0 0 2px rgba(208, 188, 255, 0.25), 0 6px 16px rgba(0, 0, 0, 0.35) !important;
}

/* ---- M3 Buttons ---- */
div[data-testid="stButton"] > button {
    border-radius: var(--md-shape-full) !important;
    background: var(--md-surface-container-high) !important;
    border: 1px solid var(--md-outline-variant) !important;
    color: var(--md-on-surface) !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.2px !important;
    padding: 8px 18px !important;
    transition: all 0.2s var(--md-easing) !important;
}

div[data-testid="stButton"] > button:hover {
    background: var(--md-surface-container-highest) !important;
    border-color: var(--md-outline) !important;
    color: #ffffff !important;
    transform: translateY(-1px);
    box-shadow: 0 2px 6px rgba(0,0,0,0.2) !important;
}

/* Export Button (M3 Filled Primary Button) */
div[data-testid="stDownloadButton"] > button {
    border-radius: var(--md-shape-full) !important;
    background: var(--md-primary) !important;
    color: var(--md-on-primary) !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    border: none !important;
    padding: 8px 18px !important;
    transition: all 0.2s var(--md-easing) !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25) !important;
}

div[data-testid="stDownloadButton"] > button:hover {
    filter: brightness(1.1);
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(208, 188, 255, 0.3) !important;
}

/* ---- M3 File Uploader Container ---- */
[data-testid="stFileUploader"] {
    background: var(--md-surface-container) !important;
    border: 1px dashed var(--md-outline) !important;
    border-radius: var(--md-shape-md) !important;
    padding: 10px !important;
}

/* Divider Styling */
hr {
    border-color: var(--md-outline-variant) !important;
    margin: 1.25rem 0 !important;
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
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 1.5rem; margin-top: 0.5rem;">
                <div style="background: var(--md-primary-container); color: var(--md-primary); width: 38px; height: 38px; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 1.2rem; font-weight: bold; border: 1px solid rgba(208,188,255,0.25);">✨</div>
                <div>
                    <div style="font-weight: 700; font-size: 1.05rem; color: #ffffff; letter-spacing: -0.2px;">SynthMind</div>
                    <div style="font-size: 0.72rem; color: var(--md-on-surface-variant); text-transform: uppercase; letter-spacing: 0.5px;">Control Panel</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 1. File Uploader Card
        st.markdown('<div class="m3-card-title">📚 Knowledge Base</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload reference documents (PDF, TXT, MD)",
            type=["pdf", "txt", "md"],
            label_visibility="collapsed",
        )
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
                        
                st.success(f"Indexed: {uploaded_file.name}")
        
        st.divider()

        # 2. System Status / Agent Health
        if workflow is not None:
            st.markdown('<div class="m3-card-title">⚡ Multi-Agent Health</div>', unsafe_allow_html=True)
            health = workflow.health_report()
            status_html = '<div class="m3-sidebar-card">'
            for name, ok in health.items():
                status_cls = "m3-status-ok" if ok else "m3-status-err"
                status_text = "Ready" if ok else "Offline"
                status_icon = "●" if ok else "▲"
                status_html += f"""
                <div class="m3-status-item">
                    <span>{name} Agent</span>
                    <span class="m3-status-pill {status_cls}">{status_icon} {status_text}</span>
                </div>
                """
            status_html += '</div>'
            st.markdown(status_html, unsafe_allow_html=True)
            st.divider()

        # 3. Session & Actions
        st.markdown('<div class="m3-card-title">🛠️ Session & Storage</div>', unsafe_allow_html=True)
        thread_id = st.session_state.get("thread_id", "unknown")
        st.markdown(
            f"""
            <div style="display:flex; justify-content:space-between; align-items:center; background: var(--md-surface-container); padding: 8px 12px; border-radius: 8px; border: 1px solid var(--md-outline-variant); margin-bottom: 12px; font-size: 0.8rem;">
                <span style="color: var(--md-on-surface-variant);">Session Key</span>
                <code style="background: transparent; color: var(--md-primary); font-family: 'JetBrains Mono', monospace;">{thread_id[:8]}</code>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Reset", use_container_width=True, key="clear_chat"):
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

        st.markdown(
            f"""
            <div style="display:flex; justify-content:space-between; font-size:0.75rem; color: var(--md-on-surface-variant);">
                <span>Messages: {len(st.session_state.messages)}</span>
                <span>Model: Gemma 4</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _export_chat() -> None:
    """Export chat history as JSON."""
    if not st.session_state.messages:
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
# Source badges (Material 3 Assist Chips)
# ---------------------------------------------------------------------------
def _render_source_badges(sources: list[str]) -> None:
    if not sources:
        return
    badges_html = '<div style="margin-top: 10px; display: flex; flex-wrap: wrap; gap: 6px;">'
    icon_map = {
        "RAG": "📄",
        "Arxiv": "📑",
        "Wikipedia": "🌐",
        "Tavily": "🔍",
    }
    for src in sources:
        css_cls = {
            "RAG": "m3-badge-rag",
            "Arxiv": "m3-badge-arxiv",
            "Wikipedia": "m3-badge-wikipedia",
            "Tavily": "m3-badge-tavily",
        }.get(src, "m3-badge-rag")
        icon = icon_map.get(src, "💡")
        badges_html += f'<span class="m3-source-badge {css_cls}"><span>{icon}</span> {src}</span>'
    badges_html += '</div>'
    st.markdown(badges_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Streaming text generator
# ---------------------------------------------------------------------------
def _stream_text(text: str):
    """Yield words one at a time with a small delay for streaming effect."""
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.018)


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

    # Header Hero (Material 3)
    st.markdown(
        """
        <div style="text-align: center; margin-top: 0.5rem;">
            <div class="m3-hero-badge">
                <span class="pulse-dot"></span>
                <span>Material 3 • Multi-Agent System</span>
            </div>
            <div class="m3-main-title">SynthMind</div>
            <div class="m3-sub-title">
                Autonomous multi-agent research assistant routing queries across textbooks, arXiv preprints, Wikipedia, and live search.
            </div>
            <div class="m3-chip-row">
                <span class="m3-chip"><span class="m3-badge-rag" style="display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:2px;"></span> RAG Textbooks</span>
                <span class="m3-chip"><span class="m3-badge-arxiv" style="display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:2px;"></span> arXiv Papers</span>
                <span class="m3-chip"><span class="m3-badge-wikipedia" style="display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:2px;"></span> Wikipedia</span>
                <span class="m3-chip"><span class="m3-badge-tavily" style="display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:2px;"></span> Tavily Search</span>
            </div>
        </div>
        """,
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