#!/usr/bin/env python3
"""
AgriMind AI - Intelligent Agriculture Knowledge Assistant
Streamlit Modern Frontend for Multi-Document Agriculture RAG System.

Connects directly to the existing RAG pipeline:
- Google Gemini LLM & Gemini Embeddings
- HyDE (Hypothetical Document Embeddings)
- FAISS Vector Store
- LangChain Orchestration
"""

import sys
import os
import time
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st
from app import config
from app.embeddings import get_embedding_model
from app.vectorstore import load_vectorstore, build_vectorstore
from app.ingestion import load_documents_from_directory
from app.chunking import split_documents_into_chunks
from app.rag_chain import AgricultureRAGAssistant, get_llm

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="AgriMind AI - Intelligent Agriculture Knowledge Assistant",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# Custom CSS for Premium Modern Aesthetics
# ==============================================================================
CUSTOM_CSS = """
<style>
/* Google Font Import */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #1a202c;
}

/* Background Subtle Gradient */
.stApp {
    background: linear-gradient(180deg, #f7faf8 0%, #edf4ef 100%);
}

/* Header Container */
.agri-header {
    background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 60%, #40916c 100%);
    padding: 2.2rem 2.5rem;
    border-radius: 18px;
    color: #ffffff;
    box-shadow: 0 10px 25px rgba(27, 67, 50, 0.15);
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}

.agri-header::after {
    content: "🌾";
    position: absolute;
    right: 25px;
    bottom: -15px;
    font-size: 7rem;
    opacity: 0.12;
    pointer-events: none;
}

.header-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255, 255, 255, 0.18);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.3);
    color: #d8f3dc;
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 5px 14px;
    border-radius: 20px;
    margin-bottom: 12px;
}

.header-title {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin: 0 0 6px 0;
    color: #ffffff;
    display: flex;
    align-items: center;
    gap: 12px;
}

.header-subtitle {
    font-size: 1.05rem;
    font-weight: 400;
    color: #b7e4c7;
    margin: 0 0 10px 0;
}

.header-desc {
    font-size: 0.92rem;
    color: #e8f5e9;
    max-width: 820px;
    line-height: 1.5;
    margin: 0;
}

/* Chat Message Bubbles */
.chat-bubble-user {
    background: linear-gradient(135deg, #e8f5e9 0%, #d8f3dc 100%);
    border: 1px solid #b7e4c7;
    border-radius: 16px 16px 4px 16px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 1.2rem;
    margin-left: 15%;
    box-shadow: 0 2px 8px rgba(45, 106, 79, 0.05);
    color: #1b4332;
    font-size: 0.98rem;
    line-height: 1.55;
}

.chat-bubble-ai {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #2d6a4f;
    border-radius: 16px 16px 16px 4px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.4rem;
    margin-right: 8%;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.04);
    color: #2d3748;
    font-size: 0.98rem;
    line-height: 1.65;
}

.chat-sender-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 8px;
}

.tag-user {
    color: #2d6a4f;
}

.tag-ai {
    color: #1b4332;
}

/* Sources Badge Container */
.sources-container {
    margin-top: 14px;
    padding-top: 12px;
    border-top: 1px dashed #e2e8f0;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
}

.source-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    color: #166534;
    font-size: 0.82rem;
    font-weight: 600;
    padding: 4px 11px;
    border-radius: 8px;
    font-family: 'JetBrains Mono', monospace;
}

.latency-pill {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    color: #64748b;
    font-size: 0.78rem;
    font-weight: 500;
    padding: 4px 10px;
    border-radius: 8px;
    margin-left: auto;
}

/* Welcome Hero Card */
.welcome-hero {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 20px;
    padding: 2.5rem 2.2rem;
    text-align: center;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.04);
    margin-bottom: 2rem;
}

.welcome-icon {
    font-size: 3.5rem;
    margin-bottom: 0.5rem;
    display: inline-block;
}

.welcome-title {
    font-size: 1.85rem;
    font-weight: 800;
    color: #1b4332;
    margin-bottom: 0.4rem;
}

.welcome-subtitle {
    font-size: 1.05rem;
    color: #4a5568;
    max-width: 650px;
    margin: 0 auto 1.5rem auto;
    line-height: 1.5;
}

.domains-row {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 10px;
    margin-bottom: 2rem;
}

.domain-pill {
    background: #f1f8f4;
    border: 1px solid #cce3d5;
    color: #2d6a4f;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
}

/* Example Question Cards */
.example-card-btn button {
    width: 100% !important;
    text-align: left !important;
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    padding: 14px 16px !important;
    transition: all 0.2s ease !important;
    color: #2d3748 !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03) !important;
}

.example-card-btn button:hover {
    border-color: #52b788 !important;
    background: #f7fcf9 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 16px rgba(45, 106, 79, 0.08) !important;
}

/* Sidebar Custom Styling */
section[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e8edf0 !important;
}

.sidebar-brand {
    padding: 0.5rem 0 1.2rem 0;
    border-bottom: 1px solid #edf2f7;
    margin-bottom: 1.2rem;
}

.status-badge-online {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #ecfdf5;
    border: 1px solid #a7f3d0;
    color: #065f46;
    font-size: 0.78rem;
    font-weight: 700;
    padding: 4px 10px;
    border-radius: 16px;
    margin-top: 6px;
}

.status-dot {
    width: 8px;
    height: 8px;
    background: #10b981;
    border-radius: 50%;
    box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.2);
}

.stat-metric-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 8px;
}

.stat-label {
    font-size: 0.75rem;
    color: #64748b;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.stat-value {
    font-size: 0.95rem;
    color: #1e293b;
    font-weight: 700;
}

/* Document Item */
.doc-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 7px 10px;
    background: #fcfdfd;
    border: 1px solid #edf2f7;
    border-radius: 8px;
    font-size: 0.82rem;
    color: #334155;
    margin-bottom: 5px;
    font-family: 'JetBrains Mono', monospace;
}

/* Footer Styling */
.agri-footer {
    text-align: center;
    padding: 2.5rem 1rem 1rem 1rem;
    color: #64748b;
    font-size: 0.84rem;
    border-top: 1px solid #e2e8f0;
    margin-top: 3rem;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ==============================================================================
# Backend Resource Initialization with Cache
# ==============================================================================
@st.cache_resource(show_spinner=False)
def get_cached_rag_assistant(temperature: float = 0.2):
    """
    Initializes and caches the FAISS vectorstore, Gemini Embeddings,
    and AgricultureRAGAssistant instance. Avoids reloading on each rerun.
    """
    config.validate_config()
    embeddings = get_embedding_model()
    
    # 1. Attempt loading existing FAISS index from disk
    vectorstore = load_vectorstore(config.FAISS_INDEX_DIR, embeddings)
    
    # 2. If index doesn't exist, auto-build from documents
    data_dir = config.DATA_DIR if config.DATA_DIR.exists() and any(config.DATA_DIR.iterdir()) else BASE_DIR
    docs = load_documents_from_directory(data_dir)
    
    if vectorstore is None:
        chunks = split_documents_into_chunks(
            docs, 
            chunk_size=config.CHUNK_SIZE, 
            chunk_overlap=config.CHUNK_OVERLAP
        )
        vectorstore = build_vectorstore(chunks, embeddings, config.FAISS_INDEX_DIR)

    llm = get_llm(temperature=temperature)
    assistant = AgricultureRAGAssistant(vectorstore=vectorstore, llm=llm)
    
    # Extract dynamic stats
    chunk_count = getattr(getattr(vectorstore, "index", None), "ntotal", len(docs) * 5)
    doc_names = sorted(list({d.metadata.get("filename", "doc.txt") for d in docs}))
    
    stats = {
        "num_documents": len(docs),
        "document_names": doc_names,
        "num_chunks": chunk_count,
        "embedding_model": config.GOOGLE_EMBEDDING_MODEL,
        "llm_model": config.GOOGLE_LLM_MODEL,
        "vectorstore": "FAISS",
        "retrieval": "HyDE"
    }
    
    return assistant, stats

# Initialize Assistant & Handle Configuration Gracefully
backend_ready = False
assistant_instance = None
kb_stats = {}

try:
    with st.spinner("🌾 Connecting to Agriculture Knowledge Base..."):
        assistant_instance, kb_stats = get_cached_rag_assistant()
        backend_ready = True
except ValueError as ve:
    st.error(f"⚠️ Configuration Error: {ve}")
    st.info("Please set your GEMINI_API_KEY or GOOGLE_API_KEY in the `.env` file and restart the application.")
except Exception as e:
    st.error(f"⚠️ Initialization Error: {e}")
    st.info("Check that documents are present in `data/` and run `python ingest.py` to build the index.")

# ==============================================================================
# Session State Management
# ==============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# ==============================================================================
# Sidebar UI: Knowledge Base Specs & Settings
# ==============================================================================
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <h2 style="margin: 0; color: #1b4332; font-size: 1.45rem; font-weight: 800; display: flex; align-items: center; gap: 8px;">
                🌾 AgriMind AI
            </h2>
            <p style="margin: 2px 0 6px 0; color: #4a5568; font-size: 0.84rem;">
                Intelligent Agriculture Knowledge Assistant
            </p>
            <div class="status-badge-online">
                <span class="status-dot"></span>
                Knowledge Base Online
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 1. Knowledge Base Specifications
    st.markdown("### 📚 Knowledge Base")
    
    num_docs = kb_stats.get("num_documents", 5)
    num_chunks = kb_stats.get("num_chunks", 26)
    emb_model = kb_stats.get("embedding_model", config.GOOGLE_EMBEDDING_MODEL)
    llm_model = kb_stats.get("llm_model", config.GOOGLE_LLM_MODEL)

    st.markdown(
        f"""
        <div class="stat-metric-card">
            <div class="stat-label">Documents Indexed</div>
            <div class="stat-value">{num_docs} Official Guides</div>
        </div>
        <div class="stat-metric-card">
            <div class="stat-label">Total Text Chunks</div>
            <div class="stat-value">{num_chunks} Vectorized Chunks</div>
        </div>
        <div class="stat-metric-card">
            <div class="stat-label">Vector Store</div>
            <div class="stat-value">FAISS (Local Disk / In-Memory)</div>
        </div>
        <div class="stat-metric-card">
            <div class="stat-label">Gemini Embeddings</div>
            <div class="stat-value" style="font-size: 0.85rem;">{emb_model}</div>
        </div>
        <div class="stat-metric-card">
            <div class="stat-label">Gemini LLM</div>
            <div class="stat-value" style="font-size: 0.85rem;">{llm_model}</div>
        </div>
        <div class="stat-metric-card">
            <div class="stat-label">Retrieval Technique</div>
            <div class="stat-value">HyDE + Semantic Search</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Ingested Files List
    with st.expander("📄 View Ingested Documents", expanded=False):
        doc_list = kb_stats.get("document_names", [
            "crop_production.txt",
            "government_schemes.txt",
            "irrigation_management.txt",
            "pest_and_disease.txt",
            "soil_health.txt"
        ])
        for doc in doc_list:
            st.markdown(f'<div class="doc-item">📄 {doc}</div>', unsafe_allow_html=True)

    st.divider()

    # 2. Settings & Controls
    st.markdown("### ⚙️ Settings")
    top_k_val = st.selectbox(
        "Top-K Document Chunks",
        options=[3, 5, 7, 10],
        index=1,
        help="Number of grounded document chunks retrieved from FAISS for each query."
    )

    use_hyde_val = st.checkbox(
        "Enable HyDE (Hypothetical Embeddings)",
        value=True,
        help="Synthesizes an expert passage first to bridge vocabulary mismatch with technical documents."
    )

    st.divider()

    # Clear Conversation Button
    if st.button("🗑️ Clear Conversation", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.session_state.pending_query = None
        st.rerun()

# ==============================================================================
# Main Screen Header
# ==============================================================================
st.markdown(
    """
    <div class="agri-header">
        <div class="header-badge">🌿 AI FOR SMART AGRICULTURE • MULTI-DOCUMENT RAG</div>
        <h1 class="header-title">🌾 AgriMind AI</h1>
        <div class="header-subtitle">Intelligent Agriculture Knowledge Assistant</div>
        <p class="header-desc">
            Ask questions about crop production, soil health, irrigation systems, 
            pest & disease management, and government schemes. Powered by Google Gemini, 
            FAISS vector search, and HyDE retrieval.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ==============================================================================
# Helper Function to Render Messages
# ==============================================================================
def render_message(msg):
    role = msg.get("role")
    content = msg.get("content", "")
    sources = msg.get("sources", [])
    chunks = msg.get("chunks", [])
    latency = msg.get("latency", 0.0)
    hypo_doc = msg.get("hypothetical_doc", "")
    used_model = msg.get("model", config.GOOGLE_LLM_MODEL)

    if role == "user":
        st.markdown(
            f"""
            <div class="chat-bubble-user">
                <div class="chat-sender-tag tag-user">👤 You</div>
                <div>{content}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        # AI Response
        sources_html = ""
        if sources:
            pills = "".join([f'<span class="source-pill">📄 {s}</span>' for s in sources])
            latency_badge = f'<span class="latency-pill">⏱ {latency}s</span>' if latency else ""
            sources_html = f"""
            <div class="sources-container">
                <span style="font-size: 0.8rem; font-weight: 700; color: #166534; margin-right: 4px;">📚 Sources:</span>
                {pills}
                {latency_badge}
            </div>
            """

        st.markdown(
            f"""
            <div class="chat-bubble-ai">
                <div class="chat-sender-tag tag-ai">🌾 AgriMind AI</div>
                <div>{content}</div>
                {sources_html}
            </div>
            """,
            unsafe_allow_html=True
        )

        # Retrieval Details Expander
        if chunks or hypo_doc:
            with st.expander("🔍 View Retrieval Details & Chunks", expanded=False):
                col_a, col_b = st.columns([1, 1])
                with col_a:
                    st.markdown(f"**Retrieved Chunks:** `{len(chunks)} chunks`")
                    st.markdown(f"**Active Model:** `{used_model}`")
                with col_b:
                    st.markdown(f"**Vector Database:** `FAISS`")
                    st.markdown(f"**HyDE Retrieval:** `{'Enabled' if hypo_doc else 'Standard'}`")

                if chunks:
                    st.markdown("#### 📑 Grounded Evidence Chunks:")
                    for idx, c in enumerate(chunks, 1):
                        fn = c.get("filename", "unknown.txt")
                        chunk_num = c.get("chunk_number", idx)
                        full_c = c.get("full_content", "")
                        st.markdown(
                            f"**Chunk {chunk_num}** — `📄 {fn}`\n"
                            f"> {full_c}\n\n---"
                        )

# ==============================================================================
# Welcome Screen (Shown when no conversation exists yet)
# ==============================================================================
if len(st.session_state.messages) == 0:
    st.markdown(
        """
        <div class="welcome-hero">
            <div class="welcome-icon">🌾</div>
            <div class="welcome-title">Welcome to AgriMind AI</div>
            <div class="welcome-subtitle">
                Your intelligent multi-document assistant for evidence-grounded agricultural knowledge.
                Ask any question below or click an example to begin.
            </div>
            <div class="domains-row">
                <span class="domain-pill">🌱 Crop Production</span>
                <span class="domain-pill">💧 Irrigation Management</span>
                <span class="domain-pill">🪨 Soil Health</span>
                <span class="domain-pill">🐛 Pest & Disease Management</span>
                <span class="domain-pill">🏛️ Government Schemes</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("#### 💡 Click any example question to ask immediately:")
    
    # 2x3 Grid of Clickable Cards
    row1_c1, row1_c2, row1_c3 = st.columns(3)
    row2_c1, row2_c2, row2_c3 = st.columns(3)

    examples = [
        ("🌱 Crop Production", "What factors affect crop production?", row1_c1),
        ("🪨 Soil Health", "How can soil health be improved?", row1_c2),
        ("💧 Irrigation", "What are efficient irrigation management practices?", row1_c3),
        ("🐛 Pest & Diseases", "What are common crop pests and diseases?", row2_c1),
        ("🏛️ Schemes", "What government schemes are mentioned in the knowledge base?", row2_c2),
        ("🔄 Multi-Domain", "How are soil health, irrigation and crop production related?", row2_c3),
    ]

    for category, question, col in examples:
        with col:
            st.markdown('<div class="example-card-btn">', unsafe_allow_html=True)
            if st.button(f"**{category}**\n\n{question}", key=f"btn_{question}", use_container_width=True):
                st.session_state.pending_query = question
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# Conversation History Rendering
# ==============================================================================
for msg in st.session_state.messages:
    render_message(msg)

# ==============================================================================
# Chat Input & Processing Pipeline
# ==============================================================================
user_prompt = st.chat_input("Ask anything about agriculture (e.g. soil fertility, drip irrigation, pest remedies, schemes)...")

# If an example card was clicked, prioritize it
if st.session_state.pending_query:
    active_query = st.session_state.pending_query
    st.session_state.pending_query = None
else:
    active_query = user_prompt

if active_query and active_query.strip():
    if not backend_ready or assistant_instance is None:
        st.error("⚠️ Backend is not initialized. Please verify your Google Gemini API key.")
    else:
        clean_query = active_query.strip()
        
        # 1. Add user message to history
        st.session_state.messages.append({
            "role": "user",
            "content": clean_query,
            "timestamp": time.time()
        })
        
        # Display immediately in current render
        st.markdown(
            f"""
            <div class="chat-bubble-user">
                <div class="chat-sender-tag tag-user">👤 You</div>
                <div>{clean_query}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # 2. Execute RAG Pipeline with informative loading states
        with st.spinner("🌾 AgriMind AI is searching agriculture knowledge base & synthesizing answer..."):
            try:
                result = assistant_instance.answer_query(
                    query=clean_query,
                    use_hyde=use_hyde_val,
                    top_k=int(top_k_val)
                )

                final_answer = result.get("answer", "No response generated.")
                sources = result.get("sources", [])
                chunks = result.get("chunks", [])
                latency = result.get("latency_seconds", 0.0)
                hypo_doc = result.get("hypothetical_doc", "")
                used_model = result.get("llm_model", config.GOOGLE_LLM_MODEL)

                # Store AI message
                ai_message = {
                    "role": "assistant",
                    "content": final_answer,
                    "sources": sources,
                    "chunks": chunks,
                    "latency": latency,
                    "hypothetical_doc": hypo_doc,
                    "model": used_model,
                    "timestamp": time.time()
                }
                st.session_state.messages.append(ai_message)

                # Rerun to render neatly with full expanders and styling
                st.rerun()

            except Exception as e:
                err_msg = f"An error occurred while retrieving information: {str(e)}"
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"⚠️ {err_msg}",
                    "sources": [],
                    "chunks": [],
                    "latency": 0.0,
                    "model": "error",
                    "timestamp": time.time()
                })
                st.rerun()

# ==============================================================================
# Footer
# ==============================================================================
st.markdown(
    """
    <div class="agri-footer">
        <strong>AgriMind AI</strong> • Multi-Document RAG • Google Gemini • LangChain • FAISS • HyDE<br>
        <span style="opacity: 0.8;">Built for intelligent agriculture research and farming decision assistance.</span>
    </div>
    """,
    unsafe_allow_html=True
)
