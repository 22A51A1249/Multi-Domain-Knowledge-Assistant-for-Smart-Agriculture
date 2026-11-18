#!/usr/bin/env python3
"""
Gradio Web Interface for Multi-Document Agriculture RAG Assistant.
Features:
- Chatbot UI with Conversational History
- Grounded Agricultural Question Answering
- HyDE (Hypothetical Document Embeddings) Transparency View
- Source Citations & Detailed Retrieved Chunks Inspection
- Dynamic Knowledge Base & System Statistics
"""

import sys
import os
import logging
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Auto-delegate to virtual environment if executed with an unconfigured Python
venv_python = BASE_DIR / "venv" / "Scripts" / "python.exe"
if venv_python.exists() and Path(sys.executable).resolve() != venv_python.resolve():
    import subprocess
    sys.exit(subprocess.call([str(venv_python)] + sys.argv))

import gradio as gr
from app import config
from app.rag_chain import AgricultureRAGAssistant, get_llm
from app.embeddings import get_embedding_model
from app.vectorstore import load_vectorstore, build_vectorstore
from app.ingestion import load_documents_from_directory
from app.chunking import split_documents_into_chunks

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GradioApp")

# Global Assistant Instance
assistant_instance = None
kb_stats = {
    "num_documents": 0,
    "num_chunks": 0,
    "document_names": [],
    "index_status": "Not Loaded"
}

def initialize_knowledge_base():
    """Initializes embeddings, vectorstore, and assistant instance."""
    global assistant_instance, kb_stats
    try:
        config.validate_config()
        embeddings = get_embedding_model()
        
        # Check if vectorstore exists
        vectorstore = load_vectorstore(config.FAISS_INDEX_DIR, embeddings)
        
        # If not, build it from documents
        data_dir = config.DATA_DIR if config.DATA_DIR.exists() and any(config.DATA_DIR.iterdir()) else BASE_DIR
        docs = load_documents_from_directory(data_dir)
        kb_stats["num_documents"] = len(docs)
        kb_stats["document_names"] = [d.metadata.get("filename", "unknown") for d in docs]

        if vectorstore is None:
            logger.info("FAISS vectorstore not found. Auto-building index...")
            chunks = split_documents_into_chunks(docs, chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
            kb_stats["num_chunks"] = len(chunks)
            vectorstore = build_vectorstore(chunks, embeddings, config.FAISS_INDEX_DIR)
            kb_stats["index_status"] = "Built & Persisted"
        else:
            # Estimate chunk count
            chunks = split_documents_into_chunks(docs, chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
            kb_stats["num_chunks"] = len(chunks)
            kb_stats["index_status"] = "Loaded from Disk"

        llm = get_llm()
        assistant_instance = AgricultureRAGAssistant(vectorstore=vectorstore, llm=llm)
        logger.info("Agriculture RAG Assistant initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing Knowledge Base: {e}", exc_info=True)
        kb_stats["index_status"] = f"Error: {str(e)}"

# Initialize at startup
initialize_knowledge_base()

def format_sources_markdown(sources, latency):
    if not sources:
        return "*No specific source documents identified.*"
    
    md = "### 📚 Verified Sources Used:\n"
    for s in sources:
        md += f"- **`{s}`**\n"
    md += f"\n*⏱ Response Latency: {latency} seconds*\n"
    return md

def format_chunks_markdown(chunks):
    if not chunks:
        return "*No chunks retrieved.*"
    
    md = f"### 📑 Retrieved Evidence Chunks ({len(chunks)} Chunks):\n\n"
    for c in chunks:
        md += f"#### 🔹 Chunk {c.get('chunk_number')} — Source: `{c.get('filename')}`\n"
        md += f"**ID**: `{c.get('chunk_id')}`\n\n"
        md += f"> {c.get('full_content')}\n\n"
        md += "---\n"
    return md

def format_hyde_markdown(hypo_doc, use_hyde):
    if not use_hyde:
        return "*HyDE is currently disabled. Standard direct query embedding was used.*"
    if not hypo_doc:
        return "*No hypothetical document was generated.*"
    
    md = "### 🔬 HyDE: Hypothetical Document Generation\n"
    md += "> **Note on HyDE**: *The hypothetical document below was synthesized by Gemini to match technical agricultural terminology for vector matching. It is purely used to query the vector database and is NEVER treated as factual evidence.*\n\n"
    md += f"```text\n{hypo_doc}\n```"
    return md

def get_stats_markdown():
    return f"""### 📊 Knowledge Base & System Specifications
- **Total Documents**: `{kb_stats['num_documents']}`
- **Ingested Files**: {", ".join([f"`{name}`" for name in kb_stats['document_names']]) if kb_stats['document_names'] else 'None'}
- **Total Text Chunks**: `{kb_stats['num_chunks']}`
- **Vector Database**: `FAISS (In-Memory / Local Disk)`
- **Vector Index Status**: `{kb_stats['index_status']}`
- **Embedding Model**: `{config.GOOGLE_EMBEDDING_MODEL}`
- **LLM Model**: `{config.GOOGLE_LLM_MODEL}`
- **Chunk Size / Overlap**: `{config.CHUNK_SIZE}` chars / `{config.CHUNK_OVERLAP}` chars
- **Default Top-K Chunks**: `{config.TOP_K}`
- **Temperature**: `{config.TEMPERATURE}`
"""

def chat_response(user_query, history, use_hyde, top_k):
    global assistant_instance
    if not user_query or not user_query.strip():
        return history, "", "*No query entered.*", "*No chunks retrieved.*", "*No HyDE data.*", get_stats_markdown()

    if assistant_instance is None:
        initialize_knowledge_base()
        if assistant_instance is None:
            err_msg = f"Knowledge base could not be initialized: {kb_stats.get('index_status')}"
            new_history = history + [{"role": "user", "content": user_query}, {"role": "assistant", "content": err_msg}]
            return new_history, "", err_msg, "", "", get_stats_markdown()

    # Query the RAG assistant
    result = assistant_instance.answer_query(
        query=user_query,
        use_hyde=use_hyde,
        top_k=int(top_k)
    )

    answer = result.get("answer", "No answer generated.")
    sources = result.get("sources", [])
    chunks = result.get("chunks", [])
    hypo_doc = result.get("hypothetical_doc", "")
    latency = result.get("latency_seconds", 0.0)

    # Format history for Gradio messages format
    new_history = list(history) if history else []
    new_history.append({"role": "user", "content": user_query})
    new_history.append({"role": "assistant", "content": answer})

    sources_md = format_sources_markdown(sources, latency)
    chunks_md = format_chunks_markdown(chunks)
    hyde_md = format_hyde_markdown(hypo_doc, use_hyde)
    stats_md = get_stats_markdown()

    return new_history, "", sources_md, chunks_md, hyde_md, stats_md

def clear_conversation():
    return [], "", "*Waiting for query...*", "*No chunks retrieved.*", "*No HyDE data.*", get_stats_markdown()

# Build the Gradio UI
custom_css = """
#main-container {max-width: 1400px; margin: 0 auto; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;}
.gradio-header {text-align: center; margin-bottom: 1.5rem;}
.header-badge {display: inline-block; padding: 4px 12px; background: #e8f5e9; color: #2e7d32; border-radius: 12px; font-weight: 600; font-size: 0.85rem; margin-bottom: 8px;}
.source-box {background: #f8fafc; border-left: 4px solid #10b981; padding: 12px; border-radius: 6px;}
"""

with gr.Blocks(title="AI Agriculture Multi-Document RAG Assistant") as demo:
    with gr.Column(elem_id="main-container"):
        # Header Section
        gr.HTML(
            """
            <div class="gradio-header">
                <span class="header-badge">🌿 AI FOR SMART AGRICULTURE • MULTI-DOCUMENT RAG</span>
                <h1 style="color: #1b5e20; margin: 0.4rem 0;">🌾 AI Agriculture Multi-Document RAG Assistant</h1>
                <p style="color: #4b5563; font-size: 1.05rem; max-width: 800px; margin: 0 auto;">
                    An intelligent research and advisory assistant powered by <b>Google Gemini</b>, <b>LangChain</b>, 
                    <b>HyDE (Hypothetical Document Embeddings)</b>, and <b>FAISS</b> vector retrieval.
                    Ask queries across crop production, soil health, irrigation systems, pest management, and government schemes.
                </p>
            </div>
            """
        )

        with gr.Row():
            # Left Column: Chatbot and Controls
            with gr.Column(scale=5):
                chatbot = gr.Chatbot(
                    label="💬 Conversation Window",
                    height=520
                )

                with gr.Row():
                    query_input = gr.Textbox(
                        placeholder="e.g. How does soil health affect crop productivity and irrigation?",
                        label="Ask an Agriculture Question",
                        lines=2,
                        scale=8
                    )
                    with gr.Column(scale=2):
                        submit_btn = gr.Button("🚀 Submit", variant="primary", size="lg")
                        clear_btn = gr.Button("🗑 Clear Chat", variant="secondary", size="sm")

                with gr.Accordion("⚙ Advanced Retrieval Settings", open=False):
                    with gr.Row():
                        use_hyde_toggle = gr.Checkbox(
                            label="Enable HyDE (Hypothetical Document Embeddings)",
                            value=True,
                            info="Synthesizes hypothetical text before vector search to bridge vocabulary mismatch."
                        )
                        top_k_slider = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=5,
                            step=1,
                            label="Top-K Retrieved Chunks",
                            info="Number of relevant chunks to retrieve from FAISS."
                        )

                # Example Questions
                gr.Markdown("#### 💡 Example Multi-Domain Questions:")
                examples = [
                    "What are the important factors affecting crop production?",
                    "How does soil health affect crop productivity?",
                    "What irrigation practices are useful for efficient water management?",
                    "What are common crop diseases and how can they be managed?",
                    "What government schemes are mentioned in the documents?",
                    "How are soil health, irrigation and crop production related?",
                    "Which documents contain information about pest management?",
                    "Compare information about irrigation and soil health from the available documents."
                ]
                gr.Examples(
                    examples=examples,
                    inputs=query_input,
                    label="Click any question to prefill:"
                )

            # Right Column: Transparency and Metadata Tabs
            with gr.Column(scale=4):
                with gr.Tabs():
                    with gr.Tab("📚 Sources & Attribution"):
                        sources_display = gr.Markdown(
                            value=format_sources_markdown([], 0.0),
                            elem_classes=["source-box"]
                        )
                    
                    with gr.Tab("📑 Retrieved Chunks"):
                        chunks_display = gr.Markdown(
                            value="*Submit a query to inspect the authentic retrieved document chunks used for grounding.*"
                        )
                    
                    with gr.Tab("🔬 HyDE Inspection"):
                        hyde_display = gr.Markdown(
                            value="*Submit a query to view the hypothetical document generated by Gemini during retrieval.*"
                        )

                    with gr.Tab("📊 System Statistics"):
                        stats_display = gr.Markdown(value=get_stats_markdown())

        # Interaction wiring
        submit_btn.click(
            fn=chat_response,
            inputs=[query_input, chatbot, use_hyde_toggle, top_k_slider],
            outputs=[chatbot, query_input, sources_display, chunks_display, hyde_display, stats_display]
        )

        query_input.submit(
            fn=chat_response,
            inputs=[query_input, chatbot, use_hyde_toggle, top_k_slider],
            outputs=[chatbot, query_input, sources_display, chunks_display, hyde_display, stats_display]
        )

        clear_btn.click(
            fn=clear_conversation,
            inputs=[],
            outputs=[chatbot, query_input, sources_display, chunks_display, hyde_display, stats_display]
        )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1", 
        server_port=7860, 
        share=False,
        theme=gr.themes.Soft(primary_hue="emerald", secondary_hue="green"),
        css=custom_css
    )
