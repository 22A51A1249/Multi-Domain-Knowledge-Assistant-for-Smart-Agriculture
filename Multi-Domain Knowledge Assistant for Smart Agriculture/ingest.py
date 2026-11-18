#!/usr/bin/env python3
"""
Standalone Ingestion Script for Multi-Document Smart Agriculture RAG.
Loads documents from data/, splits into chunks, computes Gemini embeddings,
and builds & saves the FAISS index to vectorstore/faiss_index.
"""

import sys
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

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app import config
from app.ingestion import load_documents_from_directory
from app.chunking import split_documents_into_chunks
from app.embeddings import get_embedding_model
from app.vectorstore import build_vectorstore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("Ingest")

def run_ingestion():
    print("=" * 65)
    print("🌾 SMART AGRICULTURE MULTI-DOCUMENT INGESTION PIPELINE")
    print("=" * 65)

    # 1. Validate environment configuration
    try:
        config.validate_config()
        print(f"✔ Environment validated. LLM: {config.GOOGLE_LLM_MODEL} | Embeddings: {config.GOOGLE_EMBEDDING_MODEL}")
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        sys.exit(1)

    # 2. Check and locate data directory
    data_dir = config.DATA_DIR
    if not data_dir.exists() or not any(data_dir.iterdir()):
        # Check root as fallback
        print(f"ℹ Checking fallback directory: {BASE_DIR}")
        data_dir = BASE_DIR

    print(f"\n📂 Step 1: Loading documents from: {data_dir.resolve()}")
    documents = load_documents_from_directory(data_dir)
    if not documents:
        print("❌ No valid documents found! Supported extensions: .txt, .pdf, .docx")
        sys.exit(1)

    print(f"✔ Loaded {len(documents)} document(s):")
    for doc in documents:
        filename = doc.metadata.get("filename", "unknown")
        doc_type = doc.metadata.get("document_type", "unknown")
        char_len = len(doc.page_content)
        print(f"   • {filename} ({doc_type.upper()}) - {char_len:,} characters")

    # 3. Chunking
    print(f"\n✂ Step 2: Chunking documents (Chunk Size={config.CHUNK_SIZE}, Overlap={config.CHUNK_OVERLAP})...")
    chunks = split_documents_into_chunks(
        documents=documents,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    print(f"✔ Generated {len(chunks)} total text chunks.")

    # 4. Initialize Gemini Embeddings
    print(f"\n🧠 Step 3: Initializing Google Gemini Embeddings ({config.GOOGLE_EMBEDDING_MODEL})...")
    try:
        embeddings = get_embedding_model()
    except Exception as e:
        print(f"❌ Failed to initialize Gemini embeddings: {e}")
        sys.exit(1)

    # 5. Build and save FAISS index
    print(f"\n⚡ Step 4: Generating embeddings & building FAISS index at: {config.FAISS_INDEX_DIR.resolve()}...")
    try:
        vectorstore = build_vectorstore(
            chunks=chunks,
            embeddings=embeddings,
            save_path=config.FAISS_INDEX_DIR
        )
        print("✔ FAISS vector index built and persisted successfully!")
    except Exception as e:
        print(f"❌ Error during vectorstore creation: {e}")
        sys.exit(1)

    print("\n" + "=" * 65)
    print("🎉 INGESTION COMPLETE! Ready to run the Gradio application.")
    print("=" * 65)

if __name__ == "__main__":
    run_ingestion()
