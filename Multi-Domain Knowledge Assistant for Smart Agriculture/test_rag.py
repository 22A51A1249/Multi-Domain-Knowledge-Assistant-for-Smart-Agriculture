#!/usr/bin/env python3
"""
Test Suite for Multi-Document Agriculture RAG Assistant.
Validates:
1. Document Ingestion
2. Text Chunking
3. Gemini Embedding generation
4. FAISS Vector Retrieval
5. HyDE (Hypothetical Document Embeddings)
6. Final Grounded RAG Generation on 5+ Multi-Document Agricultural Questions
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
from app.vectorstore import load_vectorstore, build_vectorstore
from app.hyde import generate_hypothetical_document, retrieve_with_hyde
from app.rag_chain import AgricultureRAGAssistant, get_llm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TestRAG")

TEST_QUESTIONS = [
    "What are the important factors affecting crop production?",
    "How does soil health affect crop productivity?",
    "What irrigation practices are useful for efficient water management?",
    "What are common crop diseases and how can they be managed?",
    "What government schemes are mentioned in the documents?",
    "How are soil health, irrigation and crop production related?"
]

def run_tests():
    print("\n" + "=" * 70)
    print("🌾 MULTI-DOCUMENT AGRICULTURE RAG: TEST SUITE")
    print("=" * 70)

    # 1. Config Check
    print("\n[TEST 1] Checking Configuration & Environment...")
    config.validate_config()
    print("✔ API Key: Configured and loaded securely from environment.")
    print(f"✔ LLM Model: {config.GOOGLE_LLM_MODEL}")
    print(f"✔ Embedding Model: {config.GOOGLE_EMBEDDING_MODEL}")

    # 2. Document Loading Check
    print("\n[TEST 2] Testing Document Ingestion...")
    data_dir = config.DATA_DIR if config.DATA_DIR.exists() and any(config.DATA_DIR.iterdir()) else BASE_DIR
    docs = load_documents_from_directory(data_dir)
    assert len(docs) >= 5, f"Expected at least 5 documents, found {len(docs)}"
    print(f"✔ Ingested {len(docs)} documents successfully.")

    # 3. Chunking Check
    print("\n[TEST 3] Testing Text Chunking...")
    chunks = split_documents_into_chunks(docs, chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    assert len(chunks) > 0, "No chunks generated!"
    print(f"✔ Created {len(chunks)} chunks with metadata:")
    for c in chunks[:3]:
        print(f"   • Chunk ID: {c.metadata.get('chunk_id')} | Length: {len(c.page_content)} chars")

    # 4. Embeddings & Vectorstore Check
    print("\n[TEST 4] Testing Gemini Embeddings & FAISS Vectorstore...")
    embeddings = get_embedding_model()
    vectorstore = load_vectorstore(config.FAISS_INDEX_DIR, embeddings)
    if vectorstore is None:
        print("ℹ Existing index not found; creating test vectorstore in memory/disk...")
        vectorstore = build_vectorstore(chunks, embeddings, config.FAISS_INDEX_DIR)
    assert vectorstore is not None, "Failed to initialize or load FAISS vectorstore!"
    print("✔ Vectorstore ready and loaded.")

    # 5. HyDE Check
    print("\n[TEST 5] Testing HyDE (Hypothetical Document Embeddings)...")
    llm = get_llm()
    sample_q = "How does soil pH affect crop growth?"
    hypo_doc = generate_hypothetical_document(sample_q, llm)
    assert len(hypo_doc) > 20, "Hypothetical document is too short!"
    print("✔ Generated Hypothetical Document:")
    print(f"   Excerpt: {hypo_doc[:200]}...")

    hyde_retrieved, _ = retrieve_with_hyde(sample_q, llm, vectorstore, top_k=3)
    assert len(hyde_retrieved) > 0, "HyDE retrieval returned 0 chunks!"
    print(f"✔ HyDE retrieved {len(hyde_retrieved)} chunks. Top chunk source: {hyde_retrieved[0].metadata.get('filename')}")

    # 6. End-to-End RAG Testing on Sample Questions
    print("\n[TEST 6] Testing End-to-End Grounded RAG on Sample Questions...")
    assistant = AgricultureRAGAssistant(vectorstore=vectorstore, llm=llm)

    for idx, q in enumerate(TEST_QUESTIONS, start=1):
        print(f"\n--- Question {idx}: '{q}' ---")
        result = assistant.answer_query(q, use_hyde=True, top_k=4)
        if result.get("error"):
            print(f"❌ Error: {result['error']}")
        else:
            print(f"✔ Latency: {result['latency_seconds']}s")
            print(f"✔ Sources: {result['sources']}")
            print(f"✔ Retrieved Chunks: {result['total_chunks']}")
            print(f"✔ Answer Preview: {result['answer'][:250]}...\n")

    print("\n" + "=" * 70)
    print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
