import logging
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def format_chunks_for_context(chunks: List[Document]) -> str:
    """
    Formats retrieved real document chunks into a clean, annotated context block
    suitable for the LLM prompt. Each chunk is tagged with its filename and chunk number.
    """
    if not chunks:
        return "No relevant context found in documents."

    context_parts = []
    for idx, doc in enumerate(chunks, start=1):
        filename = doc.metadata.get("filename", "unknown_source.txt")
        chunk_num = doc.metadata.get("chunk_number", idx)
        content = doc.page_content.strip()
        context_parts.append(
            f"--- [Document: {filename} | Chunk: {chunk_num}] ---\n{content}"
        )

    return "\n\n".join(context_parts)

def extract_sources_and_metadata(chunks: List[Document]) -> Dict[str, Any]:
    """
    Extracts structured source information from retrieved chunks.
    Returns unique sources list, chunk breakdown, and summary stats.
    """
    unique_sources = sorted(list({doc.metadata.get("filename", "unknown.txt") for doc in chunks}))
    
    chunk_details = []
    for idx, doc in enumerate(chunks, start=1):
        chunk_details.append({
            "index": idx,
            "filename": doc.metadata.get("filename", "unknown.txt"),
            "chunk_number": doc.metadata.get("chunk_number", idx),
            "chunk_id": doc.metadata.get("chunk_id", f"chunk_{idx}"),
            "content_preview": doc.page_content[:150] + ("..." if len(doc.page_content) > 150 else ""),
            "full_content": doc.page_content
        })

    return {
        "unique_sources": unique_sources,
        "total_retrieved_chunks": len(chunks),
        "chunk_details": chunk_details
    }

def standard_retrieve(
    query: str,
    vectorstore: FAISS,
    top_k: int = 5
) -> List[Document]:
    """
    Standard retrieval without HyDE: embeds the raw user query directly.
    Useful for comparison and baseline testing.
    """
    return vectorstore.similarity_search(query, k=top_k)
