import logging
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def split_documents_into_chunks(
    documents: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 100
) -> List[Document]:
    """
    Splits documents into smaller overlapping text chunks using RecursiveCharacterTextSplitter.
    Attaches chunk metadata including chunk number, document source, and character count.
    """
    if not documents:
        logger.warning("No documents provided for chunking.")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )

    all_chunks: List[Document] = []
    
    for doc in documents:
        raw_chunks = text_splitter.split_documents([doc])
        total_doc_chunks = len(raw_chunks)
        filename = doc.metadata.get("filename", "unknown_doc")

        for idx, chunk in enumerate(raw_chunks, start=1):
            chunk.metadata.update({
                "chunk_number": idx,
                "total_chunks": total_doc_chunks,
                "chunk_id": f"{filename}_chunk_{idx}",
                "char_count": len(chunk.page_content)
            })
            all_chunks.append(chunk)

    logger.info(f"Split {len(documents)} document(s) into {len(all_chunks)} chunks (size={chunk_size}, overlap={chunk_overlap}).")
    return all_chunks
