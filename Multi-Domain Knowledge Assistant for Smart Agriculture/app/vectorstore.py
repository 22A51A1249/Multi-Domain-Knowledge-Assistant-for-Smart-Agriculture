import logging
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from app import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_vectorstore(
    chunks: List[Document],
    embeddings: Embeddings,
    save_path: Optional[Path] = None
) -> FAISS:
    """
    Creates a FAISS vector store from document chunks and saves it locally if save_path is provided.
    """
    if not chunks:
        raise ValueError("Cannot build vector store with empty document chunks list.")

    logger.info(f"Creating FAISS vector index for {len(chunks)} chunks...")
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(save_path))
        logger.info(f"FAISS vector store successfully saved to: {save_path}")

    return vectorstore

def load_vectorstore(
    load_path: Path,
    embeddings: Embeddings
) -> Optional[FAISS]:
    """
    Loads an existing FAISS vector store from the local filesystem.
    """
    load_path = Path(load_path)
    index_file = load_path / "index.faiss"
    pkl_file = load_path / "index.pkl"

    if not (index_file.exists() and pkl_file.exists()):
        logger.warning(f"No existing FAISS index found at {load_path}")
        return None

    try:
        logger.info(f"Loading FAISS vector store from: {load_path}")
        vectorstore = FAISS.load_local(
            folder_path=str(load_path),
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info("FAISS vector store loaded successfully.")
        return vectorstore
    except Exception as e:
        logger.error(f"Error loading FAISS vector store from {load_path}: {e}")
        return None

def get_or_create_vectorstore(
    embeddings: Embeddings,
    chunks: Optional[List[Document]] = None,
    save_path: Path = config.FAISS_INDEX_DIR,
    force_rebuild: bool = False
) -> FAISS:
    """
    Retrieves the existing FAISS index or builds a new one from chunks.
    """
    if not force_rebuild:
        existing_vs = load_vectorstore(save_path, embeddings)
        if existing_vs is not None:
            return existing_vs

    if chunks is None or len(chunks) == 0:
        raise ValueError("Index not found and no chunks provided to build vector store.")

    return build_vectorstore(chunks, embeddings, save_path)
