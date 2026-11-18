import os
import re
import logging
from pathlib import Path
from typing import List
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """Cleans unnecessary whitespace and normalizes line breaks."""
    if not text:
        return ""
    # Replace carriage returns
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Replace multiple empty lines with double newline
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Strip leading and trailing whitespace
    return text.strip()

def load_txt_file(file_path: Path) -> List[Document]:
    """Loads a plaintext file."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        cleaned = clean_text(content)
        if not cleaned:
            logger.warning(f"File {file_path.name} is empty.")
            return []
        
        doc = Document(
            page_content=cleaned,
            metadata={
                "source": file_path.name,
                "filename": file_path.name,
                "document_type": "txt",
                "file_path": str(file_path)
            }
        )
        return [doc]
    except Exception as e:
        logger.error(f"Error loading TXT file {file_path}: {e}")
        return []

def load_pdf_file(file_path: Path) -> List[Document]:
    """Loads a PDF file if pypdf is installed."""
    docs = []
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(file_path))
        full_text = []
        for page_idx, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if page_text.strip():
                full_text.append(page_text.strip())
        
        combined_text = clean_text("\n\n".join(full_text))
        if combined_text:
            docs.append(Document(
                page_content=combined_text,
                metadata={
                    "source": file_path.name,
                    "filename": file_path.name,
                    "document_type": "pdf",
                    "file_path": str(file_path),
                    "total_pages": len(reader.pages)
                }
            ))
    except ImportError:
        logger.warning(f"pypdf not installed. Skipping PDF file: {file_path.name}")
    except Exception as e:
        logger.error(f"Error reading PDF file {file_path}: {e}")
    return docs

def load_docx_file(file_path: Path) -> List[Document]:
    """Loads a DOCX file if python-docx is installed."""
    docs = []
    try:
        import docx
        doc_obj = docx.Document(str(file_path))
        paragraphs = [p.text for p in doc_obj.paragraphs if p.text.strip()]
        combined_text = clean_text("\n\n".join(paragraphs))
        if combined_text:
            docs.append(Document(
                page_content=combined_text,
                metadata={
                    "source": file_path.name,
                    "filename": file_path.name,
                    "document_type": "docx",
                    "file_path": str(file_path)
                }
            ))
    except ImportError:
        logger.warning(f"python-docx not installed. Skipping DOCX file: {file_path.name}")
    except Exception as e:
        logger.error(f"Error reading DOCX file {file_path}: {e}")
    return docs

def load_documents_from_directory(dir_path: Path) -> List[Document]:
    """
    Loads all supported documents (.txt, .pdf, .docx, .md) from the target directory and subdirectories.
    Excludes system, script, and requirement files.
    """
    if not dir_path.exists():
        logger.error(f"Directory {dir_path} does not exist.")
        return []

    documents: List[Document] = []
    supported_extensions = {".txt", ".pdf", ".docx", ".md"}
    ignored_filenames = {
        "requirements.txt", "system prompt", "userprompt", 
        "readme.md", ".env", ".env.example", ".gitignore"
    }

    # Support both flat and nested directories
    all_files = [p for p in dir_path.rglob("*") if p.is_file()] if dir_path.is_dir() else []
    
    for file_path in all_files:
        filename_lower = file_path.name.lower()
        if filename_lower in ignored_filenames or filename_lower.startswith("."):
            continue

        ext = file_path.suffix.lower()
        if ext in supported_extensions:
            logger.info(f"Loading document: {file_path.name} ({ext})")
            if ext in {".txt", ".md"}:
                documents.extend(load_txt_file(file_path))
            elif ext == ".pdf":
                documents.extend(load_pdf_file(file_path))
            elif ext == ".docx":
                documents.extend(load_docx_file(file_path))

    logger.info(f"Loaded {len(documents)} document(s) from {dir_path}.")
    return documents
