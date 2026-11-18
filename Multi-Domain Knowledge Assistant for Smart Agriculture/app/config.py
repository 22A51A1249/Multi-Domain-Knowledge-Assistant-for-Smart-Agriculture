import os
from pathlib import Path
from dotenv import load_dotenv

# Base Directory of Project
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env file securely
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)
load_dotenv()  # Fallback to current working directory

# Google Gemini API Key (supports either GEMINI_API_KEY or GOOGLE_API_KEY from environment)
GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
GOOGLE_API_KEY = GEMINI_API_KEY

# Gemini Models Configuration
GOOGLE_LLM_MODEL = os.getenv("GOOGLE_LLM_MODEL", "gemini-3.1-flash-lite").strip()
GOOGLE_EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001").strip()

# Chunking Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# Retrieval Configuration
TOP_K = int(os.getenv("TOP_K", "5"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

# Paths
DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
FAISS_INDEX_DIR = VECTORSTORE_DIR / "faiss_index"

def validate_config() -> bool:
    """
    Validates essential environment variables and configurations.
    Returns True if valid, raises ValueError if essential configuration is missing.
    """
    if not GOOGLE_API_KEY:
        raise ValueError(
            "Gemini API key is missing! Please set your valid API key (GEMINI_API_KEY or GOOGLE_API_KEY) in the .env file."
        )
    return True
