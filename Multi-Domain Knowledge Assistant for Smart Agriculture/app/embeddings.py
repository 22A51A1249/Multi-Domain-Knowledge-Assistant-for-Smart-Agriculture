import logging
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_embedding_model(model_name: str = None) -> GoogleGenerativeAIEmbeddings:
    """
    Initializes and returns the Google Gemini embeddings model using LangChain.
    """
    config.validate_config()
    target_model = model_name or config.GOOGLE_EMBEDDING_MODEL
    logger.info(f"Initializing Gemini Embedding Model: {target_model}")

    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model=target_model,
            google_api_key=config.GOOGLE_API_KEY
        )
        return embeddings
    except Exception as e:
        logger.error(f"Failed to initialize Gemini Embeddings: {e}")
        raise RuntimeError(f"Error initializing Gemini Embeddings: {e}")
