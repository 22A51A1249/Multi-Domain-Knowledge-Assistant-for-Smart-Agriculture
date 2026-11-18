import time
import logging
from typing import Dict, Any, List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS

from app import config
from app.embeddings import get_embedding_model
from app.vectorstore import get_or_create_vectorstore
from app.prompts import RAG_PROMPT
from app.hyde import retrieve_with_hyde, generate_hypothetical_document, extract_text_content
from app.retriever import format_chunks_for_context, extract_sources_and_metadata, standard_retrieve

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FALLBACK_LLM_MODELS = ["gemini-3.1-flash-lite", "gemini-3.5-flash", "gemini-3.8-flash", "gemini-flash-latest"]

def get_llm(model_name: Optional[str] = None, temperature: Optional[float] = None) -> ChatGoogleGenerativeAI:
    """
    Initializes and returns the Google Gemini Chat model via LangChain.
    """
    config.validate_config()
    target_model = model_name or config.GOOGLE_LLM_MODEL
    temp = temperature if temperature is not None else config.TEMPERATURE
    
    logger.info(f"Initializing Gemini Chat Model: {target_model} (temperature={temp})")
    return ChatGoogleGenerativeAI(
        model=target_model,
        google_api_key=config.GOOGLE_API_KEY,
        temperature=temp
    )

class AgricultureRAGAssistant:
    """
    Production-ready Multi-Document Agriculture RAG Assistant.
    Coordinates:
    - Google Gemini LLM
    - Google Gemini Embeddings
    - FAISS Vector Store
    - HyDE (Hypothetical Document Embeddings)
    - Source Attribution & Grounded Response Generation
    """
    def __init__(
        self,
        vectorstore: Optional[FAISS] = None,
        llm: Optional[ChatGoogleGenerativeAI] = None
    ):
        config.validate_config()
        self.embeddings = get_embedding_model()
        self.vectorstore = vectorstore or get_or_create_vectorstore(embeddings=self.embeddings)
        self.llm = llm or get_llm()

    def answer_query(
        self,
        query: str,
        use_hyde: bool = True,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Executes the RAG pipeline for a given agricultural question.
        
        Steps:
        1. Query validation
        2. HyDE hypothetical document generation (if enabled)
        3. FAISS vector search using hypothetical document embedding
        4. Context formatting with chunk annotations
        5. Grounded synthesis via Gemini LLM
        6. Source metadata extraction
        """
        if not query or not query.strip():
            return {
                "answer": "Please enter a valid agricultural question.",
                "sources": [],
                "chunks": [],
                "hypothetical_doc": "",
                "latency_seconds": 0.0,
                "error": "Empty query"
            }

        start_time = time.time()
        k = top_k or config.TOP_K
        hypothetical_doc = ""

        try:
            # Step 1: Retrieval (HyDE or Standard)
            if use_hyde:
                logger.info(f"Executing HyDE retrieval for: '{query}'")
                retrieved_chunks, hypothetical_doc = retrieve_with_hyde(
                    query=query,
                    llm=self.llm,
                    vectorstore=self.vectorstore,
                    top_k=k
                )
            else:
                logger.info(f"Executing standard retrieval for: '{query}'")
                retrieved_chunks = standard_retrieve(
                    query=query,
                    vectorstore=self.vectorstore,
                    top_k=k
                )

            # Step 2: Format context for final LLM prompt
            context_text = format_chunks_for_context(retrieved_chunks)
            sources_info = extract_sources_and_metadata(retrieved_chunks)

            # Step 3: Generate strictly grounded final answer with Gemini
            prompt_input = RAG_PROMPT.format(
                context=context_text,
                question=query.strip()
            )

            logger.info("Generating final grounded answer via Gemini...")
            active_model_name = getattr(self.llm, "model", config.GOOGLE_LLM_MODEL)
            models_to_try = [active_model_name] + [m for m in FALLBACK_LLM_MODELS if m != active_model_name]
            
            response = None
            last_err = None
            used_model = active_model_name
            
            for m_name in models_to_try:
                try:
                    if getattr(self.llm, "model", "") != m_name:
                        logger.info(f"Attempting invocation with fallback model: {m_name}")
                        self.llm = get_llm(model_name=m_name)
                    response = self.llm.invoke(prompt_input)
                    used_model = m_name
                    break
                except Exception as e:
                    err_str = str(e)
                    logger.warning(f"Model {m_name} invocation error: {err_str[:120]}")
                    last_err = e
                    if any(term in err_str for term in ["429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "NOT_FOUND"]):
                        continue
                    else:
                        raise e

            if response is None:
                if last_err:
                    raise last_err
                raise RuntimeError("Failed to generate response from available Gemini models.")

            content = response.content if hasattr(response, "content") else response
            final_answer = extract_text_content(content)

            elapsed = round(time.time() - start_time, 2)
            logger.info(f"Successfully processed query in {elapsed}s using {used_model}.")

            return {
                "answer": final_answer,
                "sources": sources_info["unique_sources"],
                "chunks": sources_info["chunk_details"],
                "total_chunks": sources_info["total_retrieved_chunks"],
                "hypothetical_doc": hypothetical_doc,
                "latency_seconds": elapsed,
                "llm_model": used_model,
                "embedding_model": config.GOOGLE_EMBEDDING_MODEL,
                "use_hyde": use_hyde,
                "error": None
            }

        except Exception as e:
            logger.error(f"Error executing RAG pipeline: {e}", exc_info=True)
            elapsed = round(time.time() - start_time, 2)
            return {
                "answer": f"An error occurred while processing your request: {str(e)}",
                "sources": [],
                "chunks": [],
                "total_chunks": 0,
                "hypothetical_doc": hypothetical_doc,
                "latency_seconds": elapsed,
                "error": str(e)
            }
