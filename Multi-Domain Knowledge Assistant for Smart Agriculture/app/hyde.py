import logging
from typing import List, Tuple, Any
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.vectorstores import FAISS
from app.prompts import HYDE_PROMPT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_text_content(content: Any) -> str:
    """Safely extracts string content from diverse LLM response representations."""
    if not content:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if "text" in item and item["text"]:
                    parts.append(str(item["text"]))
            elif hasattr(item, "text"):
                parts.append(str(item.text))
            else:
                parts.append(str(item))
        return "".join(parts).strip()
    if hasattr(content, "text"):
        return str(content.text).strip()
    return str(content).strip()

def generate_hypothetical_document(query: str, llm: BaseChatModel) -> str:
    """
    Step 1 of HyDE:
    Generates a hypothetical technical document/answer using Gemini LLM.
    This creates an expanded text dense with relevant agricultural terminology
    which bridges the semantic gap between a brief query and dense technical passages.
    If generation encounters an API limitation or error, gracefully falls back to the original query.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty for HyDE document generation.")

    logger.info(f"Generating hypothetical document for query: '{query}'")
    prompt_input = HYDE_PROMPT.format(question=query.strip())
    
    try:
        response = llm.invoke(prompt_input)
        content = response.content if hasattr(response, "content") else response
        hypo_text = extract_text_content(content)
        if not hypo_text:
            return query.strip()
        logger.info(f"Generated hypothetical document ({len(hypo_text)} chars).")
        return hypo_text
    except Exception as e:
        logger.warning(f"HyDE document generation issue: {e}. Falling back to direct query embedding.")
        return query.strip()

def retrieve_with_hyde(
    query: str,
    llm: BaseChatModel,
    vectorstore: FAISS,
    top_k: int = 5
) -> Tuple[List[Document], str]:
    """
    Complete HyDE Retrieval Pipeline:
    1. Receives user query.
    2. Generates a hypothetical agricultural document via Gemini LLM.
    3. Embeds the hypothetical document using Gemini Embeddings under the hood in FAISS.
    4. Performs similarity search in the FAISS vector store.
    5. Returns top-k authentic, real document chunks and the hypothetical document text.
    
    NOTE: The hypothetical document is ONLY used for embedding/retrieval.
          It is NEVER passed as factual truth to the final answer generator.
    """
    # 1. Generate hypothetical document
    hypothetical_doc = generate_hypothetical_document(query=query, llm=llm)

    # 2. Search FAISS vector store using the hypothetical document text
    logger.info(f"Performing FAISS similarity search with hypothetical document (top_k={top_k})...")
    retrieved_docs = vectorstore.similarity_search(hypothetical_doc, k=top_k)

    logger.info(f"Retrieved {len(retrieved_docs)} authentic chunks via HyDE.")
    return retrieved_docs, hypothetical_doc
