# 🌾 AI-Powered Multi-Document Agriculture RAG Assistant
### *Hypothetical Document Embeddings (HyDE) + Google Gemini + LangChain + FAISS + Gradio*

An end-to-end, production-grade **Multi-Document Retrieval-Augmented Generation (RAG)** system designed specifically for the agricultural domain. It enables farmers, agronomists, extension officers, and agricultural students to perform grounded queries across heterogeneous agricultural documents (crop management, soil science, irrigation protocols, pest/pathogen control, and government subsidy schemes).

---

## 🏗 System Architecture

```
                 ┌───────────────────────────────────────────────────────────┐
                 │                   AGRICULTURE DOCUMENTS                   │
                 │ 5 Core TXT Files (Crop, Soil, Irrigation, Pest, Schemes)  │
                 └─────────────────────────────┬─────────────────────────────┘
                                               │
                                               ▼
                                 ┌───────────────────────────┐
                                 │ Document Ingestion Module │
                                 │     (app/ingestion.py)    │
                                 └─────────────┬─────────────┘
                                               │
                                               ▼
                                 ┌───────────────────────────┐
                                 │  Recursive Text Splitter  │
                                 │ (Chunk: 800, Overlap: 100)│
                                 └─────────────┬─────────────┘
                                               │
                                               ▼
                                 ┌───────────────────────────┐
                                 │ Google Gemini Embeddings  │
                                 │ (models/text-embedding-004)
                                 └─────────────┬─────────────┘
                                               │
                                               ▼
                                 ┌───────────────────────────┐
                                 │   FAISS Vector Database   │
                                 │  (vectorstore/faiss_index)│
                                 └─────────────┬─────────────┘
                                               │
═══════════════════════════════════════════════╪═══════════════════════════════════════════════
                           RUNTIME QUERY RETRIEVAL PIPELINE
═══════════════════════════════════════════════╪═══════════════════════════════════════════════
                                               │
                 ┌─────────────────────────────┴─────────────────────────────┐
                 │                                                           │
        [Direct Retrieval]                                           [HyDE Retrieval]
                 │                                                           │
                 │                                             ┌─────────────▼─────────────┐
                 │                                             │   User Query & Question   │
                 │                                             └─────────────┬─────────────┘
                 │                                                           │
                 │                                             ┌─────────────▼─────────────┐
                 │                                             │   Gemini LLM (HyDE Step)  │
                 │                                             │ Generates Hypothetical Doc│
                 │                                             └─────────────┬─────────────┘
                 │                                                           │
                 │                                             ┌─────────────▼─────────────┐
                 │                                             │  Embed Hypothetical Doc   │
                 │                                             │(Gemini Embedding Vector)  │
                 │                                             └─────────────┬─────────────┘
                 │                                                           │
                 └─────────────────────────────┬─────────────────────────────┘
                                               │
                                               ▼
                                 ┌───────────────────────────┐
                                 │   FAISS Similarity Search │
                                 │     (Top-K Chunks: 5)     │
                                 └─────────────┬─────────────┘
                                               │
                                               ▼
                                 ┌───────────────────────────┐
                                 │ Context & Metadata Filter │
                                 │ (Tags: filename, chunk_id)│
                                 └─────────────┬─────────────┘
                                               │
                                               ▼
                                 ┌───────────────────────────┐
                                 │  Gemini LLM Generation    │
                                 │ (Strict Grounding Prompt) │
                                 └─────────────┬─────────────┘
                                               │
                                               ▼
                                 ┌───────────────────────────┐
                                 │ Grounded Agriculture Ans  │
                                 │ + Verified Citations      │
                                 └─────────────┬─────────────┘
                                               │
                                               ▼
                                 ┌───────────────────────────┐
                                 │    Gradio Web Dashboard   │
                                 └───────────────────────────┘
```

---

## 🌟 Key Features

1. **True HyDE (Hypothetical Document Embeddings) Integration**:
   - Short user questions often suffer from a semantic mismatch with dense agricultural textbooks.
   - The assistant asks Gemini to hallucinate a technical textbook passage first.
   - Vector search is conducted on the *hypothetical passage's embedding*, matching rich agricultural vocabulary.
   - The hypothetical passage is strictly isolated and **never presented as factual evidence**.
2. **Multi-Document Synthesis**:
   - Answers questions that span multiple domains (e.g., how soil health affects irrigation and crop yield).
3. **Strict Grounding & Hallucination Prevention**:
   - The system prompt forbids fabricating schemes, chemical dosages, or agricultural practices not found in the retrieved context.
   - If the corpus lacks the answer, the assistant states clearly: *"I could not find sufficient information about this in the available documents."*
4. **Source Attribution & Transparency**:
   - Every answer displays verified source files and chunk numbers (no fake page numbers).
   - Dedicated inspection tabs for retrieved chunks and the HyDE text.
5. **Persistent Vector Database**:
   - FAISS index is saved locally in `vectorstore/faiss_index` and loaded rapidly without re-embedding on every run.
6. **Production Gradio Dashboard**:
   - Real-time chatbot with history, sample question buttons, HyDE toggles, and live system metrics.

---

## 📁 Project Structure

```
├── data/
│   ├── crop_production.txt       # Husbandry, seed rates, phenology, harvest
│   ├── government_schemes.txt    # PM-KISAN, KCC, PMFBY, PM-KUSUM, PKVY
│   ├── irrigation_management.txt # CRI stage, drip WUE, sprinklers, farm ponds
│   ├── pest_and_disease.txt      # IPM, bio-controls, EIL thresholds, PHI/MRL
│   └── soil_health.txt           # pH, EC, SOC, NPK metabolism, INM
│
├── app/
│   ├── __init__.py               # Package metadata
│   ├── config.py                 # Centralized configuration & environment loader
│   ├── ingestion.py              # Ingestion pipeline (.txt, .pdf, .docx)
│   ├── chunking.py               # Recursive text chunking with metadata
│   ├── embeddings.py             # Google Gemini Embeddings initialization
│   ├── vectorstore.py            # FAISS index persistence and retrieval
│   ├── hyde.py                   # HyDE generation & semantic search pipeline
│   ├── retriever.py              # Context formatting and source extraction
│   ├── rag_chain.py              # End-to-end RAG orchestrator class
│   └── prompts.py                # Grounded prompt templates & HyDE prompts
│
├── vectorstore/
│   └── faiss_index/              # Local persisted FAISS index files
│
├── .env                          # Secure API keys & configuration (ignored by git)
├── .env.example                  # Template configuration file
├── .gitignore                    # Prevents leaking keys, venv, and cache
├── requirements.txt              # Production dependency specifications
├── ingest.py                     # Standalone CLI script to build FAISS index
├── test_rag.py                   # Automated end-to-end test suite
├── app.py                        # Gradio interactive web interface
└── README.md                     # Comprehensive documentation
```

---

## ⚙ Installation & Setup

### 1. Prerequisites
- **Python 3.10+** (Python 3.12 recommended)
- A **Google Gemini API Key** ([Google AI Studio](https://aistudio.google.com/))

### 2. Clone and Create Virtual Environment
```bash
# Windows
py -3.12 -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory (or copy from `.env.example`):
```env
# Google Gemini API Key (set either GEMINI_API_KEY or GOOGLE_API_KEY)
GEMINI_API_KEY=your_api_key_here
GOOGLE_API_KEY=your_api_key_here
GOOGLE_LLM_MODEL=gemini-3.5-flash
GOOGLE_EMBEDDING_MODEL=models/gemini-embedding-001
CHUNK_SIZE=800
CHUNK_OVERLAP=100
TOP_K=5
```

---

## 🚀 Running the Project

### Step 1: Run Ingestion & Build Vector Database
```bash
python ingest.py
```
*Output:*
- Ingests all 5 documents from `data/`.
- Splits them into overlapping chunks.
- Computes Gemini embeddings.
- Saves the persisted FAISS vector index to `vectorstore/faiss_index/`.

### Step 2: Run Automated Tests
```bash
python test_rag.py
```
*Validates ingestion, chunking, embeddings, HyDE retrieval, and runs 6 multi-domain agricultural questions.*

### Step 3: Launch the Gradio Web Interface
```bash
python app.py
```
Open your browser at: **`http://127.0.0.1:7860`**

---

## 💡 Example Questions to Try

1. **Crop Production**:
   > *"What are the critical growth stages of crops where moisture stress causes severe yield loss?"*
2. **Soil Health**:
   > *"What is the optimal soil pH range, and how should acidic vs saline soils be treated?"*
3. **Multi-Domain Synthesis (Soil + Irrigation + Crops)**:
   > *"How do soil physical properties affect irrigation requirements and crop root penetration?"*
4. **Pest Management**:
   > *"Explain the Integrated Pest Management (IPM) hierarchy and biological control agents mentioned in the documents."*
5. **Government Subsidies & Schemes**:
   > *"What financial assistance and insurance coverage does PMFBY provide for Kharif and Rabi crops?"*
6. **Cross-Document Scheme & Pests**:
   > *"Which government scheme supports organic input procurement and biopesticides?"*

---

## 🔍 Deep-Dive: How HyDE Works

```
Standard RAG:
[User Query: "rice blast cure"] ──(Embed)──> [Vector Q] ──(Search FAISS)──> [Chunks]
Problem: The short query lacks technical keywords like "Magnaporthe oryzae", "fungicides", "lesions".

HyDE (Hypothetical Document Embeddings):
[User Query: "rice blast cure"]
         │
         ▼
[Gemini LLM]: Generates hypothetical text:
"Rice blast, caused by the fungal pathogen Magnaporthe oryzae, produces diamond-shaped spindle lesions. Management includes seed treatment with Carbendazim, balanced nitrogen application, and resistant cultivars..."
         │
         ▼
[Embed Hypothetical Text] ──> [Vector Dense in Domain Terms]
         │
         ▼
[FAISS Search]: Matches exact technical chunks with high semantic similarity!
```

---

## 🎓 10 Technical Interview Questions & Answers

#### Q1: What is RAG, and why is it needed for this smart agriculture system?
**Answer:** Retrieval-Augmented Generation (RAG) is an architectural framework where an LLM is paired with an external knowledge retrieval mechanism. Without RAG, general LLMs may hallucinate agricultural facts, recommend outdated chemical dosages, or make up government subsidy rates. RAG restricts the LLM to verify and ground its answer strictly in vetted agricultural documents.

#### Q2: How does HyDE (Hypothetical Document Embeddings) improve agricultural retrieval?
**Answer:** Farmers frequently ask brief questions (e.g., *"white insects on cotton"*). A standard vector search on this short phrase may match irrelevant documents. HyDE prompts an LLM to generate a hypothetical technical answer first. That hypothetical document contains dense domain vocabulary (e.g., *"Bemisia tabaci"*, *"sucking pests"*, *"sticky traps"*). Embedding this hypothetical passage brings the search query significantly closer in vector space to the actual technical chunks.

#### Q3: Why is the hypothetical document never passed directly to the final answer LLM?
**Answer:** The hypothetical document is synthesized by the LLM before retrieving real documents, meaning it could contain plausible-sounding hallucinations. To guarantee 100% factual accuracy, it is strictly used as an intermediate semantic search vector. The final generation prompt receives *only* authentic retrieved chunks.

#### Q4: Why choose Google Gemini `models/text-embedding-004`?
**Answer:** `text-embedding-004` provides state-of-the-art 768-dimensional dense representations optimized for semantic retrieval, fast cosine similarity indexing, and seamless native integration with Google's GenAI ecosystem.

#### Q5: Why is FAISS used instead of a relational database or cloud vector database?
**Answer:** FAISS (Facebook AI Similarity Search) is an ultra-fast, lightweight vector indexing engine that runs locally with minimal overhead. For focused domain document collections, FAISS allows instantaneous similarity searches without recurring cloud database hosting costs.

#### Q6: How does this project handle Multi-Document queries?
**Answer:** When a user asks: *"How does soil health affect irrigation and crop yield?"*, the top-k retrieved chunks (k=5) naturally pull segments from `soil_health.txt`, `irrigation_management.txt`, and `crop_production.txt`. The RAG prompt instructs Gemini to perform multi-source synthesis, merging these disparate domains into one coherent answer.

#### Q7: How are source citations tracked accurately?
**Answer:** During chunking, each text segment receives metadata including its source file name and 1-indexed chunk number (`crop_production.txt_chunk_2`). These metadata tags are injected into the context block given to the LLM, enabling verbatim attribution without generating hallucinated page numbers.

#### Q8: What role does `RecursiveCharacterTextSplitter` play?
**Answer:** It recursively splits text by priority separators (`["\n\n", "\n", ". ", " ", ""]`), preserving paragraphs and sentences intact wherever possible. This ensures agricultural instructions (such as a 3-step seed treatment protocol) remain coherent within an 800-character window.

#### Q9: How does the application safeguard API keys?
**Answer:** The API key is stored in an ignored `.env` file, loaded at runtime via `python-dotenv`, and accessed through `app/config.py`. Source code, example templates, and git histories contain zero hardcoded secrets.

#### Q10: How do you prevent hallucinated government schemes or chemical doses?
**Answer:** By enforcing a strict system prompt with negative constraints: requiring the LLM to output *"I could not find sufficient information about this in the available documents"* whenever the retrieved context is insufficient, coupled with low generation temperature (0.2).
