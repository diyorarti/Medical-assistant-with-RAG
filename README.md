# Medical Assistant with RAG

> A LLM(s)-powered medical assistant that answers med-related questions using **Retrieval-Augmented Generation (RAG)** over your curated PDF knowledge base.

[![Built with FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green)](https://fastapi.tiangolo.com/)
[![ChromaDB](https://img.shields.io/badge/Vector%20Store-ChromaDB-blue)](https://www.trychroma.com/)
[![SentenceTransformers](https://img.shields.io/badge/Embeddings-all--MiniLM--L6--v2-purple)](https://www.sbert.net/)
[![HF Endpoint](https://img.shields.io/badge/HF--Endpoint-diyorarti%2Fmed--mixed--merged--qbi-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/diyorarti/med-mixed-merged)
[![Dockerized](https://img.shields.io/badge/Docker-ready-informational)](https://www.docker.com/)
[![License](https://img.shields.io/badge/license-See%20LICENSE-lightgray)](#-license)

---

## ✨ Features

- **Production Ready RAG API** (FastAPI) with routes to 
      1.  **health** 
      2.  **index** 
      3.  **query** 
      4.  **upload** 
      5.  **delete** 
      6.  **stats**
- **Deterministic chunk IDs** and **stable metadata** for robust incremental indexing & deduplication.
- **Configurable chunking** (RecursiveCharacterTextSplitter + optional tiktoken length) with normalization/cleaning of PDF text.
- **Sentence-Transformers embeddings** (`all-MiniLM-L6-v2`) with optional normalization and batch encoding.
- **Persistent Vector Store** via **ChromaDB** (HNSW cosine) under `data/vector_store/`.
- **Two LLM providers**:
  - **Hugging Face Inference Endpoint** (default) via `langchain-huggingface`
  - **xAI Grok** via `langchain-xai` (optional)
- **Secure by default**: all (except health and stats) endpoints require `X-API-Key` header.
- **Docker-ready** image with healthcheck and `uvicorn` entrypoint.
- **Utilities & Labs**: caching demo (`data/cache`) and a development pipeline script under `rag/test/`.

---

## 🧭 Project Overview

- **Status:** MVP / research-ready (version `0.1.0`)
- **Language/Stack:** Python 3.11, FastAPI, LangChain, SentenceTransformers, ChromaDB
- **LLMs:**
  - HF Endpoint (task: `text-generation`) – configurable via `.env`
  - Grok (`grok-4`) – optional, requires `GROK_API_KEY`
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (configurable)
- **Persistence:** ChromaDB at `data/vector_store/` (collection-name: `pdf_documents`)
- **Data sources:** PDFs (used to build RAG) under `data/` (users upload PDFs are stored `data/uploads/`)
- **Security:** `X-API-Key` checked by dependency (`rag.core.security.verify_api_key`)
- **API Prefix:** `/v1`
- **Key Routes:** 
  - `GET /health` – service health & active collection
  - `GET /v1/stats` – collection size
  - `POST /v1/index` – (re)index PDFs from `data/` (or a provided directory)
  - `POST /v1/upload` – upload **PDFs** then auto-index
  - `POST /v1/query` – RAG question answering (provider: `hf` or `grok`)
  - `DELETE /v1/delete` – remove vectors by source file path

---

## 🧱 Architecture (High Level)

```bash
             ┌─────────────────────────────┐
   PDFs -->  │ data/ (RAD-dev-pdfs)        │
             │ data/uploads (user-uploaded)│ 
             └───────────────┬─────────────┘
                             │  load_data() - (langchain_community.document_loaders -> PyPDFLoader)
                             ▼
                      ┌───────────────┐
                      │  CHUNKER      │  ← cleans & splits PDF text (langchain -> RecursiveCharacterTextSplitter)
                      │ (Recursive)   │     (helpers.normalize_text)
                      └───────┬───────┘
                              │  texts + metadata
                              ▼
                      ┌───────────────┐
                      │  EMBEDDER     │  ← SentenceTransformers ("all-MiniLM-L6-v2" model )
                      │               │     (batch, normalized)
                      └───────┬───────┘
                              │  vectors + metadata (deterministic IDs)
                              ▼
                      ┌────────────────┐
                      │   ChromaDB     │  ← persistent vector store
                      │ (collection)   │
                      └───────┬────────┘
                              ▲
                       retrieve(top_k, threshold)
                              │
                      ┌───────┴─────────┐
                      │   FastAPI       │
                      │   Routers:      │
                      │  /v1/index      │   (build index from default PDFs)
                      │  /v1/upload     │   (upload PDFs + index)
   client question -> │  /v1/query -----│─► (LLM Providers: • HF Endpoint(default) (GROK optional))
                      │  /v1/delete     │   (Delete the indexed source)
                      │  /health        │   
                      │  /stats         │      
                      └─────────────────┘
```
**Modules**
- `rag/api/` – FastAPI app, routers, schemas, service layers.
- `rag/core/` – configuration (config) API-key security.
- `rag/pipeline/` – data-loader, chunker, embedder, retriever, vector-store,RAG pipelines(HF and GROK).
- `rag/pipeline/LLM` - grok-llm, hf-endpoint.
- `rag/utility/helpers.py` – hashing, normalization,getting-chunk-id,text and meta extraftor ,ID generation, context formatting.
- `rag/test/` – development pipeline with caching demo.
- `data/` – PDFs, vector store persistence.
- `data/uploads/` - user uploaded pdfs.
- `data/cache` - chunks.pkl, embeddings.npy
- `labs/` – development notebooks & lab `project-lab.ipynb` and  `requirements.txt`.

---

## 📁 Project Structure
```bash
medical-assistant-with-rag/
│
├── .vscode/ # VSCode workspace settings
├── data/ # Knowledge base and vector store
│ ├── cache/ # Cache of chunks and embeddings
│ │ ├── chunks.pkl
│ │ ├── embeddings.npy
│ │ └── manifest.json
│ ├── uploads/ # User Uploaded PDFs for knowledge base
│ │ └── 1706.03762v7.pdf
│ └── vector_store/ # ChromaDB persistence
│ ├── chroma.sqlite3
│ ├── Aging_natural_or_disease.pdf # RAG-dev-knowledge base
│ ├── Genes_and_Disease.pdf # RAG-dev-knowledge base
│ └── basic_epidemiology.pdf # RAG-dev-knowledge base
│
├── hf-cache/ # Local Hugging Face model cache
│
├── labs/ # Research notebooks and experiments
│ ├── project-lab.ipynb # RAG development lab
│ └── requirements.txt # packages used in LAB experiment
│
├── rag/ # Core application package
│ ├── api/ # FastAPI endpoints, routers, and services
│ │ ├── routers/
│ │ ├── schemas/
│ │ ├── services/
│ │ └── main.py # FastAPI entry point
│ │
│ ├── core/ # App configuration and security
│ │ ├── config.py # Loads environment variables
│ │ └── security.py # API key verification
│ │
│ ├── pipeline/ # RAG pipeline components
│ │ ├── LLM/ # Large Language Model interfaces
│ │ ├── chunker.py # Text chunking logic
│ │ ├── data_loader.py # PDF loader and parser
│ │ ├── embedder.py # Embedding generation
│ │ ├── retriever.py # Retrieves relevant chunks from vector store
│ │ ├── vector_store.py # Handles ChromaDB operations
│ │ ├── hf_rag_pipeline.py # RAG pipeline using Hugging Face models
│ │ └── grok_rag_pipeline.py # Optional RAG pipeline using Grok
│ │
│ ├── test/ # Unit & dev-level tests
│ │ └── rag_pipeline_dev.py
│ │
│ └── utility/ # Helper utilities
│   └── helpers.py # Text normalization, hashing, etc. 
│
├── .dockerignore
├── .gitignore
├── .env # Environment variables (not for Git)
├── Dockerfile # Docker setup
├── LICENSE
├── pyproject.toml # Project dependencies and metadata
└── README.md # Project documentation
```


## ⚙️ Installation

### 🧩 Prerequisites
Before starting, make sure you have:

- **Python 3.10 or higher**
- **pip / venv** or **conda**
- *(optional)* **Docker 24+** if you prefer containerized deployment

---

### 🗂️ Clone the Repository
```bash
git clone https://github.com/diyorarti/Medical-assistant-with-RAG.git
cd Medical-assistant-with-RAG

```bash
# Create venv
python -m venv .venv
# Activate (Linux/Mac)
source .venv/bin/activate
# Activate (Windows)
.venv\Scripts\activate
# installing packages
pip install -e .
```
or if Anaconda installed
```bash
# creating new virtual environment
conda create -n evnName python=3.10 -y
# activating created env
conda activate evnName
# installing packages
pip install -e .
```

### running project locally
```bash
uvicorn rag.api.main:app --reload
```
then Visit
➡️ Swagger UI: `http://127.0.0.1:8000/docs`
➡️ Healthcheck: `http://127.0.0.1:8000/health`

## 💻 Usage & Examples
You can use the **Medical Assistant with RAG** in two ways:

1. **Through the REST API** (recommended for most users)
2. **As a Python module** (for developers building custom pipelines)
---

### 🌐 1. Using the API

Once the FastAPI app is running (`uvicorn rag.api.main:app --reload`), open:

➡️ **Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

There, you can test all endpoints interactively.

#### Example — Upload & Query via API

**Upload a PDF**
```bash
curl -X POST "http://127.0.0.1:8000/v1/upload" \
     -H "X-API-Key: your_api_key_here" \
     -F "file=@data/uploads/Aging_natural_or_disease.pdf"
```

### 🧠 2. Using the Python API
```bash
from rag.pipeline.hf_rag_pipeline import RAG_Simple_HF
from rag.pipeline.retriever import Retriever
from rag.pipeline.vector_store import VectorStore
from rag.pipeline.embedder import Embedder

embedder = Embedder()
vs = VectorStore()
retriever = Retriever(vs, embedder)

answer = RAG_Simple_HF("What causes neural degeneration?", retriever=retriever)
print(answer)

```

## ☁️ Deployment
### 🐳 2. Building and Run  Docker image
```bash
# buiding docker image
docker build -t medical-assistant-rag .
# running image
docker run --rm -it `
  --env-file .env `
  -p 8000:8000 `
  -v "${PWD}/data:/app/data" `
  -v "${PWD}/hf-cache:/root/.cache/huggingface" `
  --name medrag medrag-api:latest
```
then Visit:
➡️ Swagger UI: `http://127.0.0.1:8000/docs`
➡️ Healthcheck: `http://127.0.0.1:8000/health`

### 🚀 Deploy on Railway

1. Push your repo to GitHub.
2. Create a new project on Railway
3. Add environment variables in project settings.
4. Click Deploy.
5. Access your API from the live URL (e.g., https://medical-assistant.up.railway.app).

### 🔐 Production Tips

Keep .env secrets private.
Use strong API_KEY.
Mount persistent volume for /data to retain embeddings.
Regularly back up data/vector_store.

### 📄 License
MIT License

### 🙏 Acknowledgements

[LangChain](https://www.langchain.com/)
[SentenceTransformers](https://www.sbert.net/)
[ChromaDB](https://www.trychroma.com/)
[FastAPI](https://fastapi.tiangolo.com/)
[Hugging Face](https://huggingface.co/)
[HF-Endpoint(LLM)](https://huggingface.co/diyorarti/med-mixed-merged)
[GROK(LLM)](https://x.ai/)