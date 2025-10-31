# RAG-API


# Medical Assistant with RAG

> An AI-powered medical assistant that answers health-related questions using **Retrieval-Augmented Generation (RAG)** over your curated PDF knowledge base.

[![Built with FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green)](https://fastapi.tiangolo.com/)
[![ChromaDB](https://img.shields.io/badge/Vector%20Store-ChromaDB-blue)](https://www.trychroma.com/)
[![SentenceTransformers](https://img.shields.io/badge/Embeddings-all--MiniLM--L6--v2-purple)](https://www.sbert.net/)
[![Dockerized](https://img.shields.io/badge/Docker-ready-informational)](https://www.docker.com/)
[![License](https://img.shields.io/badge/license-See%20LICENSE-lightgray)](#-license)

---

## ✨ Features

- **End-to-end RAG API** (FastAPI) with routes to **index**, **query**, **upload**, **delete**, and **inspect stats**.
- **Deterministic chunk IDs** and **stable metadata** for robust incremental indexing & deduplication.
- **Configurable chunking** (RecursiveCharacterTextSplitter + optional tiktoken length) with normalization/cleaning of PDF text.
- **Sentence-Transformers embeddings** (`all-MiniLM-L6-v2`) with optional normalization and batch encoding.
- **Persistent Vector Store** via **ChromaDB** (HNSW cosine) under `data/vector_store/`.
- **Two LLM providers**:
  - **Hugging Face Inference Endpoint** (default) via `langchain-huggingface`
  - **xAI Grok** via `langchain-xai` (optional)
- **Secure by default**: all write/query endpoints require `X-API-Key` header.
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
- **Persistence:** ChromaDB at `data/vector_store/` (collection: `pdf_documents`)
- **Data sources:** PDFs under `data/` (including `data/uploads/` for user uploads)
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
   PDFs ---> │ data/         data/uploads/ │
  (knowledge)└───────────────┬─────────────┘
                             │  load_data()
                             ▼
                      ┌───────────────┐
                      │  CHUNKER      │  ← cleans & splits PDF text
                      │ (Recursive)   │     (helpers.normalize_text)
                      └───────┬───────┘
                              │  texts + metadata
                              ▼
                      ┌───────────────┐
                      │  EMBEDDER     │  ← SentenceTransformers
                      │ (MiniLM etc.) │     (batch, normalized)
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
                      ┌───────┴────────┐
                      │   FastAPI       │
                      │   Routers       │
                      │  /v1/index      │  (build index from PDFs)
                      │  /v1/upload     │  (upload PDFs + index)
   client question -> │  /v1/query  ----┼─► LLM Providers (LangChain):
                      │  /v1/delete     │      • HF Endpoint (default)
                      │  /health, /stats│      • Grok (optional)
                      └─────────────────┘


```













Docker file runinning
```bash
docker run --rm -it `
  --env-file .env `
  -p 8000:8000 `
  -v "${PWD}/data:/app/data" `
  -v "${PWD}/hf-cache:/root/.cache/huggingface" `
  --name medrag medrag-api:latest
```