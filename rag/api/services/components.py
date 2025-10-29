# api/services/components.py
from __future__ import annotations
from typing import Optional, Tuple
from threading import RLock

from rag.core.config import settings
from rag.pipeline.embedder import Embedder
from rag.pipeline.vector_store import VectorStore
from rag.pipeline.retriever import Retriever

# Singleton instances
_EMBEDDER: Optional[Embedder] = None
_VSTORE: Optional[VectorStore] = None
_RETRIEVER: Optional[Retriever] = None

# Simple lock for thread-safe lazy init under Uvicorn/Gunicorn workers
_INIT_LOCK = RLock()


def _init_components() -> Tuple[Embedder, VectorStore, Retriever]:
    """
    Internal initializer. Call inside the lock.
    """
    global _EMBEDDER, _VSTORE, _RETRIEVER

    if _EMBEDDER is None:
        _EMBEDDER = Embedder(
            model_name=settings.EMBEDDER_MODEL_NAME,
            normalize=settings.NORMALIZE,
            batch_size=settings.BATCH_SIZE,
            # device can be passed if you want: device="cpu" or "cuda"
        )

    if _VSTORE is None:
        _VSTORE = VectorStore(
            collection_name=settings.COLLECTION_NAME,
            persist_directory=settings.PERSIST_DIRECTORY_VS,
            embedder_model_name=settings.EMBEDDER_MODEL_NAME,
        )

    if _RETRIEVER is None:
        _RETRIEVER = Retriever(vector_store=_VSTORE, embedding_manager=_EMBEDDER)

    return _EMBEDDER, _VSTORE, _RETRIEVER


def ensure_components() -> Tuple[Embedder, VectorStore, Retriever]:
    """
    Public entry: lazily initialize (once per process) and return all components.
    Safe to call from any request handler or other service.
    """
    # Fast path: already initialized
    if _EMBEDDER and _VSTORE and _RETRIEVER:
        return _EMBEDDER, _VSTORE, _RETRIEVER

    with _INIT_LOCK:
        # Check again inside the lock (double-checked locking)
        if _EMBEDDER and _VSTORE and _RETRIEVER:
            return _EMBEDDER, _VSTORE, _RETRIEVER
        return _init_components()


def get_embedder() -> Embedder:
    emb, _, _ = ensure_components()
    return emb


def get_vector_store() -> VectorStore:
    _, vs, _ = ensure_components()
    return vs


def get_retriever() -> Retriever:
    _, _, rt = ensure_components()
    return rt


def reset_components() -> None:
    """
    Optional: Clear singletons (useful for tests or hot-reload scenarios).
    """
    global _EMBEDDER, _VSTORE, _RETRIEVER
    with _INIT_LOCK:
        _EMBEDDER = None
        _VSTORE = None
        _RETRIEVER = None
