from __future__ import annotations
from typing import Optional

from rag.core.config import settings
from rag.pipeline.embedder import Embedder
from rag.pipeline.vector_store import VectorStore
from rag.pipeline.retriever import Retriever

_EMBEDDER:Optional[Embedder] = None
_VSTORE:Optional[VectorStore] = None
_RETRIEVER:Optional[Retriever] = None

def ensure_components():
    """
    Initializing sinletons once per process.
    """
    global _EMBEDDER, _VSTORE, _RETRIEVER

    if _EMBEDDER is None:
        _EMBEDDER = Embedder(
            model_name=settings.EMBEDDER_MODEL_NAME,
            normalize=settings.NORMALIZE,
            batch_size=settings.BATCH_SIZE,
        )
    if _VSTORE is None:
        _VSTORE = VectorStore(
            collection_name=settings.COLLECTION_NAME,
            persist_directory=settings.PERSIST_DIRECTORY_VS,
            embedder_model_name=settings.EMBEDDER_MODEL_NAME
        )
    
    if _RETRIEVER is None:
        _RETRIEVER = Retriever(
            vector_store=_VSTORE,
            embedding_manager=_EMBEDDER
        )

    return _EMBEDDER, _VSTORE, _RETRIEVER


def get_vector_store() -> VectorStore:
    _, vstore, _ = ensure_components()

    return vstore