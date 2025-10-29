# api/services/indexing.py
from __future__ import annotations
from pathlib import Path
from typing import Tuple, Iterable, List

from rag.pipeline.data_loader import load_data
from rag.pipeline.chunker import chunk_document
from rag.utility.helpers import extract_text_and_metas
from rag.api.services.components import ensure_components
from rag.core.logging import get_logger
logger = get_logger(__name__)

def build_index(data_dir: Path) -> Tuple[int, int]:
    """
    Load PDFs from `data_dir` -> chunk -> embed -> upsert into vector store.
    Returns (added_count_estimate, total_in_collection_after).
    """
    logger.info(f"Indexing started for {data_dir}")
    embedder, vstore, _ = ensure_components()

    # 1) load + chunk
    docs = load_data(data_dir=data_dir)
    chunks = chunk_document(docs)

    # 2) embed
    texts, _ = extract_text_and_metas(chunks)
    embeddings = embedder.generate_embeddings(texts)

    # 3) upsert
    before = None
    try:
        before = vstore.collection.count()
    except Exception:
        pass

    vstore.add_documents(chunks, embeddings)

    after = 0
    try:
        after = vstore.collection.count()
    except Exception:
        pass

    # Upsert may overwrite existing IDs; this is an estimate
    added = len(chunks) if before is None else max(0, (after or 0) - (before or 0))
    logger.info(f"Indexing finished: added={added}, total ={after}")
    return added, after


def build_index_for_upload_dir(upload_dir: Path) -> Tuple[int, int]:
    """
    Same as build_index, but for a subfolder (e.g., data/uploads).
    Useful for /v1/upload endpoint.
    """
    return build_index(upload_dir)
