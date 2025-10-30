from pathlib import Path
from typing import Tuple

from rag.api.services.components import ensure_components
from rag.pipeline.data_loader import load_data
from rag.pipeline.chunker import chunk_document
from rag.utility.helpers import extract_text_and_metas


def build_index(data_dir:Path) ->Tuple[int, int]:
    """
    Loads PDF files from provided dir -> embed -> upsert to VS
    returns number of added embeddings and total 
     
    """
    embedder, vs , _ = ensure_components()

    docs = load_data(data_dir=data_dir)
    chunks = chunk_document(docs)
    
    texts, _ = extract_text_and_metas(chunks)
    embeddings = embedder.generate_embeddings(texts)

    before_adding = None
    try:
        before_adding = vs.collection.count()
    except Exception:
        before_adding = None
    
    vs.add_documents(chunks, embeddings)

    after_adding = 0
    try:
        after_adding = vs.collection.count()
    except Exception:
        pass

    added = max(0, (after_adding or 0) - (before_adding or 0) if before_adding is not None else len(chunks))

    return added , after_adding


