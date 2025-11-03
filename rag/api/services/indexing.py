# from pathlib import Path
# from typing import Tuple

# from rag.api.services.components import ensure_components
# from rag.pipeline.data_loader import load_data
# from rag.pipeline.chunker import chunk_document
# from rag.utility.helpers import extract_text_and_metas


# def build_index(data_dir:Path) ->Tuple[int, int]:
#     """
#     Loads PDF files from provided dir -> embed -> upsert to VS
#     returns number of added embeddings and total 
     
#     """
#     embedder, vs , _ = ensure_components()

#     docs = load_data(data_dir=data_dir)
#     chunks = chunk_document(docs)
    
#     texts, _ = extract_text_and_metas(chunks)
#     embeddings = embedder.generate_embeddings(texts)

#     before_adding = None
#     try:
#         before_adding = vs.collection.count()
#     except Exception:
#         before_adding = None
    
#     vs.add_documents(chunks, embeddings)

#     after_adding = 0
#     try:
#         after_adding = vs.collection.count()
#     except Exception:
#         pass

#     added = max(0, (after_adding or 0) - (before_adding or 0) if before_adding is not None else len(chunks))

#     return added , after_adding


from pathlib import Path
from typing import Tuple
import os, time

from rag.api.services.components import ensure_components
from rag.pipeline.data_loader import load_data
from rag.pipeline.chunker import chunk_document
from rag.utility.helpers import extract_text_and_metas

SLICE = 64  # 32–64 is safe on small/medium instances
LOCK_PATH = "/app/storage/index.lock"  # make sure DATA_DIR is /app/storage

def _acquire_lock(path: str, wait_secs: int = 0) -> bool:
    # simple cross-process lock using the disk
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return True
    except FileExistsError:
        # optionally wait a little if another job is running
        if wait_secs > 0:
            time.sleep(wait_secs)
            return _acquire_lock(path, 0)
        return False

def _release_lock(path: str):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass

def build_index(data_dir: Path) -> Tuple[int, int]:
    if not _acquire_lock(LOCK_PATH, wait_secs=0):
        # another index is already running; return without doing heavy work
        try:
            # best-effort count
            embedder, vs, _ = ensure_components()
            return 0, vs.collection.count()
        except Exception:
            return 0, None

    try:
        embedder, vs, _ = ensure_components()

        docs = load_data(data_dir=data_dir)
        chunks = chunk_document(docs)

        # count BEFORE
        try:
            before = vs.collection.count()
        except Exception:
            before = 0

        added = 0
        # stream in small slices to avoid OOM and save progress incrementally
        for i in range(0, len(chunks), SLICE):
            part = chunks[i:i + SLICE]
            texts, _ = extract_text_and_metas(part)
            embs = embedder.generate_embeddings(texts)  # small batch only
            vs.add_documents(part, embs)                # persist immediately
            added += len(part)

        # count AFTER
        try:
            after = vs.collection.count()
        except Exception:
            after = before + added

        return max(0, (after or 0) - (before or 0)), after
    finally:
        _release_lock(LOCK_PATH)
