from pathlib import Path
import json
import pickle
import numpy as np
from typing import Dict, Any, List

from rag.core.config import settings, PROJECT_ROOT

from rag.utility.helpers import sha256_file
from rag.pipeline.data_loader import load_data
from rag.pipeline.chunker import chunk_document
from rag.utility.helpers import extract_text_and_metas
from rag.pipeline.grok_rag_pipeline import RAG_Simple_Grok
from rag.pipeline.hf_rag_pipeline import RAG_Simple_HF

from rag.pipeline.embedder import Embedder
from rag.pipeline.vector_store import VectorStore
from rag.pipeline.retriever import Retriever

# -------------------------------
# Cache helpers
# -------------------------------

CACHE_DIR = PROJECT_ROOT / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_PKL = CACHE_DIR / "chunks.pkl"
EMBS_NPY   = CACHE_DIR / "embeddings.npy"
MANIFEST   = CACHE_DIR / "manifest.json"

def build_manifest(data_dir: Path) -> Dict[str, Any]:
    """Scan PDFs; compute per-file sha256 + mtime and a global digest."""
    entries: List[Dict[str, Any]] = []
    for p in data_dir.rglob("*.pdf"):
        try:
            sha = sha256_file(p)
            mtime = int(p.stat().st_mtime)
            entries.append({"path": str(p.resolve()), "sha256": sha, "mtime": mtime})
        except Exception:
            # Skip unreadable files but still be robust
            continue
    # stable sort for deterministic digest
    entries.sort(key=lambda e: e["path"])
    import hashlib, json as _json
    digest_src = _json.dumps(entries, separators=(",", ":"), sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(digest_src).hexdigest()
    return {"entries": entries, "digest": digest}

def load_manifest(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def save_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

# -------------------------------
# Pipeline with caching
# -------------------------------

# 1) Compare dataset manifest to decide whether to reuse caches
current_manifest = build_manifest(settings.DATA_DIR)
cached_manifest  = load_manifest(MANIFEST)

dataset_changed = (cached_manifest is None) or (cached_manifest.get("digest") != current_manifest["digest"])

chunks = None
embeddings = None

# 2) Load or (re)create chunks
if (not dataset_changed) and CHUNKS_PKL.exists():
    print("✅ Using cached chunks")
    with open(CHUNKS_PKL, "rb") as f:
        chunks = pickle.load(f)
else:
    print("🔄 Rebuilding chunks (data changed or no cache)")
    docs = load_data()                         # uses settings.DATA_DIR
    chunks = chunk_document(docs)              # uses your config defaults
    with open(CHUNKS_PKL, "wb") as f:
        pickle.dump(chunks, f)
    # Since chunks changed, embeddings cache is invalid
    if EMBS_NPY.exists():
        try:
            EMBS_NPY.unlink()
        except Exception:
            pass

# 3) Load or (re)create embeddings
if (not dataset_changed) and EMBS_NPY.exists():
    print("✅ Using cached embeddings")
    embeddings = np.load(EMBS_NPY)
else:
    print("🔄 Rebuilding embeddings (data changed or no cache)")
    texts, metas = extract_text_and_metas(chunks)
    embedder = Embedder()
    embeddings = embedder.generate_embeddings(texts)
    np.save(EMBS_NPY, embeddings)

# 4) Persist the manifest (now that caches are valid)
save_manifest(MANIFEST, current_manifest)

# 5) Upsert to vector store (idempotent due to deterministic IDs)
vs = VectorStore()
vs.add_documents(chunks, embeddings)

# 6) Quick retrieval test (optional)
retriever = Retriever(vector_store=vs, embedding_manager=Embedder())
query = "What is an anti-aging intervention?"
hits = retriever.retrieve(query)
for h in hits:
    m = h["metadata"]
    print(f"{h['similarity_score']:.3f} | page {m.get('page')} | {m.get('source_name')}")

# 7) GROK LLM output generation
# answer = RAG_Simple_Grok("What is an anti-aging intervention?", retriever)
# print("\nFinal Answer:\n", answer)

# # 7) HF Endpoint output generation
answer = RAG_Simple_HF(query, retriever=retriever)
print(f"\n FINAL ANSWER {answer}")
