from typing import List, Dict, Any

from rag.core.config import settings

from rag.pipeline.vector_store import VectorStore
from rag.pipeline.embedder import Embedder

class Retriever:
    """Handles query-based retrieval from the vector store"""

    def __init__(self, vector_store: VectorStore, embedding_manager: Embedder):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        md = getattr(self.vector_store.collection, "metadata", None) or {}
        self.metric = str(md.get("hnsw:space", "cosine")).lower()

    def _to_similarity(self, distance: float) -> float:
        if self.metric == "cosine":
            return 1.0 - float(distance)
        elif self.metric in {"l2", "euclidean"}:
            return 1.0 / (1.0 + float(distance))
        elif self.metric in {"ip", "dot"}:
            return -float(distance) 
        return -float(distance)

    def retrieve(
            self, 
            query: str, 
            top_k: int = settings.TOP_K, 
            score_threshold: float = settings.SCORE_THRESHOLD
        ) -> List[Dict[str, Any]]:
        
        print(f"Retrieving documents for query: '{query}'")
        print(f"Top K: {top_k}, Score threshold: {score_threshold}")

        if not query or not query.strip():
            return []

        q_emb = self.embedding_manager.generate_embedding(query)
        if hasattr(q_emb, "tolist"):
            q_emb = q_emb.tolist()

        try:
            results = self.vector_store.collection.query(
                query_embeddings=[q_emb],
                n_results=top_k,
                include=["documents", "metadatas", "distances"] 
            )

            docs_batches = results.get("documents") or []
            if not docs_batches or not docs_batches[0]:
                print("No documents found")
                return []

            documents = docs_batches[0]
            metadatas = (results.get("metadatas") or [[]])[0] or [{}] * len(documents)
            distances = (results.get("distances") or [[]])[0] or [float("inf")] * len(documents)
            
            ids       = (results.get("ids") or [[]])[0] or [None] * len(documents)

            retrieved_docs: List[Dict[str, Any]] = []
            for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                sim = self._to_similarity(distance)
                if sim >= score_threshold:
                    retrieved_docs.append({
                        "id": doc_id,
                        "content": document,
                        "metadata": metadata or {},
                        "similarity_score": sim,
                        "distance": distance,
                        "rank": i + 1
                    })

            
            retrieved_docs.sort(key=lambda r: (-r["similarity_score"], r["rank"]))

        
            seen = set()
            unique = []
            for r in retrieved_docs:
                m = r.get("metadata") or {}
                sig = (r["content"], m.get("page"), m.get("source_file") or m.get("source"))
                if sig in seen:
                    continue
                seen.add(sig)
                unique.append(r)

            print(f"Retrieved {len(unique)} documents (after filtering & dedup)")
            return unique

        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []