# rag/api/services/retrieval.py
from __future__ import annotations
from typing import List, Dict, Any, Tuple
from rag.core.config import settings
from rag.pipeline.grok_rag_pipeline import RAG_Simple_Grok
# IMPORTANT: ensure the import path is correct and the typo is fixed in your file:
# from rag.pipeline.LLM.hf_endpoint import get_hf_llm  (inside your hf_rag_pipeline.py)
from rag.pipeline.hf_rag_pipeline import RAG_Simple_HF
from rag.api.services.components import ensure_components

def run_rag_query(
    question: str,
    provider: str = "hf",
    top_k: int = settings.TOP_K,
    score_threshold: float = settings.SCORE_THRESHOLD,
    max_ctx_chars: int = settings.MAX_CTX_CHARS,
) -> Tuple[str, List[Dict[str, Any]], str]:
    """
    Returns (answer, retriever_results, used_provider)
    """
    _, _, retriever = ensure_components()

    results = retriever.retrieve(
        question, top_k=top_k, score_threshold=score_threshold
    )

    if not results:
        return settings.GUARD_SENTENCE, [], provider

    if provider == "grok":
        # Will raise earlier in router if GROK_API_KEY missing
        answer = RAG_Simple_Grok(
            query=question,
            retriever=retriever,
            top_k=top_k,
            score_threshold=score_threshold,
            max_ctx_chars=max_ctx_chars,
        )
        used = "grok"
    else:
        # HF endpoint
        answer = RAG_Simple_HF(
            query=question,
            retriever=retriever,
            top_k=top_k,
            score_threshold=score_threshold,
            max_ctx_chars=max_ctx_chars,
        )
        used = "hf"

    return answer, results, used


def citations_from_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Transform retriever results into a list of citation dicts compatible with Citation model.
    """
    cites = []
    for r in results:
        m = r.get("metadata") or {}
        cites.append({
            "source_name": m.get("source_name"),
            "source_file": m.get("source_file") or m.get("source"),
            "page": m.get("page"),
            "similarity": float(r.get("similarity_score")) if r.get("similarity_score") is not None else None,
            "id": r.get("id"),
        })
    return cites
