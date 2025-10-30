from __future__ import annotations

from typing import Tuple, List, Dict, Any

from rag.core.config import settings
from rag.api.services.components import ensure_components
from rag.pipeline.grok_rag_pipeline import RAG_Simple_Grok
from rag.pipeline.hf_rag_pipeline import RAG_Simple_HF

def run_rag_query(
        question:str,
        provider:str="hf",
        top_k:int=settings.TOP_K,
        score_threshold:float=settings.SCORE_THRESHOLD,
        max_ctx_chars:int=settings.MAX_CTX_CHARS,
    ) -> Tuple[str, List[Dict[str, Any]], str]:
    """
    return (answer, retriever_results, used_provider)
    """
    _, _, retriever = ensure_components()
    results = retriever.retrieve(
        question,
        top_k=top_k,
        score_threshold=score_threshold,
    )
    
    if not results:
        return settings.GUARD_SENTENCE, [], provider
    
    if provider == "grok":
        answer = RAG_Simple_Grok(
            query=question,
            retriever=retriever,
            top_k=top_k,
            score_threshold=score_threshold,
            max_ctx_chars=max_ctx_chars
        )
        used="grok"
    else:
        answer = RAG_Simple_HF(
            query=question,
            retriever=retriever,
            top_k=top_k,
            score_threshold=score_threshold,
            max_ctx_chars=max_ctx_chars
        )
        use="hf"
    
    return answer, results, used

def citations_from_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Transform retriever results into a list of citation dicts compatible with Citation model.
    """

    cites = []

    for r in results:
        m = r.get("metadata") or {}
        cites.append({
            "source_name":m.get("source"),
            "source_file":m.get("source_name"),
            "page":m.get("page"),
            "similarity":float(r.get("similarity_score")),
            "id":r.get("id"),
        })
    return cites