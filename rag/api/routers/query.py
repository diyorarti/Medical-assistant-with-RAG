from __future__ import annotations
from fastapi import APIRouter, HTTPException, Depends

from rag.api.schemas.models import QueryResponse, QueryRequest
from rag.core.config import settings 
from rag.api.services.retrieval import run_rag_query, citations_from_results
from rag.core.security import verify_api_key

router = APIRouter(prefix="/v1")

@router.post("/query", response_model=QueryResponse, dependencies=[Depends(verify_api_key)])
def query_rag(req:QueryRequest):
    """
    Ask any question related to Knowledge base
    returns Knowledge + LLM -> Answer and citations
    """
    if req.provider == "grok" and not settings.GROK_API_KEY:
        HTTPException(status_code=400, detail="GROK request, but API-key is not set")
    if req.provider == "hf" and not settings.HF_TOKEN:
        raise HTTPException(status_code=400, detail="HF Endpoint request, but HF token is not set")
    
    answer, results, used = run_rag_query(
        question=req.question,
        provider=req.provider,
        top_k=req.top_k,
        score_threshold=req.score_threshold,
        max_ctx_chars=req.max_ctx_chars
    )
    cites = citations_from_results(results)

    return QueryResponse(answer=answer, citations=cites, used_provider=used)