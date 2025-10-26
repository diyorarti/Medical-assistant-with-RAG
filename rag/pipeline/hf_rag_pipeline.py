from typing import Any, Dict, List, Optional, Union

from rag.utility.helpers import format_context
from rag.core.config import settings
from rag.pipeline.LLM.hf_endpoit import get_hf_llm
from rag.pipeline.retriever import Retriever

PRIMARY_PROMPT = """You are a careful medical assistant.
Answer the QUESTION using ONLY the CONTEXT. If something is not in the CONTEXT, do not invent it.
Prefer concise bullet points. Include concrete measures if present in the CONTEXT.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

# A bit looser: encourages synthesis when principles are present
BACKUP_PROMPT = """You are a careful medical assistant.
Using ONLY the CONTEXT, provide a practical answer to the QUESTION. Summarize relevant principles and turn them into actionable steps.
If details are missing, say what is known from CONTEXT and avoid speculation.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

def RAG_Simple_HF(
    query: str,
    retriever: Retriever,
    llm: Any | None = None,
    top_k: int = settings.TOP_K,
    score_threshold: Optional[float] = settings.SCORE_THRESHOLD,
    max_ctx_chars: int = settings.MAX_CTX_CHARS,
    stop: Optional[List[str]] = None,
    max_new_tokens: int = settings.HF_MAX_TOKENS,
    temperature: float = settings.HF_TEMPERATURE,
) -> str:
    # HF llm initialization
    llm = get_hf_llm()
    # 1) retrieve
    if hasattr(retriever, "retrieve"):
        results = retriever.retrieve(query, top_k=top_k, score_threshold=score_threshold)
        results = [r if isinstance(r, dict) else {"content": str(r)} for r in (results or []) if r]
    elif hasattr(retriever, "get_relevant_documents"):
        docs = retriever.get_relevant_documents(query) or []
        processed = []
        for d in docs:
            score = None
            if isinstance(getattr(d, "metadata", None), dict):
                score = d.metadata.get("score") or d.metadata.get("similarity")
            if score_threshold is None or score is None or float(score) >= float(score_threshold):
                processed.append({"content": d.page_content, "metadata": getattr(d, "metadata", {})})
        results = processed[:top_k]
    else:
        raise TypeError("Retriever must implement .retrieve(...) or .get_relevant_documents(...).")

    if not results:
        return settings.GUARD_SENTENCE

    context = format_context(results, max_ctx_chars)
    print("retrieved Context >>>>>>>>>>>>>>", context[:600], "...\n")

    # 2) build prompts
    p1 = PRIMARY_PROMPT.format(context=context, question=query)
    p2 = BACKUP_PROMPT.format(context=context, question=query)
    print(f"PROMPT 1 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> {p1}")
    print(f"PROMPT 2 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> {p2}")

    if hasattr(llm, "max_new_tokens"): llm.max_new_tokens = max_new_tokens
    if hasattr(llm, "temperature"): llm.temperature = temperature
    if hasattr(llm, "return_full_text"): llm.return_full_text = False
    if stop and hasattr(llm, "stop"): llm.stop = stop

    def _invoke(prompt: str) -> str:
        try:
            out = llm.invoke(prompt)
        except TypeError:
            out = llm(prompt)
        if hasattr(out, "content"):
            out = out.content
        return (out or "").strip()

    # 4) try primary, then backup if empty/guarded
    ans = _invoke(p1)
    if not ans or ans.strip().lower() in {"i don't know.", settings.GUARD_SENTENCE.lower()}:
        ans = _invoke(p2)

    return ans or settings.GUARD_SENTENCE
