from rag.pipeline.LLM.grok_llm import get_grok_llm
from rag.core.config import settings

from langchain_core.messages import SystemMessage, HumanMessage

def RAG_Simple_Grok(query, 
                    retriever, 
                    top_k=settings.TOP_K, 
                    score_threshold=settings.SCORE_THRESHOLD, 
                    max_ctx_chars=settings.MAX_CTX_CHARS):
    # llm calling
    llm = get_grok_llm()

    # 1) retrieve
    results = retriever.retrieve(query, top_k=top_k, score_threshold=score_threshold)
    if not results:
        return "I don't know from the provided documents."

    # 2) build context (cap to avoid truncation)
    context = "\n\n".join(doc["content"] for doc in results)
    context = context[:max_ctx_chars]
    print(f"retrieved Context >>>>>>>>>>>>>> {context}")

    # 3) chat messages (system + user)
    system_msg = (
        "You are a careful medical assistant. Use ONLY the provided context. "
        'If the answer is not in the context, reply exactly: "I don\'t know from the provided documents."'
    )
    user_msg = f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
    print(f"\n prompt >>>>>>>>>>>>>>>>\n {user_msg} ")

    # 4) invoke Grok
    resp = llm.invoke([SystemMessage(content=system_msg), HumanMessage(content=user_msg)])
    text = getattr(resp, "content", resp)
    return (text or "").strip() or "I don't know from the provided documents."
