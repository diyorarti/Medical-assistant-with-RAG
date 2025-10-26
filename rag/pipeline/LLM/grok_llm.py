from langchain_xai import ChatXAI
from rag.core.config import settings

def get_grok_llm() -> ChatXAI:
    if not settings.GROK_API_KEY:
        raise RuntimeError(
            "GROK_API_KEY is not set. Add it to your environment or .env file."
        )
    return ChatXAI(
        model=settings.GROK_MODEL,
        xai_api_key=settings.GROK_API_KEY,
        temperature=settings.GROK_TEMPERATURE,
        max_tokens=settings.GROK_MAX_TOKENS,
    )
