from langchain_huggingface import HuggingFaceEndpoint
from rag.core.config import settings

def get_hf_llm() -> HuggingFaceEndpoint:
    if not settings.HF_TOKEN:
        raise RuntimeError("HF_TOKEN isn't set. Add HF_TOKEN to your environment/.env.")
    return HuggingFaceEndpoint(
        endpoint_url=settings.HF_ENDPOINT_URL,
        huggingfacehub_api_token=settings.HF_TOKEN,
        task="text-generation",
        max_new_tokens=settings.HF_MAX_TOKENS,
        temperature=settings.HF_TEMPERATURE,
        return_full_text=settings.HF_RETURN_FULL_TEXT,
    )
