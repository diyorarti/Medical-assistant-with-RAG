from langchain_huggingface import HuggingFaceEndpoint
from rag.core.config import settings

def get_hf_llm():
    llm_hf = HuggingFaceEndpoint(
        endpoint_url=settings.HF_ENDPOINT_URL,
        huggingfacehub_api_token=settings.HF_TOKEN,
        task=settings.TASK,
        max_new_tokens=settings.HF_MAX_TOKENS,
        temperature=settings.HF_TEMPERATURE,
        return_full_text=settings.HF_RETURN_FULL_TEXT,
    )
    return llm_hf