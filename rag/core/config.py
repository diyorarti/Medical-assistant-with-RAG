from __future__ import annotations
from pathlib import Path
import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv
load_dotenv()

# project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    # loading envs
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # data loader configs 
    DATA_DIR:Path=PROJECT_ROOT / "data"
    MIN_CHARS:int=30

    #chunker func configs
    #chunking
    CHUNK_SIZE:int=1000
    CHUNK_OVERLAP:int=120
    MIN_CHUNK_CHARS:int=80
    # tokenization
    TIKTOKEN_ENCODING:str="cl100k_base"
    USE_TIKTOKEN: bool = True
    #stabel chunk-id 
    CHUNK_ID_PREFIX_LEN: int = 12
    #splitter behavior
    CHUNK_SEPARATORS: list[str] = ["\n\n", "\n", " ", ""]
    KEEP_SEPARATOR: bool = False

    #embedder class configs
    EMBEDDER_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    NORMALIZE:bool=True
    BATCH_SIZE:int=64

    # Vector Store configs
    COLLECTION_NAME:str = "pdf_documents"
    PERSIST_DIRECTORY_VS:Path = PROJECT_ROOT / "data" / "vector_store"

    # retriever configs
    TOP_K:int=5
    SCORE_THRESHOLD:float=0.35

    #GROK LLM configs
    GROK_API_KEY: str | None = os.getenv("GROK_API_KEY")
    GROK_MODEL: str = "grok-4" 
    GROK_TEMPERATURE:float=0.1
    GROK_MAX_TOKENS:int=512

    MAX_CTX_CHARS:int= 8000

    # HF Endpoint configs
    HF_TOKEN: str = os.getenv("HF_TOKEN")
    HF_ENDPOINT_URL: str = "https://u8hxw751s4q6eqhs.eu-west-1.aws.endpoints.huggingface.cloud"
    HF_MAX_TOKENS:int=512
    HF_TEMPERATURE:float=0.1
    HF_RETURN_FULL_TEXT:bool=False
    GUARD_SENTENCE:str = "I don't know from the provided documents."
    TASK:str="text-generation"

    # api configs
    API_KEY:str| None = os.getenv("API_KEY")


settings = Settings()