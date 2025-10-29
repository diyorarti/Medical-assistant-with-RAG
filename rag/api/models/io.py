# api/models/io.py
from __future__ import annotations
from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from rag.core.config import settings

class IndexResponse(BaseModel):
    added: int
    total_in_collection: Optional[int] = None
    message: str

class UploadResponse(BaseModel):
    saved_files: List[str]
    added: int
    total_in_collection: Optional[int] = None
    message: str

class Citation(BaseModel):
    source_name: Optional[str] = None
    source_file: Optional[str] = None
    page: Optional[int] = None
    similarity: Optional[float] = None
    id: Optional[str] = None

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    provider: Literal["hf", "grok"] = "hf"
    top_k: int = Field(default=settings.TOP_K, ge=1, le=20)
    score_threshold: float = Field(default=settings.SCORE_THRESHOLD, ge=0.0, le=1.0)
    max_ctx_chars: int = Field(default=settings.MAX_CTX_CHARS, ge=500, le=50000)

class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation] = []
    used_provider: Literal["hf", "grok"]

class StatsResponse(BaseModel):
    collection: str
    count: Optional[int] = None

class StatsResponse(BaseModel):
    collection: str
    count: Optional[int] = None

class DeleteResponse(BaseModel):
    deleted_source: str
    message: str