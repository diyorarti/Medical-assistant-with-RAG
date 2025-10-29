# api/routers/index.py
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query
from rag.core.config import settings
from rag.api.models.io import IndexResponse
from rag.api.services.indexing import build_index

router = APIRouter(prefix="/v1")

@router.post("/index", response_model=IndexResponse)
def index_corpus(data_dir: str | None = Query(default=None, description="Optional override for data dir")):
    """
    (Re)index PDFs under settings.DATA_DIR (or a provided directory).
    """
    dir_path = Path(data_dir) if data_dir else settings.DATA_DIR
    if not dir_path.exists():
        raise HTTPException(status_code=400, detail=f"Data directory not found: {dir_path}")

    added, total = build_index(dir_path)
    return IndexResponse(added=added, total_in_collection=total, message="Indexing complete.")
