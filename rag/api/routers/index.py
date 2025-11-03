# from fastapi import APIRouter, Query, HTTPException, Depends

# from pathlib import Path

# from rag.api.schemas.models import IndexResponse
# from rag.core.config import settings
# from rag.core.security import verify_api_key
# from rag.api.services.indexing import build_index


# router = APIRouter(prefix="/v1")

# @router.post("/index", response_model=IndexResponse, dependencies=[Depends(verify_api_key)])
# def index_corpus(data_dir:str | None = Query(default=None, description="Optional override for data dir")):
#     """
#     (Re)-indexing PDFs undex settings.DATA_DIR (or a provided directory).
#     """

#     dir_path = Path(data_dir) if data_dir else settings.DATA_DIR
#     if not dir_path.exists():
#         raise HTTPException(status_code=400, detail=f"data directory not found: {dir_path}")
#     added, total = build_index(dir_path)
#     return IndexResponse(
#         added=added,
#         total_in_collection=total,
#         message="Indexing complete",
#     )

from fastapi import APIRouter, Query, HTTPException, Depends, BackgroundTasks
from pathlib import Path
from rag.api.schemas.models import IndexResponse
from rag.core.config import settings
from rag.core.security import verify_api_key
from rag.api.services.indexing import build_index

router = APIRouter(prefix="/v1")

def _do_index(dir_path: Path):
    # long-running job
    build_index(dir_path)

@router.post("/index", response_model=IndexResponse, dependencies=[Depends(verify_api_key)])
def index_corpus(
    background: BackgroundTasks,
    data_dir: str | None = Query(default=None, description="Optional override for data dir")
):
    dir_path = Path(data_dir) if data_dir else settings.DATA_DIR
    if not dir_path.exists():
        raise HTTPException(status_code=400, detail=f"data directory not found: {dir_path}")

    # enqueue the heavy job so the request returns quickly
    background.add_task(_do_index, dir_path)

    # We haven’t embedded yet, so report a “started” message.
    return IndexResponse(
        added=0,
        total_in_collection=None,
        message="Indexing started (running in background). Check /v1/stats for progress.",
    )


