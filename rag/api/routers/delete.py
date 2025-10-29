from pathlib import Path
from fastapi import APIRouter, HTTPException, Query
from rag.api.models.io import DeleteResponse
from rag.api.services.components import get_vector_store

router = APIRouter(prefix="/v1")

@router.delete("/delete", response_model=DeleteResponse)
def delete_by_source(source: str = Query(..., description="Path to the PDF to remove")):
    p = Path(source).resolve()
    if not p.exists():
        # we can still allow deletion by path even if file missing; adjust if you prefer strict
        # raise HTTPException(status_code=404, detail=f"File not found: {p}")
        pass
    vs = get_vector_store()
    try:
        vs.delete_by_source(str(p))
        msg = f"Deleted items where source_file == {p}"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {e}")
    return DeleteResponse(deleted_source=str(p), message=msg)
