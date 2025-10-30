from fastapi import APIRouter, Query, HTTPException, Depends
from pathlib import Path

from rag.api.schemas.models import DeleteResponse
from rag.api.services.components import get_vector_store
from rag.core.security import verify_api_key

router = APIRouter(prefix="/v1")

@router.delete("/delete", response_model=DeleteResponse, dependencies=[Depends(verify_api_key)])
def delete_by_source(source:str=Query(..., description="Path to the PDF to remove")):
    p = Path(source).resolve()
    if not p.exists():
        pass

    vs = get_vector_store()

    try:
        vs.delete_by_source(str(p))
        msg = f"Deleted items where source_file == {p}"
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {e}")
    
    return DeleteResponse(
        deleted_source=str(p),
        message=msg
    )