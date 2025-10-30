from fastapi import APIRouter
from rag.api.schemas.models import StatsResponse
from rag.api.services.components import get_vector_store
from rag.core.config import settings

router = APIRouter(prefix="/v1")

@router.get("/stats", response_model=StatsResponse)
def stats():
    """
    return vector store collection name and number of stored chuns
    """
    vs = get_vector_store()
    count = None

    try:
        count = vs.collection.count()
    
    except Exception:
        count = None
    
    return StatsResponse(
        collecion=settings.COLLECTION_NAME,
        count=count
    )