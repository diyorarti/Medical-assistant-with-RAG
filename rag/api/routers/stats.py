from fastapi import APIRouter
from rag.core.config import settings
from rag.api.models.io import StatsResponse
from rag.api.services.components import get_vector_store

router = APIRouter(prefix="/v1")

@router.get("/stats", response_model=StatsResponse)
def stats():
    vstore = get_vector_store()
    count = None
    try:
        count = vstore.collection.count()
    except Exception:
        pass
    return StatsResponse(collection=settings.COLLECTION_NAME, count=count)
