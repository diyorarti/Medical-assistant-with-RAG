from fastapi import APIRouter

from rag.core.config import settings

router = APIRouter()

@router.get("/health")
def health():
    return {
        "status":True,
        "collection":settings.COLLECTION_NAME,
    }