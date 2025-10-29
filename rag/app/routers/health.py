# api/routers/health.py
from fastapi import APIRouter
from rag.core.config import settings

router = APIRouter(tags=["health"])

@router.get("/health")
def health():
    """
    Liveness probe: returns 200 if the process is alive.
    Keep this endpoint trivial—no I/O, no dependencies.
    """
    return {"status": "ok"}

@router.get("/ready")
def ready():
    """
    Readiness probe: light, but proves we can import config and read settings.
    Later you can extend this to check vector store counts, etc.
    """
    return {
        "status": "ready",
        "collection": settings.COLLECTION_NAME,
        "persist_dir": str(settings.PERSIST_DIRECTORY_VS),
    }
