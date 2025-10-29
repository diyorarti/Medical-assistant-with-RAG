from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from rag.core.logging import get_logger

logger = get_logger(__name__)

def _rid(request: Request) -> str:
    return getattr(request.state, "request_id", "-")

async def http_exception_handler(request: Request, exc: HTTPException):
    rid = _rid(request)
    logger.warning(f"HTTPException[{exc.status_code}] rid={rid} detail={exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "request_id": rid},
    )

async def validation_exception_handler(request: Request, exc: ValidationError):
    rid = _rid(request)
    logger.warning(f"ValidationError rid={rid}: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "request_id": rid},
    )

async def unhandled_exception_handler(request: Request, exc: Exception):
    rid = _rid(request)
    logger.exception(f"UnhandledError rid={rid}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "request_id": rid},
    )
