from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.middleware import Middleware
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.middleware.cors import CORSMiddleware

from rag.api.routers import health, index, upload, query, stats, delete
from rag.core.logging import init_logging
from rag.core.middleware import RequestIDMiddleware, AccessLogMiddleware
from rag.core.errors import (
    http_exception_handler,
    validation_exception_handler,
    unhandled_exception_handler,
)

def create_app() -> FastAPI:
    app = FastAPI(title="Med assistant by RAG API", version="1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(AccessLogMiddleware)

    app.include_router(health.router)
    app.include_router(index.router)
    app.include_router(upload.router)
    app.include_router(query.router)
    app.include_router(stats.router)
    app.include_router(delete.router)

    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
    
    return app

app = create_app()