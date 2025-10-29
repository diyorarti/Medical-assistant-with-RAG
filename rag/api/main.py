from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag.api.routers import health

def create_app() -> FastAPI:
    app = FastAPI(title="Med assistant by RAG API", version="1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router)

    return app

app = create_app()