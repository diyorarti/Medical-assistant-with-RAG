from fastapi import FastAPI

from rag.api.routers import health


def create_app():
    app = FastAPI(title="Medical assistant by RAG api", version="0.1.0")

    app.include_router(health.router)

    return app

app = create_app()