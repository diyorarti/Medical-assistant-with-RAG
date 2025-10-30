from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag.api.routers import health, stats

def create_app():
    # app initialization
    app = FastAPI(title="Medical assistant by RAG api", version="0.1.0")

    # CORS 
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # routers 
    app.include_router(health.router)
    app.include_router(stats.router)

    return app

app = create_app()