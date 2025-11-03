from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from rag.api.routers import health, stats, index, upload, query, delete

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
    app.include_router(index.router)
    app.include_router(upload.router)
    app.include_router(query.router)
    app.include_router(delete.router)

    @app.on_event("startup")
    def _warm():
        # load SentenceTransformer + Chroma once so requests are fast
        from rag.api.services.components import ensure_components
        ensure_components()

    return app

app = create_app()