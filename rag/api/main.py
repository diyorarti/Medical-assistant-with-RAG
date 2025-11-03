from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from rag.api.routers import health, stats, index, upload, query, delete
from rag.core.config import settings

from pathlib import Path

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

    @app.get("/v1/debug/ls")
    def debug_ls():
        p = Path(settings.DATA_DIR)
        files = [str(x) for x in p.glob("**/*.pdf")] if p.exists() else []
        return {
            "DATA_DIR": str(settings.DATA_DIR),
            "PERSIST_DIRECTORY_VS": str(settings.PERSIST_DIRECTORY_VS),
            "exists": p.exists(),
            "pdf_count": len(files),
            "pdfs": files[:20],  # first few
        }

    return app

app = create_app()