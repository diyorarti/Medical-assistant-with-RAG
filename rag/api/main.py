from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from rag.api.routers import health, stats, index, upload, query, delete
from rag.api.services.components import ensure_components
from rag.core.config import settings

from pathlib import Path
import shutil

def create_app():
    app = FastAPI(title="Medical assistant by RAG api", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    def root():
        return {"ok": True, "docs": "/docs"}
    
    # routers
    app.include_router(health.router)
    app.include_router(stats.router)
    app.include_router(index.router)
    app.include_router(upload.router)
    app.include_router(query.router)
    app.include_router(delete.router)

    # ✅ Register as startup hook so it runs on every boot
    @app.on_event("startup")
    def _warm_and_seed():
        # warm components (loads SentenceTransformer + Chroma once)
        ensure_components()

        # copy default PDFs from image (/app/data) -> persistent DATA_DIR (e.g., /app/storage)
        image_data = Path("/app/data")       # baked from repo
        data_dir   = Path(settings.DATA_DIR) # should be /app/storage via env var
        data_dir.mkdir(parents=True, exist_ok=True)

        for name in [
            "Aging_natural_or_disease.pdf",
            "Genes_and_Disease.pdf",
            "basic_epidemiology.pdf",
        ]:
            src = image_data / name
            dst = data_dir / name
            if src.exists() and not dst.exists():
                shutil.copy(src, dst)

    # (optional) quick debug listing you can remove later
    @app.get("/v1/debug/ls")
    def debug_ls():
        p = Path(settings.DATA_DIR)
        files = [str(x) for x in p.glob("**/*.pdf")] if p.exists() else []
        return {
            "DATA_DIR": str(settings.DATA_DIR),
            "PERSIST_DIRECTORY_VS": str(settings.PERSIST_DIRECTORY_VS),
            "exists": p.exists(),
            "pdf_count": len(files),
            "pdfs": files[:20],
        }

    return app

app = create_app()
