from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from rag.api.routers import health, stats, index, upload, query, delete

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

    return app

app = create_app()
