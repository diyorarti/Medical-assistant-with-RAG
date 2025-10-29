from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import health

def create_app() -> FastAPI:
    app = FastAPI(title="Medical RAG API", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )

    app.include_router(health.router)

    @app.get("/")
    def root():
        return {"message":"RAG API is up. See /docs"}
    
    return app

app = create_app()