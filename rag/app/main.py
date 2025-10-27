from fastapi import FastAPI

def create_app() -> FastAPI:
    app = FastAPI(title="RAG API", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status":"ok"}
    
    @app.get("/")
    def root():
        return {"message":"RAG API is running"}
    
    return app

app = create_app()
