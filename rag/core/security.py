from fastapi import Header, HTTPException, status
from rag.core.config import settings

API_KEY = settings.API_KEY

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
