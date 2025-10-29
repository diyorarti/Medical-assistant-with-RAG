# rag/api/routers/upload.py
from __future__ import annotations
from pathlib import Path
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException
from rag.core.config import settings
from rag.api.models.io import UploadResponse  # or use IndexResponse
from rag.api.services.indexing import build_index  # reuses your existing pipeline

router = APIRouter(prefix="/v1")

@router.post("/upload", response_model=UploadResponse)
async def upload_and_index(files: List[UploadFile] = File(...)):
    """
    Accept one or more PDFs, save them under data/uploads, then index them.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    # Only allow PDFs
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Only PDF files are allowed. Got: {f.filename}")

    upload_dir = Path(settings.DATA_DIR) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: List[str] = []
    for f in files:
        dest = upload_dir / f.filename
        # If you want to avoid collisions, you can uniquify here.
        content = await f.read()
        with dest.open("wb") as out:
            out.write(content)
        saved_paths.append(str(dest.resolve()))

    # Index only the uploads directory (fast and scoped)
    added, total = build_index(upload_dir)

    return UploadResponse(
        saved_files=saved_paths,
        added=added,
        total_in_collection=total,
        message="Uploaded & indexed.",
    )
