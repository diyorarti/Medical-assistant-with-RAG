from __future__ import annotations

from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from pathlib import Path

from rag.api.schemas.models import UploadResponse
from rag.core.config import settings
from rag.api.services.indexing import build_index

router = APIRouter(prefix="/v1")

@router.post("/upload", response_model=UploadResponse)
async def upload_and_index(files:List[UploadFile]=File(...)):
    """
    Upload extra knowledge bases (pdf files), save them under data/uploads folder.
    Then, Index them.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No file is provided")
    
    # only PDFs are allowed
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"ONLY PDF files are allowed. You Provided {f.filename}")
    
    upload_dir = Path(settings.DATA_DIR)/"uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved_paths:List[str] = []
    for f in files:
        dest = upload_dir / f.filename
        content = await f.read()
        with dest.open("wb") as out:
            out.write(content)
        saved_paths.append(str(dest.resolve()))
    
    added, total = build_index(upload_dir)

    return UploadResponse(
        saved_files=saved_paths,
        added=added,
        total_in_collection=total,
        message="Uploaded & Indexed !!!",
    )