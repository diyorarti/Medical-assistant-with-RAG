from __future__ import annotations
from pydantic import BaseModel
from typing import Optional, List

class StatsResponse(BaseModel):
    collecion:str
    count:Optional[int]=None

class IndexResponse(BaseModel):
    added:int
    total_in_collection:Optional[int] = None
    message:str

class UploadResponse(BaseModel):
    saved_files:List[str]
    added:int
    total_in_collection: Optional[int] = None
    message: str