from __future__ import annotations
from pydantic import BaseModel
from typing import Optional

class StatsResponse(BaseModel):
    collecion:str
    count:Optional[int]=None

