from pydantic import BaseModel
from typing import Optional

class ChangeRequest(BaseModel):
    session_id: Optional[str] = None
    message : str

class ChangeResponse(BaseModel):
    session_id : Optional[str] = None
    response : str
    