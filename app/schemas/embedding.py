from pydantic import BaseModel
from typing import List, Dict, Any


class DocumentInput(BaseModel):
    doc_id: str
    text: str
    data: Dict[str, Any]
    hash: str


class UpsertRequest(BaseModel):
    business_id: str
    chatbot_id: str
    provider: str = "openai"
    docs: List[DocumentInput]
    sync_run_id: str = None 
    

class UpsertResponse(BaseModel):
    success: bool
    message: str
    collection: str
    count: int
    failed_count: int = 0
    failed_items: List[Dict[str, Any]] = []
    processing_stats: Dict[str, int] = {}  


class HealthResponse(BaseModel):
    status: str
    service: str
