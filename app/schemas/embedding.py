# app/schemas/embedding.py
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


class UpsertResponse(BaseModel):
    success: bool
    message: str
    collection: str
    count: int


class HealthResponse(BaseModel):
    status: str
    service: str
