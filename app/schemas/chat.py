from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class HistoryMessage(BaseModel):
    role: str  # 'user' or 'assistant'
    message: str
    timestamp: str


class ChatbotConfig(BaseModel):
    personality: Optional[str] = None
    fallback_message: Optional[str] = None
    max_conversation_length: Optional[int] = 50


class ChatRequest(BaseModel):
    business_id: str
    chatbot_id: str
    message: str
    session_id: str
    history: List[HistoryMessage] = []
    chatbot_config: Dict[str, Any] = {}
    connection_id: Optional[str] = None  
    customer_data: Optional[Dict[str, Any]] = None  
    is_authenticated: bool = False  


class SourceDocument(BaseModel):
    doc_id: str
    text: str
    confidence: float
    table: Optional[str] = None


class ChatResponse(BaseModel):
    text: str
    sources: List[SourceDocument] = []
    metadata: Dict[str, Any] = {}