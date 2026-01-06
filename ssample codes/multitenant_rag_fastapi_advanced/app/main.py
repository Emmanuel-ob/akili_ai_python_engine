
from fastapi import FastAPI, Depends
from app.schemas import ChatRequest, ChatResponse
from app.core.security import get_tenant_context
from app.rag.orchestrator import handle_query

app = FastAPI(title="Advanced Multi-Tenant RAG")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, tenant=Depends(get_tenant_context)):
    return ChatResponse(
        answer=await handle_query(tenant.tenant_id, request.query)
    )
