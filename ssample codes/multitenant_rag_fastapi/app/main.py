
from fastapi import FastAPI, Depends
from app.schemas import ChatRequest, ChatResponse
from app.core.security import get_tenant_context
from app.rag.orchestrator import handle_query

app = FastAPI(title="Multi-Tenant RAG Chatbot")

@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    tenant=Depends(get_tenant_context)
):
    answer = await handle_query(
        tenant_id=tenant.tenant_id,
        query=request.query
    )
    return ChatResponse(answer=answer)
