
from app.rag.classifier import classify_query
from app.rag.retriever import retrieve_vector_context, retrieve_sql_context
from app.core.llm import generate_answer

SYSTEM_PROMPT = (
    "You are a customer support assistant. "
    "Only answer using provided context. "
    "If unsure, say you don't know."
)

async def handle_query(tenant_id: str, query: str) -> str:
    route = classify_query(query)
    context_parts = []

    if route in ("VECTOR", "HYBRID"):
        vector_ctx = retrieve_vector_context(tenant_id, query)
        context_parts.append("FAQ:\n" + "\n".join(vector_ctx))

    if route in ("SQL", "HYBRID"):
        sql_ctx = retrieve_sql_context(tenant_id, query)
        context_parts.append(f"DB RESULT:\n{sql_ctx}")

    context = "\n\n".join(context_parts)

    return generate_answer(
        system_prompt=SYSTEM_PROMPT,
        context=context,
        query=query
    )
