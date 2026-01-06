
import time
from app.core.llm import classify, answer, generate_sql
from app.db.vector_store import search
from app.rag.sql_executor import execute
from app.analytics.logger import log

SYSTEM = "Answer using provided context only."

async def handle_query(tenant_id: str, query: str) -> str:
    start = time.time()
    route = classify(query)
    ctx = []

    if route in ("VECTOR","HYBRID"):
        ctx.append("\n".join(search(tenant_id, query)))

    if route in ("SQL","HYBRID"):
        sql = generate_sql(query)
        ctx.append(str(execute(sql)))

    response = answer(f"""{SYSTEM}

Context:
{ctx}

Question:
{query}
""")

    log(tenant_id, query, route, int((time.time()-start)*1000))
    return response
