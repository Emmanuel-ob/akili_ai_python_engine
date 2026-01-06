
from sqlalchemy import text
from app.db.pg import get_engine
from app.core.llm import embed

_embedding_cache = {}

def search(tenant_id: str, query: str):
    if query not in _embedding_cache:
        _embedding_cache[query] = embed(query)

    sql = '''
    SELECT content FROM faq_embeddings
    WHERE tenant_id=:tenant
    ORDER BY embedding <-> :vec
    LIMIT 5;
    '''
    with get_engine().connect() as c:
        r = c.execute(
            text(sql),
            {"tenant": tenant_id, "vec": _embedding_cache[query]}
        ).fetchall()
    return [x[0] for x in r]
