
from sqlalchemy import text
from app.db.vector_store import search_vectors
from app.db.tenant_registry import get_engine, get_schema
from app.rag.sql_generator import generate_sql
from app.rag.sql_validator import validate_sql

def retrieve_vector_context(tenant_id: str, query: str):
    return search_vectors(tenant_id, query)

def retrieve_sql_context(tenant_id: str, query: str):
    schema = get_schema(tenant_id)
    sql = generate_sql(query, schema)
    validate_sql(sql)

    engine = get_engine(tenant_id)
    with engine.connect() as conn:
        rows = conn.execute(text(sql)).fetchmany(50)

    return {"sql": sql, "rows": rows}
