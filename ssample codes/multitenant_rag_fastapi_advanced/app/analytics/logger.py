
from sqlalchemy import text
from app.db.pg import get_engine

def log(tenant_id, query, route, latency):
    with get_engine().connect() as c:
        c.execute(text("""
        INSERT INTO analytics_logs (tenant_id, query, route, latency_ms)
        VALUES (:t,:q,:r,:l)
        """), {
            "t": tenant_id, "q": query, "r": route, "l": latency
        })
