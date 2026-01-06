
import sqlparse
from sqlalchemy import text
from app.db.pg import get_engine

FORBIDDEN = {"insert","update","delete","drop","alter"}

def execute(sql: str):
    if not sql.lower().startswith("select"):
        raise ValueError("Only SELECT allowed")

    tokens = {t.value.lower() for t in sqlparse.parse(sql)[0].tokens if t.value}
    if tokens & FORBIDDEN:
        raise ValueError("Forbidden SQL")

    with get_engine().connect() as c:
        return c.execute(text(sql)).fetchmany(50)
