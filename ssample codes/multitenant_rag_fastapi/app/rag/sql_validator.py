
import sqlparse

FORBIDDEN = {"insert", "update", "delete", "drop", "alter"}

def validate_sql(sql: str):
    if not sql.lower().startswith("select"):
        raise ValueError("Only SELECT statements allowed")

    parsed = sqlparse.parse(sql)[0]
    tokens = {t.value.lower() for t in parsed.tokens if t.value}

    if tokens & FORBIDDEN:
        raise ValueError("Forbidden SQL detected")

    if "limit" not in sql.lower():
        raise ValueError("LIMIT clause required")
