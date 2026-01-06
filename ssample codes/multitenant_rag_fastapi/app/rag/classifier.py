
def classify_query(query: str) -> str:
    q = query.lower()
    if any(k in q for k in ["how long", "policy", "refund"]):
        return "VECTOR"
    if any(k in q for k in ["how many", "status", "count"]):
        return "SQL"
    return "HYBRID"
