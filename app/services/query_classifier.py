from typing import Literal, List, Dict, Any, Optional
from app.core.logging_config import logger

QueryRoute = Literal["FAQ", "SQL", "HYBRID"]


class QueryClassifier:
    """
    Routes user queries to FAQ, SQL, or HYBRID.

    Point 3 upgrade: context-aware routing.
    Receives recent conversation history so it can detect follow-up
    questions that reference a previous SQL result — even when the
    follow-up message itself contains no SQL keywords.
    """

    SQL_KEYWORDS = [
        "how many",
        "count",
        "total",
        "sum",
        "average",
        "avg",
        "show me",
        "list",
        "find",
        "get",
        "fetch",
        "retrieve",
        "recent",
        "last",
        "latest",
        "status of",
        "order",
        "invoice",
        "transaction",
        "compare",
        "breakdown",
        "distribution",
        "trend",
        "top",
        "bottom",
        "most",
        "least",
        "highest",
        "lowest",
        "which ones",
        "which projects",
        "what projects",
        "what orders",
        "who has",
        "who are",
        "when was",
        "when did",
    ]

    FAQ_KEYWORDS = [
        "how to",
        "what is",
        "policy",
        "refund",
        "return",
        "shipping",
        "contact",
        "hours",
        "location",
        "about",
        "explain",
        "describe",
        "tell me about",
        "what does",
    ]

    # Phrases that signal a follow-up to a previous data result
    FOLLOWUP_SIGNALS = [
        "which ones",
        "show me those",
        "can you show",
        "list them",
        "tell me more",
        "more details",
        "what about",
        "and what",
        "any of those",
        "give me two",
        "give me some",
        "pick two",
        "which of those",
        "from those",
        "of those",
        "among those",
        "from the",
        "of the",
        "break it down",
        "which one",
        "now show",
        "now list",
        "also show",
        "what are they",
        "can you tell me which",
        "i want to see",
        "let me see",
    ]

    @staticmethod
    def classify(
        query: str,
        has_database: bool = False,
        recent_history: Optional[List[Dict[str, Any]]] = None,
    ) -> QueryRoute:
        """
        Classify query into FAQ, SQL, or HYBRID.

        Args:
            query:          User's question
            has_database:   Whether chatbot has database connected
            recent_history: Last N messages [{role, message, metadata}]
                            Used to detect follow-ups to SQL results.
        """
        if not has_database:
            return "FAQ"

        query_lower = query.lower().strip()

        sql_score = sum(1 for kw in QueryClassifier.SQL_KEYWORDS if kw in query_lower)
        faq_score = sum(1 for kw in QueryClassifier.FAQ_KEYWORDS if kw in query_lower)

        # ── Point 3: Context boost ────────────────────────────────────────────
        # If the previous assistant turn was a SQL/hybrid result, check whether
        # this message is a natural follow-up that should also hit the database.
        context_sql_boost = 0
        last_sql_data = None

        if recent_history:
            last_assistant = next(
                (m for m in reversed(recent_history) if m.get("role") == "assistant"),
                None,
            )
            if last_assistant:
                meta = last_assistant.get("metadata") or {}
                last_type = meta.get("type", "")

                if last_type in ("sql", "hybrid"):
                    # Previous turn had data — check if this is a follow-up
                    is_followup = any(
                        sig in query_lower for sig in QueryClassifier.FOLLOWUP_SIGNALS
                    )
                    # Also boost for very short messages (≤6 words) after SQL
                    # e.g. "which ones?", "show me those", "and fastapi?"
                    is_short_followup = len(query_lower.split()) <= 6

                    if is_followup or (
                        is_short_followup and sql_score == 0 and faq_score == 0
                    ):
                        context_sql_boost = 2
                        logger.info(
                            f"Context boost applied: previous turn was {last_type}, "
                            f"detected follow-up query"
                        )

                    # Store last SQL data for context injection (used by chat.py)
                    if meta.get("row_count", 0) > 0:
                        last_sql_data = meta

        effective_sql_score = sql_score + context_sql_boost

        logger.info(
            f"Query classification - SQL score: {sql_score} (+{context_sql_boost} context boost = {effective_sql_score}), "
            f"FAQ score: {faq_score}"
        )

        if effective_sql_score > 0 and faq_score > 0:
            return "HYBRID"
        elif effective_sql_score > 0:
            return "SQL"
        else:
            return "FAQ"

    @staticmethod
    def extract_last_sql_context(
        recent_history: Optional[List[Dict[str, Any]]],
    ) -> Optional[Dict]:
        """
        Extract the most recent SQL result metadata from history.
        Used by chat.py to inject data context into FAQ follow-up responses.
        """
        if not recent_history:
            return None

        for msg in reversed(recent_history):
            if msg.get("role") != "assistant":
                continue
            meta = msg.get("metadata") or {}
            if meta.get("type") in ("sql", "hybrid") and meta.get("row_count", 0) > 0:
                return {
                    "type": meta.get("type"),
                    "row_count": meta.get("row_count"),
                    "sql": meta.get("sql", ""),
                    "explanation": meta.get("explanation", ""),
                    "assistant_response": msg.get("message", ""),
                }

        return None
