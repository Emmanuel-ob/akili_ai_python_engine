"""
Conversation-history helpers.

Engine Revamp Phase 3 removed this module's keyword classifier. Routing is now
decided by app/services/intent_router.py, which asks a model what the message
means instead of counting substring matches.

What was deleted and why it could not be repaired in place:

  SQL_KEYWORDS / FAQ_KEYWORDS  Routing summed substring hits from two
      hand-written lists. "order" matched "in order to". "last" matched "last
      time". "contact" sat in the FAQ list while also being a handover
      keyword, so one word pulled the same message two ways at once. Adding
      word boundaries would have fixed the substring cases and left the
      deeper problem: word presence is not meaning.

  FOLLOWUP_SIGNALS  A list of phrases like "which ones" used to detect a
      follow-up to a previous data answer. The router sees the recent
      conversation directly, so it resolves follow-ups from context rather
      than from a phrasebook.

extract_last_sql_context survives because it reads conversation METADATA that
the engine itself wrote, not the customer's wording. That is a lookup, not a
guess, and _handle_faq_query still depends on it.
"""

from typing import Any, Dict, List, Optional


class QueryClassifier:
    """Retained for extract_last_sql_context. The classifier is gone."""

    @staticmethod
    def extract_last_sql_context(
        recent_history: Optional[List[Dict[str, Any]]],
    ) -> Optional[Dict]:
        """Find the most recent assistant turn that returned rows.

        Used to answer a short follow-up ("which ones?") from the data already
        fetched, instead of running a second query. Reads metadata the engine
        attached to its own earlier reply, so there is no guessing involved.
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
