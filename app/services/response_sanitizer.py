"""
Response Sanitizer — External User Protection
==============================================

Ensures that customer-facing (external) responses never expose:
  - Database table or column names
  - Raw SQL queries
  - Schema information
  - Internal metadata like row counts with technical context
  - Error details that could aid enumeration

Internal (dashboard) users receive the full unsanitized response.
"""

import re
from typing import Dict, Any, List
from app.schemas.chat import ChatResponse, SourceDocument
from app.core.logging_config import logger


class ResponseSanitizer:
    """
    Sanitizes ChatResponse objects for external (widget) users.

    Usage:
        sanitized = ResponseSanitizer.sanitize(response)
    """

    # Patterns that suggest technical/schema leakage in response text
    _SQL_PATTERNS = [
        r"\bSELECT\b.{0,200}\bFROM\b",  # SELECT ... FROM
        r"\bWHERE\b\s+\w+\s*[=<>!]",  # WHERE col = ...
        r"\bJOIN\b\s+\w+",  # JOIN table
        r"\bGROUP BY\b",  # GROUP BY
        r"\bORDER BY\b",  # ORDER BY
        r"\bLIMIT\s+\d+",  # LIMIT 50
        r"```sql",  # SQL code blocks
        r"db\.\w+\.find\(",  # MongoDB queries
    ]

    # Metadata keys that must never reach external users
    _BLOCKED_METADATA_KEYS = {
        "sql",
        "schema",
        "table_name",
        "column_names",
        "tables_used",
        "database_type",
        "validation_errors",
        "value_assumptions",
    }

    # Metadata keys that are safe to pass through to external users
    _ALLOWED_METADATA_KEYS = {
        "type",
        "confidence",
        "source",
        "handover_requested",
        "handover_active",
        "fallback",
        "error",
        "subscription",
        "usage_warning",
    }

    @classmethod
    def sanitize(cls, response: ChatResponse) -> ChatResponse:
        """
        Sanitize a ChatResponse for external user consumption.

        - Strips SQL/schema from response text
        - Removes sensitive metadata keys
        - Cleans source documents (no table names from schema)
        - Keeps the response natural and helpful
        """
        sanitized_text = cls._sanitize_text(response.text)
        sanitized_metadata = cls._sanitize_metadata(response.metadata)
        sanitized_sources = cls._sanitize_sources(response.sources)

        if sanitized_text != response.text:
            logger.info("Response text sanitized for external user")

        return ChatResponse(
            text=sanitized_text,
            sources=sanitized_sources,
            metadata=sanitized_metadata,
        )

    @classmethod
    def _sanitize_text(cls, text: str) -> str:
        """Remove SQL queries and technical jargon from response text."""
        if not text:
            return text

        sanitized = text

        # Remove any raw SQL blocks
        for pattern in cls._SQL_PATTERNS:
            if re.search(pattern, sanitized, re.IGNORECASE):
                # Remove the matched SQL segment
                sanitized = re.sub(
                    pattern, "[data retrieved]", sanitized, flags=re.IGNORECASE
                )

        # Remove markdown SQL code blocks entirely
        sanitized = re.sub(
            r"```sql.*?```", "", sanitized, flags=re.DOTALL | re.IGNORECASE
        )

        # Remove any lines that look like raw column/table references
        # e.g. "From the orders.id column..." → strip technical part
        sanitized = re.sub(
            r"\b(table|column|schema|database|query)\s*[:\-]\s*`?\w+`?",
            "",
            sanitized,
            flags=re.IGNORECASE,
        )

        # Clean up any double spaces or blank lines created by removals
        sanitized = re.sub(r"\n{3,}", "\n\n", sanitized)
        sanitized = re.sub(r"  +", " ", sanitized)

        return sanitized.strip()

    @classmethod
    def _sanitize_metadata(cls, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Keep only safe metadata keys for external users.

        Internal metadata (sql queries, schema info, etc.) is stripped.
        The response type is preserved but generalised if it was 'sql'.
        """
        if not metadata:
            return {}

        sanitized = {}

        for key, value in metadata.items():
            if key in cls._ALLOWED_METADATA_KEYS:
                sanitized[key] = value

        # Normalise the type so frontend doesn't expose "sql" to customers
        if "type" in sanitized:
            if sanitized["type"] in ("sql", "hybrid"):
                sanitized["type"] = "data_response"

        # Keep row_count but rename it to something neutral if present
        if "row_count" in metadata and metadata.get("row_count", 0) > 0:
            # Don't expose raw counts — let the LLM text convey the result
            pass

        return sanitized

    @classmethod
    def _sanitize_sources(cls, sources: List[SourceDocument]) -> List[SourceDocument]:
        """
        Clean source documents for external users.

        SQL sources (which contain the raw query) are removed entirely.
        FAQ/knowledge base sources are kept but table names are genericised.
        """
        if not sources:
            return []

        clean_sources = []

        for source in sources:
            # Remove SQL result sources entirely from external responses
            if source.doc_id == "sql_result":
                continue

            # Keep FAQ sources but strip any raw SQL from the text snippet
            clean_text = cls._sanitize_text(source.text)

            # Genericise the table field (don't expose DB table names)
            clean_table = source.table
            if clean_table and clean_table not in ("FAQ", "Knowledge Base", "Document"):
                # It looks like a real DB table name — replace it
                clean_table = "Knowledge Base"

            clean_sources.append(
                SourceDocument(
                    doc_id=source.doc_id,
                    text=clean_text,
                    confidence=source.confidence,
                    table=clean_table,
                )
            )

        return clean_sources
