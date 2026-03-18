import re
from typing import Dict, Any, Optional, List
from app.core.logging_config import logger
from app.services.value_detector import SchemaEnricher


class SQLGenerator:
    """Generates SQL queries from natural language with smart value detection"""

    def __init__(self, llm_service):
        self.llm = llm_service
        self.enricher = SchemaEnricher()

    async def generate_sql(
        self,
        question: str,
        schema: Dict[str, Any],
        database_type: str,
        customer_id: Optional[str] = None,
        is_authenticated: bool = False,
        # ── Point 2: allowlist enforcement ────────────────────────────────────
        selected_tables: Optional[List[str]] = None,
        table_classifications: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Generate SQL query from natural language with smart value detection.

        Point 2 additions:
        - selected_tables: tables the merchant explicitly allowed. SQL that references
          anything outside this list is rejected before execution.
        - table_classifications: {table: "selected"|"system"|"business"} — passed into
          the prompt so the LLM knows upfront which tables are off-limits.
        """
        enum_hints = self.enricher.extract_enum_values(schema)
        schema_text = self._format_schema_with_hints(schema, enum_hints)

        # ── Build classification context block for prompt ─────────────────────
        classification_block = self._format_classification_context(
            selected_tables, table_classifications
        )

        logger.info(f"Enum hints extracted: {enum_hints}")

        prompt = f"""You are a SQL generator. Convert natural language to SQL.

DATABASE TYPE: {database_type}

{classification_block}
SCHEMA (ALLOWED TABLES ONLY):
{schema_text}

IMPORTANT VALUE HINTS:
{self._format_enum_hints(enum_hints)}

USER QUESTION: "{question}"

USER CONTEXT:
- Authenticated: {is_authenticated}
- Customer ID: {customer_id or "None"}

CRITICAL RULES:
1. ONLY SELECT queries allowed
2. ALWAYS include "LIMIT 50" at the end
3. Use proper {database_type} syntax
4. ONLY query tables listed in ALLOWED TABLES above. Never reference system/framework tables.
5. **IMPORTANT**: When filtering by status/type columns, use the EXACT values shown in "VALUE HINTS" above
6. **IMPORTANT**: If you're unsure about a value, use flexible matching:
   - For PostgreSQL: Use ILIKE '%value%' for case-insensitive partial matching
   - For MySQL: Use LIKE '%value%' with LOWER()
   - Always combine exact match OR flexible match for better results
7. If user asks about "my" data and is authenticated, filter by customer_id
8. Table/column names MUST match schema exactly (including schema prefix if shown)
9. If the question cannot be answered using only the allowed tables, set safe=false

SMART VALUE MATCHING EXAMPLES:
- User asks "how many projects are done"
- Schema shows status default is 'completed'
- Good SQL: WHERE (status = 'completed' OR status ILIKE '%done%' OR status ILIKE '%complete%')

Return JSON format:
{{
    "sql": "SELECT COUNT(*) FROM my_portfolio.projects WHERE (status = 'completed' OR status ILIKE '%done%') LIMIT 50",
    "explanation": "Counting projects with completed status",
    "tables_used": ["my_portfolio.projects"],
    "requires_auth": false,
    "safe": true,
    "value_assumptions": ["Assuming 'done' maps to 'completed' based on schema default"]
}}

If query is unsafe or impossible, set "safe": false and explain why in "explanation".
"""

        try:
            response = await self.llm.generate_structured_response(prompt)

            if not response.get("sql"):
                raise ValueError("No SQL generated")

            # ── Point 2: Validate generated SQL against allowlist ─────────────
            if selected_tables:
                violation = self._check_table_violations(
                    response["sql"], selected_tables, table_classifications
                )
                if violation:
                    logger.warning(f"SQL validation blocked query: {violation}")
                    return {
                        "sql": "",
                        "explanation": violation,
                        "tables_used": [],
                        "requires_auth": False,
                        "safe": False,
                        "blocked_reason": "table_not_allowed",
                    }

            logger.info(f"Generated SQL: {response.get('sql')}")

            if response.get("value_assumptions"):
                logger.info(f"Value assumptions: {response.get('value_assumptions')}")

            return response

        except Exception as e:
            logger.error(f"SQL generation failed: {str(e)}")
            return {
                "sql": "",
                "explanation": f"Cannot generate query: {str(e)}",
                "tables_used": [],
                "requires_auth": False,
                "safe": False,
            }

    async def generate_sql_with_retry(
        self,
        question: str,
        schema: Dict[str, Any],
        database_type: str,
        customer_id: Optional[str] = None,
        is_authenticated: bool = False,
        previous_attempt: Optional[Dict] = None,
        selected_tables: Optional[List[str]] = None,
        table_classifications: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Generate SQL with automatic retry using flexible matching."""

        if not previous_attempt:
            return await self.generate_sql(
                question,
                schema,
                database_type,
                customer_id,
                is_authenticated,
                selected_tables,
                table_classifications,
            )

        logger.info("Retrying SQL generation with flexible matching")

        failed_sql = previous_attempt.get("sql", "")
        enum_hints = self.enricher.extract_enum_values(schema)
        schema_text = self._format_schema(schema)

        classification_block = self._format_classification_context(
            selected_tables, table_classifications
        )

        prompt = f"""The previous query returned 0 results:
{failed_sql}

Generate a MORE FLEXIBLE version of this query that's more likely to find results.

Original question: "{question}"

Database type: {database_type}

{classification_block}
Schema:
{schema_text}

VALUE HINTS:
{self._format_enum_hints(enum_hints)}

MAKE IT MORE FLEXIBLE BY:
1. Using ILIKE '%value%' (PostgreSQL) or LIKE '%value%' (MySQL) for partial text matching
2. Adding OR conditions to try multiple possible values
3. Consider common synonyms:
   - "done" → "completed", "finished", "complete"
   - "pending" → "processing", "in_progress", "active"
   - "failed" → "error", "cancelled", "rejected"
4. Remove overly restrictive filters if they might be causing 0 results
5. For boolean columns, try both true and false values

Still ONLY use tables from the ALLOWED TABLES list above.

Return JSON with the improved SQL.
"""

        try:
            response = await self.llm.generate_structured_response(prompt)

            # Validate retry result too
            if selected_tables and response.get("sql"):
                violation = self._check_table_violations(
                    response["sql"], selected_tables, table_classifications
                )
                if violation:
                    logger.warning(f"Retry SQL also blocked: {violation}")
                    return previous_attempt

            response["is_retry"] = True
            response["original_sql"] = failed_sql
            logger.info(f"Retry SQL generated: {response.get('sql')}")
            return response

        except Exception as e:
            logger.error(f"Retry SQL generation failed: {str(e)}")
            return previous_attempt

    # ─────────────────────────────────────────────────────────────────────────
    # Point 2: Table validation
    # ─────────────────────────────────────────────────────────────────────────

    def _check_table_violations(
        self,
        sql: str,
        selected_tables: List[str],
        table_classifications: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        """
        Parse table references out of a SQL string and verify every one is
        in the selected_tables allowlist.

        Returns an error message string if a violation is found, else None.

        Handles:
        - Fully qualified names:  my_portfolio.projects
        - Bare names:             projects  (matched against schema-suffix of selected tables)
        - Case differences
        """
        referenced = self._extract_table_references(sql)

        if not referenced:
            return None  # couldn't parse — let validator catch it downstream

        # Build lookup: bare name → full selected name, and full name set
        selected_lower = {t.lower() for t in selected_tables}
        selected_bare_lower = {t.lower().split(".")[-1]: t for t in selected_tables}

        for ref in referenced:
            ref_lower = ref.lower()
            ref_bare = ref_lower.split(".")[-1]

            # Check full match first
            if ref_lower in selected_lower:
                continue

            # Check bare name match (LLM dropped schema prefix)
            if ref_bare in selected_bare_lower:
                continue  # valid — just missing prefix, executor will handle it

            # Not in allowlist — figure out why for a good error message
            if table_classifications:
                classification = table_classifications.get(
                    ref
                ) or table_classifications.get(
                    next(
                        (k for k in table_classifications if k.lower() == ref_lower), ""
                    ),
                    "",
                )
                if classification == "system":
                    return (
                        f"I can't query system table '{ref}' — it contains framework "
                        f"internals, not business data."
                    )

            return (
                f"Table '{ref}' is not in the list of analysed tables. "
                f"I can only query: {', '.join(selected_tables)}."
            )

        return None

    def _extract_table_references(self, sql: str) -> List[str]:
        """
        Extract table names from SQL using regex.
        Covers FROM, JOIN, UPDATE, INTO patterns.
        Returns a deduplicated list of raw references as they appear in the SQL.
        """
        # Remove string literals to avoid false positives like 'FROM the store'
        clean = re.sub(r"'[^']*'", "''", sql)
        clean = re.sub(r'"[^"]*"', '""', clean)

        # Match schema.table or bare table after FROM/JOIN/INTO/UPDATE keywords
        pattern = re.compile(
            r'\b(?:FROM|JOIN|UPDATE|INTO)\s+(["\w]+(?:\.["\w]+)?)',
            re.IGNORECASE,
        )

        refs = []
        for match in pattern.finditer(clean):
            name = match.group(1).strip('"').strip("`")
            if name.upper() not in ("SELECT", "WHERE", "SET", "VALUES"):
                refs.append(name)

        return list(dict.fromkeys(refs))  # deduplicate preserving order

    # ─────────────────────────────────────────────────────────────────────────
    # Point 2: Classification context block for prompt
    # ─────────────────────────────────────────────────────────────────────────

    def _format_classification_context(
        self,
        selected_tables: Optional[List[str]],
        table_classifications: Optional[Dict[str, str]],
    ) -> str:
        """
        Builds the ALLOWED TABLES section injected at the top of every prompt.
        Explicitly lists system tables as forbidden so the LLM doesn't guess.
        """
        if not selected_tables:
            return ""

        lines = ["ALLOWED TABLES (the ONLY tables you may query):"]
        for t in selected_tables:
            lines.append(f"  ✓ {t}")

        if table_classifications:
            system_tables = [
                t for t, cls in table_classifications.items() if cls == "system"
            ]
            if system_tables:
                lines.append("")
                lines.append("FORBIDDEN — system/framework tables (never query these):")
                for t in system_tables[:10]:  # cap display length
                    lines.append(f"  ✗ {t}")
                if len(system_tables) > 10:
                    lines.append(f"  ... and {len(system_tables) - 10} more")

        lines.append("")
        return "\n".join(lines) + "\n"

    # ─────────────────────────────────────────────────────────────────────────
    # Schema formatting helpers (unchanged from original)
    # ─────────────────────────────────────────────────────────────────────────

    def _format_schema_with_hints(
        self, schema: Dict[str, Any], enum_hints: Dict[str, List[str]]
    ) -> str:
        lines = []

        for table_name, table_info in schema.items():
            lines.append(f"\n{'=' * 60}")
            lines.append(f"Table: {table_name}")
            lines.append(f"Entity Type: {table_info.get('entity_type', 'unknown')}")
            lines.append(f"{'=' * 60}")

            columns = table_info.get("columns", [])
            lines.append("\nColumns:")
            for col in columns[:20]:
                col_name = col.get("name", "")
                col_type = col.get("type", "")
                nullable = col.get("nullable", True)
                default = col.get("default", "")

                col_line = f"  • {col_name} ({col_type})"

                if not nullable:
                    col_line += " NOT NULL"
                if default:
                    col_line += f" [default: {default}]"

                enum_key = f"{table_name}.{col_name}"
                if enum_key in enum_hints and enum_hints[enum_key]:
                    col_line += f" [possible values: {', '.join(enum_hints[enum_key])}]"

                lines.append(col_line)

            fks = table_info.get("foreign_keys", [])
            if fks:
                lines.append("\nForeign Keys:")
                for fk in fks:
                    lines.append(f"  • {fk['column']} → {fk['references_table']}")

            customer_id = table_info.get("customer_identifier")
            if customer_id:
                lines.append(f"\nCustomer Identifier: {customer_id}")

        return "\n".join(lines)

    def _format_enum_hints(self, enum_hints: Dict[str, List[str]]) -> str:
        if not enum_hints:
            return "No specific value hints available from schema defaults"

        lines = ["Columns with known values:"]
        for key, values in enum_hints.items():
            if values:
                lines.append(f"  • {key}: {', '.join(values)}")

        if len(lines) == 1:
            return "No specific value hints available from schema defaults"

        return "\n".join(lines)

    def _format_schema(self, schema: Dict[str, Any]) -> str:
        lines = []

        for table_name, table_info in schema.items():
            lines.append(f"\nTable: {table_name}")
            lines.append(f"Entity Type: {table_info.get('entity_type', 'unknown')}")

            columns = table_info.get("columns", [])
            lines.append("Columns:")
            for col in columns[:15]:
                col_name = col.get("name", "")
                col_type = col.get("type", "")
                lines.append(f"  - {col_name} ({col_type})")

            fks = table_info.get("foreign_keys", [])
            if fks:
                lines.append("Foreign Keys:")
                for fk in fks:
                    lines.append(f"  - {fk['column']} → {fk['references_table']}")

        return "\n".join(lines)
