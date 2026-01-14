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
    ) -> Dict[str, Any]:
        """
        Generate SQL query from natural language with smart value detection

        This method:
        1. Extracts possible values from schema defaults
        2. Provides hints to the LLM about actual column values
        3. Generates accurate SQL with proper value matching

        Args:
            question: User's natural language question
            schema: Database schema with tables and columns
            database_type: mysql, postgresql, or mongodb
            customer_id: Optional customer ID for filtering
            is_authenticated: Whether user is authenticated

        Returns:
            Dict with:
                - sql: Generated SQL query
                - explanation: What the query does
                - tables_used: List of tables referenced
                - requires_auth: Whether query needs authentication
                - safe: Whether query is safe to execute
                - value_assumptions: Any assumptions made about values
        """

        # Extract enum hints from schema (e.g., status defaults)
        enum_hints = self.enricher.extract_enum_values(schema)

        logger.info(f"Enum hints extracted: {enum_hints}")

        # Format schema with enum hints
        schema_text = self._format_schema_with_hints(schema, enum_hints)

        prompt = f"""You are a SQL generator. Convert natural language to SQL.

DATABASE TYPE: {database_type}

SCHEMA:
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
4. **IMPORTANT**: When filtering by status/type columns, use the EXACT values shown in "VALUE HINTS" above
5. **IMPORTANT**: If you're unsure about a value (like "done" vs "completed"), use flexible matching:
   - For PostgreSQL: Use ILIKE '%value%' for case-insensitive partial matching
   - For MySQL: Use LIKE '%value%' with LOWER()
   - Always combine exact match OR flexible match for better results
6. If user asks about "my" data and is authenticated, filter by customer_id
7. Table/column names MUST match schema exactly (including schema prefix if shown)

SMART VALUE MATCHING EXAMPLES:
- User asks "how many projects are done"
- Schema shows status default is 'completed'
- Good SQL: WHERE (status = 'completed' OR status ILIKE '%done%' OR status ILIKE '%complete%')
- This catches both exact matches and variations

- User asks "show active users"  
- Schema shows is_active default is 'false' (boolean)
- Good SQL: WHERE is_active = true

Return JSON format:
{{
    "sql": "SELECT COUNT(*) FROM my_portfolio.projects WHERE (status = 'completed' OR status ILIKE '%done%') LIMIT 50",
    "explanation": "Counting projects with completed status (using flexible matching to catch 'done', 'completed', etc.)",
    "tables_used": ["my_portfolio.projects"],
    "requires_auth": false,
    "safe": true,
    "value_assumptions": ["Assuming 'done' maps to 'completed' based on schema default"]
}}

If query is unsafe or impossible, set "safe": false and explain why in "explanation".
"""

        try:
            response = await self.llm.generate_structured_response(prompt)

            # Validate response
            if not response.get("sql"):
                raise ValueError("No SQL generated")

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
    ) -> Dict[str, Any]:
        """
        Generate SQL with automatic retry using flexible matching

        This is called when the first query returns 0 results.
        It generates a more flexible version using:
        - ILIKE/LIKE with wildcards
        - OR conditions for multiple possible values
        - Synonym matching

        Args:
            question: User's question
            schema: Database schema
            database_type: Database type
            customer_id: Optional customer ID
            is_authenticated: Auth status
            previous_attempt: Previous SQL generation result

        Returns:
            Dict with new SQL attempt (or original if retry fails)
        """

        # First attempt - use standard generation
        if not previous_attempt:
            return await self.generate_sql(
                question, schema, database_type, customer_id, is_authenticated
            )

        # Retry with more flexible matching
        logger.info("Retrying SQL generation with flexible matching")

        # Extract failed SQL
        failed_sql = previous_attempt.get("sql", "")

        # Get enum hints for synonym matching
        enum_hints = self.enricher.extract_enum_values(schema)
        schema_text = self._format_schema(schema)

        prompt = f"""The previous query returned 0 results:
{failed_sql}

Generate a MORE FLEXIBLE version of this query that's more likely to find results.

Original question: "{question}"

Database type: {database_type}

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

EXAMPLE TRANSFORMATION:
Bad:  WHERE status = 'done'
Good: WHERE (status ILIKE '%done%' OR status ILIKE '%complet%' OR status ILIKE '%finish%')

Return JSON with the improved SQL.
"""

        try:
            response = await self.llm.generate_structured_response(prompt)

            # Mark this as a retry
            response["is_retry"] = True
            response["original_sql"] = failed_sql

            logger.info(f"Retry SQL generated: {response.get('sql')}")

            return response

        except Exception as e:
            logger.error(f"Retry SQL generation failed: {str(e)}")
            return previous_attempt  # Return original if retry fails

    def _format_schema_with_hints(
        self, schema: Dict[str, Any], enum_hints: Dict[str, List[str]]
    ) -> str:
        """
        Format schema for LLM prompt with value hints

        This creates a comprehensive schema description including:
        - Column names and types
        - Default values
        - Possible enum values
        - Foreign key relationships
        """
        lines = []

        for table_name, table_info in schema.items():
            lines.append(f"\n{'=' * 60}")
            lines.append(f"Table: {table_name}")
            lines.append(f"Entity Type: {table_info.get('entity_type', 'unknown')}")
            lines.append(f"{'=' * 60}")

            columns = table_info.get("columns", [])
            lines.append("\nColumns:")
            for col in columns[:20]:  # Limit to 20 columns
                col_name = col.get("name", "")
                col_type = col.get("type", "")
                nullable = col.get("nullable", True)
                default = col.get("default", "")

                col_line = f"  • {col_name} ({col_type})"

                # Add nullable info
                if not nullable:
                    col_line += " NOT NULL"

                # Add default value hint
                if default:
                    col_line += f" [default: {default}]"

                # Add enum hint if available
                enum_key = f"{table_name}.{col_name}"
                if enum_key in enum_hints and enum_hints[enum_key]:
                    col_line += f" [possible values: {', '.join(enum_hints[enum_key])}]"

                lines.append(col_line)

            # Add foreign keys
            fks = table_info.get("foreign_keys", [])
            if fks:
                lines.append("\nForeign Keys:")
                for fk in fks:
                    lines.append(f"  • {fk['column']} → {fk['references_table']}")

            # Add customer identifier if exists
            customer_id = table_info.get("customer_identifier")
            if customer_id:
                lines.append(f"\nCustomer Identifier: {customer_id}")

        return "\n".join(lines)

    def _format_enum_hints(self, enum_hints: Dict[str, List[str]]) -> str:
        """
        Format enum hints for prompt

        Creates a readable list of columns with their possible values
        """
        if not enum_hints:
            return "No specific value hints available from schema defaults"

        lines = ["Columns with known values:"]
        for key, values in enum_hints.items():
            if values:
                lines.append(f"  • {key}: {', '.join(values)}")

        if len(lines) == 1:  # Only header
            return "No specific value hints available from schema defaults"

        return "\n".join(lines)

    def _format_schema(self, schema: Dict[str, Any]) -> str:
        """Format schema for LLM prompt (basic version without hints)"""
        lines = []

        for table_name, table_info in schema.items():
            lines.append(f"\nTable: {table_name}")
            lines.append(f"Entity Type: {table_info.get('entity_type', 'unknown')}")

            columns = table_info.get("columns", [])
            lines.append("Columns:")
            for col in columns[:15]:  # Limit to 15 cols
                col_name = col.get("name", "")
                col_type = col.get("type", "")
                lines.append(f"  - {col_name} ({col_type})")

            # Add foreign keys
            fks = table_info.get("foreign_keys", [])
            if fks:
                lines.append("Foreign Keys:")
                for fk in fks:
                    lines.append(f"  - {fk['column']} → {fk['references_table']}")

        return "\n".join(lines)
