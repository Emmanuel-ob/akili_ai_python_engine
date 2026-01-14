from typing import Dict, List, Optional
from app.core.logging_config import logger


class ValueDetector:
    """Detects actual column values to help with accurate SQL generation"""

    @staticmethod
    async def get_distinct_values(
        connection_id: str,
        business_id: str,
        chatbot_id: str,
        table: str,
        column: str,
        limit: int = 10,
    ) -> Dict[str, any]:
        """
        Fetch distinct values from a column to help AI make better decisions

        Args:
            connection_id: Database connection ID
            business_id: Business ID
            chatbot_id: Chatbot ID
            table: Table name
            column: Column name
            limit: Max number of distinct values to fetch

        Returns:
            {
                "success": bool,
                "values": ["completed", "pending", "cancelled"],
                "count": 3,
                "sample_query": "SELECT DISTINCT status FROM projects LIMIT 10"
            }
        """
        from app.services.sql_executor import SQLExecutor

        # Generate safe query to get distinct values
        sql = f"SELECT DISTINCT {column} FROM {table} WHERE {column} IS NOT NULL LIMIT {limit}"

        try:
            result = await SQLExecutor.execute(
                sql=sql,
                connection_id=connection_id,
                business_id=business_id,
                chatbot_id=chatbot_id,
            )

            if result.get("success"):
                values = [row[column] for row in result.get("data", [])]
                return {
                    "success": True,
                    "values": values,
                    "count": len(values),
                    "sample_query": sql,
                }
            else:
                return {
                    "success": False,
                    "values": [],
                    "count": 0,
                    "error": result.get("error"),
                }

        except Exception as e:
            logger.error(f"Value detection failed: {str(e)}")
            return {"success": False, "values": [], "count": 0, "error": str(e)}


class SchemaEnricher:
    """Enriches schema with sample values and defaults"""

    @staticmethod
    def extract_enum_values(schema: Dict) -> Dict[str, List[str]]:
        """
        Extract possible values from schema defaults and types

        This helps the AI understand what values exist in enum-like columns
        by extracting them from default values and type definitions.

        Args:
            schema: Database schema dict with tables and columns

        Returns:
            Dict mapping "table.column" to list of possible values

        Example:
            Input: column with default: "'completed'::character varying"
            Output: {"my_portfolio.projects.status": ["completed"]}
        """
        enum_hints = {}

        for table, info in schema.items():
            for column in info.get("columns", []):
                col_name = column.get("name")
                col_type = column.get("type", "").lower()
                default = column.get("default", "")

                # Extract from default values (PostgreSQL style: 'value'::type)
                if default and isinstance(default, str):
                    # Pattern: 'value'::type or "value"
                    if "::" in default:
                        # Extract value before ::
                        value = default.split("::")[0].strip("'\"")
                        if value:
                            key = f"{table}.{col_name}"
                            enum_hints[key] = [value]
                    elif default.startswith("'") or default.startswith('"'):
                        # Simple quoted value
                        value = default.strip("'\"")
                        if value:
                            key = f"{table}.{col_name}"
                            enum_hints[key] = [value]

                # Identify columns that likely have enum-like values
                if (
                    "enum" in col_type
                    or "varchar" in col_type
                    or "character varying" in col_type
                ):
                    # Common status/type columns
                    if any(
                        keyword in col_name.lower()
                        for keyword in ["status", "state", "type", "category"]
                    ):
                        key = f"{table}.{col_name}"
                        if key not in enum_hints:
                            enum_hints[key] = []

        logger.info(f"Extracted enum hints for {len(enum_hints)} columns")
        return enum_hints

    @staticmethod
    def get_common_value_synonyms() -> Dict[str, List[str]]:
        """
        Return common synonyms for status values

        This helps the AI understand that "done" might mean "completed", etc.

        Returns:
            Dict mapping common terms to their synonyms
        """
        return {
            "done": ["completed", "finished", "complete", "success", "successful"],
            "pending": ["processing", "in_progress", "active", "ongoing", "waiting"],
            "failed": ["error", "cancelled", "canceled", "rejected", "declined"],
            "active": ["enabled", "live", "running", "open"],
            "inactive": ["disabled", "closed", "archived", "deleted"],
            "paid": ["complete", "successful", "cleared", "processed"],
            "unpaid": ["pending", "outstanding", "due", "open"],
        }

    @staticmethod
    def find_synonym_matches(
        user_term: str, schema_hints: Dict[str, List[str]]
    ) -> List[str]:
        """
        Find possible schema values that match user term

        Args:
            user_term: Term used in user's question (e.g., "done")
            schema_hints: Extracted enum hints from schema

        Returns:
            List of possible matching values from schema

        Example:
            user_term = "done"
            schema_hints = {"projects.status": ["completed"]}
            returns: ["completed"]
        """
        user_term_lower = user_term.lower()
        synonyms = SchemaEnricher.get_common_value_synonyms()

        # Get synonyms for the user term
        possible_matches = synonyms.get(user_term_lower, [])
        possible_matches.append(user_term_lower)  # Include original term

        # Find matching values in schema hints
        schema_values = []
        for hint_values in schema_hints.values():
            for hint_value in hint_values:
                hint_lower = hint_value.lower()
                # Check if any synonym matches
                if any(
                    synonym in hint_lower or hint_lower in synonym
                    for synonym in possible_matches
                ):
                    schema_values.append(hint_value)

        return list(set(schema_values))  # Remove duplicates
