from typing import Dict, List, Any, Optional
from app.core.logging_config import logger
import re


class SecureQueryGenerator:
    """Generates safe, filtered database queries"""

    def __init__(self, llm_service):
        self.llm = llm_service

    async def generate_query(
        self,
        intent: Dict[str, Any],
        schema: Dict[str, Any],
        customer_id: Optional[str],
        connection_type: str,
    ) -> Dict[str, Any]:
        """
        Generate database query based on intent
        ALWAYS includes customer filter for safety
        """

        # Security check
        if intent.get("safety_level") == "admin_only":
            return {
                "success": False,
                "error": "This action requires administrator privileges",
                "requires_auth": False,
                "safe": False,
            }

        if intent.get("requires_customer_filter") and not customer_id:
            return {
                "success": False,
                "error": "Authentication required",
                "requires_auth": True,
                "safe": False,
            }

        # Get table information
        target_tables = intent.get("target_tables", [])
        if not target_tables:
            return {
                "success": False,
                "error": "No tables identified for query",
                "safe": False,
            }

        # Build prompt for query generation
        table_info = {}
        for table in target_tables:
            if table in schema.get("tables", {}):
                table_data = schema["tables"][table]
                columns = [col["name"] for col in table_data.get("columns", [])]
                table_info[table] = {
                    "columns": columns,
                    "customer_identifier": table_data.get("customer_identifier"),
                    "entity_type": table_data.get("entity_type"),
                }

        if connection_type == "mongodb":
            return await self._generate_mongo_query(intent, table_info, customer_id)
        else:
            return await self._generate_sql_query(
                intent, table_info, customer_id, connection_type
            )

    async def _generate_sql_query(
        self, intent: Dict, table_info: Dict, customer_id: Optional[str], db_type: str
    ) -> Dict[str, Any]:
        """Generate SQL query"""

        primary_table = intent["target_tables"][0]
        customer_field = table_info[primary_table].get(
            "customer_identifier", "customer_id"
        )

        prompt = f"""Generate a safe SQL query for {db_type}:

Intent: {intent["description"]}
Tables: {list(table_info.keys())}
Customer ID (MUST FILTER): {customer_id}

Table Details:
{self._format_table_info(table_info)}

CRITICAL SAFETY RULES:
1. MUST include: WHERE {customer_field} = '{customer_id}'
2. Only SELECT queries - NO INSERT, UPDATE, DELETE, DROP, ALTER
3. MUST include: LIMIT 50
4. Only query the specified tables
5. Use proper {db_type} syntax

Return JSON:
{{
    "query": "SELECT ... WHERE {customer_field} = '{customer_id}' LIMIT 50",
    "explanation": "This query retrieves...",
    "tables_used": ["table1"]
}}"""

        try:
            response = await self.llm.generate_structured_response(prompt)

            # Validate query safety
            validation = self._validate_sql_query(
                response.get("query", ""), customer_id, list(table_info.keys())
            )

            if not validation["safe"]:
                return {
                    "success": False,
                    "error": f"Query failed safety check: {validation['reason']}",
                    "safe": False,
                }

            return {
                "success": True,
                "query": response["query"],
                "explanation": response.get("explanation", ""),
                "tables_used": response.get("tables_used", []),
                "query_type": "sql",
                "safe": True,
            }

        except Exception as e:
            logger.error(f"SQL query generation failed: {str(e)}")
            return {"success": False, "error": str(e), "safe": False}

    async def _generate_mongo_query(
        self, intent: Dict, table_info: Dict, customer_id: Optional[str]
    ) -> Dict[str, Any]:
        """Generate MongoDB query"""

        primary_table = intent["target_tables"][0]
        customer_field = table_info[primary_table].get(
            "customer_identifier", "customer_id"
        )

        prompt = f"""Generate a safe MongoDB query:

Intent: {intent["description"]}
Collection: {primary_table}
Customer ID (MUST FILTER): {customer_id}

Available Fields: {table_info[primary_table].get("columns", [])}

CRITICAL SAFETY RULES:
1. MUST include filter: {{"{customer_field}": "{customer_id}"}}
2. Only find() queries - NO insert, update, delete
3. MUST include: limit: 50
4. Return proper JSON format

Return JSON:
{{
    "collection": "{primary_table}",
    "filter": {{"{customer_field}": "{customer_id}"}},
    "projection": {{}},
    "limit": 50
}}"""

        try:
            response = await self.llm.generate_structured_response(prompt)

            # Validate MongoDB query
            if customer_id and customer_field not in str(response.get("filter", {})):
                return {
                    "success": False,
                    "error": "Customer filter missing from MongoDB query",
                    "safe": False,
                }

            return {
                "success": True,
                "query": response,
                "explanation": f"Find documents in {primary_table} for customer {customer_id}",
                "tables_used": [primary_table],
                "query_type": "mongodb",
                "safe": True,
            }

        except Exception as e:
            logger.error(f"MongoDB query generation failed: {str(e)}")
            return {"success": False, "error": str(e), "safe": False}

    def _validate_sql_query(
        self, query: str, customer_id: Optional[str], allowed_tables: List[str]
    ) -> Dict[str, Any]:
        """Multi-layer SQL query validation"""

        query_lower = query.lower().strip()

        # Must start with SELECT
        if not query_lower.startswith("select"):
            return {"safe": False, "reason": "Only SELECT queries allowed"}

        # No dangerous keywords
        dangerous = [
            "drop",
            "truncate",
            "delete",
            "update",
            "insert",
            "alter",
            "grant",
            "revoke",
            "create",
            "exec",
            "execute",
            "call",
            "declare",
            "procedure",
            "function",
        ]

        for keyword in dangerous:
            if re.search(rf"\b{keyword}\b", query_lower):
                return {"safe": False, "reason": f"Forbidden keyword: {keyword}"}

        # Customer ID must be present
        if customer_id and customer_id not in query:
            return {"safe": False, "reason": "Customer filter missing"}

        # Must have LIMIT
        if not re.search(r"\bLIMIT\s+\d+", query, re.IGNORECASE):
            return {"safe": False, "reason": "LIMIT clause required"}

        # LIMIT must not exceed 100
        limit_match = re.search(r"\bLIMIT\s+(\d+)", query, re.IGNORECASE)
        if limit_match and int(limit_match.group(1)) > 100:
            return {"safe": False, "reason": "LIMIT cannot exceed 100"}

        return {"safe": True}

    def _format_table_info(self, table_info: Dict) -> str:
        """Format table information for prompt"""
        lines = []
        for table, info in table_info.items():
            columns = ", ".join(info.get("columns", [])[:10])
            lines.append(f"{table}: {columns}")
        return "\n".join(lines)
