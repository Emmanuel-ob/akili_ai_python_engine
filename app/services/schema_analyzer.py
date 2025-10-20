from typing import Dict, List, Any, Optional
from app.core.logging_config import logger
import re


class SchemaAnalyzer:
    """Analyzes database schemas to understand structure and relationships"""

    def __init__(self, llm_service):
        self.llm = llm_service

    async def analyze_table_structure(
        self, table: str, columns: List[str], all_tables: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze a table's structure and infer relationships
        """

        prompt = f"""Analyze this database table structure:

                        Table: {table}
                        Columns: {", ".join(columns)}
                        Other tables in database: {", ".join(all_tables)}

                        Determine:
                        1. Primary key column (usually 'id', '_id', or '{table}_id')
                        2. Foreign key columns (columns ending in '_id' that reference other tables)
                        3. Customer/user identifier column (user_id, customer_id, email, etc.)
                        4. Entity type (users, orders, products, payments, etc.)
                        5. Relationships to other tables

                        Return JSON format:
                        {{
                            "primary_key": "id",
                            "foreign_keys": [{{"column": "user_id", "references_table": "users"}}],
                            "customer_identifier": "user_id",
                            "entity_type": "orders",
                            "relationships": [{{"type": "belongs_to", "table": "users", "via": "user_id"}}]
                        }}
                    """

        try:
            response = await self.llm.generate_structured_response(prompt)
            return response
        except Exception as e:
            logger.error(f"Schema analysis failed: {str(e)}")
            return self._fallback_analysis(table, columns, all_tables)

    def _fallback_analysis(
        self, table: str, columns: List[str], all_tables: List[str]
    ) -> Dict[str, Any]:
        """Heuristic-based analysis when AI fails"""

        analysis = {
            "primary_key": "id",
            "foreign_keys": [],
            "customer_identifier": None,
            "entity_type": "unknown",
            "relationships": [],
        }

        for col in columns:
            col_lower = col.lower()

            # Detect primary key
            if col_lower in ["id", "_id", f"{table}_id"]:
                analysis["primary_key"] = col

            # Detect foreign keys
            if col_lower.endswith("_id") and col_lower != "id":
                ref_table = col_lower.replace("_id", "")
                if ref_table in [t.lower() for t in all_tables]:
                    analysis["foreign_keys"].append(
                        {"column": col, "references_table": ref_table}
                    )
                    analysis["relationships"].append(
                        {"type": "belongs_to", "table": ref_table, "via": col}
                    )

            # Detect customer identifier
            if col_lower in [
                "user_id",
                "customer_id",
                "client_id",
                "account_id",
                "email",
                "user_email",
            ]:
                analysis["customer_identifier"] = col

        # Infer entity type
        table_lower = table.lower()
        if "user" in table_lower or "customer" in table_lower:
            analysis["entity_type"] = "users"
        elif "order" in table_lower:
            analysis["entity_type"] = "orders"
        elif "product" in table_lower or "item" in table_lower:
            analysis["entity_type"] = "products"
        elif "payment" in table_lower or "transaction" in table_lower:
            analysis["entity_type"] = "payments"

        return analysis


class BusinessOverviewGenerator:
    """Generates comprehensive business context from schema"""

    def __init__(self, llm_service):
        self.llm = llm_service

    async def generate_overview(
        self, schema_analysis: Dict[str, Any], database_type: str
    ) -> Dict[str, Any]:
        """Generate business overview and discover possible actions"""

        # Format schema for prompt
        tables_summary = []
        for table, info in schema_analysis.get("tables", {}).items():
            columns = [col["name"] for col in info.get("columns", [])]
            entity_type = info.get("entity_type", "unknown")
            tables_summary.append(
                f"- {table} ({entity_type}): {', '.join(columns[:8])}"
            )

        tables_text = "\n".join(tables_summary)

        prompt = f"""Analyze this business database and create a comprehensive overview:

Database Type: {database_type}
Tables:
{tables_text}

Relationships:
{schema_analysis.get("relationships", [])}

Generate:
1. Business Type: What kind of business is this? (e-commerce, SaaS, service provider, etc.)
2. Core Operations: What are the main business processes?
3. Customer Journey: How do customers interact with this business?
4. Data Structure: Key entities and their relationships
5. Possible AI Actions: What actions could an AI assistant safely perform with this data?
   Examples: view_products, check_order_status, generate_payment_link, etc.
   Consider what requires authentication vs. what's public

Return JSON format:
{{
    "business_type": "E-commerce Platform",
    "overview": "Detailed paragraph about the business...",
    "core_operations": ["order management", "inventory", "payments"],
    "customer_journey": "Customers browse products, place orders...",
    "discovered_actions": [
        {{
            "key": "view_products",
            "label": "View Products",
            "description": "Browse product catalog",
            "permission": "public",
            "tables_needed": ["products"]
        }},
        {{
            "key": "check_order_status",
            "label": "Check Order Status",
            "description": "View status of customer orders",
            "permission": "authenticated",
            "tables_needed": ["orders", "customers"]
        }}
    ]
}}"""

        try:
            response = await self.llm.generate_structured_response(prompt)
            return response
        except Exception as e:
            logger.error(f"Business overview generation failed: {str(e)}")
            return {
                "business_type": "Unknown",
                "overview": f"A business with {len(schema_analysis.get('tables', {}))} data tables.",
                "discovered_actions": [],
            }
