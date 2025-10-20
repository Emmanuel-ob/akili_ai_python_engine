from token import OP
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

IMPORTANT CONSTRAINTS:
- You can ONLY reference the table "{table}" and the other tables listed above
- Do NOT mention any tables that are not in the "Other tables in database" list
- Foreign keys can ONLY reference tables from the "Other tables in database" list
- If a column ends with _id but the referenced table is not in the list, ignore it

Determine:
1. Primary key column (usually 'id', '_id', or '{table}_id')
2. Foreign key columns (columns ending in '_id' that reference OTHER TABLES FROM THE LIST ABOVE)
3. Customer/user identifier column (user_id, customer_id, email, etc.)
4. Entity type (users, orders, products, payments, etc.)
5. Relationships to OTHER TABLES FROM THE LIST ABOVE ONLY

Return JSON format:
{{
    "primary_key": "id",
    "foreign_keys": [{{"column": "user_id", "references_table": "users"}}],
    "customer_identifier": "user_id",
    "entity_type": "orders",
    "relationships": [{{"type": "belongs_to", "table": "users", "via": "user_id"}}]
}}

CRITICAL: Ensure all table names in your response exist in this list: {", ".join(all_tables)}
"""

        try:
            response = await self.llm.generate_structured_response(prompt)

            # Post-validation: Filter out any invalid table references
            response = self._validate_table_references(response, all_tables)

            return response
        except Exception as e:
            logger.error(f"Schema analysis failed: {str(e)}")
            return self._fallback_analysis(table, columns, all_tables)

    def _validate_table_references(
        self, analysis: Dict[str, Any], valid_tables: List[str]
    ) -> Dict[str, Any]:
        """Remove any references to tables not in the valid_tables list"""
        valid_tables_lower = [t.lower() for t in valid_tables]

        # Filter foreign keys
        if "foreign_keys" in analysis:
            analysis["foreign_keys"] = [
                fk
                for fk in analysis["foreign_keys"]
                if fk.get("references_table", "").lower() in valid_tables_lower
            ]

        # Filter relationships
        if "relationships" in analysis:
            analysis["relationships"] = [
                rel
                for rel in analysis["relationships"]
                if rel.get("table", "").lower() in valid_tables_lower
            ]

        return analysis

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

        all_tables_lower = [t.lower() for t in all_tables]

        for col in columns:
            col_lower = col.lower()

            # Detect primary key
            if col_lower in ["id", "_id", f"{table}_id"]:
                analysis["primary_key"] = col

            # Detect foreign keys - ONLY if referenced table exists
            if col_lower.endswith("_id") and col_lower != "id":
                ref_table = col_lower.replace("_id", "")
                # Check both singular and plural forms
                if ref_table in all_tables_lower:
                    actual_table = all_tables[all_tables_lower.index(ref_table)]
                    analysis["foreign_keys"].append(
                        {"column": col, "references_table": actual_table}
                    )
                    analysis["relationships"].append(
                        {"type": "belongs_to", "table": actual_table, "via": col}
                    )
                elif (ref_table + "s") in all_tables_lower:
                    actual_table = all_tables[all_tables_lower.index(ref_table + "s")]
                    analysis["foreign_keys"].append(
                        {"column": col, "references_table": actual_table}
                    )
                    analysis["relationships"].append(
                        {"type": "belongs_to", "table": actual_table, "via": col}
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
        self, schema_analysis: Dict[str, Any], database_type: str,selected_tables: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Generate business overview and discover possible actions"""

        # Extract ONLY the selected tables
        selected_tables = list(schema_analysis.get("tables", {}).keys())

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

SELECTED TABLES (these are the ONLY tables available):
{tables_text}

CRITICAL CONSTRAINTS:
- You can ONLY analyze and reference the tables listed above
- Do NOT mention or suggest actions for tables that are not in the SELECTED TABLES list
- All discovered actions MUST use ONLY the tables listed above
- The "tables_needed" field in actions can ONLY contain tables from the SELECTED TABLES list

Relationships (validated):
{schema_analysis.get("relationships", [])}

Generate:
1. Business Type: What kind of business is this based ONLY on the selected tables?
2. Core Operations: What operations can be performed with ONLY these tables?
3. Customer Journey: How do customers interact based ONLY on these tables?
4. Data Structure: Key entities and relationships in the SELECTED TABLES
5. Possible AI Actions: Actions that can be performed using ONLY the SELECTED TABLES

VALIDATION RULES:
- Every action's "tables_needed" must be a subset of: {", ".join(selected_tables)}
- Do not infer the existence of tables not in the SELECTED TABLES list
- If critical tables are missing (e.g., no user/customer table), acknowledge the limitation

Return JSON format:
{{
    "business_type": "E-commerce Platform",
    "overview": "This business manages... based on the available tables: {", ".join(selected_tables[:3])}...",
    "core_operations": ["order management", "inventory"],
    "customer_journey": "Based on available data...",
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
            "description": "View customer order status",
            "permission": "authenticated",
            "tables_needed": ["orders"]
        }}
    ]
}}

REMEMBER: All table references must be from this list: {", ".join(selected_tables)}
"""

        try:
            response = await self.llm.generate_structured_response(prompt)

            # Post-validation: Filter actions with invalid table references
            response = self._validate_discovered_actions(response, selected_tables)

            # Add metadata about what tables were analyzed
            response["analyzed_tables"] = selected_tables
            response["table_count"] = len(selected_tables)

            return response
        except Exception as e:
            logger.error(f"Business overview generation failed: {str(e)}")
            return {
                "business_type": "Unknown",
                "overview": f"A business with {len(selected_tables)} selected tables: {', '.join(selected_tables)}",
                "discovered_actions": [],
                "analyzed_tables": selected_tables,
                "table_count": len(selected_tables),
            }

    def _validate_discovered_actions(
        self, response: Dict[str, Any], valid_tables: List[str]
    ) -> Dict[str, Any]:
        """Remove actions that reference invalid tables"""
        if "discovered_actions" not in response:
            return response

        valid_tables_lower = [t.lower() for t in valid_tables]
        validated_actions = []

        for action in response["discovered_actions"]:
            tables_needed = action.get("tables_needed", [])

            # Check if all tables needed are in valid_tables
            invalid_tables = [
                t for t in tables_needed if t.lower() not in valid_tables_lower
            ]

            if not invalid_tables:
                # All tables are valid
                validated_actions.append(action)
            else:
                logger.warning(
                    f"Removed action '{action.get('key')}' - references invalid tables: {invalid_tables}"
                )

        response["discovered_actions"] = validated_actions
        return response
