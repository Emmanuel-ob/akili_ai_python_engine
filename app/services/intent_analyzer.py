from typing import Dict, List, Any, Optional
from app.core.logging_config import logger


class IntentAnalyzer:
    """Analyzes user intent to determine required actions"""

    def __init__(self, llm_service):
        self.llm = llm_service

    async def analyze_intent(
        self,
        message: str,
        schema: Dict[str, Any],
        customer_context: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Analyze user message to determine intent and required actions
        """

        # Format available tables
        available_tables = list(schema.get("tables", {}).keys())
        tables_with_types = []
        for table, info in schema.get("tables", {}).items():
            entity_type = info.get("entity_type", "unknown")
            tables_with_types.append(f"{table} ({entity_type})")

        customer_info = "Anonymous user"
        if customer_context:
            customer_info = (
                f"Authenticated user: {customer_context.get('email', 'N/A')}"
            )

        prompt = f"""Analyze this user message and determine the intent:

User Message: "{message}"

Available Data (ONLY USE THESE TABLES):
{chr(10).join(["- " + t for t in tables_with_types])}

Customer Context: {customer_info}

Classify the intent:
1. Intent Category: 
   - "information_request" (user wants data/info)
   - "action_request" (user wants to perform action)
   - "general_chat" (greeting, thanks, general conversation)

2. Requires Database Query: true/false
3. Target Tables: which tables from the AVAILABLE DATA above?
4. Customer Filter Needed: true/false (must filter by customer_id?)
5. Safety Level:
   - "public" (anyone can access)
   - "authenticated" (requires user login)
   - "admin_only" (only business owner)

6. Specific Fields Needed: what data columns?

IMPORTANT: target_tables MUST only contain tables from the Available Data list above.

Return JSON:
{{
    "intent_category": "information_request",
    "requires_database_query": true,
    "target_tables": ["orders", "products"],
    "requires_customer_filter": true,
    "safety_level": "authenticated",
    "specific_fields": ["order_id", "status", "product_name"],
    "description": "User wants to check their order status"
}}"""

        try:
            response = await self.llm.generate_structured_response(prompt)

            # Filter target_tables to only include available tables
            target_tables = response.get("target_tables", [])
            filtered_tables = [t for t in target_tables if t in available_tables]
            response["target_tables"] = filtered_tables

            # If no valid tables remain but query is required, mark as general_chat
            if response.get("requires_database_query") and not filtered_tables:
                response["requires_database_query"] = False
                response["intent_category"] = "general_chat"

            # Validate and enhance response
            if response.get("requires_customer_filter") and not customer_context:
                response["requires_auth"] = True
            else:
                response["requires_auth"] = False

            return response

        except Exception as e:
            logger.error(f"Intent analysis failed: {str(e)}")
            return {
                "intent_category": "general_chat",
                "requires_database_query": False,
                "target_tables": [],
                "requires_customer_filter": False,
                "safety_level": "public",
                "description": "Unable to analyze intent",
            }
