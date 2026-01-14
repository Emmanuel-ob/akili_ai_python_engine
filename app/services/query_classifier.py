from typing import Literal
from app.core.logging_config import logger

QueryRoute = Literal["FAQ", "SQL", "HYBRID"]


class QueryClassifier:
    """Routes user queries to appropriate handler"""

    # Keywords that indicate SQL needs
    SQL_KEYWORDS = [
        "how many",
        "count",
        "total",
        "sum",
        "average",
        "show me",
        "list",
        "find all",
        "get my",
        "recent",
        "last",
        "status of",
        "order #",
        "invoice",
        "transaction",
    ]

    # Keywords that indicate FAQ search
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
    ]

    @staticmethod
    def classify(query: str, has_database: bool = False) -> QueryRoute:
        """
        Classify query into FAQ, SQL, or HYBRID

        Args:
            query: User's question
            has_database: Whether chatbot has database connected

        Returns:
            "FAQ", "SQL", or "HYBRID"
        """
        query_lower = query.lower()

        # If no database, always FAQ
        if not has_database:
            return "FAQ"

        # Check for SQL indicators
        sql_score = sum(
            1 for keyword in QueryClassifier.SQL_KEYWORDS if keyword in query_lower
        )

        # Check for FAQ indicators
        faq_score = sum(
            1 for keyword in QueryClassifier.FAQ_KEYWORDS if keyword in query_lower
        )

        logger.info(
            f"Query classification - SQL score: {sql_score}, FAQ score: {faq_score}"
        )

        # Decision logic
        if sql_score > 0 and faq_score > 0:
            return "HYBRID"
        elif sql_score > 0:
            return "SQL"
        else:
            return "FAQ"
