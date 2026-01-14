import re
from typing import Dict, Any
from app.core.logging_config import logger


class SQLValidator:
    """Validates SQL queries for security"""

    FORBIDDEN_KEYWORDS = {
        "insert",
        "update",
        "delete",
        "drop",
        "truncate",
        "alter",
        "create",
        "grant",
        "revoke",
        "exec",
        "execute",
        "xp_",
        "--",
        "/*",
        "*/",
        ";",
    }

    @staticmethod
    def validate(sql: str, database_type: str) -> Dict[str, Any]:
        """
        Validate SQL query for security

        Returns:
            {
                "valid": bool,
                "errors": List[str],
                "warnings": List[str]
            }
        """
        errors = []
        warnings = []

        sql_lower = sql.lower().strip()

        # Check 1: Must start with SELECT (or find for MongoDB)
        if database_type == "mongodb":
            if not sql_lower.startswith(("db.", "{")):
                errors.append("MongoDB queries must start with db. or {")
        else:
            if not sql_lower.startswith("select"):
                errors.append("Only SELECT queries allowed")

        # Check 2: Forbidden keywords
        for keyword in SQLValidator.FORBIDDEN_KEYWORDS:
            if keyword in sql_lower:
                errors.append(f"Forbidden keyword detected: {keyword}")

        # Check 3: LIMIT clause required (SQL only)
        if database_type != "mongodb":
            if "limit" not in sql_lower:
                errors.append("LIMIT clause required")

        # Check 4: No multiple statements
        if ";" in sql and sql_lower.count(";") > 1:
            errors.append("Multiple statements not allowed")

        # Check 5: No SQL injection patterns
        injection_patterns = [
            r"union\s+select",
            r"(or|and)\s+1\s*=\s*1",
            r"(or|and)\s+\'.*\'\s*=\s*\'.*\'",
            r"sleep\s*\(",
            r"benchmark\s*\(",
        ]

        for pattern in injection_patterns:
            if re.search(pattern, sql_lower):
                errors.append(f"Potential SQL injection detected: {pattern}")

        # Warnings (not blocking)
        if "*" in sql:
            warnings.append("SELECT * may return large results")

        if "like" in sql_lower and "%" in sql:
            warnings.append("LIKE with % can be slow on large tables")

        valid = len(errors) == 0

        logger.info(
            f"SQL validation - Valid: {valid}, Errors: {len(errors)}, Warnings: {len(warnings)}"
        )

        return {"valid": valid, "errors": errors, "warnings": warnings}