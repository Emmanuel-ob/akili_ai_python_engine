
        from app.core.llm import call_llm

        def generate_sql(query: str, schema: dict) -> str:
            schema_text = ""
            for table, meta in schema.items():
                schema_text += f"Table: {table}\nColumns:\n"
                for col in meta["columns"]:
                    schema_text += f"- {col}\n"

            prompt = f"""
You are a SQL generator.

Rules:
- Only SELECT queries
- Use only provided tables and columns
- Always include LIMIT 50
- Output SQL only

Schema:
{schema_text}

User question:
{query}
"""
            return call_llm(prompt).strip()
