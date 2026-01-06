
        # Replace with OpenAI / Claude SDK in production
        def call_llm(prompt: str) -> str:
            # Mocked response for demo
            return "SELECT COUNT(*) FROM orders WHERE status = 'delayed' LIMIT 50;"

        def generate_answer(system_prompt: str, context: str, query: str) -> str:
            return f"""{system_prompt}

Context:
{context}

Answer:
Based on the information above, here is the response."""
