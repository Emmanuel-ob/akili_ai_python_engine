from typing import List, Dict, Any
from huggingface_hub import InferenceClient
from app.core.config import settings
from app.core.logging_config import logger
import re
import json


class LLMServiceHuggingFace:
    def __init__(self):
        self.model = "HuggingFaceTB/SmolLM3-3B"
        self.client = InferenceClient(
            model=self.model,
            token=settings.HUGGING_FACE_KEY,
        )

    async def generate_response(
        self,
        message: str,
        context: List[str],
        history: str,
        personality: str = "helpful and professional",
        business_overview: str = None,
    ) -> str:
        """Generate a response using Hugging Face Inference API"""

        try:
            messages = self._build_messages(
                message=message,
                context=context,
                history=history,
                personality=personality,
                business_overview=business_overview,
            )

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
            )

            reply = response.choices[0].message["content"]
            reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL).strip()

            logger.info(f"Generated response: {reply[:100]}...")
            return reply

        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            return "I apologize, but I'm having trouble generating a response right now. Please try again."

    async def generate_structured_response(self, prompt: str) -> Dict[str, Any]:
        """Generate structured JSON response for analysis tasks"""

        system_prompt = """You are a data analysis assistant. 
Always respond with valid JSON only. No other text before or after the JSON.
Ensure all JSON is properly formatted with correct quotes and brackets."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,  # Lower temperature for structured output
            )

            reply = response.choices[0].message["content"]
            reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL).strip()

            # Extract JSON from response
            json_match = re.search(r"\{.*\}", reply, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            else:
                logger.error(f"No JSON found in response: {reply}")
                return {}

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {str(e)}, Response: {reply}")
            return {}
        except Exception as e:
            logger.error(f"Structured response generation failed: {str(e)}")
            return {}

    async def format_query_results(
        self, question: str, data: List[Dict], explanation: str
    ) -> str:
        """Format database query results into natural language"""

        if not data:
            return "I couldn't find any matching records for your request."

        data_summary = f"Found {len(data)} results. "
        if len(data) <= 3:
            # Show all results
            data_summary += f"Details: {json.dumps(data, indent=2)}"
        else:
            # Summarize
            data_summary += f"First 3 results: {json.dumps(data[:3], indent=2)}"

        prompt = f"""Convert this database query result into a natural, conversational response.

User Question: "{question}"
Query Purpose: {explanation}
Data: {data_summary}

Write a helpful, natural response. Include key information but keep it conversational.
If there are multiple results, summarize them clearly."""

        try:
            response = await self.generate_response(
                message=prompt, context=[], history="", personality="helpful and clear"
            )
            return response
        except Exception as e:
            logger.error(f"Query result formatting failed: {str(e)}")
            return f"I found {len(data)} results. {explanation}"

    def _build_messages(
        self,
        message: str,
        context: List[str],
        history: str,
        personality: str,
        business_overview: str = None,
    ) -> List[dict]:
        """Build OpenAI-style messages with context and history"""

        context_text = (
            "\n".join([f"- {ctx}" for ctx in context[:3]])
            if context
            else "No specific context available."
        )

        system_prompt = f"""You are a {personality} AI assistant for a business.
"""

        if business_overview:
            system_prompt += f"""
BUSINESS CONTEXT:
{business_overview}

"""

        system_prompt += f"""
RELEVANT DATA:
{context_text}

Guidelines:
- Be concise and helpful
- If the answer is not in the data provided, say so politely
- Reference the data when answering
- Keep responses natural and conversational
- If you need more information, ask clarifying questions
"""

        messages = [{"role": "system", "content": system_prompt}]

        if history:
            messages.append(
                {"role": "system", "content": f"Previous conversation:\n{history}"}
            )

        messages.append({"role": "user", "content": message})

        return messages
