from typing import List, Dict, Any
from google import genai
from google.genai import types
from groq import Groq
from app.core.config import settings
from app.core.logging_config import logger
from app.services.rate_limiter import rate_limiter
import re
import json


# Initialize Groq client
groq_client = Groq(api_key=settings.GROQ_API_KEY)


class LLMService:
    """
    Unified LLM service with intelligent provider switching

    Primary: Gemini (gemini-2.0-flash-exp or newer)
    Fallback: Groq (llama-3.3-70b-versatile)

    Rate Limiting Strategy:
    - Track requests per minute
    - Switch to Groq after 14 Gemini requests/minute
    - Auto-reset when minute window expires
    """

    def __init__(self):
        # Initialize the new genai client
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.gemini_model = settings.GEMINI_LLM_MODEL
        self.groq_model = settings.GROQ_LLM_MODEL
        self.rate_limiter = rate_limiter

        logger.info(
            f"Initialized LLM service - Gemini: {self.gemini_model}, Groq: {self.groq_model}"
        )

    def _should_use_groq(self) -> bool:
        """
        Determine if we should use Groq instead of Gemini

        Returns:
            True if Gemini rate limit reached, False otherwise
        """
        return self.rate_limiter.is_rate_limited("gemini")

    async def generate_response(
        self,
        message: str,
        context: List[str],
        history: str,
        personality: str = "helpful and professional",
        business_overview: str = None,
    ) -> str:
        """
        Generate a response using Gemini or Groq (with fallback)

        Args:
            message: User's message
            context: List of relevant context strings
            history: Conversation history
            personality: AI personality/tone
            business_overview: Business context (optional)

        Returns:
            Generated response text
        """
        try:
            messages = self._build_messages(
                message=message,
                context=context,
                history=history,
                personality=personality,
                business_overview=business_overview,
            )

            # Check rate limit and choose provider
            use_groq = self._should_use_groq()

            if use_groq:
                logger.info("Using Groq (Gemini rate limit reached)")
                reply = await self._generate_with_groq(messages)
            else:
                logger.info("Using Gemini (primary)")
                try:
                    reply = await self._generate_with_gemini(messages)
                    # Track successful Gemini request
                    self.rate_limiter.add_request("gemini")
                except Exception as gemini_error:
                    logger.warning(
                        f"Gemini failed, falling back to Groq: {str(gemini_error)}"
                    )
                    reply = await self._generate_with_groq(messages)

            logger.info(f"Generated response: {reply[:100]}...")
            return reply

        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            return "I apologize, but I'm having trouble generating a response right now. Please try again."

    async def _generate_with_gemini(self, messages: List[dict]) -> str:
        """Generate response using Gemini with new SDK"""
        try:
            # Convert OpenAI-style messages to Gemini format
            contents = self._convert_to_gemini_contents(messages)

            # Generate content using new SDK
            response = self.client.models.generate_content(
                model=self.gemini_model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                ),
            )

            return response.text.strip()

        except Exception as e:
            logger.error(f"Gemini generation failed: {str(e)}")
            raise

    async def _generate_with_groq(self, messages: List[dict]) -> str:
        """Generate response using Groq"""
        try:
            response = groq_client.chat.completions.create(
                model=self.groq_model,
                messages=messages,
                temperature=0.7,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Groq generation failed: {str(e)}")
            raise

    async def generate_structured_response(self, prompt: str) -> Dict[str, Any]:
        """
        Generate structured JSON response for analysis tasks

        Args:
            prompt: Analysis prompt

        Returns:
            Parsed JSON dictionary
        """
        system_prompt = """You are a data analysis assistant. 
Always respond with valid JSON only. No other text before or after the JSON.
Ensure all JSON is properly formatted with correct quotes and brackets."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        try:
            # Check rate limit
            use_groq = self._should_use_groq()

            if use_groq:
                logger.info("Using Groq for structured response")
                reply = await self._generate_with_groq(messages)
            else:
                logger.info("Using Gemini for structured response")
                try:
                    reply = await self._generate_with_gemini(messages)
                    self.rate_limiter.add_request("gemini")
                except Exception as gemini_error:
                    logger.warning(f"Gemini failed, using Groq: {str(gemini_error)}")
                    reply = await self._generate_with_groq(messages)

            # Clean response
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
            data_summary += f"Details: {json.dumps(data, indent=2)}"
        else:
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
        """Build messages array for LLM (OpenAI-style for Groq compatibility)"""

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

    def _convert_to_gemini_contents(self, messages: List[dict]) -> str:
        """
        Convert OpenAI-style messages to Gemini content format

        For the new SDK, we can use a simple text concatenation approach
        since Gemini handles system instructions differently
        """
        combined_text = ""

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                combined_text += f"{content}\n\n"
            elif role == "user":
                combined_text += f"User: {content}\n\n"
            elif role == "assistant":
                combined_text += f"Assistant: {content}\n\n"

        return combined_text.strip()

    def get_rate_limit_stats(self) -> Dict[str, Any]:
        """Get current rate limiting statistics"""
        stats = self.rate_limiter.get_stats()
        stats["current_provider"] = "groq" if self._should_use_groq() else "gemini"
        return stats


# Backward compatibility alias
LLMServiceHuggingFace = LLMService
