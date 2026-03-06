from typing import List, Dict, Any
from google import genai
from google.genai import types
from groq import Groq
from app.core.config import settings
from app.core.logging_config import logger
from app.services.rate_limiter import rate_limiter
import re
import json


groq_client = Groq(api_key=settings.GROQ_API_KEY)


class LLMService:
    """
    Unified LLM service with intelligent provider switching.

    Primary:  Gemini (gemini-2.0-flash-exp)
    Fallback: Groq (llama-3.3-70b-versatile)

    Modes:
      generate_response()           → Standard FAQ/chat responses
      generate_analytics_response() → Internal "Digital Brain" analytics (NEW)
      generate_structured_response() → JSON output for SQL generation
      format_query_results()        → Convert DB results to natural language
    """

    def __init__(self):
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.gemini_model = settings.GEMINI_LLM_MODEL
        self.groq_model = settings.GROQ_LLM_MODEL
        self.rate_limiter = rate_limiter

        logger.info(
            f"Initialized LLM service — Gemini: {self.gemini_model}, Groq: {self.groq_model}"
        )

    def _should_use_groq(self) -> bool:
        return self.rate_limiter.is_rate_limited("gemini")

    # ─────────────────────────────────────────────────────────────────
    # Standard response (external users + general FAQ)
    # ─────────────────────────────────────────────────────────────────

    async def generate_response(
        self,
        message: str,
        context: List[str],
        history: str,
        personality: str = "helpful and professional",
        business_overview: str = None,
    ) -> str:
        try:
            messages = self._build_messages(
                message=message,
                context=context,
                history=history,
                personality=personality,
                business_overview=business_overview,
            )

            use_groq = self._should_use_groq()

            if use_groq:
                logger.info("Using Groq (Gemini rate limited)")
                reply = await self._generate_with_groq(messages)
            else:
                logger.info("Using Gemini (primary)")
                try:
                    reply = await self._generate_with_gemini(messages)
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

    # ─────────────────────────────────────────────────────────────────
    # ✅ NEW: Analytics response (internal "Digital Brain" users only)
    # ─────────────────────────────────────────────────────────────────

    async def generate_analytics_response(
        self,
        message: str,
        data: List[Dict],
        sql_explanation: str,
        business_overview: str,
        history: str = "",
        row_count: int = 0,
    ) -> str:
        """
        Generate a rich, analytical response for internal (staff/owner) users.

        Unlike format_query_results() which produces a simple summary,
        this method acts as a business intelligence analyst:
          - Identifies trends and patterns in the data
          - Highlights top/bottom performers
          - Notes anomalies or notable figures
          - Provides business context and interpretation
          - Suggests follow-up questions when appropriate
          - Can comment on what the data implies for strategy

        Args:
            message:          The original question from the staff member
            data:             Raw query results from the database
            sql_explanation:  What the SQL query was doing
            business_overview: Business context for interpretation
            history:          Conversation history
            row_count:        Number of records returned

        Returns:
            Rich analytical narrative as a string
        """
        if not data:
            return (
                "I couldn't find any data matching your query. This could mean:\n"
                "- The data hasn't been synced yet\n"
                "- The time range has no records\n"
                "- The filter criteria didn't match any entries\n\n"
                "Try adjusting your question or checking a different time period."
            )

        # Prepare a rich data summary for the prompt
        data_summary = self._prepare_analytics_summary(data, row_count)

        system_prompt = f"""You are an expert business intelligence analyst and data scientist for an African SME.
You have access to the company's live database and your job is to help leadership make data-driven decisions.

COMPANY CONTEXT:
{business_overview or "A growing African business looking for operational insights."}

YOUR ANALYTICAL STYLE:
- Lead with the most important insight, not with pleasantries
- Use specific numbers and percentages from the data
- Identify trends, not just raw figures (e.g. "up 23% vs last quarter")
- Flag anomalies or outliers worth attention
- Connect data points to business implications
- Use clear formatting: bold key figures, use bullet points for comparisons
- Suggest relevant follow-up questions at the end when appropriate
- Think like a CFO or COO reviewing a dashboard

IMPORTANT:
- You have the FULL data — be specific, not vague
- Never say "I don't have enough information" if the data is right there
- If the data shows a problem, name it directly
- Keep responses concise but substantive — no filler sentences
"""

        history_section = f"\nPrevious conversation:\n{history}\n" if history else ""

        user_prompt = f"""Question: "{message}"

Data retrieved ({row_count} records):
{data_summary}

Query context: {sql_explanation}

Provide a clear, analytical answer. Include:
1. Direct answer to the question with key figures
2. Notable trends or patterns you observe
3. Any anomalies or figures worth flagging
4. Brief strategic implication (1-2 sentences)
5. One relevant follow-up question they might want to explore next
{history_section}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            use_groq = self._should_use_groq()

            if use_groq:
                logger.info("Using Groq for analytics response")
                reply = await self._generate_with_groq(messages)
            else:
                logger.info("Using Gemini for analytics response")
                try:
                    reply = await self._generate_with_gemini(messages)
                    self.rate_limiter.add_request("gemini")
                except Exception as e:
                    logger.warning(f"Gemini failed for analytics, using Groq: {str(e)}")
                    reply = await self._generate_with_groq(messages)

            logger.info(f"Analytics response generated: {reply[:100]}...")
            return reply

        except Exception as e:
            logger.error(f"Analytics response generation failed: {str(e)}")
            # Graceful fallback to basic formatting
            return await self.format_query_results(message, data, sql_explanation)

    def _prepare_analytics_summary(self, data: List[Dict], row_count: int) -> str:
        """
        Prepare a rich data summary for the analytics prompt.
        Shows full data for small sets, structured summaries for large sets.
        """
        if not data:
            return "No data returned."

        if row_count <= 10:
            # Show full data as a formatted table
            return f"Full dataset:\n{json.dumps(data, indent=2, default=str)}"

        elif row_count <= 50:
            # Show first 10 + last 5 + summary stats
            summary_parts = [
                f"First 10 records:\n{json.dumps(data[:10], indent=2, default=str)}",
                f"\nLast 5 records:\n{json.dumps(data[-5:], indent=2, default=str)}",
            ]

            # Try to compute basic stats for numeric columns
            stats = self._compute_basic_stats(data)
            if stats:
                summary_parts.append(
                    f"\nAggregate stats:\n{json.dumps(stats, indent=2)}"
                )

            return "\n".join(summary_parts)

        else:
            # Large dataset — show top/bottom performers + stats
            summary_parts = [
                f"Top 10 records:\n{json.dumps(data[:10], indent=2, default=str)}",
            ]

            stats = self._compute_basic_stats(data)
            if stats:
                summary_parts.append(
                    f"\nAggregate stats across all {row_count} records:\n{json.dumps(stats, indent=2)}"
                )

            return "\n".join(summary_parts)

    def _compute_basic_stats(self, data: List[Dict]) -> Dict[str, Any]:
        """Compute min/max/sum/avg for numeric columns."""
        if not data:
            return {}

        stats = {}
        numeric_cols = [
            k
            for k, v in data[0].items()
            if isinstance(v, (int, float)) and v is not None
        ]

        for col in numeric_cols[:5]:  # Limit to 5 cols
            values = [row[col] for row in data if row.get(col) is not None]
            if values:
                stats[col] = {
                    "min": min(values),
                    "max": max(values),
                    "sum": sum(values),
                    "avg": round(sum(values) / len(values), 2),
                    "count": len(values),
                }

        return stats

    # ─────────────────────────────────────────────────────────────────
    # Structured JSON response (SQL generation)
    # ─────────────────────────────────────────────────────────────────

    async def generate_structured_response(self, prompt: str) -> Dict[str, Any]:
        system_prompt = """You are a data analysis assistant. 
Always respond with valid JSON only. No other text before or after the JSON.
Ensure all JSON is properly formatted with correct quotes and brackets."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        try:
            use_groq = self._should_use_groq()

            if use_groq:
                reply = await self._generate_with_groq(messages)
            else:
                try:
                    reply = await self._generate_with_gemini(messages)
                    self.rate_limiter.add_request("gemini")
                except Exception as gemini_error:
                    logger.warning(f"Gemini failed, using Groq: {str(gemini_error)}")
                    reply = await self._generate_with_groq(messages)

            reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL).strip()

            json_match = re.search(r"\{.*\}", reply, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                logger.error(f"No JSON found in response: {reply}")
                return {}

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {str(e)}")
            return {}
        except Exception as e:
            logger.error(f"Structured response generation failed: {str(e)}")
            return {}

    # ─────────────────────────────────────────────────────────────────
    # Basic query result formatting (external users / simple cases)
    # ─────────────────────────────────────────────────────────────────

    async def format_query_results(
        self, question: str, data: List[Dict], explanation: str
    ) -> str:
        """
        Format database results into plain natural language.
        Used for external users where the response must be customer-friendly.
        """
        if not data:
            return "I couldn't find any matching records for your request."

        data_summary = f"Found {len(data)} results. "
        if len(data) <= 3:
            data_summary += f"Details: {json.dumps(data, indent=2, default=str)}"
        else:
            data_summary += (
                f"First 3 results: {json.dumps(data[:3], indent=2, default=str)}"
            )

        prompt = f"""Convert this database query result into a natural, conversational response.

User Question: "{question}"
Query Purpose: {explanation}
Data: {data_summary}

Write a helpful, natural response. Include key information but keep it conversational.
Do NOT mention table names, column names, or SQL. Speak as if you just know the answer.
If there are multiple results, summarize them clearly."""

        try:
            response = await self.generate_response(
                message=prompt,
                context=[],
                history="",
                personality="helpful and clear",
            )
            return response
        except Exception as e:
            logger.error(f"Query result formatting failed: {str(e)}")
            return f"I found {len(data)} results. {explanation}"

    # ─────────────────────────────────────────────────────────────────
    # Provider implementations
    # ─────────────────────────────────────────────────────────────────

    async def _generate_with_gemini(self, messages: List[dict]) -> str:
        try:
            contents = self._convert_to_gemini_contents(messages)
            response = self.client.models.generate_content(
                model=self.gemini_model,
                contents=contents,
                config=types.GenerateContentConfig(temperature=0.7),
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini generation failed: {str(e)}")
            raise

    async def _generate_with_groq(self, messages: List[dict]) -> str:
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

    # ─────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────

    def _build_messages(
        self,
        message: str,
        context: List[str],
        history: str,
        personality: str,
        business_overview: str = None,
    ) -> List[dict]:
        context_text = (
            "\n".join([f"- {ctx}" for ctx in context[:3]])
            if context
            else "No specific context available."
        )

        system_prompt = f"You are a {personality} AI assistant for a business.\n"

        if business_overview:
            system_prompt += f"\nBUSINESS CONTEXT:\n{business_overview}\n"

        system_prompt += f"""
RELEVANT DATA:
{context_text}

Guidelines:
- Be concise and helpful
- If the answer is not in the data provided, say so politely
- Reference the data when answering
- Keep responses natural and conversational
- Do NOT mention database tables, column names, or SQL
"""

        messages = [{"role": "system", "content": system_prompt}]

        if history:
            messages.append(
                {"role": "system", "content": f"Previous conversation:\n{history}"}
            )

        messages.append({"role": "user", "content": message})
        return messages

    def _convert_to_gemini_contents(self, messages: List[dict]) -> str:
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
        stats = self.rate_limiter.get_stats()
        stats["current_provider"] = "groq" if self._should_use_groq() else "gemini"
        return stats


# Backward compatibility
LLMServiceHuggingFace = LLMService
