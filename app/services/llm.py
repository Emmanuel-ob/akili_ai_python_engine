from typing import List, Dict, Any, Optional
from google import genai
from google.genai import types
from groq import Groq
from app.core.config import settings
from app.core.logging_config import logger
from app.services.rate_limiter import rate_limiter
from app.services.llm_provider import choose_provider, generate_bedrock
import asyncio
import re
import json


groq_client = Groq(api_key=settings.GROQ_API_KEY)


class LLMService:
    """
    Unified LLM service with intelligent provider switching.

    Primary:  Gemini
    Fallback: Groq (llama-3.3-70b-versatile)

    Point 3 additions:
      answer_with_sql_context() → grounded follow-up answers using last SQL result
      generate_analytics_response() → adaptive verbosity based on question complexity

    Point 5C/5D additions:
      generate_internal_response() → richer BI analyst system prompt for internal users
      _dispatch() → single provider-selection point for every generation path
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
    # Point 5C — Smarter Internal Prompting
    # ─────────────────────────────────────────────────────────────────

    async def generate_internal_response(
        self,
        message: str,
        context: List[str],
        history: str,
        business_name: str = "",
        user_role: str = "",
        recent_insights: str = "",
    ) -> str:
        """
        Point 5C: BI-analyst system prompt for internal staff users.

        Differences from the standard generate_response():
        - Role: data analyst & business intelligence advisor
        - Leads with the key metric/number
        - Adds context vs last period where applicable
        - Identifies the WHY if detectable
        - Gives a concrete recommendation
        - Never gives generic answers — always specific with numbers
        - Injects recent proactive insights as additional context
        """
        try:
            system_lines = [
                f"ROLE: You are an expert data analyst and business intelligence advisor"
                f"{' for ' + business_name if business_name else ''}.",
                "MODE: Internal Analytics — full unrestricted access.",
            ]
            if user_role:
                system_lines.append(f"USER ROLE: {user_role}")

            system_lines += [
                "",
                "When answering questions:",
                "1. Lead with the key metric/number immediately.",
                "2. Add context (vs last period, vs target if inferable).",
                "3. Identify the WHY if detectable from data.",
                "4. Give a concrete recommendation.",
                "5. Show what data was queried (transparency).",
                "",
                "Rules:",
                "- Never give generic answers. Always be specific with numbers.",
                "- If you don't have the data, say so clearly and suggest how to get it.",
                "- Format numbers with commas. Use ₦, $, € based on context.",
                "- Use markdown tables for multi-row comparisons.",
            ]

            if recent_insights:
                system_lines += ["", recent_insights]

            system_prompt = "\n".join(system_lines)

            context_text = "\n\n".join(context) if context else ""
            user_content_parts = []
            if history:
                user_content_parts.append(f"Conversation history:\n{history}")
            if context_text:
                user_content_parts.append(f"Relevant data/knowledge:\n{context_text}")
            user_content_parts.append(f"Question: {message}")

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "\n\n".join(user_content_parts)},
            ]

            # Internal BI answers are grounded in retrieved reports and SQL
            # results, so they take the grounded temperature like any other.
            return await self._dispatch(messages, settings.TEMPERATURE_GROUNDED)

        except Exception as e:
            logger.error(f"generate_internal_response error: {str(e)}")
            return "I encountered an error generating the analysis. Please try again."

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
        temperature: Optional[float] = None,
    ) -> str:
        try:
            # Default to grounded. Callers doing small talk pass the
            # conversational value explicitly; everything else in this engine
            # answers from retrieved context, where creative phrasing is
            # exactly the failure mode.
            if temperature is None:
                temperature = settings.TEMPERATURE_GROUNDED

            messages = self._build_messages(
                message=message,
                context=context,
                history=history,
                personality=personality,
                business_overview=business_overview,
            )
            reply = await self._dispatch(messages, temperature)
            logger.info(f"Generated response: {reply[:100]}...")
            return reply
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            return "I apologize, but I'm having trouble generating a response right now. Please try again."

    # ─────────────────────────────────────────────────────────────────
    # Point 3: Grounded follow-up answers using last SQL context
    # ─────────────────────────────────────────────────────────────────

    async def answer_with_sql_context(
        self,
        question: str,
        sql_context: Dict[str, Any],
        business_overview: str = "",
    ) -> Optional[str]:
        """
        Answer a follow-up question grounded in the previous SQL result.

        Called when:
        - The previous assistant turn returned SQL data
        - The current message is a short follow-up with no new SQL keywords
        - e.g. "which ones use laravel?" after "there are 7 completed projects"

        Returns None if the context isn't helpful (caller falls back to FAQ).
        """
        prev_response = sql_context.get("assistant_response", "")
        row_count = sql_context.get("row_count", 0)
        explanation = sql_context.get("explanation", "")

        if not prev_response or row_count == 0:
            return None

        system_prompt = f"""You are a helpful assistant with access to live business data.
{f"Business context: {business_overview}" if business_overview else ""}

The user just received this data result:
---
{prev_response[:600]}
---
(This came from a database query: {explanation})

Now they are asking a follow-up question. Answer it using the data shown above.
- Be direct and specific — use the actual data, not generic statements
- If the follow-up asks to filter or narrow the previous results, do so
- If the data genuinely can't answer the follow-up, say so briefly and suggest they ask differently
- Keep it concise — no boilerplate, no "Strategic Implications"
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        try:
            reply = await self._dispatch(messages, settings.TEMPERATURE_GROUNDED)

            # Sanity check — if the LLM says it can't answer, return None
            cant_answer_signals = [
                "i don't have",
                "i cannot",
                "not available",
                "no information",
                "cannot answer",
                "don't have access",
                "unable to",
            ]
            if any(s in reply.lower()[:150] for s in cant_answer_signals):
                logger.info(
                    "answer_with_sql_context: LLM indicated it can't answer, falling back to FAQ"
                )
                return None

            logger.info(f"Grounded follow-up answered: {reply[:100]}...")
            return reply

        except Exception as e:
            logger.warning(f"answer_with_sql_context failed: {e}")
            return None

    # ─────────────────────────────────────────────────────────────────
    # Analytics response (internal "Digital Brain" users)
    # Point 3: adaptive verbosity — simple questions get concise answers
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
        if not data:
            return (
                "No data found for that query. This could mean the records don't exist yet, "
                "or the filter criteria didn't match anything. Try a different time range or rephrasing."
            )

        # Point 3: detect question complexity to calibrate response depth
        is_simple_count = row_count == 1 and any(
            kw in message.lower() for kw in ("how many", "count", "total", "number of")
        )
        is_lookup = row_count <= 5 and any(
            kw in message.lower()
            for kw in ("show me", "list", "which", "what", "find", "get")
        )

        data_summary = self._prepare_analytics_summary(data, row_count)

        if is_simple_count:
            # For simple count questions, skip the full BI narrative
            depth_instruction = (
                "Give a direct, concise answer with the number. "
                "Add one insight if it's genuinely interesting (e.g. breakdown by status). "
                "Skip 'Strategic Implications' and 'Follow-up Questions' unless the data is surprising."
            )
        elif is_lookup:
            # For listing/lookup questions, present the data clearly
            depth_instruction = (
                "Present the data clearly and concisely. Highlight the most relevant fields. "
                "Skip generic business commentary. One follow-up suggestion is fine if relevant."
            )
        else:
            # Complex analytical questions — full BI treatment
            depth_instruction = (
                "Provide a thorough analytical response:\n"
                "1. Direct answer with key figures\n"
                "2. Trends or patterns in the data\n"
                "3. Any anomalies worth flagging\n"
                "4. Brief strategic implication (1-2 sentences)\n"
                "5. One relevant follow-up question"
            )

        system_prompt = f"""You are a business intelligence analyst.
{f"Business context: {business_overview}" if business_overview else ""}

Style:
- Lead with the answer, not pleasantries
- Use specific numbers from the data
- Bold key figures
- No filler sentences
- {depth_instruction}
"""

        history_section = f"\nPrevious conversation:\n{history}\n" if history else ""

        user_prompt = f"""Question: "{message}"

Data ({row_count} records):
{data_summary}

Query context: {sql_explanation}
{history_section}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            reply = await self._dispatch(messages, settings.TEMPERATURE_GROUNDED)

            logger.info(f"Analytics response generated: {reply[:100]}...")
            return reply

        except Exception as e:
            logger.error(f"Analytics response generation failed: {str(e)}")
            return await self.format_query_results(message, data, sql_explanation)

    def _prepare_analytics_summary(self, data: List[Dict], row_count: int) -> str:
        if not data:
            return "No data returned."
        if row_count <= 10:
            return f"Full dataset:\n{json.dumps(data, indent=2, default=str)}"
        elif row_count <= 50:
            parts = [
                f"First 10 records:\n{json.dumps(data[:10], indent=2, default=str)}",
                f"\nLast 5 records:\n{json.dumps(data[-5:], indent=2, default=str)}",
            ]
            stats = self._compute_basic_stats(data)
            if stats:
                parts.append(f"\nAggregate stats:\n{json.dumps(stats, indent=2)}")
            return "\n".join(parts)
        else:
            parts = [f"Top 10 records:\n{json.dumps(data[:10], indent=2, default=str)}"]
            stats = self._compute_basic_stats(data)
            if stats:
                parts.append(
                    f"\nAggregate stats across all {row_count} records:\n{json.dumps(stats, indent=2)}"
                )
            return "\n".join(parts)

    def _compute_basic_stats(self, data: List[Dict]) -> Dict[str, Any]:
        if not data:
            return {}
        stats = {}
        numeric_cols = [
            k
            for k, v in data[0].items()
            if isinstance(v, (int, float)) and v is not None
        ]
        for col in numeric_cols[:5]:
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
            reply = await self._dispatch(messages, settings.TEMPERATURE_GROUNDED)

            reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL).strip()
            json_match = re.search(r"\{.*\}", reply, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            logger.error(f"No JSON found in response: {reply}")
            return {}

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {str(e)}")
            return {}
        except Exception as e:
            logger.error(f"Structured response generation failed: {str(e)}")
            return {}

    # ─────────────────────────────────────────────────────────────────
    # External user query formatting
    # ─────────────────────────────────────────────────────────────────

    async def format_query_results(
        self, question: str, data: List[Dict], explanation: str
    ) -> str:
        if not data:
            return "I couldn't find any matching records for your request."

        data_summary = f"Found {len(data)} results. "
        if len(data) <= 3:
            data_summary += f"Details: {json.dumps(data, indent=2, default=str)}"
        else:
            data_summary += (
                f"First 3 results: {json.dumps(data[:3], indent=2, default=str)}"
            )

        prompt = f"""Convert this database result into a natural, conversational response.

User Question: "{question}"
Query Purpose: {explanation}
Data: {data_summary}

Write a helpful, natural response. Do NOT mention table names, column names, or SQL.
Speak as if you just know the answer."""

        try:
            return await self.generate_response(
                message=prompt, context=[], history="", personality="helpful and clear"
            )
        except Exception as e:
            logger.error(f"Query result formatting failed: {str(e)}")
            return f"I found {len(data)} results. {explanation}"

    # ─────────────────────────────────────────────────────────────────
    # Provider implementations
    # ─────────────────────────────────────────────────────────────────

    async def _dispatch(self, messages: List[dict], temperature: float) -> str:
        """Single provider-selection point for every generation path.

        This logic used to be copy-pasted into five methods, which is how the
        engine ended up with a hardcoded 0.7 in two places and no single point
        at which a provider could be switched. Groq is the last resort on
        every path, so a provider outage degrades rather than fails.
        """
        provider = choose_provider(
            settings.LLM_PROVIDER,
            gemini_rate_limited=self._should_use_groq(),
        )
        logger.info(f"Generating with provider: {provider} (temp={temperature})")

        if provider == "bedrock":
            try:
                return await self._to_thread(self._generate_with_bedrock, messages, temperature)
            except Exception as e:
                logger.warning(f"Bedrock failed, falling back to Groq: {str(e)}")
                return await self._to_thread(self._generate_with_groq, messages, temperature)

        if provider == "groq":
            return await self._to_thread(self._generate_with_groq, messages, temperature)

        try:
            reply = await self._to_thread(self._generate_with_gemini, messages, temperature)
            self.rate_limiter.add_request("gemini")
            return reply
        except Exception as e:
            logger.warning(f"Gemini failed, falling back to Groq: {str(e)}")
            return await self._to_thread(self._generate_with_groq, messages, temperature)

    async def _to_thread(self, fn, *args):
        """Run a blocking SDK call off the event loop.

        Every provider SDK here is synchronous: boto3, google.genai and groq
        all block. Awaiting them directly inside a coroutine stalls the whole
        worker for the duration of the call, so concurrent chats serialise
        behind each other. This is the same threadpool discipline Trivia's
        voice intent router uses for exactly this reason.
        """
        return await asyncio.get_running_loop().run_in_executor(None, fn, *args)

    def _generate_with_bedrock(
        self, messages: List[dict], temperature: float
    ) -> str:
        try:
            return generate_bedrock(
                messages=messages,
                model=settings.BEDROCK_LLM_MODEL,
                region=settings.AWS_REGION,
                temperature=temperature,
            )
        except Exception as e:
            logger.error(f"Bedrock generation failed: {str(e)}")
            raise

    def _generate_with_gemini(
        self, messages: List[dict], temperature: float
    ) -> str:
        try:
            contents = self._convert_to_gemini_contents(messages)
            response = self.client.models.generate_content(
                model=self.gemini_model,
                contents=contents,
                config=types.GenerateContentConfig(temperature=temperature),
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini generation failed: {str(e)}")
            raise

    def _generate_with_groq(
        self, messages: List[dict], temperature: float
    ) -> str:
        try:
            response = groq_client.chat.completions.create(
                model=self.groq_model,
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Groq generation failed: {str(e)}")
            raise

    # ─────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────

    def _build_messages(
        self, message, context, history, personality, business_overview=None
    ):
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
        combined = ""
        for msg in messages:
            role, content = msg.get("role", ""), msg.get("content", "")
            if role == "system":
                combined += f"{content}\n\n"
            elif role == "user":
                combined += f"User: {content}\n\n"
            elif role == "assistant":
                combined += f"Assistant: {content}\n\n"
        return combined.strip()

    def get_rate_limit_stats(self) -> Dict[str, Any]:
        stats = self.rate_limiter.get_stats()
        # Must reflect the same decision _dispatch makes, or this reports a
        # provider that is not actually serving traffic.
        stats["current_provider"] = choose_provider(
            settings.LLM_PROVIDER,
            gemini_rate_limited=self._should_use_groq(),
        )
        return stats


LLMServiceHuggingFace = LLMService
