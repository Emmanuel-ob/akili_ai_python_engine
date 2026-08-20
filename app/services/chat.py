import asyncio
from typing import List, Dict, Any, Optional, Literal
from app.schemas.chat import ChatResponse, HistoryMessage, SourceDocument
from app.services.vectorstore import VectorStoreService
from app.services.embeddings import EmbeddingService
from app.services.llm import LLMService
from app.services.conversation_memory import ConversationMemory, ConversationSummarizer
from app.services.query_classifier import QueryClassifier
from app.services.intent_router import DEFAULT_INTENT, classify as classify_intent
from app.services.sql_generator import SQLGenerator
from app.services.sql_validator import SQLValidator
from app.services.sql_executor import DirectSQLExecutor
from app.services.config_cache import ConfigCache
from app.services.response_sanitizer import ResponseSanitizer
from app.services.retrieval_gate import gate_and_order
from app.services.insight_engine import get_recent_insights_summary  # Point 5A
from app.core.config import settings
from app.core.logging_config import logger


def _build_visitor_context(identity: dict) -> str:
    """Build a one-line visitor context string for prompt injection."""
    if not identity:
        return ""
    parts = []
    if identity.get("name"):
        parts.append(f"Name: {identity['name']}")
    if identity.get("email"):
        parts.append(f"Email: {identity['email']}")
    if identity.get("user_id"):
        parts.append(f"ID: {identity['user_id']}")
    if identity.get("plan"):
        parts.append(f"Plan: {identity['plan']}")
    if identity.get("phone"):
        parts.append(f"Phone: {identity['phone']}")
    if identity.get("custom") and isinstance(identity["custom"], dict):
        for k, v in list(identity["custom"].items())[:3]:
            parts.append(f"{k}: {v}")
    return " | ".join(parts) if parts else ""


class ChatService:
    def __init__(
        self,
        vectorstore_service: VectorStoreService,
        embedding_service: EmbeddingService,
        llm_service: LLMService,
    ):
        self.vectorstore = vectorstore_service
        self.embedding = embedding_service
        self.llm = llm_service
        self.memory = ConversationMemory()
        self.summarizer = ConversationSummarizer(llm_service)
        self.sql_generator = SQLGenerator(llm_service)
        self.config_cache = ConfigCache()


    # ─────────────────────────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────────────────────────

    async def generate_response(
        self,
        business_id: str,
        chatbot_id: str,
        message: str,
        history: List[HistoryMessage],
        chatbot_config: Dict[str, Any],
        user_type: Literal["internal", "external"] = "external",  # ✅ NEW
    ) -> ChatResponse:
        """
        HYBRID response generation with user-type-aware routing.

        user_type="internal"  → Digital Brain mode
            Full analytics, trend analysis, predictions, unrestricted data access.
            Acts as a business intelligence analyst for staff/owners.

        user_type="external"  → Customer support mode
            Sanitized responses, customer-scoped data only, no schema leakage.
            Acts as a friendly customer service agent.
        """
        try:
            logger.info(f"generate_response called with user_type={user_type}")

            business_overview = chatbot_config.get("business_overview", "")
            handoff_enabled = chatbot_config.get("handoff_to_human", False)

            # Item 7: Inject chatbot_type persona prefix so the LLM adopts the right role and tone.
            # This runs for external (customer-facing) users only — internal/dashboard users get
            # unrestricted analyst mode regardless of chatbot_type.
            if user_type == "external":
                chatbot_type = chatbot_config.get("chatbot_type", "general")
                type_persona_prefix = self._get_type_persona(chatbot_type)
                if type_persona_prefix and business_overview:
                    business_overview = f"{type_persona_prefix}\n\n{business_overview}"
                elif type_persona_prefix:
                    business_overview = type_persona_prefix
                chatbot_config["business_overview"] = business_overview

            # Cache database config from payload
            connection_id = chatbot_config.get("connection_id")
            database_config = chatbot_config.get("database_config")

            if connection_id and database_config:
                self.config_cache.set(connection_id, database_config)
                logger.info(f"Cached config for connection: {connection_id}")

            # Point 1: Prefer rich DatabaseProfile over legacy schema_analysis.
            # database_profile is built once at sync time and contains full schema,
            # real FK constraints, enum hints, entity types, and business narrative.
            # Falls back gracefully to legacy schema_analysis if profile not yet built.
            database_profile = chatbot_config.get("database_profile")
            if database_profile:
                schema_analysis = self._profile_to_schema(database_profile)
                # Point 2: pass allowlist into SQL generator
                chatbot_config["_selected_tables"] = database_profile.get(
                    "selected_tables", []
                )
                chatbot_config["_table_classifications"] = database_profile.get(
                    "table_classifications", {}
                )
                logger.info(
                    f"Using DatabaseProfile (Point 1) for connection {connection_id}"
                )
            else:
                schema_analysis = chatbot_config.get("schema_analysis", {})
                logger.info(
                    f"Using legacy schema_analysis for connection {connection_id}"
                )

            # Point 4: extract visitor identity from widget XeliAI.identify()
            # Used as customer_id for SQL filtering + injected into prompts
            visitor_identity = chatbot_config.get("visitor_identity") or {}
            if visitor_identity:
                logger.info(
                    f"Visitor identified: name={visitor_identity.get('name')}, "
                    f"email={visitor_identity.get('email')}, id={visitor_identity.get('user_id')}"
                )
                # Use email or user_id as the customer_id for SQL filtering
                chatbot_config.setdefault(
                    "customer_id",
                    visitor_identity.get("user_id") or visitor_identity.get("email"),
                )
                # Enrich business_overview with visitor context so LLM knows who it's talking to
                visitor_ctx = _build_visitor_context(visitor_identity)
                if visitor_ctx:
                    existing = chatbot_config.get("business_overview", "")
                    chatbot_config["business_overview"] = (
                        f"{existing}\n\nCURRENT VISITOR: {visitor_ctx}"
                        if existing
                        else f"CURRENT VISITOR: {visitor_ctx}"
                    )

            # B1: Cross-session visitor memory — inject accumulated profile for returning visitors
            visitor_context = chatbot_config.get("visitor_context")
            if visitor_context and user_type == "external":
                existing = chatbot_config.get("business_overview", "")
                chatbot_config["business_overview"] = (
                    f"{existing}\n\nRETURNING VISITOR PROFILE: {visitor_context}"
                    if existing
                    else f"RETURNING VISITOR PROFILE: {visitor_context}"
                )
                logger.info(f"B1: Injected returning visitor context into prompt")

            has_database = bool(connection_id and schema_analysis and database_config)

            # ONE classifier call decides both routing and handover, so the
            # two can never disagree about the same message. Previously each
            # ran its own substring keyword match, and "contact" appeared in
            # both lists at once.
            #
            # boto3 blocks, so this runs in a threadpool. Timing out or
            # failing yields DEFAULT_INTENT: route FAQ, no handover.
            intent = await self._classify_intent(message, history, has_database)
            route = intent["route"]
            logger.info(f"Query route: {route}, user_type: {user_type}")

            # Handover is external-only and requires the business to have
            # enabled it. An uncertain classification never hands over: see
            # the confidence gate in intent_router._normalise.
            if user_type == "external" and handoff_enabled and intent["wants_human"]:
                logger.info(
                    f"Handover requested (confidence {intent['confidence']}): "
                    f"{intent['reason']}"
                )
                return ChatResponse(
                    text="I understand you'd like to speak with a human agent. Let me connect you with someone from our team.",
                    sources=[],
                    metadata={
                        "type": "handover_request",
                        "confidence": intent["confidence"],
                        "handover_requested": True,
                    },
                )

            # ─────────────────────────────────────────────────
            # Route to appropriate handler
            # ─────────────────────────────────────────────────

            if route == "SQL" and has_database:
                response = await self._handle_sql_query(
                    message,
                    business_id,
                    chatbot_id,
                    connection_id,
                    schema_analysis,
                    chatbot_config,
                    user_type=user_type,
                )
            elif route == "HYBRID" and has_database:
                response = await self._handle_hybrid_query(
                    message,
                    business_id,
                    chatbot_id,
                    connection_id,
                    schema_analysis,
                    history,
                    chatbot_config,
                    user_type=user_type,
                )
            else:
                response = await self._handle_faq_query(
                    message,
                    business_id,
                    chatbot_id,
                    history,
                    business_overview,
                    user_type=user_type,
                    chatbot_config=chatbot_config,
                )

            # ─────────────────────────────────────────────────
            # ✅ Sanitize response for external users
            # Internal users get the full response.
            # ─────────────────────────────────────────────────
            if user_type == "external":
                response = ResponseSanitizer.sanitize(response)
                logger.info("Response sanitized for external user")

            return response

        except Exception as e:
            logger.error(f"Error in chat service: {str(e)}", exc_info=True)
            fallback_message = chatbot_config.get(
                "fallback_message",
                "I'm sorry, I'm having trouble responding right now. Could you please rephrase?",
            )
            return ChatResponse(
                text=fallback_message,
                sources=[],
                metadata={"error": True, "fallback": True},
            )

    # ─────────────────────────────────────────────────────────────────
    # SQL handler — forks on user_type
    # ─────────────────────────────────────────────────────────────────

    async def _handle_sql_query(
        self,
        message: str,
        business_id: str,
        chatbot_id: str,
        connection_id: str,
        schema_analysis: Dict[str, Any],
        chatbot_config: Dict[str, Any],
        user_type: str = "external",
    ) -> ChatResponse:
        """
        SQL handler with user-type-aware response formatting.

        Internal users  → generate_analytics_response() — rich BI narrative
        External users  → format_query_results() — plain customer-friendly text
        """
        try:
            database_config = self.config_cache.get(connection_id)

            if not database_config:
                logger.error(f"Config not in cache for connection: {connection_id}")
                return ChatResponse(
                    text="Database configuration not available. Please try again.",
                    sources=[],
                    metadata={"type": "config_error"},
                )

            tables = schema_analysis.get("tables", {})
            database_type = database_config.get("type", "postgresql")

            logger.info(f"Generating SQL for: {message} (user_type={user_type})")

            sql_result = await self.sql_generator.generate_sql(
                question=message,
                schema=tables,
                database_type=database_type,
                customer_id=chatbot_config.get("customer_id"),
                is_authenticated=chatbot_config.get("is_authenticated", False),
                # Point 2: allowlist enforcement
                selected_tables=chatbot_config.get("_selected_tables"),
                table_classifications=chatbot_config.get("_table_classifications"),
            )

            if not sql_result.get("safe"):
                explanation = sql_result.get("explanation", "")
                blocked_reason = sql_result.get("blocked_reason", "")

                # Point 3: friendly error messages instead of raw technical errors
                if blocked_reason == "table_not_allowed":
                    # Our validator caught a forbidden table — use the clean message
                    user_message = explanation
                elif "No SQL generated" in explanation or not explanation:
                    # LLM refused — likely because the question references unavailable data
                    user_message = (
                        "I can only answer questions about the data in this system. "
                        "That information isn't available here — try asking about something else."
                    )
                else:
                    user_message = explanation

                return ChatResponse(
                    text=user_message,
                    sources=[],
                    metadata={"type": "sql_error", "reason": "unsafe_query"},
                )

            validation = SQLValidator.validate(sql_result["sql"], database_type)
            if not validation["valid"]:
                return ChatResponse(
                    text="I couldn't process that query. Please try rephrasing your question.",
                    sources=[],
                    metadata={
                        "type": "sql_error",
                        "validation_errors": validation["errors"],
                    },
                )

            logger.info(f"Executing SQL: {sql_result['sql']}")

            execution_result = await DirectSQLExecutor.execute_with_config(
                sql=sql_result["sql"],
                database_config=database_config,
            )

            if not execution_result.get("success"):
                logger.error(f"SQL execution failed: {execution_result.get('error')}")
                return ChatResponse(
                    text="I had trouble retrieving that data. Please try again.",
                    sources=[],
                    metadata={
                        "type": "sql_error",
                        "error": execution_result.get("error"),
                    },
                )

            row_count = execution_result.get("row_count", 0)
            logger.info(f"Query returned {row_count} rows")

            # Retry with flexible matching if no results
            if row_count == 0:
                logger.info("Retrying with flexible matching...")
                retry_result = await self.sql_generator.generate_sql_with_retry(
                    question=message,
                    schema=tables,
                    database_type=database_type,
                    customer_id=chatbot_config.get("customer_id"),
                    is_authenticated=chatbot_config.get("is_authenticated", False),
                    previous_attempt=sql_result,
                    selected_tables=chatbot_config.get("_selected_tables"),
                    table_classifications=chatbot_config.get("_table_classifications"),
                )

                if retry_result.get("sql") and retry_result["sql"] != sql_result["sql"]:
                    retry_execution = await DirectSQLExecutor.execute_with_config(
                        sql=retry_result["sql"],
                        database_config=database_config,
                    )
                    if (
                        retry_execution.get("success")
                        and retry_execution.get("row_count", 0) > 0
                    ):
                        execution_result = retry_execution
                        sql_result = retry_result
                        row_count = execution_result.get("row_count", 0)
                        logger.info(f"Retry successful: {row_count} rows")

            # ─────────────────────────────────────────────────
            # ✅ FORK: Internal vs External response formatting
            # ─────────────────────────────────────────────────

            if user_type == "internal":
                # Digital Brain mode: rich BI analysis
                logger.info("Formatting analytics response for internal user")
                response_text = await self.llm.generate_analytics_response(
                    message=message,
                    data=execution_result.get("data", []),
                    sql_explanation=sql_result.get("explanation", ""),
                    business_overview=chatbot_config.get("business_overview", ""),
                    history=self._format_history(chatbot_config.get("history", [])),
                    row_count=row_count,
                )

                return ChatResponse(
                    text=response_text,
                    sources=[
                        SourceDocument(
                            doc_id="sql_result",
                            text=f"SQL: {sql_result['sql']}",
                            confidence=0.95,
                            table="database",
                        )
                    ],
                    metadata={
                        "type": "sql",
                        "sql": sql_result["sql"],  # Internal users can see SQL
                        "row_count": row_count,
                        "explanation": sql_result.get("explanation", ""),
                        "tables_used": sql_result.get("tables_used", []),
                        "execution_time_ms": execution_result.get("execution_time_ms"),
                    },
                )

            else:
                # External mode: plain, customer-friendly formatting
                logger.info("Formatting plain response for external user")
                response_text = await self.llm.format_query_results(
                    question=message,
                    data=execution_result.get("data", []),
                    explanation=sql_result.get("explanation", ""),
                )

                return ChatResponse(
                    text=response_text,
                    sources=[],  # No sources exposed to external users
                    metadata={
                        "type": "sql",
                        "row_count": row_count,
                        # sql, tables_used, explanation NOT included — sanitizer
                        # will catch anything that slips through
                    },
                )

        except Exception as e:
            logger.error(f"SQL query handling failed: {str(e)}", exc_info=True)
            return ChatResponse(
                text="I encountered an error while retrieving that information. Please try rephrasing your question.",
                sources=[],
                metadata={"type": "sql_error", "error": str(e)},
            )

    # ─────────────────────────────────────────────────────────────────
    # Hybrid handler
    # ─────────────────────────────────────────────────────────────────

    async def _handle_hybrid_query(
        self,
        message: str,
        business_id: str,
        chatbot_id: str,
        connection_id: str,
        schema_analysis: Dict[str, Any],
        history: List[HistoryMessage],
        chatbot_config: Dict[str, Any],
        user_type: str = "external",
    ) -> ChatResponse:
        sql_response = await self._handle_sql_query(
            message,
            business_id,
            chatbot_id,
            connection_id,
            schema_analysis,
            chatbot_config,
            user_type=user_type,
        )

        faq_response = await self._search_faq(
            business_id,
            chatbot_id,
            message,
            history,
            chatbot_config.get("business_overview", ""),
        )

        combined_parts = []
        if sql_response and sql_response.text:
            combined_parts.append(sql_response.text)

        if faq_response and faq_response.text:
            if user_type == "internal":
                combined_parts.append(
                    "\n\n---\n**Additional context from knowledge base:**"
                )
            else:
                combined_parts.append("\n\nAdditional information:")
            combined_parts.append(faq_response.text)

        combined_text = (
            "\n".join(combined_parts)
            if combined_parts
            else "I couldn't find relevant information."
        )

        all_sources = sql_response.sources.copy()
        if faq_response:
            all_sources.extend(faq_response.sources)

        return ChatResponse(
            text=combined_text,
            sources=all_sources,
            metadata={
                "type": "hybrid",
                "sql_metadata": sql_response.metadata,
                "faq_metadata": faq_response.metadata if faq_response else {},
            },
        )

    # ─────────────────────────────────────────────────────────────────
    # FAQ handler (same for both user types — sanitizer handles cleanup)
    # ─────────────────────────────────────────────────────────────────

    async def _handle_faq_query(
        self,
        message: str,
        business_id: str,
        chatbot_id: str,
        history: List[HistoryMessage],
        business_overview: str,
        user_type: str = "external",
        chatbot_config: Optional[Dict[str, Any]] = None,
    ) -> ChatResponse:
        # Point 5C: Internal users get the richer BI-analyst prompt path —
        # BUT only for actual data/business questions, not greetings or small talk.
        if user_type == "internal" and not self._is_conversational(message):
            recent_insights = await get_recent_insights_summary(
                (chatbot_config or {}).get("recent_insights", [])
            )
            # Search internal_report chunks as well as FAQ
            faq_ctx = await self._search_knowledge(
                business_id,
                chatbot_id,
                message,
                history,
                business_overview,
                include_internal_reports=True,
            )
            context_texts = [r.payload["text"] for r in (faq_ctx or [])]

            history_text = self._format_history(history[-8:])
            response_text = await self.llm.generate_internal_response(
                message=message,
                context=context_texts,
                history=history_text,
                business_name=(chatbot_config or {}).get("business_name", ""),
                user_role=((chatbot_config or {}).get("visitor_identity") or {}).get(
                    "plan", ""
                ),
                recent_insights=recent_insights,
            )
            return ChatResponse(
                text=response_text,
                sources=[],
                metadata={"type": "internal_faq", "internal_mode": True},
            )

        # External path — same as before
        # Point 3: if last turn was a SQL result, try to answer grounded in that data
        last_sql_ctx = QueryClassifier.extract_last_sql_context(
            [m.dict() if hasattr(m, "dict") else m for m in (history or [])]
        )
        if last_sql_ctx and len(message.split()) <= 12:
            # Short follow-up after SQL — answer using the actual data context
            grounded = await self.llm.answer_with_sql_context(
                question=message,
                sql_context=last_sql_ctx,
                business_overview=business_overview,
            )
            if grounded:
                return ChatResponse(
                    text=grounded,
                    sources=[],
                    metadata={"type": "sql_followup", "grounded": True},
                )

        faq_response = await self._search_faq(
            business_id, chatbot_id, message, history, business_overview
        )

        if faq_response:
            return faq_response

        if business_overview:
            return await self._handle_business_context(
                message, history, business_overview
            )

        return await self._handle_general_chat(message, history, business_overview)

    async def _search_knowledge(
        self,
        business_id: str,
        chatbot_id: str,
        message: str,
        history,
        business_overview: str,
        include_internal_reports: bool = False,
    ):
        """Search FAQ chunks + optionally internal_report chunks (Point 5B)."""
        collection_name = self.vectorstore.get_collection_name(business_id, chatbot_id)
        query_embedding = self.embedding.generate_query_embedding(message)

        # Build source_type filter
        source_filter = "faq" if not include_internal_reports else None

        results = self.vectorstore.search_similar(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=settings.DEFAULT_SEARCH_LIMIT,
            business_id=business_id,
            chatbot_id=chatbot_id,
            source_type_filter=source_filter,
        )

        # Relevance floor and priority ordering both live in retrieval_gate, so
        # this path and _search_faq can never drift apart again.
        return gate_and_order(results, threshold=settings.MIN_CONFIDENCE_THRESHOLD)

    async def _search_faq(
        self,
        business_id: str,
        chatbot_id: str,
        message: str,
        history: List[HistoryMessage],
        business_overview: str,
    ) -> Optional[ChatResponse]:
        collection_name = self.vectorstore.get_collection_name(business_id, chatbot_id)
        query_embedding = self.embedding.generate_query_embedding(message)

        search_results = self.vectorstore.search_similar(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=settings.DEFAULT_SEARCH_LIMIT,
            business_id=business_id,
            chatbot_id=chatbot_id,
            source_type_filter="faq",
        )

        # Gate on relevance FIRST, then order by admin priority among the
        # survivors. The reverse used to be true: everything was sorted by
        # priority and only the first item was score-checked, so a pinned but
        # irrelevant chunk became both the gate and the answer.
        search_results = gate_and_order(
            search_results, threshold=settings.MIN_CONFIDENCE_THRESHOLD
        )

        if not search_results:
            logger.info(
                f"No FAQ chunk cleared the relevance floor "
                f"({settings.MIN_CONFIDENCE_THRESHOLD})"
            )
            return None

        context_texts = []
        sources = []

        for result in search_results[:3]:
            context_texts.append(result.payload["text"])
            sources.append(
                SourceDocument(
                    doc_id=result.payload["doc_id"],
                    text=result.payload["text"][:200] + "...",
                    confidence=result.score,
                    table=result.payload.get("metadata", {}).get(
                        "faq_source_name", "FAQ"
                    ),
                )
            )

        history_context = self._format_history(history[-6:])

        response_text = await self.llm.generate_response(
            message=message,
            context=context_texts,
            history=history_context,
            personality="helpful and professional",
            business_overview=business_overview,
        )

        confidence = self._calculate_confidence(search_results[:3])

        return ChatResponse(
            text=response_text,
            sources=sources,
            metadata={
                "type": "faq",
                "confidence": confidence,
                "source": "knowledge_base",
            },
        )

    async def _handle_business_context(
        self, message: str, history: List[HistoryMessage], business_overview: str
    ) -> ChatResponse:
        history_text = self._format_history(history[-5:])
        response_text = await self.llm.generate_response(
            message=message,
            context=[business_overview],
            history=history_text,
            personality="helpful and professional",
            business_overview=business_overview,
        )
        return ChatResponse(
            text=response_text, sources=[], metadata={"type": "business_context"}
        )

    async def _handle_general_chat(
        self, message: str, history: List[HistoryMessage], business_overview: str
    ) -> ChatResponse:
        history_text = self._format_history(history[-5:])
        response_text = await self.llm.generate_response(
            message=message,
            context=[],
            history=history_text,
            personality="friendly and helpful",
            business_overview=business_overview,
            # The only path with no retrieved context to be faithful to, so it
            # keeps the warmer setting. Every other path inherits the grounded
            # default, where creative phrasing is the hallucination.
            temperature=settings.TEMPERATURE_CONVERSATIONAL,
        )
        return ChatResponse(
            text=response_text, sources=[], metadata={"type": "general_chat"}
        )

    # ─────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────

    async def _classify_intent(
        self, message: str, history, has_database: bool
    ) -> Dict[str, Any]:
        """Classify one message: which source answers it, and does the customer
        want a person.

        Runs in a threadpool because boto3 is synchronous and this sits on the
        critical path of every reply; awaiting it directly would stall the
        worker for the whole round trip.

        Bounded by INTENT_TIMEOUT_SECONDS. On timeout, failure, or the router
        being disabled, returns DEFAULT_INTENT: route FAQ, no handover. A
        classifier outage must never start ejecting customers into a queue.
        """
        if not settings.INTENT_ROUTER_ENABLED:
            return DEFAULT_INTENT

        turns = [m.dict() if hasattr(m, "dict") else m for m in (history or [])]

        try:
            return await asyncio.wait_for(
                asyncio.get_running_loop().run_in_executor(
                    None,
                    classify_intent,
                    message,
                    turns,
                    has_database,
                    settings.INTENT_ROUTER_MODEL,
                    settings.AWS_REGION,
                    settings.INTENT_CONFIDENCE_THRESHOLD,
                ),
                timeout=settings.INTENT_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"Intent router exceeded {settings.INTENT_TIMEOUT_SECONDS}s; "
                f"using the safe default"
            )
            return DEFAULT_INTENT

    def _profile_to_schema(self, profile: dict) -> dict:
        """
        Point 1: Convert DatabaseProfile to the schema dict SQLGenerator expects.
        This means sql_generator.py needs zero changes — it still receives the same
        {tables: {table: {columns, foreign_keys, entity_type, ...}}} structure.
        """
        schema = {"tables": {}}

        for table, summary in profile.get("schema_summary", {}).items():
            schema["tables"][table] = {
                "columns": summary.get("columns", []),
                "foreign_keys": summary.get("foreign_keys", []),
                "entity_type": summary.get("entity_type", "unknown"),
                "customer_identifier": summary.get("customer_identifier"),
                "enum_hints": summary.get("enum_hints", {}),
                "sample_values": summary.get("sample_values", []),
            }

        schema["business_type"] = profile.get("business_type", "unknown")
        schema["relationships"] = profile.get("relationships", [])
        return schema

    def _is_conversational(self, message: str) -> bool:
        """
        Returns True if the message is a greeting, small talk, or too short
        to be a meaningful business/data question.
        Internal users who say "hi" should get a normal greeting,
        not a BI analyst response.
        """
        msg = message.strip().lower()

        # Too short to be a real question
        if len(msg) <= 4:
            return True

        # Greeting patterns
        greetings = [
            "hi",
            "hello",
            "hey",
            "hiya",
            "howdy",
            "good morning",
            "good afternoon",
            "good evening",
            "good day",
            "what's up",
            "whats up",
            "sup",
            "yo",
            "greetings",
            "how are you",
            "how r u",
            "how do you do",
            "nice to meet you",
        ]
        for g in greetings:
            if (
                msg == g
                or msg.startswith(g + " ")
                or msg.startswith(g + "!")
                or msg.startswith(g + ",")
            ):
                return True

        # Pure small talk — no data keywords at all
        data_keywords = [
            "how many",
            "how much",
            "count",
            "total",
            "revenue",
            "sales",
            "orders",
            "customers",
            "users",
            "products",
            "show me",
            "list",
            "which",
            "what is",
            "what are",
            "when",
            "where",
            "who",
            "compare",
            "trend",
            "report",
            "data",
            "analytics",
            "metric",
            "average",
            "top",
            "best",
            "worst",
            "last",
            "today",
            "week",
            "month",
            "year",
            "pending",
            "completed",
            "projects",
        ]
        has_data_intent = any(kw in msg for kw in data_keywords)
        return not has_data_intent

    def _get_type_persona(self, chatbot_type: str) -> str:
        """
        Item 7: Returns a system prompt prefix based on the chatbot type.
        This shapes the AI's focus and tone without overriding the business's
        custom personality or FAQ knowledge.
        """
        personas = {
            "customer_service": (
                "You are a customer service specialist. Your priorities are: "
                "resolving the customer's issue quickly and clearly, showing empathy when they are frustrated, "
                "providing actionable next steps, and escalating to a human agent when the issue is beyond your scope. "
                "Keep responses concise and solution-focused. Never make the customer feel dismissed."
            ),
            "sales": (
                "You are a knowledgeable sales assistant. Your priorities are: "
                "understanding what the customer is looking for, clearly communicating product benefits relevant to their needs, "
                "addressing objections with honest and confident answers, and guiding interested customers toward "
                "the next step (demo, free trial, or speaking with the team). "
                "Be enthusiastic but never pushy."
            ),
            "general": "",  # No prefix — default behaviour
        }
        return personas.get(chatbot_type, "")

    def _format_history(self, history) -> str:
        if not history:
            return ""

        formatted = []
        for msg in history:
            if hasattr(msg, "role"):
                role = "User" if msg.role == "user" else "Assistant"
                message = msg.message if hasattr(msg, "message") else str(msg)
            elif isinstance(msg, dict):
                role = "User" if msg.get("role") == "user" else "Assistant"
                message = msg.get("message", "")
            else:
                continue
            formatted.append(f"{role}: {message}")

        return "\n".join(formatted)

    def _calculate_confidence(self, search_results) -> float:
        if not search_results:
            return 0.0
        top_scores = [result.score for result in search_results[:3]]
        avg_score = sum(top_scores) / len(top_scores)
        return min(max(avg_score, 0.0), 1.0)
