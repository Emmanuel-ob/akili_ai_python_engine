from typing import List, Dict, Any, Optional, Literal
from app.schemas.chat import ChatResponse, HistoryMessage, SourceDocument
from app.services.vectorstore import VectorStoreService
from app.services.embeddings import EmbeddingService
from app.services.llm import LLMService
from app.services.conversation_memory import ConversationMemory, ConversationSummarizer
from app.services.query_classifier import QueryClassifier
from app.services.sql_generator import SQLGenerator
from app.services.sql_validator import SQLValidator
from app.services.sql_executor import DirectSQLExecutor
from app.services.config_cache import ConfigCache
from app.services.response_sanitizer import ResponseSanitizer  # ✅ NEW
from app.core.config import settings
from app.core.logging_config import logger


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

        self.handover_keywords = [
            "human",
            "agent",
            "person",
            "representative",
            "support",
            "speak to someone",
            "talk to someone",
            "real person",
            "live chat",
            "live agent",
            "customer service",
            "operator",
            "help desk",
            "escalate",
            "supervisor",
            "manager",
        ]

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

            # Handover detection (only relevant for external users)
            if (
                user_type == "external"
                and handoff_enabled
                and self._is_handover_request(message)
            ):
                logger.info("Handover request detected (external user)")
                return ChatResponse(
                    text="I understand you'd like to speak with a human agent. Let me connect you with someone from our team.",
                    sources=[],
                    metadata={
                        "type": "handover_request",
                        "confidence": 1.0,
                        "handover_requested": True,
                    },
                )

            # Cache database config from payload
            connection_id = chatbot_config.get("connection_id")
            database_config = chatbot_config.get("database_config")

            if connection_id and database_config:
                self.config_cache.set(connection_id, database_config)
                logger.info(f"Cached config for connection: {connection_id}")

            schema_analysis = chatbot_config.get("schema_analysis", {})
            has_database = bool(connection_id and schema_analysis and database_config)

            route = QueryClassifier.classify(message, has_database)
            logger.info(f"Query route: {route}, user_type: {user_type}")

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
            )

            if not sql_result.get("safe"):
                return ChatResponse(
                    text=sql_result.get("explanation", "Cannot process this query"),
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
    ) -> ChatResponse:
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
            limit=5,
            business_id=business_id,
            chatbot_id=chatbot_id,
            source_type_filter="faq",
        )

        if not search_results:
            return None

        search_results.sort(
            key=lambda x: x.payload.get("metadata", {}).get("faq_priority", 5),
            reverse=True,
        )

        top_result = search_results[0]
        if top_result.score < 0.20:
            logger.info(f"FAQ confidence too low: {top_result.score}")
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
        )
        return ChatResponse(
            text=response_text, sources=[], metadata={"type": "general_chat"}
        )

    # ─────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────

    def _is_handover_request(self, message: str) -> bool:
        message_lower = message.lower()

        for keyword in self.handover_keywords:
            if keyword in message_lower:
                negative_indicators = ["no", "not", "don't", "without"]
                has_negation = any(neg in message_lower for neg in negative_indicators)
                if not has_negation:
                    logger.info(f"Handover keyword detected: {keyword}")
                    return True

        handover_phrases = [
            "speak with",
            "talk to",
            "connect me",
            "transfer me",
            "chat with",
            "contact",
            "i need help from",
            "can i speak",
            "is there someone",
            "get a person",
        ]

        for phrase in handover_phrases:
            if phrase in message_lower:
                logger.info(f"Handover phrase detected: {phrase}")
                return True

        return False

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
