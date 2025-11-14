from typing import List, Dict, Any, Optional
from app.schemas.chat import ChatResponse, HistoryMessage, SourceDocument
from app.services.vectorstore import VectorStoreService
from app.services.embeddings import EmbeddingServiceOpenai, EmbeddingServiceHuggingFace
from app.services.llm import LLMServiceHuggingFace
from app.services.conversation_memory import ConversationMemory, ConversationSummarizer
from app.core.config import settings
from app.core.logging_config import logger


class ChatService:
    def __init__(
        self,
        vectorstore_service: VectorStoreService,
        embedding_service_openai: EmbeddingServiceOpenai,
        embedding_service_huggingface: EmbeddingServiceHuggingFace,
        llm_service: LLMServiceHuggingFace,
    ):
        self.vectorstore = vectorstore_service
        self.embedding_openai = embedding_service_openai
        self.embedding_huggingface = embedding_service_huggingface
        self.llm = llm_service
        self.memory = ConversationMemory()
        self.summarizer = ConversationSummarizer(llm_service)

        # Handover keywords
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
            "representative",
            "operator",
            "help desk",
            "escalate",
            "supervisor",
            "manager",
        ]

    async def generate_response(
        self,
        business_id: str,
        chatbot_id: str,
        message: str,
        history: List[HistoryMessage],
        chatbot_config: Dict[str, Any],
    ) -> ChatResponse:
        """
        FAQ-first response generation with handover detection
        """

        try:
            business_overview = chatbot_config.get("business_overview", "")

            # ===== NEW: Check if handover is enabled =====
            handoff_enabled = chatbot_config.get("handoff_to_human", False)

            # ===== NEW: Detect handover request =====
            if handoff_enabled and self._is_handover_request(message):
                logger.info("Handover request detected")
                return ChatResponse(
                    text="I understand you'd like to speak with a human agent. Let me connect you with someone from our team. Please wait a moment while I find an available representative.",
                    sources=[],
                    metadata={
                        "type": "handover_request",
                        "confidence": 1.0,
                        "handover_requested": True,
                    },
                )

            # Check for follow-up context
            reference_context = self.memory.resolve_reference(message)
            if reference_context.get("is_followup"):
                logger.info(
                    f"Detected follow-up question about: {reference_context.get('current_topic')}"
                )

            # STEP 1: Search FAQ first (high confidence threshold)
            faq_response = await self._search_faq(
                business_id, chatbot_id, message, history, business_overview
            )

            if faq_response:
                logger.info("Returning FAQ-based response")
                return faq_response

            # STEP 2: Use business context if available
            if business_overview:
                logger.info("Using business context for response")
                return await self._handle_business_context(
                    message, history, business_overview
                )

            # STEP 3: General chat fallback
            logger.info("Using general chat response")
            return await self._handle_general_chat(message, history, business_overview)

        except Exception as e:
            logger.error(f"Error in chat service: {str(e)}")

            fallback_message = chatbot_config.get(
                "fallback_message",
                "I'm sorry, I'm having trouble responding right now. Could you please rephrase?",
            )

            return ChatResponse(
                text=fallback_message,
                sources=[],
                metadata={"error": True, "fallback": True},
            )

    def _is_handover_request(self, message: str) -> bool:
        """
        Detect if user is requesting to speak with a human agent
        Uses keyword matching with context awareness
        """
        message_lower = message.lower()

        # Direct matches
        for keyword in self.handover_keywords:
            if keyword in message_lower:
                # Additional context check to avoid false positives
                negative_indicators = ["no", "not", "don't", "without"]

                # Check if negation is present
                has_negation = any(neg in message_lower for neg in negative_indicators)

                if not has_negation:
                    logger.info(f"Handover keyword detected: {keyword}")
                    return True

        # Check for common phrases
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

    async def _search_faq(
        self,
        business_id: str,
        chatbot_id: str,
        message: str,
        history: List[HistoryMessage],
        business_overview: str,
    ) -> Optional[ChatResponse]:
        """
        Search FAQ embeddings with high confidence threshold
        Returns None if no good match found
        """

        collection_name = self.vectorstore.get_collection_name(business_id, chatbot_id)
        query_embedding = self.embedding_huggingface.generate_single_embeddings(message)

        # Search with FAQ filter
        search_results = self.vectorstore.search_similar(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=5,
            business_id=business_id,
            chatbot_id=chatbot_id,
            source_type_filter="faq",  # Only search FAQ
        )

        if not search_results:
            return None

        # Sort by priority (stored in metadata)
        search_results.sort(
            key=lambda x: x.payload.get("metadata", {}).get("faq_priority", 5),
            reverse=True,
        )

        # Check confidence threshold
        top_result = search_results[0]
        if top_result.score < 0.20:  # 70% confidence threshold for FAQ
            logger.info(f"FAQ confidence too low: {top_result.score}")
            return None

        # Build context from top results
        context_texts = []
        sources = []

        for result in search_results[:3]:  # Use top 3 results
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
        """Handle queries using business context/overview"""

        history_text = self._format_history(history[-5:])

        response_text = await self.llm.generate_response(
            message=message,
            context=[business_overview],
            history=history_text,
            personality="helpful and professional",
            business_overview=business_overview,
        )

        return ChatResponse(
            text=response_text,
            sources=[],
            metadata={"type": "business_context"},
        )

    async def _handle_general_chat(
        self, message: str, history: List[HistoryMessage], business_overview: str
    ) -> ChatResponse:
        """Handle general conversation"""

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

    def _format_history(self, history: List[HistoryMessage]) -> str:
        """Format conversation history"""
        if not history:
            return ""

        formatted = []
        for msg in history:
            role = "User" if msg.role == "user" else "Assistant"
            formatted.append(f"{role}: {msg.message}")

        return "\n".join(formatted)

    def _calculate_confidence(self, search_results) -> float:
        """Calculate confidence score"""
        if not search_results:
            return 0.0

        top_scores = [result.score for result in search_results[:3]]
        avg_score = sum(top_scores) / len(top_scores)

        return min(max(avg_score, 0.0), 1.0)
