from typing import List, Dict, Any, Optional
from app.schemas.chat import ChatResponse, HistoryMessage, SourceDocument
from app.services.vectorstore import VectorStoreService
from app.services.embeddings import EmbeddingServiceOpenai, EmbeddingServiceHuggingFace
from app.services.llm import LLMServiceHuggingFace
from app.services.intent_analyzer import IntentAnalyzer
from app.services.conversation_memory import ConversationMemory, ConversationSummarizer
from app.core.config import settings
from app.core.logging_config import logger
import httpx


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
        self.intent_analyzer = IntentAnalyzer(llm_service)
        self.memory = ConversationMemory()
        self.summarizer = ConversationSummarizer(llm_service)

    async def generate_response(
        self,
        business_id: str,
        chatbot_id: str,
        message: str,
        history: List[HistoryMessage],
        chatbot_config: Dict[str, Any],
        customer_data: Optional[Dict] = None,
        is_authenticated: bool = False,
        connection_id: Optional[str] = None,
    ) -> ChatResponse:
        """AI-orchestrated response with vector search and conditional auth"""

        try:
            schema_analysis = chatbot_config.get("schema_analysis", {})
            business_overview = chatbot_config.get("business_overview", "")

            # Check for follow-up context
            reference_context = self.memory.resolve_reference(message)
            if reference_context.get("is_followup"):
                logger.info(
                    f"Detected follow-up question about: {reference_context.get('current_topic')}"
                )

            # STEP 1: Analyze Intent
            intent = await self.intent_analyzer.analyze_intent(
                message=message, schema=schema_analysis, customer_context=customer_data
            )

            logger.info(
                f"Intent: {intent.get('intent_category')}, Safety: {intent.get('safety_level')}"
            )

            # STEP 2: Check if authentication is required
            requires_auth = intent.get("safety_level") in [
                "authenticated",
                "admin_only",
            ]

            if requires_auth and not is_authenticated:
                return ChatResponse(
                    text="This information requires authentication. Please verify your identity to continue.",
                    sources=[],
                    metadata={"type": "auth_required", "requires_auth": True},
                )

            # STEP 3: Route based on intent
            if intent["intent_category"] == "general_chat":
                response = await self._handle_general_chat(
                    message, history, business_overview
                )
            else:
                # Use vector search for all data queries
                response = await self._handle_rag_flow(
                    message,
                    business_id,
                    chatbot_id,
                    history,
                    business_overview,
                    customer_data,
                    is_authenticated,
                )

            # Update conversation memory
            self.memory.update_context(message, intent, response.text)

            return response

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

    async def _handle_rag_flow(
        self,
        message: str,
        business_id: str,
        chatbot_id: str,
        history: List[HistoryMessage],
        business_overview: str,
        customer_data: Optional[Dict] = None,
        is_authenticated: bool = False,
    ) -> ChatResponse:
        """Vector search-based RAG flow"""

        collection_name = self.vectorstore.get_collection_name(business_id, chatbot_id)

        query_embedding = self.embedding_huggingface.generate_single_embeddings(message)

        search_results = self.vectorstore.search_similar(
            collection_name=collection_name, query_vector=query_embedding, limit=5
        )

        context_texts = []
        sources = []

        for result in search_results:
            # Filter by customer if authenticated and customer_id is available
            if is_authenticated and customer_data:
                result_customer = result.payload.get("metadata", {}).get(
                    "customer_value"
                )
                if result_customer and result_customer != customer_data.get("id"):
                    continue  # Skip data not belonging to this customer

            context_texts.append(result.payload["text"])
            sources.append(
                SourceDocument(
                    doc_id=result.payload["doc_id"],
                    text=result.payload["text"][:200] + "...",
                    confidence=result.score,
                    table=result.payload.get("metadata", {}).get("table"),
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

        confidence = self._calculate_confidence(search_results)

        return ChatResponse(
            text=response_text,
            sources=sources,
            metadata={
                "type": "rag",
                "confidence": confidence,
                "context_used": len(context_texts) > 0,
                "authenticated": is_authenticated,
            },
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
