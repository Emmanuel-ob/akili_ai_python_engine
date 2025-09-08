from typing import List, Dict, Any, Optional
from app.schemas.chat import ChatResponse, HistoryMessage, SourceDocument
from app.services.vectorstore import VectorStoreService
from app.services.embeddings import EmbeddingServiceOpenai, EmbeddingServiceHuggingFace
from app.services.llm import LLMServiceHuggingFace
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

    async def generate_response(
        self,
        business_id: str,
        chatbot_id: str,
        message: str,
        history: List[HistoryMessage],
        chatbot_config: Dict[str, Any],
    ) -> ChatResponse:
        """Generate chatbot response with context retrieval"""

        try:
            # 1. Get collection name
            collection_name = self.vectorstore.get_collection_name(
                business_id, chatbot_id
            )
            logger.info(f"Using collection: {collection_name}")

            # 2. Generate embedding for user query
            query_embedding = self.embedding_huggingface.generate_single_embeddings(
                message
            )
            logger.info(f"Generated query embedding")

            # 3. Search vector database for relevant context
            search_results = self.vectorstore.search_similar(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=5,  # Top 5 most relevant documents
            )

            logger.info(f"Found {len(search_results)} relevant documents")

            # 4. Prepare context and sources
            context_texts = []
            sources = []

            for result in search_results:
                context_texts.append(result.payload["text"])
                sources.append(
                    SourceDocument(
                        doc_id=result.payload["doc_id"],
                        text=result.payload["text"][:200]
                        + "...",  # Truncate for display
                        confidence=result.score,
                        table=result.payload.get("data", {}).get("table"),
                    )
                )

            # 5. Build conversation history context
            history_context = self._format_history(
                history[-6:]
            )  # Last 6 messages for context

            # 6. Generate LLM response
            response_text = await self.llm.generate_response(
                message=message,
                context=context_texts,
                history=history_context,
                personality=chatbot_config.get(
                    "personality", "helpful and professional"
                ),
            )

            # 7. Calculate confidence based on search results
            confidence = self._calculate_confidence(search_results)

            return ChatResponse(
                text=response_text,
                sources=sources,
                metadata={
                    "confidence": confidence,
                    "context_used": len(context_texts) > 0,
                    "history_length": len(history),
                },
            )

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")

            # Fallback response
            fallback_message = chatbot_config.get(
                "fallback_message",
                "I'm sorry, I didn't understand that. Could you please rephrase?",
            )

            return ChatResponse(
                text=fallback_message,
                sources=[],
                metadata={"error": True, "fallback": True, "confidence": 0.0},
            )

    def _format_history(self, history: List[HistoryMessage]) -> str:
        """Format conversation history for LLM context"""
        if not history:
            return ""

        formatted = []
        for msg in history:
            role = "Human" if msg.role == "user" else "Assistant"
            formatted.append(f"{role}: {msg.message}")

        return "\n".join(formatted)

    def _calculate_confidence(self, search_results) -> float:
        """Calculate confidence score based on search results"""
        if not search_results:
            return 0.0

        # Average of top 3 scores, normalized
        top_scores = [result.score for result in search_results[:3]]
        avg_score = sum(top_scores) / len(top_scores)

        # Convert to 0-1 range (assuming cosine similarity)
        return min(max(avg_score, 0.0), 1.0)
