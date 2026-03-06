from fastapi import APIRouter, HTTPException, Depends, Header, Request
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat import ChatService
from app.services.vectorstore import VectorStoreService
from app.services.embeddings import EmbeddingService
from app.services.llm import LLMService
from app.core.config import settings
from app.core.logging_config import logger

from app.services.subscription_validator import (
    get_subscription_context,
    validate_subscription_active,
    validate_rate_limit,
    SubscriptionContext,
)

router = APIRouter()

embedding_service = EmbeddingService()
vectorstore_service = VectorStoreService()
llm_service = LLMService()

chat_service = ChatService(
    vectorstore_service=vectorstore_service,
    embedding_service=embedding_service,
    llm_service=llm_service,
)


def verify_api_key(x_akili_key: str = Header(...)):
    if x_akili_key != settings.FASTAPI_SHARED_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


@router.post("/respond", response_model=ChatResponse)
async def chat_respond(
    request_data: Request,
    chat_request: ChatRequest,
    _: bool = Depends(verify_api_key),
    subscription: SubscriptionContext = Depends(get_subscription_context),
):
    """
    Unified chat endpoint supporting both internal (staff) and external (customer) users.

    user_type = "internal"  → Digital Brain mode (analytics, trends, full data access)
    user_type = "external"  → Customer support mode (sanitized, scoped responses)
    """
    try:
        validate_subscription_active(subscription)
        await validate_rate_limit(subscription)

        logger.info(
            f"Chat request — Business: {chat_request.business_id}, "
            f"Chatbot: {chat_request.chatbot_id}, "
            f"User type: {chat_request.user_type}, "  # ✅ Log user type
            f"Plan: {subscription.plan_id}"
        )

        # ✅ Pass user_type into generate_response
        response = await chat_service.generate_response(
            business_id=chat_request.business_id,
            chatbot_id=chat_request.chatbot_id,
            message=chat_request.message,
            history=chat_request.history,
            chatbot_config=chat_request.chatbot_config,
            user_type=chat_request.user_type,  # ✅ NEW
        )

        response_dict = response.dict()
        response_dict["metadata"] = response_dict.get("metadata", {})
        response_dict["metadata"]["subscription"] = {
            "plan": subscription.plan_id,
            "conversations_used": subscription.get_usage("conversations"),
            "conversations_limit": subscription.get_limit("conversations"),
        }

        conversations_limit = subscription.get_limit("conversations")
        conversations_used = subscription.get_usage("conversations")

        if conversations_limit != float("inf"):
            usage_percentage = (conversations_used / conversations_limit) * 100
            if usage_percentage >= 90:
                response_dict["metadata"]["usage_warning"] = {
                    "type": "conversations",
                    "percentage": round(usage_percentage, 1),
                    "message": f"You have used {round(usage_percentage)}% of your conversation limit",
                    "upgrade_recommended": True,
                }

        logger.info(
            f"Response generated [{chat_request.user_type}]: {response.text[:100]}..."
        )
        return ChatResponse(**response_dict)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat_respond: {str(e)}")

        fallback_message = chat_request.chatbot_config.get(
            "fallback_message",
            "I'm sorry, I'm having trouble responding right now. Please try again.",
        )

        return ChatResponse(
            text=fallback_message,
            sources=[],
            metadata={"error": True, "fallback": True, "confidence": 0.0},
        )
