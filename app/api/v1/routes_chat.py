from fastapi import APIRouter, HTTPException, Depends, Header, Request
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat import ChatService
from app.services.vectorstore import VectorStoreService
from app.services.embeddings import EmbeddingServiceOpenai, EmbeddingServiceHuggingFace
from app.services.llm import LLMServiceHuggingFace
from app.core.config import settings
from app.core.logging_config import logger

# Import subscription validation
from app.services.subscription_validator import (
    get_subscription_context,
    validate_subscription_active,
    validate_rate_limit,
    SubscriptionContext,
)

router = APIRouter()

# Initialize services
embedding_service_openai = EmbeddingServiceOpenai()
embedding_service_huggingface = EmbeddingServiceHuggingFace()
vectorstore_service = VectorStoreService()
llm_service = LLMServiceHuggingFace()
chat_service = ChatService(
    vectorstore_service=vectorstore_service,
    embedding_service_openai=embedding_service_openai,
    embedding_service_huggingface=embedding_service_huggingface,
    llm_service=llm_service,
)


def verify_api_key(x_akili_key: str = Header(...)):
    """Verify the API key from request header"""
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
    Chat endpoint with subscription validation

    Laravel sends subscription context in headers:
    - X-Subscription-Plan: starter|professional|enterprise
    - X-Subscription-Status: active|trialing|canceled
    - X-Subscription-Limits: {"conversations": 1000, ...}
    - X-Subscription-Usage: {"conversations_used": 150, ...}
    - X-Business-Id: business_id_here
    """
    try:
        # 1. Validate subscription is active
        validate_subscription_active(subscription)

        # 2. Check rate limiting based on plan
        await validate_rate_limit(subscription)

        # 3. Log request with subscription context
        logger.info(
            f"Chat request - Business: {chat_request.business_id}, "
            f"Chatbot: {chat_request.chatbot_id}, "
            f"Plan: {subscription.plan_id}, "
            f"Conversations used: {subscription.get_usage('conversations')}"
        )

        # 4. Generate response using chat service
        response = await chat_service.generate_response(
            business_id=chat_request.business_id,
            chatbot_id=chat_request.chatbot_id,
            message=chat_request.message,
            history=chat_request.history,
            chatbot_config=chat_request.chatbot_config,
        )

        # 5. Add subscription context to response
        response_dict = response.dict()
        response_dict["metadata"] = response_dict.get("metadata", {})
        response_dict["metadata"]["subscription"] = {
            "plan": subscription.plan_id,
            "conversations_used": subscription.get_usage("conversations"),
            "conversations_limit": subscription.get_limit("conversations"),
        }

        # 6. Optional: Warn if near conversation limit
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

        logger.info(f"Generated response: {response.text[:100]}...")
        return ChatResponse(**response_dict)

    except HTTPException:
        # Re-raise subscription/auth errors
        raise
    except Exception as e:
        logger.error(f"Error in chat_respond: {str(e)}")

        # Return fallback response
        fallback_message = chat_request.chatbot_config.get(
            "fallback_message",
            "I'm sorry, I'm having trouble responding right now. Please try again.",
        )

        return ChatResponse(
            text=fallback_message,
            sources=[],
            metadata={"error": True, "fallback": True, "confidence": 0.0},
        )
