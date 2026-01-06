from sqlite3 import connect
from fastapi import APIRouter, HTTPException, Depends, Header
from app.schemas.chat import ChatRequest, ChatResponse, HistoryMessage
from app.services.chat import ChatService
from app.services.vectorstore import VectorStoreService
from app.services.embeddings import EmbeddingServiceOpenai, EmbeddingServiceHuggingFace
from app.services.llm import LLMServiceHuggingFace
from app.core.config import settings
from app.core.logging_config import logger

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
    llm_service=llm_service
)

def verify_api_key(x_akili_key: str = Header(...)):

    """Verify the API key from request header"""
    if x_akili_key != settings.FASTAPI_SHARED_KEY:

        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return True


@router.post("/respond", response_model=ChatResponse)
async def chat_respond(
    request: ChatRequest,
    _: bool = Depends(verify_api_key)
):
    try:

        logger.info(f"Received chat request for business {request.business_id}, chatbot {request.chatbot_id}")
        logger.info(f"User message: {request.message}")
        
        # Generate response using chat service
        response = await chat_service.generate_response(
        business_id=request.business_id,
        chatbot_id=request.chatbot_id,
        message=request.message,
        history=request.history,
        chatbot_config=request.chatbot_config,
        )

        
        logger.info(f"Generated response: {response.text[:100]}...")
        return response

    except Exception as e:

        logger.error(f"Error in chat_respond: {str(e)}")
        
        # Return fallback response
        fallback_message = request.chatbot_config.get('fallback_message', 
            "I'm sorry, I'm having trouble responding right now. Please try again.")
        
        return ChatResponse(
            text=fallback_message,
            sources=[],
            metadata={
                "error": True,
                "fallback": True,
                "confidence": 0.0
            }
        )