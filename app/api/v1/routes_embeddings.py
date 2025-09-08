from fastapi import APIRouter, HTTPException, Depends, Header
from app.schemas.embedding import UpsertRequest, UpsertResponse
from app.services.embeddings import EmbeddingServiceHuggingFace, EmbeddingServiceOpenai
from app.services.vectorstore import VectorStoreService
from app.services.chunking import ChunkingService
from app.core.config import settings
from app.core.logging_config import logger

router = APIRouter()

# Initialize services
embedding_service = EmbeddingServiceOpenai()
vectorstore_service = VectorStoreService()
chunking_service = ChunkingService()
embedding_service_huggingface = EmbeddingServiceHuggingFace()

def verify_api_key(x_akili_key: str = Header(...)):
    """Verify the API key from request header"""
    if x_akili_key != settings.FASTAPI_SHARED_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


@router.post("/upsert", response_model=UpsertResponse)
async def upsert_embeddings(
    request: UpsertRequest,
    _: bool = Depends(verify_api_key)
):
    """
    Upsert documents with embeddings to vector database
    
    This endpoint:
    1. Receives document batch from Laravel
    2. Optionally chunks long documents
    3. Generates embeddings using OpenAI
    4. Stores everything in Qdrant vector database
    """
    try:
        # Get collection name
        collection_name = vectorstore_service.get_collection_name(
            request.business_id,
            request.chatbot_id
        )
        print("Request Received:", request)
        logger.info(f"Request Received: {request}")
        print("Collection Name:", collection_name)
        logger.info(f"Collection Name: {collection_name}")
        
        
        # print("request documents 0", request.docs)
        logger.info(f"request documents {request.docs}")
        # Convert Pydantic models to dictionaries for processing
        documents = [doc.dict() for doc in request.docs]
        print("request documents", documents)
        

        # Optional: Chunk documents if they're too long
        chunked_documents = chunking_service.chunk_documents(documents)
        print("chunked documents", chunked_documents)
        logger.info(f"chunked documents {chunked_documents}")

        # Extract texts for embedding generation
        texts = [doc['text'] for doc in chunked_documents]
        
        embeddings = []

        print("texts", texts)
        # Generate embeddings
        if(request.provider == "openai"):
            embeddings = embedding_service.generate_embeddings(
                texts, request.provider)
        
            print("embeddings openai", embeddings)
            logger.info(f"Generated OpenAI embeddings for {len(texts)} texts")
        elif(request.provider == "huggingface"):
            embeddings = embedding_service_huggingface.generate_embeddings(texts)
            logger.info(f"Generated HuggingFace embeddings for {len(texts)} texts")
            print("embeddings huggingface", embeddings)

        # Store in vector database
        count = vectorstore_service.upsert_documents(
            collection_name=collection_name,
            documents=chunked_documents,
            embeddings=embeddings,
            business_id=request.business_id,
            chatbot_id=request.chatbot_id
        )

        return UpsertResponse(
            success=True,
            message=f"Successfully processed {count} documents",
            collection=collection_name,
            count=count
        )

    except Exception as e:
        logger.error(f"Error in upsert_embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
