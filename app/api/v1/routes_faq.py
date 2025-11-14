from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Header
from pydantic import BaseModel
from typing import List, Dict, Any
from app.services.faq_manager import FAQManager
from app.services.vectorstore import VectorStoreService
from app.services.embeddings import EmbeddingServiceHuggingFace
from app.core.config import settings
from app.core.logging_config import logger

router = APIRouter()

# Initialize services
vectorstore_service = VectorStoreService()
embedding_service = EmbeddingServiceHuggingFace()
faq_manager = FAQManager(vectorstore_service, embedding_service)


def verify_api_key(x_akili_key: str = Header(...)):
    """Verify API key"""
    if x_akili_key != settings.FASTAPI_SHARED_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


# Schemas
class FAQEmbedRequest(BaseModel):
    business_id: str
    chatbot_id: str
    faq_source_id: str
    content: List[Dict[str, Any]]
    source_name: str
    priority: int = 5


class FAQDeleteRequest(BaseModel):
    business_id: str
    chatbot_id: str
    faq_source_id: str


@router.post("/process-file")
async def process_file(
    file: UploadFile = File(...),
    business_id: str = Form(...),
    chatbot_id: str = Form(...),
    source_type: str = Form(...),
    _: bool = Depends(verify_api_key),
):
    """
    Process uploaded file and extract content
    Returns structured content for preview
    """
    try:
        logger.info(f"Processing {source_type} file: {file.filename}")

        # Read file content
        file_content = await file.read()

        # Process file
        result = await faq_manager.process_file(
            file_content=file_content, source_type=source_type, file_path=file.filename
        )

        return {
            "success": True,
            "content": result["content"],
            "metadata": result["metadata"],
            "total_items": len(result["content"]),
        }

    except ValueError as e:
        logger.error(f"File processing validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"File processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@router.post("/embed")
async def embed_faq(request: FAQEmbedRequest, _: bool = Depends(verify_api_key)):
    """
    Embed FAQ content into vector database
    """
    try:
        logger.info(
            f"Embedding FAQ source {request.faq_source_id}: "
            f"{len(request.content)} items"
        )

        result = await faq_manager.embed_faq(
            business_id=request.business_id,
            chatbot_id=request.chatbot_id,
            faq_source_id=request.faq_source_id,
            content=request.content,
            source_name=request.source_name,
            priority=request.priority,
        )

        return {
            "success": True,
            "embedded_count": result["embedded_count"],
            "total_items": result["total_items"],
            "skipped_items": result["skipped_items"],
        }

    except Exception as e:
        logger.error(f"FAQ embedding failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")


@router.delete("/embeddings")
async def delete_faq_embeddings(
    request: FAQDeleteRequest, _: bool = Depends(verify_api_key)
):
    """
    Delete FAQ embeddings from vector database
    """
    try:
        logger.info(f"Deleting embeddings for FAQ source {request.faq_source_id}")

        deleted_count = await faq_manager.delete_faq_embeddings(
            business_id=request.business_id,
            chatbot_id=request.chatbot_id,
            faq_source_id=request.faq_source_id,
        )

        return {
            "success": True,
            "deleted_count": deleted_count,
            "message": f"Deleted {deleted_count} embeddings",
        }

    except Exception as e:
        logger.error(f"FAQ deletion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")
