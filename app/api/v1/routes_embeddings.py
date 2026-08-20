from fastapi import APIRouter, HTTPException, Depends, Header
from app.schemas.embedding import UpsertRequest, UpsertResponse
from app.services.embeddings import EmbeddingService
from app.services.vectorstore import VectorStoreService
from app.services.chunking import CHUNKER_VERSION, ChunkingService
from app.core.config import settings
from app.core.logging_config import logger

router = APIRouter()

# Initialize services (unified embedding service)
embedding_service = EmbeddingService()
vectorstore_service = VectorStoreService()
chunking_service = ChunkingService()


def verify_api_key(x_akili_key: str = Header(...)):
    """Verify the API key from request header"""
    if x_akili_key != settings.FASTAPI_SHARED_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


@router.post("/upsert", response_model=UpsertResponse)
async def upsert_embeddings(request: UpsertRequest, _: bool = Depends(verify_api_key)):
    """
    Enhanced upsert with granular error tracking
    Uses Gemini for all embeddings
    """
    try:
        # Get collection name
        collection_name = vectorstore_service.get_collection_name(
            request.business_id, request.chatbot_id
        )

        logger.info(
            f"Enhanced upsert request: {len(request.docs)} docs for collection {collection_name}"
        )

        # Convert Pydantic models to dictionaries for processing
        documents = [doc.dict() for doc in request.docs]

        # Clear each source document's previous chunks BEFORE writing the new
        # ones. Chunk ids are positional ("{doc_id}_chunk_{i}") and the upsert
        # keys on doc_id, so re-chunking a document into FEWER pieces leaves
        # the surplus behind, still searchable. Structure-aware chunking
        # produces fewer chunks than the old fixed-width slicing on most
        # prose, so without this every re-ingest would leave mid-word
        # fragments of the old scheme competing with the new chunks.
        for source_doc_id in {d.get("doc_id") for d in documents if d.get("doc_id")}:
            try:
                removed = vectorstore_service.delete_by_pattern(
                    VectorStoreService.chunk_pattern_for(source_doc_id)
                )
                if removed:
                    logger.info(
                        f"Cleared {removed} previous chunk(s) for {source_doc_id}"
                    )
            except Exception as e:
                # Hard Rule 8: a cleanup failure must not abort the ingest.
                # Worst case is the orphan behaviour we had before.
                logger.error(f"Chunk cleanup failed for {source_doc_id}: {e}")

        # Optional: Chunk documents if they're too long
        chunked_documents = chunking_service.chunk_documents(documents)
        logger.info(f"After chunking: {len(chunked_documents)} total documents")

        # Track processing results
        failed_docs = []
        successful_docs = []
        processing_stats = {
            "chunking_failed": 0,
            "embedding_failed": 0,
            "storage_failed": 0,
            "total_processed": 0,
        }

        # Process each document individually for granular error tracking
        for i, doc in enumerate(chunked_documents):
            doc_id = doc.get("doc_id", f"unknown_{i}")
            processing_stats["total_processed"] += 1

            try:
                # Step 1: Extract text (chunking already done)
                text = doc["text"]
                if not text or len(text.strip()) == 0:
                    raise ValueError("Empty text content")

                # Step 2: Generate embedding using Gemini
                embedding = None
                try:
                    embedding = embedding_service.generate_single_embedding(text)

                    # Validate embedding format
                    if not embedding or not isinstance(embedding, list):
                        raise ValueError("Embedding generation returned invalid format")

                    if len(embedding) == 0:
                        raise ValueError("Embedding generation returned empty result")

                    # Ensure all values are float
                    embedding = [float(x) for x in embedding]

                except Exception as e:
                    processing_stats["embedding_failed"] += 1
                    failed_docs.append(
                        {
                            "doc_id": doc_id,
                            "error": f"Embedding error: {str(e)}",
                            "stage": "embedding_generation",
                            "provider": "gemini",
                            "text_length": len(text),
                        }
                    )
                    logger.error(f"Embedding failed for {doc_id}: {str(e)}")
                    continue

                # Step 3: Store in vector database
                #
                # Record which provider, model and chunker produced this row.
                # Vectors from two different embedding models are not
                # comparable, so without provenance a mixed collection
                # degrades silently with nothing to diagnose it from. The
                # chunker version is also what makes a re-embed resumable: a
                # row already at the current version can be skipped.
                doc.setdefault("metadata", {})
                doc["metadata"].update(
                    {
                        **{
                            f"embedding_{k}": v
                            for k, v in embedding_service.get_provider_info().items()
                        },
                        "chunker_version": CHUNKER_VERSION,
                    }
                )

                try:
                    count = vectorstore_service.upsert_documents(
                        collection_name=collection_name,
                        documents=[doc],
                        embeddings=[embedding],
                        business_id=request.business_id,
                        chatbot_id=request.chatbot_id,
                    )

                    if count == 0:
                        raise ValueError("Vector storage returned zero count")

                except Exception as e:
                    processing_stats["storage_failed"] += 1
                    failed_docs.append(
                        {"doc_id": doc_id, "error": str(e), "stage": "vector_storage"}
                    )
                    logger.error(f"Vector storage failed for {doc_id}: {str(e)}")
                    continue

                # Success - document fully processed
                successful_docs.append(doc_id)
                logger.debug(f"Successfully processed document: {doc_id}")

            except Exception as e:
                # Catch any other processing errors
                processing_stats["chunking_failed"] += 1
                failed_docs.append(
                    {"doc_id": doc_id, "error": str(e), "stage": "document_processing"}
                )
                logger.error(f"Document processing failed for {doc_id}: {str(e)}")

        # Prepare response
        success_count = len(successful_docs)
        failure_count = len(failed_docs)
        total_attempted = len(chunked_documents)

        is_success = failure_count == 0

        response_message = (
            f"Processed {success_count}/{total_attempted} documents successfully"
        )
        if failure_count > 0:
            response_message += f", {failure_count} failed"

        logger.info(
            f"Upsert completed - Success: {success_count}, Failed: {failure_count}"
        )
        logger.info(f"Processing stats: {processing_stats}")

        return UpsertResponse(
            success=is_success,
            message=response_message,
            collection=collection_name,
            count=success_count,
            failed_count=failure_count,
            failed_items=failed_docs[
                :50
            ],  # Limit to first 50 failed items to avoid huge responses
            processing_stats=processing_stats,
        )

    except Exception as e:
        logger.error(f"Critical error in upsert_embeddings: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "stage": "critical_failure",
                "message": "Upsert operation failed completely",
            },
        )
