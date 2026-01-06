from typing import Dict, Any, List
from app.services.file_processors.pdf_processor import PDFProcessor
from app.services.file_processors.docx_processor import DOCXProcessor
from app.services.file_processors.excel_processor import ExcelProcessor
from app.services.file_processors.text_processor import TextProcessor
from app.services.file_processors.json_processor import JSONProcessor
from app.services.vectorstore import VectorStoreService
from app.services.embeddings import EmbeddingServiceHuggingFace
from app.services.chunking import ChunkingService
from app.core.logging_config import logger
import hashlib


class FAQManager:
    """Orchestrates FAQ file processing and embedding"""

    def __init__(
        self,
        vectorstore_service: VectorStoreService,
        embedding_service: EmbeddingServiceHuggingFace,
    ):
        self.vectorstore = vectorstore_service
        self.embedding_service = embedding_service
        self.chunking_service = ChunkingService()

        # Initialize processors
        self.processors = {
            "pdf": PDFProcessor(),
            "docx": DOCXProcessor(),
            "excel": ExcelProcessor(),
            "csv": ExcelProcessor(),  # Reuse Excel processor
            "txt": TextProcessor(),
            "json": JSONProcessor(),
        }

    async def process_file(
        self, file_content: bytes, source_type: str, file_path: str = ""
    ) -> Dict[str, Any]:
        """Process file and extract structured content"""

        if source_type not in self.processors:
            raise ValueError(f"Unsupported file type: {source_type}")

        processor = self.processors[source_type]

        try:
            result = await processor.extract_text(file_path, file_content)
            logger.info(
                f"Successfully processed {source_type} file: "
                f"{len(result['content'])} items extracted"
            )
            return result

        except Exception as e:
            logger.error(f"File processing failed: {str(e)}")
            raise

    async def embed_faq(
        self,
        business_id: str,
        chatbot_id: str,
        faq_source_id: str,
        content: List[Dict[str, Any]],
        source_name: str,
        priority: int = 5,
    ) -> Dict[str, int]:
        """Embed FAQ content into vector store"""

        collection_name = self.vectorstore.get_collection_name(business_id, chatbot_id)

        # Prepare documents for embedding
        documents = []
        texts_to_embed = []

        for idx, item in enumerate(content):
            text = item.get("text", "")

            if not text or len(text.strip()) < 10:
                logger.warning(f"Skipping empty/short content at index {idx}")
                continue

            # Generate doc_id
            doc_id = f"{business_id}:{chatbot_id}:faq:{faq_source_id}:{idx}"

            # Generate hash
            text_hash = hashlib.sha256(text.encode()).hexdigest()

            # Prepare metadata
            metadata = {
                "source_type": "faq",
                "faq_source_id": faq_source_id,
                "faq_source_name": source_name,
                "faq_priority": priority,
                "faq_item_index": idx,
            }

            # Add item-specific metadata
            if "metadata" in item:
                metadata.update(item["metadata"])

            # Store question separately if it's Q&A format
            if "question" in item.get("metadata", {}):
                metadata["faq_question"] = item["metadata"]["question"]
                metadata["faq_answer_preview"] = item["metadata"].get("answer", "")[
                    :200
                ]

            documents.append(
                {
                    "doc_id": doc_id,
                    "text": text,
                    "data": item.get("metadata", {}),
                    "hash": text_hash,
                    "metadata": metadata,
                }
            )

            texts_to_embed.append(text)

        if not documents:
            raise ValueError("No valid content to embed")

        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts_to_embed)} FAQ items")
        embeddings = self.embedding_service.generate_embeddings(texts_to_embed)

        # Upsert to vector store
        logger.info(f"Storing {len(documents)} FAQ embeddings in vector database")
        count = self.vectorstore.upsert_documents(
            collection_name=collection_name,
            documents=documents,
            embeddings=embeddings,
            business_id=business_id,
            chatbot_id=chatbot_id,
        )

        logger.info(f"Successfully embedded {count} FAQ items")

        return {
            "embedded_count": count,
            "total_items": len(content),
            "skipped_items": len(content) - count,
        }

    async def delete_faq_embeddings(
        self, business_id: str, chatbot_id: str, faq_source_id: str
    ) -> int:
        """
        Delete all embeddings for a specific FAQ source

        Returns: Number of deleted documents
        """
        try:
            # Use the vectorstore's delete method
            deleted_count = self.vectorstore.delete_faq_source(
                business_id=business_id,
                chatbot_id=chatbot_id,
                faq_source_id=faq_source_id,
            )

            logger.info(
                f"Deleted {deleted_count} embeddings for FAQ source {faq_source_id}"
            )

            return deleted_count

        except Exception as e:
            logger.error(f"Failed to delete FAQ embeddings: {str(e)}")
            raise

    async def get_faq_stats(self, business_id: str, chatbot_id: str) -> Dict[str, Any]:
        """
        Get statistics about FAQ embeddings

        Returns: FAQ-specific stats
        """
        try:
            collection_info = self.vectorstore.get_collection_info(
                business_id=business_id,
                chatbot_id=chatbot_id,
            )

            if not collection_info:
                return {
                    "total_faq_documents": 0,
                    "faq_sources": 0,
                }

            real_time = collection_info.get("real_time_stats", {})

            return {
                "total_faq_documents": real_time.get("faq_count", 0),
                "faq_sources": real_time.get("faq_sources", 0),
                "collection_name": collection_info.get("collection_name"),
                "last_updated": collection_info.get("updated_at"),
            }

        except Exception as e:
            logger.error(f"Failed to get FAQ stats: {str(e)}")
            return {
                "total_faq_documents": 0,
                "faq_sources": 0,
            }
