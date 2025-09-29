from token import OP
from typing import List, Dict, Any, Optional
from venv import logger
from app.core.config import settings
from app.core.logging_config import logger


class ChunkingService:

    @staticmethod
    def chunk_text(text: str, chunk_size: Optional[int] = None, overlap: Optional[int] = None) -> List[str]:
        """Split text into chunks with overlap"""
        if chunk_size is None:
            chunk_size = settings.CHUNK_SIZE
        if overlap is None:
            overlap = settings.CHUNK_OVERLAP

        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)

            # Move start position with overlap
            start = end - overlap

            if end >= len(text):
                break

        return chunks

    @staticmethod
    def chunk_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process documents and create chunks if needed"""
        chunked_docs = []

        for doc in documents:
            text = doc["text"]
            doc_id = doc.get("doc_id", "unknown")

            # Debug logging
            logger.info(f"Processing document {doc_id} with text length {len(text)}")
            logger.info(f"Text preview: {text[:200]}...")

            # Check for problematic text content
            if not text or not isinstance(text, str):
                logger.error(f"Invalid text type for doc {doc_id}: {type(text)}")
                continue

            if len(text.strip()) == 0:
                logger.error(f"Empty text for doc {doc_id}")
                continue

            chunks = ChunkingService.chunk_text(text)
            logger.info(f"Chunked document {doc_id} into {len(chunks)} chunks")

            if len(chunks) == 1:
                # No chunking needed
                chunked_docs.append(doc)
            else:
                # Create multiple documents for chunks
                for i, chunk in enumerate(chunks):
                    if not chunk.strip():
                        logger.warning(f"Empty chunk {i} for doc {doc_id}")
                        continue

                    chunk_doc = doc.copy()
                    chunk_doc["text"] = chunk
                    chunk_doc["doc_id"] = f"{doc['doc_id']}_chunk_{i}"
                    chunked_docs.append(chunk_doc)

        logger.info(f"Total documents after chunking: {len(chunked_docs)}")
        return chunked_docs