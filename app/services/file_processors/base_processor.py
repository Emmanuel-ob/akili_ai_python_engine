from abc import ABC, abstractmethod
from typing import Dict, Any, List
from app.core.logging_config import logger


class BaseFileProcessor(ABC):
    """Abstract base class for file processors"""

    def __init__(self, max_size_mb: int = 10):
        self.max_size_mb = max_size_mb

    @abstractmethod
    async def extract_text(self, file_path: str, file_content: bytes) -> Dict[str, Any]:
        """
        Extract text and structure from file

        Returns:
        {
            'content': [
                {
                    'text': str,
                    'metadata': dict (optional)
                }
            ],
            'metadata': {
                'total_pages': int (optional),
                'extraction_method': str,
                'warnings': list (optional)
            }
        }
        """
        pass

    def validate_file(self, file_size: int, mime_type: str) -> Dict[str, Any]:
        """Validate file size and type"""

        # Check size
        max_size_bytes = self.max_size_mb * 1024 * 1024
        if file_size > max_size_bytes:
            return {
                "valid": False,
                "error": f"File size exceeds {self.max_size_mb}MB limit",
            }

        return {"valid": True}

    def clean_text(self, text: str) -> str:
        """Clean extracted text"""
        if not text:
            return ""

        # Remove excessive whitespace
        text = " ".join(text.split())

        # Remove control characters
        text = "".join(char for char in text if ord(char) >= 32 or char == "\n")

        return text.strip()

    def chunk_by_size(self, text: str, max_chunk_size: int = 1000) -> List[str]:
        """Split text into chunks by size"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0

        for word in words:
            word_size = len(word) + 1  # +1 for space

            if current_size + word_size > max_chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_size = word_size
            else:
                current_chunk.append(word)
                current_size += word_size

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks
