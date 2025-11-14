from typing import Dict, Any
from app.services.file_processors.base_processor import BaseFileProcessor
from app.core.logging_config import logger


class TextProcessor(BaseFileProcessor):
    """Extract text from plain text files"""

    async def extract_text(self, file_path: str, file_content: bytes) -> Dict[str, Any]:
        """Extract and structure plain text"""

        try:
            # Try different encodings
            encodings = ["utf-8", "latin-1", "cp1252"]
            text = None

            for encoding in encodings:
                try:
                    text = file_content.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue

            if text is None:
                raise ValueError("Could not decode text file")

            # Clean text
            cleaned_text = self.clean_text(text)

            if not cleaned_text:
                raise ValueError("No text content found")

            # Split into paragraphs (by double newline)
            paragraphs = [p.strip() for p in cleaned_text.split("\n\n") if p.strip()]

            content = []

            # If paragraphs are small, combine them
            if paragraphs and len(paragraphs[0]) < 200:
                # Small paragraphs - chunk by size
                chunks = self.chunk_by_size(cleaned_text, max_chunk_size=1000)

                for idx, chunk in enumerate(chunks):
                    content.append(
                        {"text": chunk, "metadata": {"source": "txt", "chunk": idx + 1}}
                    )
            else:
                # Large paragraphs - keep separate
                for idx, para in enumerate(paragraphs):
                    content.append(
                        {
                            "text": para,
                            "metadata": {"source": "txt", "paragraph": idx + 1},
                        }
                    )

            return {
                "content": content,
                "metadata": {
                    "extraction_method": "plain_text",
                    "total_characters": len(cleaned_text),
                    "total_sections": len(content),
                },
            }

        except Exception as e:
            logger.error(f"Text processing failed: {str(e)}")
            raise ValueError(f"Failed to process text file: {str(e)}")
