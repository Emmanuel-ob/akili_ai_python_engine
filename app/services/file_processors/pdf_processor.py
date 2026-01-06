import io
import pdfplumber
from typing import Dict, Any
from app.services.file_processors.base_processor import BaseFileProcessor
from app.core.logging_config import logger


class PDFProcessor(BaseFileProcessor):
    """Extract text from PDF files"""

    async def extract_text(self, file_path: str, file_content: bytes) -> Dict[str, Any]:
        """Extract text from PDF with page-by-page structure"""

        try:
            content = []
            warnings = []

            with pdfplumber.open(io.BytesIO(file_content)) as pdf:
                total_pages = len(pdf.pages)

                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        text = page.extract_text()

                        if text:
                            cleaned_text = self.clean_text(text)

                            if cleaned_text:
                                content.append(
                                    {
                                        "text": cleaned_text,
                                        "metadata": {"page": page_num, "source": "pdf"},
                                    }
                                )
                        else:
                            warnings.append(f"Page {page_num}: No text extracted")

                    except Exception as e:
                        logger.error(f"Error extracting page {page_num}: {str(e)}")
                        warnings.append(f"Page {page_num}: Extraction failed")

            if not content:
                raise ValueError("No text could be extracted from PDF")

            return {
                "content": content,
                "metadata": {
                    "total_pages": total_pages,
                    "extraction_method": "pdfplumber",
                    "warnings": warnings if warnings else None,
                },
            }

        except Exception as e:
            logger.error(f"PDF processing failed: {str(e)}")
            raise ValueError(f"Failed to process PDF: {str(e)}")
