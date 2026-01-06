import json
from typing import Dict, Any
from app.services.file_processors.base_processor import BaseFileProcessor
from app.core.logging_config import logger


class JSONProcessor(BaseFileProcessor):
    """Extract text from JSON files"""

    async def extract_text(self, file_path: str, file_content: bytes) -> Dict[str, Any]:
        """Extract structured Q&A or data from JSON"""

        try:
            # Parse JSON
            data = json.loads(file_content.decode("utf-8"))

            content = []

            # Check if it's a Q&A format
            if isinstance(data, list):
                # Array of objects
                for idx, item in enumerate(data):
                    if isinstance(item, dict):
                        # Check for Q&A keys
                        if "question" in item and "answer" in item:
                            content.append(
                                {
                                    "text": f"Q: {item['question']}\nA: {item['answer']}",
                                    "metadata": {
                                        "source": "json",
                                        "index": idx,
                                        "type": "qa_pair",
                                        "question": item["question"],
                                        "answer": item["answer"],
                                    },
                                }
                            )
                        else:
                            # Generic object - convert to text
                            text = " | ".join([f"{k}: {v}" for k, v in item.items()])
                            content.append(
                                {
                                    "text": text,
                                    "metadata": {
                                        "source": "json",
                                        "index": idx,
                                        "type": "object",
                                    },
                                }
                            )

            elif isinstance(data, dict):
                # Single object or nested structure
                if "faqs" in data or "questions" in data or "qa_pairs" in data:
                    # Extract FAQ array
                    faq_key = (
                        "faqs"
                        if "faqs" in data
                        else ("questions" if "questions" in data else "qa_pairs")
                    )
                    faq_data = data[faq_key]

                    for idx, item in enumerate(faq_data):
                        if (
                            isinstance(item, dict)
                            and "question" in item
                            and "answer" in item
                        ):
                            content.append(
                                {
                                    "text": f"Q: {item['question']}\nA: {item['answer']}",
                                    "metadata": {
                                        "source": "json",
                                        "index": idx,
                                        "type": "qa_pair",
                                        "question": item["question"],
                                        "answer": item["answer"],
                                    },
                                }
                            )
                else:
                    # Flatten nested structure
                    flattened_text = self._flatten_dict(data)
                    content.append(
                        {
                            "text": flattened_text,
                            "metadata": {"source": "json", "type": "nested_object"},
                        }
                    )

            if not content:
                raise ValueError("No extractable content found in JSON")

            return {
                "content": content,
                "metadata": {"extraction_method": "json", "total_items": len(content)},
            }

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON: {str(e)}")
            raise ValueError(f"Invalid JSON file: {str(e)}")
        except Exception as e:
            logger.error(f"JSON processing failed: {str(e)}")
            raise ValueError(f"Failed to process JSON: {str(e)}")

    def _flatten_dict(self, d: dict, parent_key: str = "", sep: str = " > ") -> str:
        """Flatten nested dictionary to readable text"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k

            if isinstance(v, dict):
                items.append(self._flatten_dict(v, new_key, sep=sep))
            elif isinstance(v, list):
                items.append(f"{new_key}: {', '.join(map(str, v))}")
            else:
                items.append(f"{new_key}: {v}")

        return " | ".join(items)
