import io
import pandas as pd
from typing import Dict, Any
from app.services.file_processors.base_processor import BaseFileProcessor
from app.core.logging_config import logger


class ExcelProcessor(BaseFileProcessor):
    """Extract text from Excel and CSV files"""

    async def extract_text(self, file_path: str, file_content: bytes) -> Dict[str, Any]:
        """Extract text from Excel/CSV as structured Q&A"""

        try:
            content = []
            warnings = []

            # Try to read as Excel first
            try:
                df = pd.read_excel(io.BytesIO(file_content), sheet_name=None)
                is_excel = True
            except:
                # Fall back to CSV
                df = {"Sheet1": pd.read_csv(io.BytesIO(file_content))}
                is_excel = False

            # Process each sheet
            for sheet_name, sheet_df in df.items():
                # Check if it's a Q&A format (has 'question' and 'answer' columns)
                columns_lower = [col.lower().strip() for col in sheet_df.columns]

                if "question" in columns_lower and "answer" in columns_lower:
                    # Q&A format
                    q_col = sheet_df.columns[columns_lower.index("question")]
                    a_col = sheet_df.columns[columns_lower.index("answer")]

                    for idx, row in sheet_df.iterrows():
                        question = str(row[q_col]).strip()
                        answer = str(row[a_col]).strip()

                        if (
                            question
                            and answer
                            and question != "nan"
                            and answer != "nan"
                        ):
                            content.append(
                                {
                                    "text": f"Q: {question}\nA: {answer}",
                                    "metadata": {
                                        "source": "excel" if is_excel else "csv",
                                        "sheet": sheet_name,
                                        "row": idx + 2,  # +2 for header and 0-index
                                        "type": "qa_pair",
                                        "question": question,
                                        "answer": answer,
                                    },
                                }
                            )
                else:
                    # Generic tabular data - convert to text
                    for idx, row in sheet_df.iterrows():
                        row_text = " | ".join(
                            [
                                f"{col}: {str(val).strip()}"
                                for col, val in row.items()
                                if str(val).strip() and str(val) != "nan"
                            ]
                        )

                        if row_text:
                            content.append(
                                {
                                    "text": row_text,
                                    "metadata": {
                                        "source": "excel" if is_excel else "csv",
                                        "sheet": sheet_name,
                                        "row": idx + 2,
                                        "type": "table_row",
                                    },
                                }
                            )

            if not content:
                raise ValueError("No data could be extracted from file")

            return {
                "content": content,
                "metadata": {
                    "extraction_method": "pandas",
                    "file_type": "excel" if is_excel else "csv",
                    "total_sheets": len(df),
                    "total_rows": sum(len(sheet) for sheet in df.values()),
                    "warnings": warnings if warnings else None,
                },
            }

        except Exception as e:
            logger.error(f"Excel/CSV processing failed: {str(e)}")
            raise ValueError(f"Failed to process file: {str(e)}")
