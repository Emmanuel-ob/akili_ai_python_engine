"""
routes_intelligence.py — Point 5A + 5B

Endpoints:
  POST /api/v1/intelligence/generate-insights  → run metric queries, return insight cards
  POST /api/v1/intelligence/process-report     → embed an uploaded internal report (5B)
  GET  /api/v1/intelligence/insights/{business_id}/{connection_id}  → fetch cached insights
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import base64
import io

from app.services.insight_engine import InsightEngine
from app.services.llm import LLMService
from app.services.vectorstore import VectorStoreService
from app.services.embeddings import EmbeddingService
from app.services.faq_manager import FAQManager
from app.core.logging_config import logger

router = APIRouter()

# Singletons
_llm = LLMService()
_insight_engine = InsightEngine(_llm)
_vectorstore = VectorStoreService()
_embedding = EmbeddingService()
_faq_manager = FAQManager(_vectorstore, _embedding)


# ── Request / Response models ──────────────────────────────────────────────────


class GenerateInsightsRequest(BaseModel):
    business_id: str
    connection_id: str
    database_config: Dict[str, Any]
    database_profile: Dict[str, Any]
    business_overview: Optional[str] = ""
    max_insights: int = 6


class ProcessReportRequest(BaseModel):
    business_id: str
    chatbot_id: str
    report_name: str
    report_id: str  # Laravel report DB id (for doc_id)
    file_content_base64: str  # base64-encoded file bytes
    file_type: str  # "pdf" | "docx" | "excel" | "csv" | "txt"


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.post("/generate-insights")
async def generate_insights(request: GenerateInsightsRequest):
    """
    Point 5A: Run proactive metric queries and generate insight cards.

    Called by Laravel's GenerateInsightsJob.
    Returns a list of insight card objects to be stored in business_insights.
    """
    try:
        logger.info(
            f"Generating insights for business={request.business_id} "
            f"connection={request.connection_id}"
        )

        insights = await _insight_engine.generate_insights(
            database_config=request.database_config,
            database_profile=request.database_profile,
            business_overview=request.business_overview or "",
            max_insights=request.max_insights,
        )

        return {
            "success": True,
            "business_id": request.business_id,
            "connection_id": request.connection_id,
            "insights": insights,
            "count": len(insights),
        }

    except Exception as e:
        logger.error(f"generate-insights failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Insight generation failed: {str(e)}"
        )


@router.post("/process-report")
async def process_report(request: ProcessReportRequest):
    """
    Point 5B: Process an uploaded internal report (PDF/DOCX/Excel/CSV) and
    embed its content into the vector store tagged as source_type='internal_report'.

    The embedded chunks are then searchable by the internal chat mode.
    """
    try:
        logger.info(
            f"Processing report '{request.report_name}' for business={request.business_id}"
        )

        # Decode the file
        file_bytes = base64.b64decode(request.file_content_base64)

        # Route to the correct file processor
        text_chunks = await _extract_text_from_file(
            file_bytes=file_bytes,
            file_type=request.file_type.lower(),
            report_name=request.report_name,
        )

        if not text_chunks:
            return {
                "success": False,
                "message": "No extractable text found in file.",
                "chunks_embedded": 0,
            }

        # Embed into vector store, tagged as internal_report
        collection_name = _vectorstore.get_collection_name(
            request.business_id, request.chatbot_id
        )
        _vectorstore.ensure_collection(collection_name)

        embedded_count = 0
        for idx, chunk_text in enumerate(text_chunks):
            if not chunk_text.strip():
                continue

            embedding = _embedding.generate_embedding(chunk_text)
            doc_id = f"report_{request.report_id}_chunk_{idx}"

            _vectorstore.upsert_vectors(
                collection_name=collection_name,
                vectors=[embedding],
                payloads=[
                    {
                        "doc_id": doc_id,
                        "text": chunk_text,
                        "business_id": request.business_id,
                        "chatbot_id": request.chatbot_id,
                        "source_type": "internal_report",
                        "metadata": {
                            "report_id": request.report_id,
                            "report_name": request.report_name,
                            "file_type": request.file_type,
                            "chunk_index": idx,
                        },
                    }
                ],
                ids=[doc_id],
            )
            embedded_count += 1

        logger.info(
            f"Embedded {embedded_count} chunks from report '{request.report_name}'"
        )

        return {
            "success": True,
            "report_name": request.report_name,
            "chunks_embedded": embedded_count,
        }

    except Exception as e:
        logger.error(f"process-report failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Report processing failed: {str(e)}"
        )


# ── File extraction helpers ────────────────────────────────────────────────────


async def _extract_text_from_file(
    file_bytes: bytes, file_type: str, report_name: str
) -> List[str]:
    """
    Route to the correct file processor and return a list of text chunks.
    Reuses the existing file_processors/ modules.
    """
    try:
        if file_type in ("pdf",):
            from app.services.file_processors.pdf_processor import PDFProcessor

            processor = PDFProcessor()
            return processor.extract_chunks(file_bytes)

        elif file_type in ("docx", "doc"):
            from app.services.file_processors.docx_processor import DocxProcessor

            processor = DocxProcessor()
            return processor.extract_chunks(file_bytes)

        elif file_type in ("xlsx", "xls", "csv"):
            from app.services.file_processors.excel_processor import ExcelProcessor

            processor = ExcelProcessor()
            return processor.extract_chunks(file_bytes)

        elif file_type in ("json",):
            from app.services.file_processors.json_processor import JSONProcessor

            processor = JSONProcessor()
            return processor.extract_chunks(file_bytes)

        elif file_type in ("txt", "text", "md"):
            from app.services.file_processors.text_processor import TextProcessor

            processor = TextProcessor()
            return processor.extract_chunks(file_bytes)

        else:
            logger.warning(f"Unsupported file type: {file_type}")
            return []

    except Exception as e:
        logger.error(f"File extraction failed for type={file_type}: {e}")
        return []


# ── Generate integration guide ─────────────────────────────────────────────────

from pydantic import BaseModel as _BaseModel


class GenerateGuideRequest(_BaseModel):
    framework: str
    widget_token: str
    chatbot_name: str = "AI Assistant"
    site_url: str = ""


@router.post("/generate-guide")
async def generate_framework_guide(request: GenerateGuideRequest):
    """
    Generate a tailored integration guide for any framework using the LLM.
    Called when a merchant's framework isn't in the pre-written list.
    """
    try:
        logger.info(f"Generating integration guide for framework: {request.framework}")

        prompt = f"""You are a developer documentation writer. Generate a concise integration guide for adding the XeliAI chat widget to a **{request.framework}** project.

Widget script tag (use exactly as-is):
<script src="{request.site_url}/widget.js?token={request.widget_token}" async></script>

The guide must have exactly these two sections, no more:

## 1. Script Installation
Explain exactly WHERE to place the script tag so it loads on every page in a {request.framework} project. Be specific about the file name and location. Show the full file snippet with the script tag in context.

## 2. User Identity (Optional but Recommended)  
Show how to call XeliAI.identify() using {request.framework}'s native auth/session pattern so the chatbot knows who it's talking to. Use realistic variable names for {request.framework}.

The identify call structure is always:
XeliAI.identify({{
  user_id: "...",
  email: "...",
  name: "...",
  plan: "..."  // optional
}});

Rules:
- Use actual {request.framework} syntax and conventions
- Keep each section short and practical — developers should be able to copy-paste
- Use markdown code blocks with the correct language tag
- Do not add any other sections or commentary"""

        messages = [{"role": "user", "content": prompt}]

        if _llm._should_use_groq():
            guide = await _llm._generate_with_groq(messages)
        else:
            try:
                guide = await _llm._generate_with_gemini(messages)
                _llm.rate_limiter.add_request("gemini")
            except Exception:
                guide = await _llm._generate_with_groq(messages)

        return {
            "success": True,
            "framework": request.framework,
            "guide": guide,
        }

    except Exception as e:
        logger.error(f"generate-guide failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Guide generation failed: {str(e)}"
        )
