from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from app.services.schema_analyzer import (
    SchemaAnalyzer,
    BusinessOverviewGenerator,
    ProfileBuilder,
)
from app.services.llm import LLMService
from app.core.logging_config import logger

router = APIRouter()

llm_service = LLMService()
profile_builder = ProfileBuilder(llm_service)


# ── Request / Response models ──────────────────────────────────────────────────


class TableProfile(BaseModel):
    columns: List[Dict[str, Any]]
    foreign_keys: List[Dict[str, Any]] = []
    sample_rows: List[Dict[str, Any]] = []
    enum_hints: Dict[str, List[str]] = {}
    row_count_estimate: Optional[int] = None


class BuildProfileRequest(BaseModel):
    connection_id: str
    business_id: str
    chatbot_id: str
    database_type: str  # mysql | postgresql | mongodb
    all_tables: List[Dict[str, Any]]  # [{name, likely_system}]
    selected_tables: List[str]
    table_profiles: Dict[str, TableProfile]  # keyed by table name


class SchemaAnalysisRequest(BaseModel):
    """Legacy endpoint — kept for backward compatibility."""

    table: str
    columns: List[str]
    all_tables: List[str]
    task: str = "analyze_schema"


class BusinessOverviewRequest(BaseModel):
    """Legacy endpoint — kept for backward compatibility."""

    schema_analysis: Dict[str, Any]
    database_type: str
    table_count: int
    selected_tables: List[str]
    task: str = "generate_business_overview"


# ── New unified endpoint ───────────────────────────────────────────────────────


@router.post("/build-profile")
async def build_database_profile(request: BuildProfileRequest):
    """
    Point 1: Build a rich DatabaseProfile from schema + samples.

    Replaces the old row-embedding approach. Called once during sync.
    The resulting profile is stored on the DatabaseConnection and passed
    to the SQL generator at query time — no vector search needed for DB queries.
    """
    try:
        logger.info(
            f"Building database profile for connection {request.connection_id} "
            f"({len(request.selected_tables)} selected tables)"
        )

        # Convert pydantic models to plain dicts for internal use
        table_profiles_dict = {
            table: profile.model_dump()
            for table, profile in request.table_profiles.items()
        }

        profile = await profile_builder.build(
            connection_id=request.connection_id,
            business_id=request.business_id,
            chatbot_id=request.chatbot_id,
            database_type=request.database_type,
            all_tables=request.all_tables,
            selected_tables=request.selected_tables,
            table_profiles=table_profiles_dict,
        )

        return profile

    except Exception as e:
        logger.error(f"Profile build failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Profile build failed: {str(e)}")


# ── Legacy endpoints (unchanged — other parts of the system may call these) ───


@router.post("/schema")
async def analyze_schema(request: SchemaAnalysisRequest):
    """Legacy: analyze a single table's structure."""
    try:
        analyzer = SchemaAnalyzer(llm_service)
        analysis = await analyzer.analyze_table_structure(
            table=request.table,
            columns=request.columns,
            all_tables=request.all_tables,
        )
        return {"analysis": analysis}
    except Exception as e:
        logger.error(f"Schema analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/business-overview")
async def generate_business_overview(request: BusinessOverviewRequest):
    """Legacy: generate business overview from schema analysis."""
    try:
        generator = BusinessOverviewGenerator(llm_service)
        overview = await generator.generate_overview(
            schema_analysis=request.schema_analysis,
            database_type=request.database_type,
            selected_tables=request.selected_tables,
        )
        return overview
    except Exception as e:
        logger.error(f"Business overview generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
