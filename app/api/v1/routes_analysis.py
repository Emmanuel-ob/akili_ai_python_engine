from fastapi import APIRouter, HTTPException, Depends, Header
from app.schemas.analysis import (
    SchemaAnalysisRequest,
    SchemaAnalysisResponse,
    BusinessOverviewRequest,
    BusinessOverviewResponse,
    QueryGenerationRequest,
    QueryGenerationResponse,
)
from app.services.schema_analyzer import SchemaAnalyzer, BusinessOverviewGenerator
from app.services.llm import LLMService
from app.core.config import settings
from app.core.logging_config import logger

router = APIRouter()

# Initialize services
llm_service = LLMService()
schema_analyzer = SchemaAnalyzer(llm_service)
overview_generator = BusinessOverviewGenerator(llm_service)


def verify_api_key(x_akili_key: str = Header(...)):
    """Verify API key"""
    if x_akili_key != settings.FASTAPI_SHARED_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


@router.post("/schema", response_model=SchemaAnalysisResponse)
async def analyze_schema(
    request: SchemaAnalysisRequest, _: bool = Depends(verify_api_key)
):
    """Analyze database table schema and detect relationships"""

    try:
        logger.info(f"Analyzing schema for table: {request.table}")
        logger.info(f"Available tables: {request.all_tables}")

        analysis = await schema_analyzer.analyze_table_structure(
            table=request.table, columns=request.columns, all_tables=request.all_tables
        )

        logger.info(f"Schema analysis result: {analysis}")

        return SchemaAnalysisResponse(success=True, analysis=analysis)

    except Exception as e:
        logger.error(f"Schema analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/business-overview", response_model=BusinessOverviewResponse)
async def generate_business_overview(
    request: BusinessOverviewRequest, _: bool = Depends(verify_api_key)
):
    """Generate comprehensive business overview from schema"""

    try:
        logger.info(
            f"Generating business overview for {request.table_count} tables: "
            f"{', '.join(request.selected_tables)}"
        )

        # Validate that schema_analysis contains the selected tables
        schema_tables = list(request.schema_analysis.get("tables", {}).keys())
        missing_tables = [t for t in request.selected_tables if t not in schema_tables]

        if missing_tables:
            logger.warning(
                f"Some selected tables not in schema_analysis: {missing_tables}"
            )

        result = await overview_generator.generate_overview(
            schema_analysis=request.schema_analysis,
            database_type=request.database_type,
            selected_tables=request.selected_tables,  # Pass explicitly
        )

        logger.info(
            f"Business overview generated successfully. "
            f"Discovered {len(result.get('discovered_actions', []))} actions."
        )

        return BusinessOverviewResponse(
            success=True,
            overview=result.get("overview", ""),
            discovered_actions=result.get("discovered_actions", []),
            analyzed_tables=result.get("analyzed_tables", request.selected_tables),
        )

    except ValueError as e:
        logger.error(f"Validation error in business overview: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Business overview generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
