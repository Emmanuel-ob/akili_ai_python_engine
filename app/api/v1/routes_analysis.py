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
from app.services.query_generator import SecureQueryGenerator
from app.services.llm import LLMServiceHuggingFace
from app.core.config import settings
from app.core.logging_config import logger

router = APIRouter()

# Initialize services
llm_service = LLMServiceHuggingFace()
schema_analyzer = SchemaAnalyzer(llm_service)
overview_generator = BusinessOverviewGenerator(llm_service)
query_generator = SecureQueryGenerator(llm_service)


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

    # try:
    logger.info(f"Analyzing schema for table: {request.table}")

    analysis = await schema_analyzer.analyze_table_structure(
            table=request.table, columns=request.columns, all_tables=request.all_tables
        )
    
    logger.info(f"Schema analysis result: {analysis}")

    return SchemaAnalysisResponse(success=True, analysis=analysis)

    # except Exception as e:
    #     logger.error(f"Schema analysis failed: {str(e)}")
    #     raise HTTPException(status_code=500, detail=str(e))


@router.post("/business-overview", response_model=BusinessOverviewResponse)
async def generate_business_overview(
    request: BusinessOverviewRequest, _: bool = Depends(verify_api_key)
):
    """Generate comprehensive business overview from schema"""

    try:
        logger.info(f"Generating business overview for {request.table_count} tables")

        result = await overview_generator.generate_overview(
            schema_analysis=request.schema_analysis, database_type=request.database_type
        )
        

        return BusinessOverviewResponse(
            success=True,
            overview=result.get("overview", ""),
            discovered_actions=result.get("discovered_actions", []),
        )

    except Exception as e:
        logger.error(f"Business overview generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/generate", response_model=QueryGenerationResponse)
async def generate_query(
    request: QueryGenerationRequest, _: bool = Depends(verify_api_key)
):
    """Generate safe database query from user intent"""

    try:
        logger.info(f"Generating query for: {request.user_question}")

        # This would typically get intent first, but for direct query generation:
        result = await query_generator.generate_query(
            intent={"description": request.user_question, "target_tables": []},
            schema=request.schema,
            customer_id=request.customer_id,
            connection_type=request.connection_type,
        )

        return QueryGenerationResponse(
            success=result.get("success", False),
            query=result.get("query", ""),
            explanation=result.get("explanation", ""),
            tables_used=result.get("tables_used", []),
            requires_auth=result.get("requires_auth", False),
            safe=result.get("safe", False),
        )

    except Exception as e:
        logger.error(f"Query generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
