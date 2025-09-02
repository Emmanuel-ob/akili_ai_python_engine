# app/api/v1/routes_health.py
from fastapi import APIRouter
from app.schemas.embedding import HealthResponse

router = APIRouter()


@router.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        service="akili-ai-engine"
    )
