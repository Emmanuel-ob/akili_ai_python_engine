from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1 import (
    routes_chat,
    routes_embeddings,
    routes_health,
    routes_analysis,
    routes_faq,
)
from app.core.logging_config import logger

akiliAi = FastAPI(
    title="Akili AI Engine",
    description="AI-powered chat and embedding service with FAQ support",
    version="2.0.0",
)

# CORS
akiliAi.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
akiliAi.include_router(routes_health.router, prefix="/api/v1/health", tags=["Health"])
akiliAi.include_router(routes_chat.router, prefix="/api/v1/chat", tags=["Chat"])
akiliAi.include_router(
    routes_embeddings.router, prefix="/api/v1/embeddings", tags=["Embeddings"]
)
akiliAi.include_router(routes_analysis.router, prefix="/api/v1/analyze", tags=["Analysis"])
akiliAi.include_router(routes_faq.router, prefix="/api/v1/faq", tags=["FAQ"])  # NEW


@akiliAi.on_event("startup")
async def startup_event():
    logger.info("Akili AI Engine started successfully with FAQ support")


@akiliAi.on_event("shutdown")
async def shutdown_event():
    logger.info("Akili AI Engine shutting down")


@akiliAi.get("/")
async def root():
    return {
        "message": "Akili AI Engine API",
        "version": "2.0.0",
        "features": ["Chat", "Embeddings", "FAQ Management", "Analysis"],
        "docs": "/docs",
    }

@akiliAi.get("/ping")
def health_check():
    return {"message": "AI Engine Ready", "version": "2.0.0"}
