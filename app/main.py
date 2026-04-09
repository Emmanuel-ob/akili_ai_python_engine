from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1 import (
    routes_chat,
    routes_embeddings,
    routes_health,
    routes_analysis,
    routes_faq,
    routes_intelligence,
)
from app.core.config import settings
from app.core.logging_config import logger

akiliAi = FastAPI(
    title="Akili AI Engine",
    description="AI-powered chat and embedding service with FAQ support",
    version="2.0.0",
)

# Item 10: Restrict CORS to explicit allowed origins from config.
# allow_origins=["*"] with allow_credentials=True is a browser spec violation and a security risk.
akiliAi.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
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
akiliAi.include_router(
    routes_analysis.router, prefix="/api/v1/analyze", tags=["Analysis"]
)
akiliAi.include_router(routes_faq.router, prefix="/api/v1/faq", tags=["FAQ"])
akiliAi.include_router(
    routes_intelligence.router, prefix="/api/v1/intelligence", tags=["Intelligence"]
)


def validate_settings():
    """
    Item 3: Refuse to start if any required secret is missing.
    This prevents the app from running in an insecure state when .env is absent or misconfigured.
    """
    required_secrets = {
        "FASTAPI_SHARED_KEY": settings.FASTAPI_SHARED_KEY,
        "GEMINI_API_KEY": settings.GEMINI_API_KEY,
        "POSTGRES_PASSWORD": settings.POSTGRES_PASSWORD,
        "POSTGRES_USER": settings.POSTGRES_USER,
    }

    missing = [name for name, value in required_secrets.items() if not value]

    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}. "
            f"Check your .env file — the application will not start without these values set."
        )


@akiliAi.on_event("startup")
async def startup_event():
    validate_settings()  # Item 3: hard-fail on missing secrets
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
