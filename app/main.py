from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import routes_embeddings
from app.api.v1 import routes_health
from app.api.v1 import routes_chat
from app.api.v1 import routes_analysis  # NEW

akiliAi = FastAPI(
    title="Akili AI Engine",
    version="2.0.0",
    description="AI-Powered Database Chat System with Dynamic Query Generation",
)

# CORS middleware
akiliAi.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
akiliAi.include_router(
    routes_embeddings.router, prefix="/api/v1/embeddings", tags=["Embeddings"]
)
akiliAi.include_router(routes_health.router, prefix="/api/v1/health", tags=["Health"])
akiliAi.include_router(routes_chat.router, prefix="/api/v1/chat", tags=["Chat"])
akiliAi.include_router(
    routes_analysis.router,  # NEW
    prefix="/api/v1/analyze",
    tags=["Analysis"],
)


@akiliAi.get("/ping")
def health_check():
    return {"message": "AI Engine Ready", "version": "2.0.0"}
