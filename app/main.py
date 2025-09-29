from fastapi import FastAPI

from app.api.v1 import routes_embeddings
from app.api.v1 import routes_health
from app.api.v1 import routes_chat

akiliAi = FastAPI(title="Akili AI Engine", version="1.0.0")

akiliAi.include_router(
    routes_embeddings.router, prefix="/api/v1/embeddings", tags=["Embeddings"]
)
akiliAi.include_router(routes_health.router, prefix="/api/v1/health", tags=["Health"])
akiliAi.include_router(routes_chat.router, prefix="/api/v1/chat", tags=["Chat"])


@akiliAi.get("/ping")
def first_func():
    return {"message": "AI Engine Ready"}
