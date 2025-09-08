import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # API Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    FASTAPI_SHARED_KEY: str = os.getenv(
        "FASTAPI_SHARED_KEY", "your-secret-key")

    # Qdrant Config
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")

    # Embedding Config
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "text-embedding-ada-002")
    EMBEDDING_SIZE: int = int(os.getenv("EMBEDDING_SIZE", "1536"))
    HUGGING_FACE_KEY: str = os.getenv("HUGGING_FACE_KEY","")

    # Chunking Config
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    # Chat Configuration
    MAX_HISTORY_LENGTH: int = 10
    DEFAULT_SEARCH_LIMIT: int = 5
    MIN_CONFIDENCE_THRESHOLD: float = 0.5
    
    


settings = Settings()
