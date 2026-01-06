import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # API Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    FASTAPI_SHARED_KEY: str = os.getenv("FASTAPI_SHARED_KEY", "your-secret-key")
    HUGGING_FACE_KEY: str = os.getenv("HUGGING_FACE_KEY", "")

    # PostgreSQL + pgvector Config (NEW - replaces Qdrant)
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "akili_vectors")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "prnzdiamond")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "diamond")
    
    # Build PostgreSQL connection string
    @property
    def POSTGRES_URL(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    # Embedding Config
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    EMBEDDING_SIZE: int = int(
        os.getenv("EMBEDDING_SIZE", "768")
    )  # 768 for HuggingFace all-mpnet-base-v2

    # Chunking Config
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    # Chat Configuration
    MAX_HISTORY_LENGTH: int = 10
    DEFAULT_SEARCH_LIMIT: int = 5
    MIN_CONFIDENCE_THRESHOLD: float = 0.5

    # Laravel API URL for query execution
    LARAVEL_API_URL: str = os.getenv("LARAVEL_API_URL", "http://localhost:8000")


settings = Settings()