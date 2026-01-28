import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # API Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    FASTAPI_SHARED_KEY: str = os.getenv("FASTAPI_SHARED_KEY", "your-secret-key")

    # Gemini API Key (Primary for embeddings and LLM)
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    # Groq API Key (Fallback for LLM when rate limit hit)
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

    # PostgreSQL + pgvector Config
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
    # NOTE: Updated to use gemini-embedding-001 (text-embedding-004 will be deprecated Jan 2026)
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "gemini-embedding-001"
    )  # Latest Gemini embedding model
    EMBEDDING_SIZE: int = int(
        os.getenv("EMBEDDING_SIZE", "768")
    )  # 768 is recommended (can go up to 3072 for gemini-embedding-001)

    # LLM Models
    GEMINI_LLM_MODEL: str = os.getenv(
        "GEMINI_LLM_MODEL", "gemini-2.5-flash"
    )  # Best free Gemini model (or use "gemini-2.0-flash" for stable)
    GROQ_LLM_MODEL: str = os.getenv(
        "GROQ_LLM_MODEL", "llama-3.3-70b-versatile"
    )  # Best Groq model

    # Rate Limiting Config
    GEMINI_RATE_LIMIT: int = int(
        os.getenv("GEMINI_RATE_LIMIT", "14")
    )  # Max requests per minute before switching to Groq
    RATE_LIMIT_WINDOW: int = 60  # Time window in seconds (1 minute)

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
