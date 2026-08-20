import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # API Keys — defaults are empty strings intentionally.
    # validate_settings() in main.py will refuse startup if any secret is unset.
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    FASTAPI_SHARED_KEY: str = os.getenv("FASTAPI_SHARED_KEY", "")  # Item 3: removed hardcoded fallback

    # Gemini API Key (Primary for embeddings and LLM)
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    # Groq API Key (Fallback for LLM when rate limit hit)
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

    # PostgreSQL + pgvector Config — non-secret connection params keep their defaults
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "akili_vectors")
    # Item 3: removed hardcoded username/password defaults
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "")

    @property
    def POSTGRES_URL(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    # Embedding Config
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
    EMBEDDING_SIZE: int = int(os.getenv("EMBEDDING_SIZE", "768"))

    # LLM Models
    GEMINI_LLM_MODEL: str = os.getenv("GEMINI_LLM_MODEL", "gemini-2.5-flash")
    GROQ_LLM_MODEL: str = os.getenv("GROQ_LLM_MODEL", "llama-3.3-70b-versatile")

    # Rate Limiting Config
    GEMINI_RATE_LIMIT: int = int(os.getenv("GEMINI_RATE_LIMIT", "14"))
    RATE_LIMIT_WINDOW: int = 60

    # Chunking Config
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    # Chat Configuration
    MAX_HISTORY_LENGTH: int = 10

    # How many chunks to pull from the vector store per query, before gating.
    DEFAULT_SEARCH_LIMIT: int = int(os.getenv("DEFAULT_SEARCH_LIMIT", "5"))

    # Minimum cosine similarity for a chunk to be treated as relevant.
    #
    # Env-overridable so this can be tuned in production without a deploy. It
    # is the single most sensitive knob in the retrieval path: too low and the
    # model is handed near-random context and answers confidently from it (the
    # engine shipped with an effective 0.20 for exactly this reason); too high
    # and the bot says it does not know things it actually knows.
    #
    # Tune against real conversation logs, not by intuition.
    #
    # Shipping at 0.35 deliberately. The engine ran at an effective 0.20 for its
    # whole life, so jumping straight to the 0.5 this file always declared would
    # silence the bot on questions it currently answers correctly. 0.35 is a
    # staging post: raise it toward 0.5 while watching real conversations, using
    # the env var so no deploy is needed to adjust it.
    MIN_CONFIDENCE_THRESHOLD: float = float(os.getenv("MIN_CONFIDENCE_THRESHOLD", "0.35"))

    # Laravel API URL for query execution
    LARAVEL_API_URL: str = os.getenv("LARAVEL_API_URL", "http://localhost:8000")

    # Item 10: CORS allowed origins — never use "*" with credentials=True
    # Local dev:   ALLOWED_ORIGINS=http://localhost:8000,http://localhost:3000
    # Production:  ALLOWED_ORIGINS=https://your-backend-domain.com
    ALLOWED_ORIGINS: list = os.getenv(
        "ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000,http://localhost:3000"
    ).split(",")


settings = Settings()
