from typing import List
from google import genai
from google.genai import types
from app.core.config import settings
from app.core.logging_config import logger


class EmbeddingService:
    """
    Unified embedding service using Google Gemini

    Uses: gemini-embedding-001 model (latest, replaces text-embedding-004)
    Free tier: 1,500 requests/minute
    Context: Up to 2,048 tokens per input
    Output: 768 dimensions (configurable up to 3072)
    """

    def __init__(self):
        # Initialize the new genai client
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.model = settings.EMBEDDING_MODEL  # "gemini-embedding-001"
        logger.info(f"Initialized Gemini embedding service with model: {self.model}")

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (each is a list of floats)
        """
        try:
            if not texts:
                raise ValueError("Empty text list provided")

            # Validate inputs
            for i, text in enumerate(texts):
                if not isinstance(text, str):
                    raise ValueError(f"Text at index {i} is not a string: {type(text)}")
                if not text.strip():
                    raise ValueError(f"Text at index {i} is empty")

            logger.info(f"Generating embeddings for {len(texts)} texts using Gemini")

            # New SDK: Use embed_content with batch of texts
            result = self.client.models.embed_content(
                model=self.model,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",  # For storing in vector DB
                    output_dimensionality=settings.EMBEDDING_SIZE,  # 768
                ),
            )

            # Extract embeddings from result
            embeddings = [emb.values for emb in result.embeddings]

            logger.info(f"Successfully generated {len(embeddings)} embeddings")

            # Validate output
            if len(embeddings) != len(texts):
                raise ValueError(
                    f"Mismatch: {len(texts)} texts but {len(embeddings)} embeddings"
                )

            return embeddings

        except Exception as e:
            logger.error(f"Gemini batch embedding failed: {str(e)}")
            raise

    def generate_single_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text

        Args:
            text: Text string to embed

        Returns:
            Embedding vector (list of floats)
        """
        try:
            # Validate input
            if not isinstance(text, str):
                raise ValueError(f"Expected string input, got {type(text)}")

            if not text.strip():
                raise ValueError("Empty text input")

            logger.info(f"Generating single embedding for text length: {len(text)}")

            # Generate embedding using new SDK
            result = self.client.models.embed_content(
                model=self.model,
                contents=text,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=settings.EMBEDDING_SIZE,
                ),
            )

            # Extract embedding values
            embedding = result.embeddings[0].values

            # Validate output
            if not isinstance(embedding, list):
                raise ValueError(f"Embedding is not a list: {type(embedding)}")

            if len(embedding) == 0:
                raise ValueError("Empty embedding vector")

            # Ensure all elements are floats
            embedding = [float(x) for x in embedding]

            logger.info(
                f"Successfully generated embedding with dimension: {len(embedding)}"
            )
            return embedding

        except Exception as e:
            logger.error(
                f"Gemini single embedding failed for text '{text[:100]}...': {str(e)}"
            )
            raise

    def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding optimized for query (search) use

        Args:
            query: Query text to embed

        Returns:
            Embedding vector (list of floats)
        """
        try:
            if not isinstance(query, str):
                raise ValueError(f"Expected string input, got {type(query)}")

            if not query.strip():
                raise ValueError("Empty query input")

            logger.info(f"Generating query embedding for: {query[:50]}...")

            # Use RETRIEVAL_QUERY task type for search queries
            result = self.client.models.embed_content(
                model=self.model,
                contents=query,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY",  # Optimized for querying
                    output_dimensionality=settings.EMBEDDING_SIZE,
                ),
            )

            embedding = result.embeddings[0].values

            # Validate and convert
            if not isinstance(embedding, list):
                raise ValueError(f"Embedding is not a list: {type(embedding)}")

            if len(embedding) == 0:
                raise ValueError("Empty embedding vector")

            embedding = [float(x) for x in embedding]

            logger.info(f"Successfully generated query embedding")
            return embedding

        except Exception as e:
            logger.error(f"Gemini query embedding failed: {str(e)}")
            raise


# Backward compatibility aliases (for minimal code changes in routes)
EmbeddingServiceOpenai = EmbeddingService  # Old name → New service
EmbeddingServiceHuggingFace = EmbeddingService  # Old name → New service
