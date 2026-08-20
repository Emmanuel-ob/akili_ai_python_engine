from typing import List
from google import genai
from google.genai import types
from app.core.config import settings
from app.core.logging_config import logger
from app.services.embedding_resilience import batch_texts, cap_input, with_retry


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

            # Batch rather than sending the whole list as one request. A large
            # document was previously a single enormous call that failed as a
            # unit; now one bad batch loses a batch, and each text inside it
            # is capped and retried individually.
            embeddings = []
            for batch in batch_texts(texts, settings.EMBED_BATCH_SIZE):
                capped = [cap_input(t, settings.EMBED_MAX_INPUT_CHARS) for t in batch]
                result = with_retry(
                    lambda payload: self.client.models.embed_content(
                        model=self.model,
                        contents=payload,
                        config=types.EmbedContentConfig(
                            task_type="RETRIEVAL_DOCUMENT",
                            output_dimensionality=settings.EMBEDDING_SIZE,  # 768
                        ),
                    ),
                    capped,
                    retries=settings.EMBED_RETRIES,
                )
                embeddings.extend(emb.values for emb in result.embeddings)

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
        """Embed one chunk, with an input cap and retries.

        The cap and the retry loop are the difference between one bad chunk
        failing and one bad chunk taking an entire document with it. See
        app/services/embedding_resilience.py for why an over-length rejection
        is retried by shrinking rather than by waiting.

        Only the VECTOR input is capped; the full chunk text is stored
        unchanged by the caller, so no answer content is lost.
        """
        if not isinstance(text, str):
            raise ValueError(f"Expected string input, got {type(text)}")
        if not text.strip():
            raise ValueError("Empty text input")

        return with_retry(
            self._embed_once,
            cap_input(text, settings.EMBED_MAX_INPUT_CHARS),
            retries=settings.EMBED_RETRIES,
        )

    def get_provider_info(self) -> dict:
        """Provider and model that produced a vector.

        Stored on every embedded row so a future provider change is
        diagnosable: vectors from two different models are not comparable, and
        without provenance a mixed collection degrades silently. Matches the
        shape Trivia records, so the two products stay consistent.
        """
        return {"provider": "gemini", "model": self.model}

    def _embed_once(self, text: str) -> List[float]:
        """One raw embedding call. Raises; retry policy lives in the caller."""
        try:
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
