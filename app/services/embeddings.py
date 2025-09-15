from re import L
from urllib import response
import openai
from typing import List
import numpy as np
from huggingface_hub import InferenceClient
from app.core.config import settings
from app.core.logging_config import logger

# Initialize OpenAI client
openai.api_key = settings.OPENAI_API_KEY


class EmbeddingServiceOpenai:
    @staticmethod
    def generate_embeddings(
        texts: List[str], provider: str = "openai"
    ) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        if provider == "openai":
            response = openai.embeddings.create(
                model=settings.EMBEDDING_MODEL, input=texts
            )
            return [item.embedding for item in response.data]
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")

    @staticmethod
    def generate_single_embedding(text: str, provider: str = "openai") -> List[float]:
        """Generate embedding for a single text"""
        embeddings = EmbeddingServiceOpenai.generate_embeddings([text], provider)
        return embeddings[0]


class EmbeddingServiceHuggingFace:
    client = InferenceClient(
        model="sentence-transformers/all-mpnet-base-v2",
        api_key=settings.HUGGING_FACE_KEY,
    )

    @staticmethod
    def generate_embeddings(texts):
        """Generate embeddings for a list of texts using Hugging Face"""
        try:
            # Feature extraction can handle a batch list directly
            response = EmbeddingServiceHuggingFace.client.feature_extraction(texts)
            logger.info(f"Generated HuggingFace embeddings for {len(texts)} texts")
            logger.info(
                f"Response type: {type(response)}, Shape: {response.shape if hasattr(response, 'shape') else 'No shape'}"
            )

            # Convert to proper format if needed
            if isinstance(response, np.ndarray):
                response = response.tolist()

            # Validate the response structure
            if not isinstance(response, list):
                raise ValueError(f"Unexpected response type: {type(response)}")

            # For batch processing, we expect a list of embeddings
            if len(response) != len(texts):
                raise ValueError(
                    f"Mismatch: {len(texts)} texts but {len(response)} embeddings"
                )

            return response
        except Exception as e:
            logger.error(f"HuggingFace batch embedding failed: {str(e)}")
            raise

    @staticmethod
    def generate_single_embeddings(text):
        """Generate embedding for a single text using Hugging Face"""
        try:
            # Validate input
            if not isinstance(text, str):
                raise ValueError(f"Expected string input, got {type(text)}")

            if not text.strip():
                raise ValueError("Empty text input")

            logger.info(f"Generating single embedding for text length: {len(text)}")

            # Call the API with a single string (not a list)
            response = EmbeddingServiceHuggingFace.client.feature_extraction(text)

            logger.info(f"Raw response type: {type(response)}")
            logger.info(
                f"Raw response shape: {response.shape if hasattr(response, 'shape') else 'No shape'}"
            )

            # Handle the response format
            if isinstance(response, np.ndarray):
                if response.ndim == 1:
                    # Single embedding vector - this is what we expect
                    embedding = response.tolist()
                elif response.ndim == 2:
                    # 2D array - this might happen if the model returns [1, embedding_dim]
                    if response.shape[0] == 1:
                        # Single embedding in a batch format
                        embedding = response[0].tolist()
                    else:
                        raise ValueError(
                            f"Unexpected 2D array shape: {response.shape}. Expected single embedding."
                        )
                else:
                    raise ValueError(f"Unexpected array dimensions: {response.ndim}")

            elif isinstance(response, list):
                if isinstance(response[0], (int, float)):
                    # Already a flat list of numbers - single embedding
                    embedding = response
                elif isinstance(response[0], list):
                    if len(response) == 1:
                        # Single embedding wrapped in a list
                        embedding = response[0]
                    else:
                        raise ValueError(
                            f"Unexpected nested list structure: {len(response)} embeddings returned for single text"
                        )
                else:
                    raise ValueError(
                        f"Unexpected list element type: {type(response[0])}"
                    )
            else:
                raise ValueError(f"Unexpected response type: {type(response)}")

            # Validate the final embedding
            if not isinstance(embedding, list):
                raise ValueError(f"Final embedding is not a list: {type(embedding)}")

            if len(embedding) == 0:
                raise ValueError("Empty embedding vector")

            # Ensure all elements are numbers
            try:
                embedding = [float(x) for x in embedding]
            except (ValueError, TypeError) as e:
                raise ValueError(f"Non-numeric values in embedding: {e}")

            logger.info(
                f"Successfully generated embedding with dimension: {len(embedding)}"
            )
            return embedding

        except Exception as e:
            logger.error(
                f"HuggingFace single embedding failed for text '{text[:100]}...': {str(e)}"
            )
            raise
