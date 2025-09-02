# app/services/embeddings.py
from re import L
from urllib import response
import openai
from typing import List
from huggingface_hub import InferenceClient
from app.core.config import settings
from app.core.logging_config import logger

# Initialize OpenAI client
openai.api_key = settings.OPENAI_API_KEY


class EmbeddingServiceOpenai:

    @staticmethod
    def generate_embeddings(texts: List[str], provider: str = "openai") -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        if provider == "openai":
            response = openai.embeddings.create(
                model=settings.EMBEDDING_MODEL,
                input=texts
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
        api_key=settings.HUGGING_FACE_KEY
    )
    
    
    @staticmethod
    def generate_embeddings(texts):
        """Generate embeddings for a list of texts using Hugging Face"""
        # Feature extraction can handle a batch list directly
        response = EmbeddingServiceHuggingFace.client.feature_extraction(texts)
        logger.info(f"Generated HuggingFace embeddings for {len(texts)} texts")
        print("response from embeddings", response)
        logger.info(f"response from embeddings {response}")
        return response    
        
    @staticmethod
    def generate_single_embeddings(text):
        
        response = EmbeddingServiceHuggingFace.client.feature_extraction(text)
        
        logger.info(f"Generated HuggingFace embeddings for {len(text)} texts")
        print("response from single  embeddings", response)
        logger.info(f"response from single  embeddings {response}")
        return response
    
    
    