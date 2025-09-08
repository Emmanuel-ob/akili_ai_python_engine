import hashlib
from typing import List, Dict, Any
from venv import logger
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from app.core.config import settings
from app.core.logging_config import logger


class VectorStoreService:

    def __init__(self):
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY if settings.QDRANT_API_KEY else None
        )

    def get_collection_name(self, business_id: str, chatbot_id: str) -> str:
        """Generate collection name from business and chatbot IDs"""
        return f"akili_{business_id}_{chatbot_id}".replace("-", "_")

    def ensure_collection_exists(self, collection_name: str):
        """Create collection if it doesn't exist"""
        try:
            self.client.get_collection(collection_name)
        except:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=settings.EMBEDDING_SIZE,
                    distance=Distance.COSINE
                )
            )

    def generate_point_id(self, doc_id: str) -> str:
        """Generate a unique point ID from document ID"""
        return hashlib.md5(doc_id.encode()).hexdigest()

    def upsert_documents(
        self,
        collection_name: str,
        documents: List[Dict[str, Any]],
        embeddings: List[List[float]],
        business_id: str,
        chatbot_id: str
    ) -> int:
        """Upsert documents with embeddings to Qdrant"""

        # Ensure collection exists
        self.ensure_collection_exists(collection_name)

        # Prepare points
        points = []
        for i, doc in enumerate(documents):
            point = PointStruct(
                id=self.generate_point_id(doc['doc_id']),
                vector=embeddings[i],
                payload={
                    "doc_id": doc['doc_id'],
                    "text": doc['text'],
                    "data": doc['data'],
                    "hash": doc['hash'],
                    "business_id": business_id,
                    "chatbot_id": chatbot_id
                }
            )
            points.append(point)
        print("points to be upserted ", points)
        logger.info(f"Prepared {len(points)} points for upsert to collection {collection_name}")

        # Upsert to Qdrant
        self.client.upsert(collection_name=collection_name, points=points)

        return len(points)
    

    def search_similar(
        self, collection_name: str, query_vector: List[float], limit: int = 5
    ):
        """Search for similar documents in the vector database"""
        try:
            # Ensure collection exists
            self.client.get_collection(collection_name)

            # Unwrap if embedding is nested
            if isinstance(query_vector[0], list):
                query_vector = query_vector[0]

            # Perform search
            search_results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                with_payload=True,
            )

            for res in search_results:
                logger.info(
                    f"Found doc_id={res.payload.get('doc_id')} score={res.score:.4f}"
                )

            return search_results

        except Exception as e:
            logger.error(f"Error searching vector database: {str(e)}")
            return []
