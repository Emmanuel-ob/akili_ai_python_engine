import hashlib
import json
from typing import List, Dict, Any, Optional
import psycopg2
from psycopg2.extras import execute_values, RealDictCursor
from pgvector.psycopg2 import register_vector
from app.core.config import settings
from app.core.logging_config import logger


class SearchResult:
    """Mimics Qdrant search result structure for compatibility"""

    def __init__(self, doc_id: str, score: float, payload: Dict[str, Any]):
        self.id = doc_id
        self.score = score
        self.payload = payload


class VectorStoreService:
    def __init__(self):
        """Initialize PostgreSQL connection pool"""
        self.connection_params = {
            "host": settings.POSTGRES_HOST,
            "port": settings.POSTGRES_PORT,
            "database": settings.POSTGRES_DB,
            "user": settings.POSTGRES_USER,
            "password": settings.POSTGRES_PASSWORD,
        }

        # Test connection and register vector type
        self._test_connection()
        logger.info("PostgreSQL + pgvector initialized successfully")

    def _test_connection(self):
        """Test database connection and register vector type"""
        try:
            conn = psycopg2.connect(**self.connection_params)
            register_vector(conn)
            conn.close()
            logger.info("PostgreSQL connection test successful")
        except Exception as e:
            logger.error(f"PostgreSQL connection failed: {str(e)}")
            raise

    def _get_connection(self):
        """Get a new database connection"""
        conn = psycopg2.connect(**self.connection_params)
        register_vector(conn)
        return conn

    def get_collection_name(self, business_id: str, chatbot_id: str) -> str:
        """Generate collection name from business and chatbot IDs"""
        return f"akili_{business_id}_{chatbot_id}".replace("-", "_")

    # ========================================
    # COLLECTIONS MANAGEMENT
    # ========================================

    def ensure_collection_exists(self, business_id: str, chatbot_id: str):
        """
        Ensure collection exists and is properly tracked
        Returns collection info
        """
        collection_name = self.get_collection_name(business_id, chatbot_id)

        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            cursor.execute(
                """
                INSERT INTO collections (
                    collection_name, 
                    business_id, 
                    chatbot_id, 
                    vector_dimension,
                    created_at,
                    updated_at
                )
                VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT (collection_name) 
                DO UPDATE SET
                    updated_at = CURRENT_TIMESTAMP
                RETURNING *
                """,
                (collection_name, business_id, chatbot_id, settings.EMBEDDING_SIZE),
            )

            collection = cursor.fetchone()
            conn.commit()
            cursor.close()
            conn.close()

            logger.debug(f"Collection ensured: {collection_name}")
            return dict(collection) if collection else None

        except Exception as e:
            if conn:
                conn.rollback()
                conn.close()
            logger.error(f"Error ensuring collection: {str(e)}")
            raise

    def update_collection_stats(self, business_id: str, chatbot_id: str):
        """
        Update collection statistics (call after bulk operations)
        """
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                WITH stats AS (
                    SELECT 
                        COUNT(*) as total_docs,
                        COUNT(*) FILTER (WHERE metadata->>'source_type' = 'faq') as faq_count,
                        COUNT(*) FILTER (WHERE metadata->>'source_type' = 'database') as db_count,
                        AVG(LENGTH(text)) as avg_text_length
                    FROM embeddings
                    WHERE business_id = %s AND chatbot_id = %s
                )
                UPDATE collections 
                SET 
                    document_count = stats.total_docs,
                    faq_document_count = stats.faq_count,
                    database_document_count = stats.db_count,
                    avg_document_size = CAST(stats.avg_text_length AS INTEGER),
                    updated_at = CURRENT_TIMESTAMP
                FROM stats
                WHERE business_id = %s AND chatbot_id = %s
                """,
                (business_id, chatbot_id, business_id, chatbot_id),
            )

            conn.commit()
            cursor.close()
            conn.close()

            logger.debug(f"Updated collection stats for {business_id}/{chatbot_id}")

        except Exception as e:
            if conn:
                conn.rollback()
                conn.close()
            logger.warning(f"Failed to update collection stats: {str(e)}")

    def get_collection_info(self, business_id: str, chatbot_id: str) -> Dict[str, Any]:
        """Get detailed collection information"""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            cursor.execute(
                """
                SELECT * FROM collections 
                WHERE business_id = %s AND chatbot_id = %s
                """,
                (business_id, chatbot_id),
            )

            collection = cursor.fetchone()

            if not collection:
                cursor.close()
                conn.close()
                return None

            cursor.execute(
                """
                SELECT 
                    COUNT(*) as total_docs,
                    COUNT(*) FILTER (WHERE metadata->>'source_type' = 'faq') as faq_count,
                    COUNT(*) FILTER (WHERE metadata->>'source_type' = 'database') as db_count,
                    COUNT(DISTINCT metadata->>'faq_source_id') as faq_sources,
                    COUNT(DISTINCT metadata->>'table') as database_tables
                FROM embeddings
                WHERE business_id = %s AND chatbot_id = %s
                """,
                (business_id, chatbot_id),
            )

            stats = cursor.fetchone()
            cursor.close()
            conn.close()

            result = dict(collection)
            result.update(
                {
                    "real_time_stats": dict(stats) if stats else {},
                    "health_check": {
                        "stats_synced": collection["document_count"]
                        == (stats["total_docs"] if stats else 0),
                        "last_updated": collection["updated_at"].isoformat()
                        if collection["updated_at"]
                        else None,
                    },
                }
            )

            return result

        except Exception as e:
            if conn:
                conn.close()
            logger.error(f"Error getting collection info: {str(e)}")
            return None

    def list_all_collections(self) -> List[Dict[str, Any]]:
        """List all collections with stats"""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            cursor.execute("SELECT * FROM collection_stats ORDER BY updated_at DESC")
            collections = cursor.fetchall()
            cursor.close()
            conn.close()

            return [dict(c) for c in collections]

        except Exception as e:
            if conn:
                conn.close()
            logger.error(f"Error listing collections: {str(e)}")
            return []

    # ========================================
    # DOCUMENT OPERATIONS
    # ========================================

    def upsert_documents(
        self,
        collection_name: str,
        documents: List[Dict[str, Any]],
        embeddings: List[List[float]],
        business_id: str,
        chatbot_id: str,
    ) -> int:
        """
        Upsert documents with embeddings to PostgreSQL
        Uses ON CONFLICT to update existing documents
        """
        if len(documents) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(documents)} documents but {len(embeddings)} embeddings"
            )

        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Ensure collection exists
            self.ensure_collection_exists(business_id, chatbot_id)

            # Prepare data for batch insert
            values = []
            for i, doc in enumerate(documents):
                # Merge all metadata into one level
                merged_metadata = {
                    "data": doc.get("data", {}),
                    "hash": doc.get("hash", ""),
                }

                # Add all fields from metadata
                if "metadata" in doc:
                    for key, value in doc["metadata"].items():
                        merged_metadata[key] = value

                values.append(
                    (
                        doc["doc_id"],
                        business_id,
                        chatbot_id,
                        embeddings[i],
                        doc["text"],
                        json.dumps(merged_metadata),
                    )
                )

            # Batch upsert with ON CONFLICT
            execute_values(
                cursor,
                """
                INSERT INTO embeddings (doc_id, business_id, chatbot_id, embedding, text, metadata)
                VALUES %s
                ON CONFLICT (doc_id) 
                DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    text = EXCLUDED.text,
                    metadata = EXCLUDED.metadata,
                    updated_at = CURRENT_TIMESTAMP
                """,
                values,
                template="(%s, %s, %s, %s, %s, %s::jsonb)",
            )

            conn.commit()
            cursor.close()
            conn.close()

            # Update collection stats (non-blocking)
            try:
                self.update_collection_stats(business_id, chatbot_id)
            except Exception as e:
                logger.warning(f"Failed to update collection stats: {str(e)}")

            logger.info(f"Successfully upserted {len(values)} documents")
            return len(values)

        except Exception as e:
            if conn:
                conn.rollback()
                conn.close()
            logger.error(f"Error upserting documents: {str(e)}")
            raise

    def search_similar(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 5,
        business_id: Optional[str] = None,
        chatbot_id: Optional[str] = None,
        source_type_filter: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Search for similar documents using cosine similarity
        Supports filtering by source_type ('faq' or 'database')
        """
        conn = None
        try:
            # Unwrap if embedding is nested
            if isinstance(query_vector[0], list):
                query_vector = query_vector[0]

            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Build query with filters
            if business_id and chatbot_id:
                if source_type_filter:
                    query = """
                        SELECT 
                            doc_id,
                            text,
                            metadata,
                            business_id,
                            chatbot_id,
                            1 - (embedding <=> %s::vector) AS similarity
                        FROM embeddings
                        WHERE business_id = %s 
                        AND chatbot_id = %s
                        AND metadata->>'source_type' = %s
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                    """
                    cursor.execute(
                        query,
                        (
                            query_vector,
                            business_id,
                            chatbot_id,
                            source_type_filter,
                            query_vector,
                            limit,
                        ),
                    )
                else:
                    query = """
                        SELECT 
                            doc_id,
                            text,
                            metadata,
                            business_id,
                            chatbot_id,
                            1 - (embedding <=> %s::vector) AS similarity
                        FROM embeddings
                        WHERE business_id = %s AND chatbot_id = %s
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                    """
                    cursor.execute(
                        query,
                        (query_vector, business_id, chatbot_id, query_vector, limit),
                    )
            else:
                query = """
                    SELECT 
                        doc_id,
                        text,
                        metadata,
                        business_id,
                        chatbot_id,
                        1 - (embedding <=> %s::vector) AS similarity
                    FROM embeddings
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """
                cursor.execute(query, (query_vector, query_vector, limit))

            rows = cursor.fetchall()
            cursor.close()
            conn.close()

            # Convert to SearchResult objects
            results = []
            for row in rows:
                payload = {
                    "doc_id": row["doc_id"],
                    "text": row["text"],
                    "data": row["metadata"].get("data", {}) if row["metadata"] else {},
                    "hash": row["metadata"].get("hash", "") if row["metadata"] else "",
                    "metadata": row["metadata"] if row["metadata"] else {},
                    "business_id": row["business_id"],
                    "chatbot_id": row["chatbot_id"],
                }

                result = SearchResult(
                    doc_id=row["doc_id"],
                    score=float(row["similarity"]),
                    payload=payload,
                )
                results.append(result)

            logger.info(f"Retrieved {len(results)} similar documents")
            return results

        except Exception as e:
            if conn:
                conn.close()
            logger.error(f"Error searching vector database: {str(e)}")
            return []

    # ========================================
    # DELETE OPERATIONS
    # ========================================

    def delete_document(self, doc_id: str) -> bool:
        """Delete a single document by doc_id"""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("DELETE FROM embeddings WHERE doc_id = %s", (doc_id,))
            deleted_count = cursor.rowcount

            conn.commit()
            cursor.close()
            conn.close()

            logger.info(f"Deleted {deleted_count} document(s) with doc_id={doc_id}")
            return deleted_count > 0

        except Exception as e:
            if conn:
                conn.rollback()
                conn.close()
            logger.error(f"Error deleting document: {str(e)}")
            return False

    def delete_by_pattern(self, doc_id_pattern: str) -> int:
        """
        Generic method to delete documents by doc_id pattern

        Args:
            doc_id_pattern: SQL LIKE pattern (use % as wildcard)

        Examples:
            "business123:chatbot456:%"  - Delete all for a chatbot
            "%:faq:%"                   - Delete all FAQ embeddings
            "business123:%:database:%"  - Delete all database embeddings

        Returns: Number of deleted documents
        """
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                "DELETE FROM embeddings WHERE doc_id LIKE %s", (doc_id_pattern,)
            )

            deleted_count = cursor.rowcount
            conn.commit()
            cursor.close()
            conn.close()

            logger.info(
                f"Deleted {deleted_count} documents matching pattern: {doc_id_pattern}"
            )
            return deleted_count

        except Exception as e:
            if conn:
                conn.rollback()
                conn.close()
            logger.error(f"Error deleting by pattern: {str(e)}")
            return 0

    def delete_faq_source(
        self, business_id: str, chatbot_id: str, faq_source_id: str
    ) -> int:
        """
        Delete all embeddings for a specific FAQ source
        Uses pattern matching on doc_id for efficiency

        Returns: Number of deleted documents
        """
        doc_id_pattern = f"{business_id}:{chatbot_id}:faq:{faq_source_id}:%"
        deleted_count = self.delete_by_pattern(doc_id_pattern)

        # Update collection stats after deletion
        try:
            self.update_collection_stats(business_id, chatbot_id)
        except Exception as e:
            logger.warning(f"Failed to update collection stats: {str(e)}")

        logger.info(
            f"Deleted {deleted_count} embeddings for FAQ source {faq_source_id}"
        )
        return deleted_count

    def delete_collection(self, business_id: str, chatbot_id: str) -> int:
        """Delete all documents for a business/chatbot"""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Delete embeddings
            cursor.execute(
                "DELETE FROM embeddings WHERE business_id = %s AND chatbot_id = %s",
                (business_id, chatbot_id),
            )
            deleted_count = cursor.rowcount

            # Delete collection metadata
            cursor.execute(
                "DELETE FROM collections WHERE business_id = %s AND chatbot_id = %s",
                (business_id, chatbot_id),
            )

            conn.commit()
            cursor.close()
            conn.close()

            logger.info(
                f"Deleted collection: {deleted_count} embeddings for {business_id}/{chatbot_id}"
            )
            return deleted_count

        except Exception as e:
            if conn:
                conn.rollback()
                conn.close()
            logger.error(f"Error deleting collection: {str(e)}")
            return 0

    # ========================================
    # STATS & UTILITIES
    # ========================================

    def get_collection_stats(self, business_id: str, chatbot_id: str) -> Dict[str, Any]:
        """Get statistics for a collection"""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            cursor.execute(
                """
                SELECT 
                    COUNT(*) as document_count,
                    AVG(LENGTH(text)) as avg_text_length,
                    COUNT(*) FILTER (WHERE metadata->>'source_type' = 'faq') as faq_count,
                    COUNT(*) FILTER (WHERE metadata->>'source_type' = 'database') as db_count
                FROM embeddings
                WHERE business_id = %s AND chatbot_id = %s
            """,
                (business_id, chatbot_id),
            )

            stats = cursor.fetchone()
            cursor.close()
            conn.close()

            return dict(stats) if stats else {"document_count": 0, "avg_text_length": 0}

        except Exception as e:
            if conn:
                conn.close()
            logger.error(f"Error getting collection stats: {str(e)}")
            return {"document_count": 0, "avg_text_length": 0}
