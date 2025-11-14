"""
Test script to verify vector database setup
Run this after creating tables

Usage: python -m app.test.test_vector_setup
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.services.vectorstore import VectorStoreService
from app.services.embeddings import EmbeddingServiceHuggingFace
from app.core.logging_config import logger


def test_connection():
    """Test 1: Basic connection"""
    print("\n" + "=" * 50)
    print("TEST 1: Database Connection")
    print("=" * 50)

    try:
        vectorstore = VectorStoreService()
        print("✅ Connected to PostgreSQL successfully")
        return True
    except Exception as e:
        print(f"❌ Connection failed: {str(e)}")
        return False


def test_collection_creation():
    """Test 2: Collection creation"""
    print("\n" + "=" * 50)
    print("TEST 2: Collection Creation")
    print("=" * 50)

    try:
        vectorstore = VectorStoreService()

        # Create test collection
        business_id = "test_business_001"
        chatbot_id = "test_chatbot_001"

        collection = vectorstore.ensure_collection_exists(business_id, chatbot_id)
        print(f"✅ Collection created: {collection['collection_name']}")

        # Get collection info
        info = vectorstore.get_collection_info(business_id, chatbot_id)
        print(f"   Document count: {info['document_count']}")
        print(f"   Created at: {info['created_at']}")

        return True
    except Exception as e:
        print(f"❌ Collection creation failed: {str(e)}")
        return False


def test_embedding_and_search():
    """Test 3: Embed documents and search"""
    print("\n" + "=" * 50)
    print("TEST 3: Embedding & Search")
    print("=" * 50)

    try:
        vectorstore = VectorStoreService()
        embedding_service = EmbeddingServiceHuggingFace()

        business_id = "test_business_001"
        chatbot_id = "test_chatbot_001"
        collection_name = vectorstore.get_collection_name(business_id, chatbot_id)

        # Test documents
        test_docs = [
            {
                "doc_id": f"{business_id}:{chatbot_id}:faq:test_001:0",
                "text": "What are your business hours? We are open Monday to Friday, 9 AM to 5 PM.",
                "metadata": {
                    "source_type": "faq",
                    "faq_source_id": "test_001",
                    "type": "qa_pair",
                    "question": "What are your business hours?",
                },
            },
            {
                "doc_id": f"{business_id}:{chatbot_id}:faq:test_001:1",
                "text": "How do I reset my password? Click on 'Forgot Password' on the login page.",
                "metadata": {
                    "source_type": "faq",
                    "faq_source_id": "test_001",
                    "type": "qa_pair",
                    "question": "How do I reset my password?",
                },
            },
            {
                "doc_id": f"{business_id}:{chatbot_id}:database:test_table:0",
                "text": "Customer: John Doe, Email: john@example.com, Status: Active",
                "metadata": {
                    "source_type": "database",
                    "table": "customers",
                },
            },
        ]

        # Generate embeddings
        texts = [doc["text"] for doc in test_docs]
        print(f"Generating embeddings for {len(texts)} documents...")
        embeddings = embedding_service.generate_embeddings(texts)
        print(f"✅ Generated {len(embeddings)} embeddings")

        # Upsert documents
        print("Storing embeddings in database...")
        count = vectorstore.upsert_documents(
            collection_name=collection_name,
            documents=test_docs,
            embeddings=embeddings,
            business_id=business_id,
            chatbot_id=chatbot_id,
        )
        print(f"✅ Stored {count} documents")

        # Test search
        print("\nTesting search...")
        query = "What time do you open?"
        query_embedding = embedding_service.generate_embeddings([query])

        results = vectorstore.search_similar(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=3,
            business_id=business_id,
            chatbot_id=chatbot_id,
            source_type_filter="faq",  # Only search FAQs
        )

        print(f"✅ Found {len(results)} results for query: '{query}'")
        for i, result in enumerate(results, 1):
            print(f"\n   Result {i}:")
            print(f"   Score: {result.score:.4f}")
            print(f"   Text: {result.payload['text'][:80]}...")
            print(f"   Source: {result.payload['metadata'].get('source_type')}")

        return True

    except Exception as e:
        print(f"❌ Embedding/Search test failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def test_deletion():
    """Test 4: Delete operations"""
    print("\n" + "=" * 50)
    print("TEST 4: Deletion")
    print("=" * 50)

    try:
        vectorstore = VectorStoreService()

        business_id = "test_business_001"
        chatbot_id = "test_chatbot_001"

        # Check current count
        stats = vectorstore.get_collection_stats(business_id, chatbot_id)
        print(f"Documents before deletion: {stats['document_count']}")

        # Delete FAQ source
        deleted = vectorstore.delete_faq_source(
            business_id=business_id, chatbot_id=chatbot_id, faq_source_id="test_001"
        )
        print(f"✅ Deleted {deleted} FAQ documents")

        # Check count after deletion
        stats = vectorstore.get_collection_stats(business_id, chatbot_id)
        print(f"Documents after deletion: {stats['document_count']}")

        # Delete entire collection
        deleted_total = vectorstore.delete_collection(business_id, chatbot_id)
        print(f"✅ Deleted entire collection: {deleted_total} documents")

        return True

    except Exception as e:
        print(f"❌ Deletion test failed: {str(e)}")
        return False


def test_collections_list():
    """Test 5: List collections"""
    print("\n" + "=" * 50)
    print("TEST 5: List Collections")
    print("=" * 50)

    try:
        vectorstore = VectorStoreService()

        collections = vectorstore.list_all_collections()
        print(f"✅ Found {len(collections)} collections:")

        for col in collections:
            print(f"\n   Collection: {col['collection_name']}")
            print(f"   Business: {col['business_id']}")
            print(
                f"   Documents: {col['tracked_count']} (tracked) / {col['actual_count']} (actual)"
            )
            print(f"   Status: {col['sync_status']}")

        return True

    except Exception as e:
        print(f"❌ List collections failed: {str(e)}")
        return False


def main():
    print("\n" + "=" * 70)
    print("VECTOR DATABASE SETUP TEST SUITE")
    print("=" * 70)

    tests = [
        ("Connection", test_connection),
        ("Collection Creation", test_collection_creation),
        ("Embedding & Search", test_embedding_and_search),
        ("Deletion", test_deletion),
        ("List Collections", test_collections_list),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {str(e)}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Vector database is ready.")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")


if __name__ == "__main__":
    main()
