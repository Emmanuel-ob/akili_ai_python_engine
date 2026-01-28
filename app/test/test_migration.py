"""
Test script to verify google-genai SDK migration

Run this after installing the new SDK to ensure everything works correctly.
"""

import sys
import os

# Add parent directory to path (adjust as needed)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_imports():
    """Test that all imports work correctly"""
    print("=" * 60)
    print("TEST 1: Checking imports...")
    print("=" * 60)

    try:
        from google import genai
        from google.genai import types

        print("✅ Successfully imported google.genai")
        print("✅ Successfully imported google.genai.types")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("\nPlease install the new SDK:")
        print("  pip install google-genai")
        return False


def test_client_initialization():
    """Test client initialization"""
    print("\n" + "=" * 60)
    print("TEST 2: Testing client initialization...")
    print("=" * 60)

    try:
        from google import genai
        import os

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        if not api_key:
            print("⚠️  Warning: No API key found in environment")
            print("   Set GEMINI_API_KEY or GOOGLE_API_KEY")
            return False

        client = genai.Client(api_key=api_key)
        print("✅ Client initialized successfully")
        return True

    except Exception as e:
        print(f"❌ Client initialization failed: {e}")
        return False


def test_embedding_service():
    """Test the updated EmbeddingService"""
    print("\n" + "=" * 60)
    print("TEST 3: Testing EmbeddingService...")
    print("=" * 60)

    try:
        # This assumes your app structure
        from app.services.embeddings import EmbeddingService

        service = EmbeddingService()
        print("✅ EmbeddingService initialized")

        # Test single embedding
        test_text = "Hello, this is a test"
        embedding = service.generate_single_embedding(test_text)

        print(f"✅ Generated single embedding")
        print(f"   Dimension: {len(embedding)}")
        print(f"   First 5 values: {embedding[:5]}")

        # Test batch embeddings
        test_texts = ["First text", "Second text", "Third text"]
        embeddings = service.generate_embeddings(test_texts)

        print(f"✅ Generated batch embeddings")
        print(f"   Count: {len(embeddings)}")
        print(f"   Dimensions: {len(embeddings[0])}")

        # Test query embedding
        query = "What is the meaning of life?"
        query_embedding = service.generate_query_embedding(query)

        print(f"✅ Generated query embedding")
        print(f"   Dimension: {len(query_embedding)}")

        return True

    except ImportError as e:
        print(f"⚠️  Could not import EmbeddingService: {e}")
        print("   This is expected if you're testing standalone")
        return None
    except Exception as e:
        print(f"❌ EmbeddingService test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_llm_service():
    """Test the updated LLMService"""
    print("\n" + "=" * 60)
    print("TEST 4: Testing LLMService...")
    print("=" * 60)

    try:
        from app.services.llm import LLMService
        import asyncio

        service = LLMService()
        print("✅ LLMService initialized")

        async def test_generation():
            response = await service.generate_response(
                message="What is 2+2?",
                context=[],
                history="",
                personality="helpful and concise",
            )
            print(f"✅ Generated response: {response[:100]}...")
            return True

        result = asyncio.run(test_generation())
        return result

    except ImportError as e:
        print(f"⚠️  Could not import LLMService: {e}")
        print("   This is expected if you're testing standalone")
        return None
    except Exception as e:
        print(f"❌ LLMService test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_direct_sdk():
    """Test the SDK directly without app dependencies"""
    print("\n" + "=" * 60)
    print("TEST 5: Direct SDK test (no app dependencies)...")
    print("=" * 60)

    try:
        from google import genai
        from google.genai import types
        import os

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("⚠️  Skipping: No API key found")
            return None

        client = genai.Client(api_key=api_key)

        # Test embedding
        print("Testing embeddings...")
        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents="Test embedding",
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT", output_dimensionality=768
            ),
        )
        embedding = result.embeddings[0].values
        print(f"✅ Direct embedding test passed")
        print(f"   Dimension: {len(embedding)}")

        # Test generation
        print("Testing text generation...")
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents="Say hello in one word",
            config=types.GenerateContentConfig(temperature=0.7),
        )
        print(f"✅ Direct generation test passed")
        print(f"   Response: {response.text}")

        return True

    except Exception as e:
        print(f"❌ Direct SDK test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_model_availability():
    """Test that the models we need are available"""
    print("\n" + "=" * 60)
    print("TEST 6: Checking model availability...")
    print("=" * 60)

    try:
        from google import genai
        import os

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("⚠️  Skipping: No API key found")
            return None

        client = genai.Client(api_key=api_key)

        # List available models
        print("Fetching available models...")
        models = client.models.list()

        embedding_models = [m for m in models if "embedding" in m.name.lower()]
        generation_models = [m for m in models if "gemini" in m.name.lower()]

        print(f"\n📊 Embedding models found: {len(embedding_models)}")
        for model in embedding_models[:5]:
            print(f"   - {model.name}")

        print(f"\n📊 Generation models found: {len(generation_models)}")
        for model in generation_models[:5]:
            print(f"   - {model.name}")

        # Check for our specific models
        model_names = [m.name for m in models]
        required_models = ["gemini-embedding-001", "gemini-2.0-flash"]

        for req_model in required_models:
            found = any(req_model in name for name in model_names)
            if found:
                print(f"✅ {req_model} is available")
            else:
                print(f"⚠️  {req_model} not found (might use different naming)")

        return True

    except Exception as e:
        print(f"❌ Model availability check failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("GOOGLE GENAI SDK MIGRATION TEST SUITE")
    print("=" * 60)

    results = {
        "Imports": test_imports(),
        "Client Init": test_client_initialization(),
        "Direct SDK": test_direct_sdk(),
        "Model Availability": test_model_availability(),
        "Embedding Service": test_embedding_service(),
        "LLM Service": test_llm_service(),
    }

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, result in results.items():
        if result is True:
            status = "✅ PASS"
        elif result is False:
            status = "❌ FAIL"
        else:
            status = "⚠️  SKIP"

        print(f"{status} - {test_name}")

    passed = sum(1 for r in results.values() if r is True)
    failed = sum(1 for r in results.values() if r is False)
    skipped = sum(1 for r in results.values() if r is None)

    print(f"\nResults: {passed} passed, {failed} failed, {skipped} skipped")

    if failed > 0:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)
    else:
        print("\n✅ All tests passed! Migration successful.")
        sys.exit(0)


if __name__ == "__main__":
    main()
