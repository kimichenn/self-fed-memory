from datetime import datetime
import os
import time

import pytest

from app.core.embeddings import get_embeddings
from app.core.memory import MemoryManager
from tests.test_config import get_test_settings

# Marker for tests that require Pinecone API credentials
requires_pinecone = pytest.mark.skipif(
    not os.getenv("PINECONE_API_KEY"), reason="requires PINECONE_API_KEY"
)


@requires_pinecone
def test_pinecone_integration(monkeypatch):
    """Full round-trip test for Pinecone: upsert, search, delete."""
    # 1. Get test configuration and set environment variables accordingly
    test_settings = get_test_settings()

    # Override environment variables to use test settings
    monkeypatch.setenv("PINECONE_INDEX", test_settings.pinecone_index)
    monkeypatch.setenv("EMBEDDING_MODEL", test_settings.embedding_model)

    # Clear any cached settings/embeddings to pick up the new env vars
    from app.core.embeddings import get_embeddings

    get_embeddings.cache_clear()

    # 2. Initialize the manager (which will use the test configuration)
    embeddings = get_embeddings()
    manager = MemoryManager(embeddings)

    # 3. Prepare test data
    test_chunks = [
        {
            "id": "v1",
            "content": "This is a test vector.",
            "source": "test.md",
            "created_at": datetime.utcnow().isoformat(),
        },
        {
            "id": "v2",
            "content": "Another test vector for similarity.",
            "source": "test.md",
            "created_at": datetime.utcnow().isoformat(),
        },
    ]

    # 4. Upsert data
    manager.add_chunks(test_chunks)

    # Poll until the index is ready
    for _ in range(10):  # Poll for up to 20 seconds
        stats = manager.store.index.describe_index_stats()
        if stats["total_vector_count"] >= len(test_chunks):
            break
        time.sleep(2)
    else:
        pytest.fail("Index did not update in time.")

    # 5. Search for a vector
    results = manager.search("A vector for testing similarity.", k=1)
    assert len(results) == 1
    assert results[0]["id"] == "v2"

    # 6. Clean up (mini test - should clean up after itself)
    manager.store.delete(ids=["v1", "v2"])
    time.sleep(5)
    print("✅ Vector store integration test completed and cleaned up test vectors")
