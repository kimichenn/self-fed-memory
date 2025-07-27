"""End-to-end integration tests for the complete ingestion pipeline.

This module tests the full workflow from markdown files in personal_notes/
through parsing, embedding, and storage in Pinecone, then retrieval.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from app.core.embeddings import get_embeddings
from app.core.memory import MemoryManager
from app.ingestion.markdown_loader import parse_markdown_file
from tests.test_config import get_test_settings

# Marker for tests that require Pinecone API credentials
requires_pinecone = pytest.mark.skipif(
    not os.getenv("PINECONE_API_KEY"), reason="requires PINECONE_API_KEY"
)


@requires_pinecone
def test_end_to_end_personal_notes_ingestion(monkeypatch):
    """Test complete pipeline: personal_notes/ → parsing → embedding → Pinecone → search."""
    # 1. Setup test environment with test Pinecone index
    test_settings = get_test_settings()
    monkeypatch.setenv("PINECONE_INDEX", test_settings.pinecone_index)
    monkeypatch.setenv("EMBEDDING_MODEL", test_settings.embedding_model)

    # Clear caches to pick up new environment
    from app.core.embeddings import get_embeddings

    get_embeddings.cache_clear()

    # 2. Initialize components
    embeddings = get_embeddings()
    manager = MemoryManager(embeddings)

    # 3. Clear the test index before starting (but preserve data after test completion)
    print("Clearing test index before starting end-to-end test...")
    try:
        # Get all vector IDs and delete them to start fresh
        stats = manager.store.index.describe_index_stats()
        if stats["total_vector_count"] > 0:
            # Unfortunately, Pinecone doesn't have a "delete all" in namespace easily
            # So we'll clear by fetching all IDs and deleting them
            # This is a more aggressive approach for the end-to-end test
            manager.store.index.delete(
                delete_all=True, namespace=manager.store.cfg.pinecone_namespace
            )
            time.sleep(5)  # Wait for deletion to propagate
            print(f"Cleared {stats['total_vector_count']} vectors from test index")
    except Exception as e:
        print(f"Note: Could not clear index (may be empty): {e}")

    # 4. Get some test files from personal_notes (limit to avoid overwhelming the test)
    personal_notes_dir = Path("personal_notes")
    markdown_files = list(personal_notes_dir.glob("*.md"))[:3]  # Just test with 3 files

    if not markdown_files:
        pytest.skip("No markdown files found in personal_notes/ directory")

        # 5. Store original file info for verification
    original_files = []
    all_test_chunk_ids = []

    # 6. Ingest each file through the complete pipeline
    for md_file in markdown_files:
        # Parse the markdown file
        chunks = parse_markdown_file(md_file)
        original_files.append(
            {"file": md_file, "chunks": chunks, "chunk_count": len(chunks)}
        )

        # Track chunk IDs for cleanup
        all_test_chunk_ids.extend([chunk["id"] for chunk in chunks])

        # Add chunks through MemoryManager (complete pipeline)
        manager.add_chunks(chunks)

    # 7. Wait for Pinecone to be eventually consistent
    total_expected_chunks = sum(f["chunk_count"] for f in original_files)
    for _ in range(15):  # Poll for up to 30 seconds
        stats = manager.store.index.describe_index_stats()
        if stats["total_vector_count"] >= total_expected_chunks:
            break
        time.sleep(2)
    else:
        pytest.fail(
            f"Index did not update in time. Expected {total_expected_chunks} chunks."
        )

    # 8. Test semantic search with content from the files
    # Try searching for concepts that should be in the personal notes
    search_queries = [
        "career goals and professional development",
        "relationships and friendships",
        "personal growth and self improvement",
    ]

    search_results_found = False
    for query in search_queries:
        results = manager.search(query, k=3)
        if results:
            search_results_found = True
            # Verify results structure
            for result in results:
                assert "content" in result
                assert "source" in result
                assert "created_at" in result
                assert "id" in result
                # Verify source is one of our test files
                source_path = Path(result["source"])
                assert source_path.name in [f.name for f in markdown_files]
            break

    assert (
        search_results_found
    ), "No search results found for any queries - pipeline may be broken"

    # 9. Test that we can find specific content
    # Pick the first file and search for some of its content
    first_file = original_files[0]
    first_chunk_content = first_file["chunks"][0]["content"]
    # Take a snippet from the middle of the content for searching
    search_snippet = " ".join(first_chunk_content.split()[:10])

    specific_results = manager.search(search_snippet, k=5)
    assert len(specific_results) > 0, "Could not find content from ingested file"

    # Verify at least one result is from the expected file
    found_expected_file = any(
        Path(result["source"]).name == first_file["file"].name
        for result in specific_results
    )
    assert (
        found_expected_file
    ), "Search did not return results from the expected source file"

    # 10. END-TO-END TEST: Do NOT delete data after completion for manual inspection
    print(f"✅ End-to-end test completed successfully!")
    print(
        f"📁 Ingested {total_expected_chunks} chunks from {len(markdown_files)} files"
    )
    print(f"🔍 Data is preserved in test index for manual inspection")
    print(f"📝 Files processed: {[f.name for f in markdown_files]}")
    # Note: Data is intentionally left in the test index for inspection


@requires_pinecone
def test_ingest_folder_script_functionality(monkeypatch):
    """Test the core functionality that ingest_folder.py script provides."""
    # This test verifies the same logic as the script but in a controlled way

    # 1. Setup test environment
    test_settings = get_test_settings()
    monkeypatch.setenv("PINECONE_INDEX", test_settings.pinecone_index)
    monkeypatch.setenv("EMBEDDING_MODEL", test_settings.embedding_model)

    from app.core.embeddings import get_embeddings

    get_embeddings.cache_clear()

    # 2. Replicate the script logic
    embeddings = get_embeddings()
    manager = MemoryManager(embeddings)

    personal_notes_dir = Path("personal_notes")

    # 3. Process files like the script does (recursive glob)
    markdown_files = list(personal_notes_dir.rglob("*.md"))
    if not markdown_files:
        pytest.skip("No markdown files found in personal_notes/")

    # Limit to 2 files for testing
    test_files = markdown_files[:2]
    processed_chunks = []
    all_chunk_ids = []

    # 4. Process each file (mimicking the script's main loop)
    for md_file in test_files:
        chunks = parse_markdown_file(md_file)
        processed_chunks.extend(chunks)
        all_chunk_ids.extend([chunk["id"] for chunk in chunks])
        manager.add_chunks(chunks)

    # 5. Verify the processing worked
    assert len(processed_chunks) > 0, "No chunks were processed from markdown files"

    # 6. Wait for index update
    for _ in range(10):
        stats = manager.store.index.describe_index_stats()
        if stats["total_vector_count"] >= len(processed_chunks):
            break
        time.sleep(2)
    else:
        pytest.fail("Index did not update after ingestion")

    # 7. Verify we can retrieve the data
    results = manager.search("personal notes content", k=3)
    assert len(results) > 0, "Could not retrieve any ingested content"

    # 8. Cleanup (mini test - should clean up after itself)
    try:
        manager.store.delete(ids=all_chunk_ids)
        time.sleep(3)
        print(
            f"✅ Script functionality test completed and cleaned up {len(all_chunk_ids)} chunks"
        )
    except Exception as e:
        print(f"Warning: Could not clean up test data: {e}")


def test_dry_run_functionality():
    """Test the dry-run mode (parsing without Pinecone upload)."""
    # This tests the parsing logic without requiring Pinecone credentials

    personal_notes_dir = Path("personal_notes")
    markdown_files = list(personal_notes_dir.glob("*.md"))

    if not markdown_files:
        pytest.skip("No markdown files found in personal_notes/")

    # Test with first file
    test_file = markdown_files[0]

    # Parse the file (this is what dry-run mode does)
    chunks = parse_markdown_file(test_file)

    # Verify parsing worked correctly
    assert len(chunks) > 0, f"No chunks parsed from {test_file}"

    for chunk in chunks:
        # Verify chunk structure matches expected format
        assert "id" in chunk
        assert "content" in chunk
        assert "source" in chunk
        assert "created_at" in chunk
        assert len(chunk["content"].strip()) > 0, "Chunk content is empty"
        assert chunk["source"] == str(test_file)

    print(
        f"✓ Dry-run test: Successfully parsed {len(chunks)} chunks from {test_file.name}"
    )
