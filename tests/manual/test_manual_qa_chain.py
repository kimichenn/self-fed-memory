"""
Manual Verification Tests for the End-to-End QA Chain

Purpose:
- To provide a structured and repeatable process for manually evaluating the quality
  of the entire Question-Answering (QA) system using REAL personal notes.
- To test the system with REAL APIs (OpenAI, Pinecone) and REAL data to ensure that the
  integration works as expected in a production-like environment.

When to Run:
- After making significant changes to the memory, retrieval, or QA chain logic.
- Periodically, to catch regressions in response quality or system behavior.
- When you need to assess the real-world impact of prompt engineering,
  model updates, or changes to the retrieval strategy.

Usage:
  pytest tests/test_manual_qa_chain.py -v -s

Prerequisites:
- You must have a .env file with valid OPENAI_API_KEY and PINECONE_API_KEY.
- You must have actual notes in the personal_notes/ directory.
- It is highly recommended to use a separate test index in Pinecone.
"""

from datetime import datetime
from datetime import timedelta
import os
from pathlib import Path
import time

from langchain_openai import ChatOpenAI
import pytest

from app.core.chains.qa_chain import IntegratedQAChain
from app.core.embeddings import get_embeddings
from app.core.memory import MemoryManager
from app.core.vector_store.pinecone import PineconeVectorStore
from app.ingestion.markdown_loader import parse_markdown_file
from tests.helpers import get_test_settings
from app.core.config import Settings

# Marker for tests that require real API keys. Skips them if keys are not found.
requires_real_apis = pytest.mark.skipif(
    not (os.getenv("OPENAI_API_KEY") and os.getenv("PINECONE_API_KEY")),
    reason="Requires OPENAI_API_KEY and PINECONE_API_KEY for manual verification",
)


def print_header(title: str):
    """Prints a standardized header for test sections."""
    print("\n" + "=" * 80)
    print(f"MANUAL VERIFICATION: {title}")
    print("=" * 80)


def print_manual_check(prompts: list[str]):
    """Prints a standardized set of manual verification prompts."""
    print("\n🔍 MANUAL CHECK:")
    for prompt in prompts:
        print(f"   - {prompt}")


def load_actual_notes():
    """Load actual notes from the personal_notes directory."""
    personal_notes_dir = Path("personal_notes")
    if not personal_notes_dir.exists() or not personal_notes_dir.is_dir():
        return []

    markdown_files = list(personal_notes_dir.glob("*.md"))
    if not markdown_files:
        return []

    all_chunks = []
    for md_file in markdown_files:
        try:
            chunks = parse_markdown_file(md_file)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"Warning: Could not parse {md_file}: {e}")

    return all_chunks


@pytest.fixture(scope="module")
def real_qa_setup(monkeypatch_module):
    """
    Module-level fixture to set up a real QA chain for manual testing.
    - Connects to the actual Pinecone and OpenAI APIs.
    - Cleans up the test namespace before running.
    - Ingests actual notes from the personal_notes directory.
    - Yields the QA chain and other components for testing.
    - Cleans up the test data after all tests in the module are complete.
    """
    print_header("Setting Up Real QA Environment with Actual Notes")

    # Load actual notes first to check if we can proceed
    real_memories = load_actual_notes()
    if not real_memories:
        pytest.skip(
            "No actual notes found in personal_notes/ directory. Cannot run tests with real data."
        )

    print(f"Loaded {len(real_memories)} chunks from actual notes")

    # Show sample of what we're working with
    print("\nSample of actual note content:")
    for i, chunk in enumerate(real_memories[:3]):  # Show first 3 chunks
        content_preview = (
            chunk["content"][:80] + "..."
            if len(chunk["content"]) > 80
            else chunk["content"]
        )
        source_name = Path(chunk.get("source", "unknown")).name
        print(f"  {i+1}. [{source_name}] {content_preview}")

    # 1. Configure test settings
    test_settings = Settings.for_testing()
    monkeypatch_module.setenv("TEST_PINECONE_NAMESPACE", "test-manual-qa-real-notes")
    test_settings.pinecone_namespace = (
        "test-manual-qa-real-notes"  # Override for this test
    )

    get_embeddings.cache_clear()

    # 2. Initialize real components
    embeddings = get_embeddings()
    vector_store = PineconeVectorStore(embeddings)
    memory_manager = MemoryManager(embeddings=embeddings, use_time_weighting=True)
    memory_manager.store = vector_store
    llm = ChatOpenAI(model="gpt-4o-2024-11-20", temperature=0.3)
    qa_chain = IntegratedQAChain(
        memory_manager=memory_manager, llm=llm, k=5, name="TestUser"
    )

    # 3. Clean up the test namespace before the test run
    print(f"Cleaning up namespace: '{vector_store.cfg.pinecone_namespace}'...")
    try:
        vector_store.index.delete(
            delete_all=True, namespace=vector_store.cfg.pinecone_namespace
        )
        # Pinecone deletion can take a moment
        time.sleep(5)
    except Exception as e:
        print(f"Note: Could not clear test namespace before test run: {e}")

    # 4. Ingest actual notes
    memory_manager.add_chunks(real_memories)
    print(f"Ingested {len(real_memories)} chunks from actual notes...")

    # 5. Wait for Pinecone to index the vectors
    for _ in range(15):
        stats = vector_store.index.describe_index_stats()
        count = (
            stats.get("namespaces", {})
            .get(vector_store.cfg.pinecone_namespace, {})
            .get("vector_count", 0)
        )
        if count >= len(real_memories):
            print(f"✓ Pinecone index is ready with {count} vectors.")
            break
        time.sleep(2)
    else:
        pytest.fail("Pinecone indexing timeout. The test cannot proceed.")

    yield qa_chain, real_memories

    # 6. Teardown: Clean up test data after all tests are done
    print_header("Tearing Down QA Environment")
    try:
        vector_store.delete([m["id"] for m in real_memories])
        print("✓ Cleaned up test data from Pinecone.")
    except Exception as e:
        print(f"Warning: Cleanup failed after test run: {e}")


@pytest.mark.manual
@requires_real_apis
def test_time_weighted_retrieval(real_qa_setup):
    """
    Verify that the QA chain correctly prioritizes more recent memories from actual notes.
    """
    print_header("Time-Weighted Retrieval with Real Notes")
    qa_chain, real_memories = real_qa_setup

    # Analyze the actual notes to suggest a good query
    sources = [Path(m.get("source", "")).name for m in real_memories]
    unique_sources = list(set(sources))

    query = "What are the most recent things I've written about or worked on?"
    print(f"Query: {query}")
    print(
        f"(Testing against {len(real_memories)} chunks from files: {', '.join(unique_sources[:5])}{'...' if len(unique_sources) > 5 else ''})"
    )

    result = qa_chain.invoke({"question": query})

    print(f"\nAnswer: {result['answer']}")
    print("\nRetrieved memories (ranked by relevance + recency):")
    for doc in result["source_documents"]:
        source_name = Path(doc.get("source", "unknown")).name
        created_date = doc.get("created_at", "unknown")[:10]  # Just the date part
        content_preview = (
            doc.get("content", "")[:60] + "..."
            if len(doc.get("content", "")) > 60
            else doc.get("content", "")
        )
        print(f"  - [{source_name}] {created_date}: {content_preview}")

    print_manual_check(
        [
            "Does the answer reflect recent content from your actual notes?",
            "Are more recent notes ranked higher in the retrieved memories?",
            "Do you recognize the content as being from your actual notes?",
        ]
    )


@pytest.mark.manual
@requires_real_apis
def test_content_based_retrieval(real_qa_setup):
    """
    Verify that the QA chain can find relevant content from actual notes.
    """
    print_header("Content-Based Retrieval from Real Notes")
    qa_chain, real_memories = real_qa_setup

    # Try to make a query that should match something in the actual notes
    test_queries = [
        "What topics or subjects have I been thinking about?",
        "What projects or work have I been involved in?",
        "What personal activities or experiences have I recorded?",
    ]

    for query in test_queries:
        print(f"\n--- Query: {query} ---")

        result = qa_chain.invoke({"question": query})
        print(f"Answer: {result['answer']}")
        print("Retrieved memories:")
        for doc in result["source_documents"]:
            source_name = Path(doc.get("source", "unknown")).name
            content_preview = (
                doc.get("content", "")[:80] + "..."
                if len(doc.get("content", "")) > 80
                else doc.get("content", "")
            )
            print(f"  - [{source_name}] {content_preview}")

    print_manual_check(
        [
            "Do the answers accurately reflect the content of your actual notes?",
            "Are the retrieved memories relevant to each query?",
            "Does the system successfully find different types of content for different queries?",
        ]
    )


@pytest.mark.manual
@requires_real_apis
def test_edge_case_queries(real_qa_setup):
    """
    Verify the system's behavior with queries that may not match actual note content.
    """
    print_header("Edge Cases and Potentially Irrelevant Queries")
    qa_chain, real_memories = real_qa_setup

    edge_cases = {
        "asdfghjkl": "Nonsensical query",
        "What is the capital of Mars?": "Query unlikely to be in personal notes",
        "Tell me about quantum physics": "Academic query that may not be in notes",
    }

    for query, description in edge_cases.items():
        print(f"\n--- {description}: '{query}' ---")
        result = qa_chain.invoke({"question": query})
        print(f"Answer: {result['answer']}")
        print(f"Retrieved {len(result['source_documents'])} memories")

        if result["source_documents"]:
            print("Top retrieved memory:")
            doc = result["source_documents"][0]
            source_name = Path(doc.get("source", "unknown")).name
            content_preview = (
                doc.get("content", "")[:100] + "..."
                if len(doc.get("content", "")) > 100
                else doc.get("content", "")
            )
            print(f"  [{source_name}] {content_preview}")

    print_manual_check(
        [
            "Does the system provide sensible responses when queries don't match note content?",
            "Does it avoid hallucinating information not in your notes?",
            "Are any retrieved memories actually relevant, or appropriately irrelevant?",
        ]
    )


@pytest.mark.manual
@requires_real_apis
def test_response_quality_and_synthesis(real_qa_setup):
    """
    Verify the overall quality and synthesis of responses using actual notes.
    """
    print_header("Response Quality and Synthesis with Real Notes")
    qa_chain, real_memories = real_qa_setup

    query = "Give me a summary of the main themes, topics, or activities reflected in my notes."
    print(f"\nQuery: {query}")

    start_time = time.time()
    result = qa_chain.invoke({"question": query})
    duration = time.time() - start_time

    print(f"\nAnswer (generated in {duration:.2f}s): {result['answer']}")
    print(
        f"Synthesized from {len(result['source_documents'])} memories from your actual notes."
    )

    print("\nSources used in synthesis:")
    source_files = set()
    for doc in result["source_documents"]:
        source_name = Path(doc.get("source", "unknown")).name
        source_files.add(source_name)
    print(f"  Files: {', '.join(sorted(source_files))}")

    print_manual_check(
        [
            "Does the summary accurately reflect the themes and topics in your actual notes?",
            "Is the answer a coherent synthesis rather than just listing individual notes?",
            "Do you recognize the content as authentically from your personal notes?",
            "Is the response time acceptable for real-world usage?",
            "Would this summary be helpful if you were trying to remember what you've been working on or thinking about?",
        ]
    )
