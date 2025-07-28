"""
Manual Verification Tests for the End-to-End QA Chain

Purpose:
- To provide a structured and repeatable process for manually evaluating the quality
  of the entire Question-Answering (QA) system.
- To test the system with REAL APIs (OpenAI, Pinecone) to ensure that the
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
- It is highly recommended to use a separate test index in Pinecone.
"""

from datetime import datetime
from datetime import timedelta
import os
import time

from langchain_openai import ChatOpenAI
import pytest

from app.core.chains.qa_chain import IntegratedQAChain
from app.core.embeddings import get_embeddings
from app.core.memory import MemoryManager
from app.core.vector_store.pinecone import PineconeVectorStore
from tests.helpers import get_test_settings

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


@pytest.fixture(scope="module")
def real_qa_setup(monkeypatch_module):
    """
    Module-level fixture to set up a real QA chain for manual testing.
    - Connects to the actual Pinecone and OpenAI APIs.
    - Cleans up the test namespace before running.
    - Ingests a standard set of diverse test memories.
    - Yields the QA chain and other components for testing.
    - Cleans up the test data after all tests in the module are complete.
    """
    print_header("Setting Up Real QA Environment")

    # 1. Configure test settings
    # Set test-specific environment variables *before* loading the settings
    monkeypatch_module.setenv("TEST_PINECONE_INDEX", "self-memory-test-manual")
    monkeypatch_module.setenv("TEST_PINECONE_NAMESPACE", "test-manual-qa-chain")
    test_settings = get_test_settings()

    # Set the final environment variables that the application will use
    monkeypatch_module.setenv("PINECONE_INDEX", test_settings.pinecone_index)
    monkeypatch_module.setenv("EMBEDDING_MODEL", test_settings.embedding_model)
    get_embeddings.cache_clear()

    # 2. Initialize real components
    embeddings = get_embeddings()
    vector_store = PineconeVectorStore(embeddings)
    memory_manager = MemoryManager(embeddings=embeddings, use_time_weighting=True)
    memory_manager.store = vector_store
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
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

    # 4. Ingest diverse test memories
    test_memories = _create_test_memories()
    memory_manager.add_chunks(test_memories)
    print(f"Ingested {len(test_memories)} test memories...")

    # 5. Wait for Pinecone to index the vectors
    for _ in range(15):
        stats = vector_store.index.describe_index_stats()
        count = (
            stats.get("namespaces", {})
            .get(vector_store.cfg.pinecone_namespace, {})
            .get("vector_count", 0)
        )
        if count >= len(test_memories):
            print(f"✓ Pinecone index is ready with {count} vectors.")
            break
        time.sleep(2)
    else:
        pytest.fail("Pinecone indexing timeout. The test cannot proceed.")

    yield qa_chain

    # 6. Teardown: Clean up test data after all tests are done
    print_header("Tearing Down QA Environment")
    try:
        vector_store.delete([m["id"] for m in test_memories])
        print("✓ Cleaned up test data from Pinecone.")
    except Exception as e:
        print(f"Warning: Cleanup failed after test run: {e}")


def _create_test_memories():
    """Helper function to create a diverse set of test memories."""
    now = datetime.utcnow()
    return [
        {
            "id": "work_recent_1",
            "content": "Completed the React dashboard refactor today. Improved loading performance by 40%.",
            "created_at": now.isoformat(),
            "source": "work_notes.md",
            "tags": ["work", "react", "performance"],
        },
        {
            "id": "work_old",
            "content": "Started the React dashboard project 3 months ago. Initial planning discussions.",
            "created_at": (now - timedelta(days=90)).isoformat(),
            "source": "project_history.md",
            "tags": ["work", "react", "planning"],
        },
        {
            "id": "personal_recent",
            "content": "Went hiking at Mount Tamalpais with Alex and Emma. Beautiful weather!",
            "created_at": (now - timedelta(days=2)).isoformat(),
            "source": "personal_diary.md",
            "tags": ["personal", "hiking"],
        },
        {
            "id": "learning_recent",
            "content": "Reading 'Designing Data-Intensive Applications'. Learning about data consistency.",
            "created_at": (now - timedelta(days=3)).isoformat(),
            "source": "reading_log.md",
            "tags": ["learning", "books"],
        },
    ]


@pytest.mark.manual
@requires_real_apis
def test_time_weighted_retrieval(real_qa_setup: IntegratedQAChain):
    """
    Verify that the QA chain correctly prioritizes more recent memories.
    """
    print_header("Time-Weighted Retrieval")
    query = "Tell me about my React dashboard work"
    print(f"Query: {query}\n(Should prioritize recent work over older work)")

    result = real_qa_setup.invoke({"question": query})

    print(f"\nAnswer: {result['answer']}")
    print("\nRetrieved memories (should be ranked by relevance + recency):")
    for doc in result["source_documents"]:
        print(
            f"  - [{doc['id']}] Created: {doc['created_at']}, "
            f"Content: {doc['content'][:80]}..."
        )

    print_manual_check(
        [
            "Does the answer focus more on the recent refactor than the old planning?",
            "Is the 'work_recent_1' memory ranked higher than 'work_old'?",
        ]
    )


@pytest.mark.manual
@requires_real_apis
def test_domain_separation(real_qa_setup: IntegratedQAChain):
    """
    Verify that the QA chain can distinguish between different life domains.
    """
    print_header("Personal vs. Work Memory Separation")
    test_cases = {
        "What personal activities have I enjoyed recently?": "Should find hiking, not work.",
        "What work projects am I involved in?": "Should find React work, not hiking.",
    }

    for query, expectation in test_cases.items():
        print(f"\n--- Query: {query} ---")
        print(f"Expected: {expectation}")

        result = real_qa_setup.invoke({"question": query})
        print(f"Answer: {result['answer']}")
        print("Retrieved memories:")
        for doc in result["source_documents"]:
            print(f"  - [{doc['id']}] Tags: {doc.get('tags', [])}")

    print_manual_check(
        [
            "Did the 'personal' query retrieve only personal memories?",
            "Did the 'work' query retrieve only work memories?",
            "Are the answers appropriate for each domain?",
        ]
    )


@pytest.mark.manual
@requires_real_apis
def test_edge_case_queries(real_qa_setup: IntegratedQAChain):
    """
    Verify the system's behavior with nonsensical or irrelevant queries.
    """
    print_header("Edge Cases and Irrelevant Queries")
    edge_cases = {
        "asdfghjkl": "Nonsensical query",
        "What is the meaning of life?": "Query with no relevant memories",
    }

    for query, description in edge_cases.items():
        print(f"\n--- {description}: '{query}' ---")
        result = real_qa_setup.invoke({"question": query})
        print(f"Answer: {result['answer']}")
        print(f"Retrieved {len(result['source_documents'])} memories")

    print_manual_check(
        [
            "Does the system provide a sensible default response for irrelevant queries?",
            "Does it avoid hallucinating or retrieving unrelated memories?",
        ]
    )


@pytest.mark.manual
@requires_real_apis
def test_response_quality_and_synthesis(real_qa_setup: IntegratedQAChain):
    """
    Verify the overall quality and synthesis of the responses.
    """
    print_header("Response Quality and Synthesis")
    query = "Summarize what I've been up to recently across work and personal life."
    print(f"\nQuery: {query}")

    start_time = time.time()
    result = real_qa_setup.invoke({"question": query})
    duration = time.time() - start_time

    print(f"\nAnswer (generated in {duration:.2f}s): {result['answer']}")
    print(f"Synthesized from {len(result['source_documents'])} memories.")

    print_manual_check(
        [
            "Is the answer a coherent and well-structured summary?",
            "Does it successfully synthesize information from multiple memories?",
            "Is the response time acceptable?",
        ]
    )
