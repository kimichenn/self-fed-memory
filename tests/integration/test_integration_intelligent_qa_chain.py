import pytest
from unittest.mock import MagicMock, patch
from app.core.chains.intelligent_qa_chain import IntelligentQAChain
from app.core.memory import MemoryManager
from app.core.vector_store.mock import MockVectorStore
from app.core.embeddings import get_embeddings


@pytest.fixture
def mock_vector_store():
    """Fixture for a MockVectorStore."""
    return MockVectorStore(embeddings=get_embeddings())


@pytest.fixture
def memory_manager(mock_vector_store):
    """Fixture for a MemoryManager using a MockVectorStore.

    The current `MemoryManager` no longer accepts a `vector_store` constructor
    argument.  We therefore initialise it with `use_time_weighting=False` (so
    that it skips creating an internal time-weighted retriever that would lock
    in the default Pinecone store) and then monkey-patch its `store` attribute
    to our in-memory `MockVectorStore`.  This keeps the public API intact while
    ensuring the test remains hermetic and fast.
    """

    mm = MemoryManager(embeddings=get_embeddings(), use_time_weighting=False)
    # Swap in the mock store.  Because time weighting is disabled the manager
    # will invoke `_basic_similarity_search`, which only relies on `self.store`.
    mm.store = mock_vector_store
    return mm


# NOTE: We patch `random.uniform` so the mock vector store returns
# deterministic similarity scores, and we stub the QA chain's LLM call so
# there is zero network dependency or non-determinism.


@patch("random.uniform", return_value=0.0)
def test_intelligent_qa_chain_with_mock_vector_store(mock_random, memory_manager):
    """
    Integration test for IntelligentQAChain with a MockVectorStore.
    Verifies the end-to-end flow from question to answer using mocked components.
    """
    # Arrange
    # Ingest some data into the mock vector store
    memory_manager.add_chunks(
        [
            {
                "content": "user likes to code in python",
                "metadata": {"source": "test.md", "type": "preference"},
            }
        ]
    )
    memory_manager.add_chunks(
        [
            {
                "content": "user lives in new york",
                "metadata": {"source": "test.md", "type": "fact"},
            }
        ]
    )

    chain = IntelligentQAChain(memory_manager=memory_manager)

    # Stub out the actual LLM call with a deterministic response so the test
    # never reaches out to OpenAI (or any remote service) and always produces
    # the same output.  We keep the retrieval logic intact so we still get
    # meaningful `context_used` and `preferences_applied` values.
    deterministic_answer = "The user likes to code in python and lives in new york."
    chain.qa_chain = MagicMock()
    chain.qa_chain.invoke.return_value = deterministic_answer

    # Act
    result = chain.invoke({"question": "What does the user like to do?"})

    # Assert
    assert result["answer"] == deterministic_answer

    # These ensure the retrieval path is still exercised even though the LLM
    # itself is mocked.
    assert len(result["context_used"]) > 0
    assert len(result["preferences_applied"]) > 0
