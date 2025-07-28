"""Unit tests for the TimeWeightedRetriever mathematical functions.

This module tests the core mathematical and logical components of time-weighted retrieval.
For integration testing with real data, use manual verification tests.
"""

from __future__ import annotations

from datetime import datetime
from datetime import timedelta
import math
from unittest.mock import MagicMock

from langchain_core.documents import Document
import pytest

from app.core.retriever import TimeWeightedRetriever


@pytest.mark.unit
class TestTimeWeightedRetrieverMath:
    """Test suite for TimeWeightedRetriever mathematical functions."""

    @pytest.fixture
    def retriever(self):
        """Simple retriever instance for testing math functions."""
        mock_vector_store = MagicMock()
        mock_embeddings = MagicMock()
        return TimeWeightedRetriever(
            vector_store=mock_vector_store,
            embeddings=mock_embeddings,
            decay_rate=0.01,
            k=3,
        )

    def test_retriever_initialization(self):
        """Test retriever initializes with correct parameters."""
        mock_vector_store = MagicMock()
        mock_embeddings = MagicMock()

        retriever = TimeWeightedRetriever(
            vector_store=mock_vector_store,
            embeddings=mock_embeddings,
            decay_rate=0.02,
            k=10,
        )

        assert retriever.vector_store == mock_vector_store
        assert retriever.embeddings == mock_embeddings
        assert retriever.decay_rate == 0.02
        assert retriever.k == 10

    @pytest.mark.parametrize(
        "position, total_docs, expected_score",
        [
            (0, 1, 1.0),  # Single document
            (0, 5, 1.0),  # First of many
            (4, 5, 0.1),  # Last of many
            (2, 5, 0.55),  # Middle document
        ],
    )
    def test_estimate_similarity_score(
        self, retriever, position, total_docs, expected_score
    ):
        """Test similarity score estimation based on position."""
        score = retriever._estimate_similarity_score(position, total_docs)
        assert score == expected_score

    def test_calculate_time_score_recent_document(self, retriever):
        """Test time score calculation for recent document."""
        now = datetime.utcnow()
        doc = Document(page_content="test", metadata={"created_at": now.isoformat()})

        score = retriever._calculate_time_score(doc, now)
        # Recent document (0 hours) should get score close to 1.0
        expected = math.pow(1 - 0.01, 0)  # (1 - 0.01)^0 = 1.0
        assert abs(score - expected) < 0.001

    def test_calculate_time_score_old_document(self, retriever):
        """Test time score calculation for old document."""
        now = datetime.utcnow()
        old_time = now - timedelta(hours=100)
        doc = Document(
            page_content="test", metadata={"created_at": old_time.isoformat()}
        )

        score = retriever._calculate_time_score(doc, now)
        # Old document should get lower score
        expected = math.pow(1 - 0.01, 100)  # (1 - 0.01)^100
        assert abs(score - expected) < 0.001
        assert score < 0.5  # Should be significantly decayed

    def test_calculate_time_score_last_accessed_priority(self, retriever):
        """Test that last_accessed_at takes priority over created_at."""
        now = datetime.utcnow()
        recent_access = now - timedelta(hours=1)
        old_creation = now - timedelta(days=30)

        doc = Document(
            page_content="test",
            metadata={
                "created_at": old_creation.isoformat(),
                "last_accessed_at": recent_access.isoformat(),
            },
        )

        score = retriever._calculate_time_score(doc, now)
        # Should use last_accessed_at (1 hour ago) not created_at (30 days ago)
        expected = math.pow(1 - 0.01, 1)  # Based on 1 hour, not 30 days
        assert abs(score - expected) < 0.001

    def test_calculate_time_score_no_timestamp(self, retriever):
        """Test time score when no timestamp is available."""
        now = datetime.utcnow()
        doc = Document(page_content="test", metadata={"id": "test"})

        score = retriever._calculate_time_score(doc, now)
        assert score == 0.0

    def test_calculate_time_score_invalid_timestamp(self, retriever):
        """Test time score with invalid timestamp format."""
        now = datetime.utcnow()
        doc = Document(
            page_content="test", metadata={"created_at": "invalid-timestamp"}
        )

        score = retriever._calculate_time_score(doc, now)
        assert score == 0.0

    def test_decay_rate_effect(self):
        """Test that different decay rates produce different time scores."""
        mock_vector_store = MagicMock()
        mock_embeddings = MagicMock()

        # Create retrievers with different decay rates
        high_decay_retriever = TimeWeightedRetriever(
            vector_store=mock_vector_store,
            embeddings=mock_embeddings,
            decay_rate=0.1,  # High decay
            k=2,
        )

        low_decay_retriever = TimeWeightedRetriever(
            vector_store=mock_vector_store,
            embeddings=mock_embeddings,
            decay_rate=0.001,  # Low decay
            k=2,
        )

        # Test with documents of different ages
        now = datetime.utcnow()
        recent_doc = Document(
            page_content="test", metadata={"created_at": now.isoformat()}
        )
        old_doc = Document(
            page_content="test",
            metadata={"created_at": (now - timedelta(hours=50)).isoformat()},
        )

        # Calculate time scores
        recent_score_high = high_decay_retriever._calculate_time_score(recent_doc, now)
        old_score_high = high_decay_retriever._calculate_time_score(old_doc, now)

        recent_score_low = low_decay_retriever._calculate_time_score(recent_doc, now)
        old_score_low = low_decay_retriever._calculate_time_score(old_doc, now)

        # High decay should create bigger difference between recent and old
        high_decay_diff = recent_score_high - old_score_high
        low_decay_diff = recent_score_low - old_score_low

        assert high_decay_diff > low_decay_diff

    def test_retriever_handles_different_timestamp_formats(self, retriever):
        """Test that retriever handles various timestamp formats gracefully."""
        now = datetime.utcnow()

        # Test ISO format with Z
        doc1 = Document(
            page_content="test1", metadata={"created_at": now.isoformat() + "Z"}
        )

        # Test ISO format without Z
        doc2 = Document(page_content="test2", metadata={"created_at": now.isoformat()})

        # Test with timezone
        doc3 = Document(
            page_content="test3", metadata={"created_at": now.isoformat() + "+00:00"}
        )

        # All should parse without errors
        score1 = retriever._calculate_time_score(doc1, now)
        score2 = retriever._calculate_time_score(doc2, now)
        score3 = retriever._calculate_time_score(doc3, now)

        # All should be valid scores (close to 1.0 for recent docs)
        assert 0.9 < score1 <= 1.0
        assert 0.9 < score2 <= 1.0
        assert 0.9 < score3 <= 1.0


@pytest.mark.unit
class TestTimeWeightedRetrieverBehaviour:
    """Behaviour-oriented tests for TimeWeightedRetriever."""

    @pytest.fixture()
    def mock_vector_store(self) -> MagicMock:  # type: ignore[return-value]
        """A mock vector store with similarity_search & upsert capabilities."""
        store = MagicMock()
        # ``upsert`` is explicitly defined so we can assert calls.
        store.upsert = MagicMock()
        return store

    @pytest.fixture()
    def retriever(self, mock_vector_store: MagicMock) -> TimeWeightedRetriever:
        """A retriever instance wired to the mock vector store."""
        return TimeWeightedRetriever(
            vector_store=mock_vector_store,
            embeddings=MagicMock(),  # Embeddings are not used in these tests
            decay_rate=0.01,
            k=3,
            use_intelligent_queries=False,  # Ensure _basic_retrieval code path
        )

    def _build_docs(self, now: datetime):  # noqa: D401 – helper
        """Helper that returns three documents with varying recency.

        The documents are purposely returned in *reverse* chronological order
        so that correct re-ranking (recency + similarity) can be asserted.
        """

        doc_newest = Document(
            page_content="I am the newest document",
            metadata={"id": "newest", "created_at": now.isoformat()},
        )
        doc_old = Document(
            page_content="I am the oldest document",
            metadata={
                "id": "old",
                "created_at": (now - timedelta(hours=10)).isoformat(),
            },
        )
        doc_mid = Document(
            page_content="I am the middle document",
            metadata={
                "id": "mid",
                "created_at": (now - timedelta(hours=2)).isoformat(),
            },
        )
        # Intentionally return in a non-ideal order to test re-ranking.
        return [doc_old, doc_mid, doc_newest]

    def test_basic_retrieval_reranks_and_updates_timestamps(
        self, retriever: TimeWeightedRetriever, mock_vector_store: MagicMock
    ) -> None:
        """The newest document should be ranked first after re-scoring.

        We also verify that ``last_accessed_at`` is added to metadata and that
        the vector store's ``upsert`` method is invoked once.
        """
        now = datetime.utcnow()

        # Arrange: similarity_search returns documents in sub-optimal order.
        docs_from_store = self._build_docs(now)
        mock_vector_store.similarity_search.return_value = docs_from_store

        # Act
        results = retriever.get_relevant_documents("arbitrary query")

        # Assert – re-ranking by combined (similarity + recency) places the
        # most recent document first.
        assert results[0].metadata["id"] == "newest"
        assert {d.metadata["id"] for d in results} == {"newest", "mid", "old"}

        # All returned docs should have a ``last_accessed_at`` timestamp.
        for doc in results:
            assert "last_accessed_at" in doc.metadata

        # The retriever should persist timestamp updates via ``upsert``.
        mock_vector_store.upsert.assert_called_once()
        updated_docs = mock_vector_store.upsert.call_args.args[0]
        # Every upserted document must carry the new timestamp as well.
        assert all("last_accessed_at" in d.metadata for d in updated_docs)

    def test_search_wrapper_returns_dicts(
        self, retriever: TimeWeightedRetriever, mock_vector_store: MagicMock
    ) -> None:
        """The ``search`` helper should return a list of dictionaries."""
        now = datetime.utcnow()
        mock_vector_store.similarity_search.return_value = self._build_docs(now)

        results = retriever.search("does not matter")

        assert isinstance(results, list)
        assert results, "search() must return at least one result"
        assert isinstance(results[0], dict)
        # The dict should include both metadata and the content field.
        assert "content" in results[0]
        assert "id" in results[0]

    def test_update_access_timestamps_direct_call(
        self, retriever: TimeWeightedRetriever, mock_vector_store: MagicMock
    ) -> None:
        """Direct exercise of the _update_access_timestamps helper.

        Ensures that the helper enriches metadata and delegates to the vector
        store exactly once.
        """
        now = datetime.utcnow()
        doc = Document(page_content="data", metadata={"id": "42"})

        retriever._update_access_timestamps([doc], accessed_at=now)

        mock_vector_store.upsert.assert_called_once()
        upserted_docs = mock_vector_store.upsert.call_args.args[0]
        assert upserted_docs[0].metadata["last_accessed_at"] == now.isoformat()
