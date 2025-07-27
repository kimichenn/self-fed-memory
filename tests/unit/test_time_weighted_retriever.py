"""Unit tests for the TimeWeightedRetriever mathematical functions.

This module tests the core mathematical and logical components of time-weighted retrieval.
For integration testing with real data, use manual verification tests.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

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
