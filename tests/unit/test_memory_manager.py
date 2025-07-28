"""Unit tests for MemoryManager retrieval, scoring, and ingestion logic.

This module tests the core functionality of the MemoryManager facade including:
- Initialization with different configurations
- Document ingestion via add_chunks()
- Retrieval logic with time weighting vs basic similarity search
- Search method parameter handling and delegation
- Retry logic in basic similarity search
"""

from __future__ import annotations

from unittest.mock import MagicMock
from unittest.mock import call
from unittest.mock import patch

from langchain_core.documents import Document
import pytest

from app.core.memory import MemoryManager


@pytest.mark.unit
class TestMemoryManagerInitialization:
    """Test MemoryManager initialization with different configurations."""

    def test_init_with_time_weighting_enabled(
        self, memory_manager, mock_embeddings, mock_store, mock_retriever
    ):
        """Test initialization with time weighting enabled (default)."""
        assert memory_manager.embeddings == mock_embeddings
        assert memory_manager.use_time_weighting is True
        assert memory_manager.store == mock_store
        assert memory_manager.retriever == mock_retriever

    def test_init_with_time_weighting_disabled(
        self, memory_manager_no_time_weight, mock_embeddings, mock_store
    ):
        """Test initialization with time weighting disabled."""
        assert memory_manager_no_time_weight.embeddings == mock_embeddings
        assert memory_manager_no_time_weight.use_time_weighting is False
        assert memory_manager_no_time_weight.store == mock_store
        assert memory_manager_no_time_weight.retriever is None

    def test_init_default_parameters(self, mock_retriever_class, mock_store_class):
        """Test initialization with default parameters."""
        mock_embeddings = MagicMock()
        manager = MemoryManager(embeddings=mock_embeddings)
        assert manager.use_time_weighting is True
        # The `TimeWeightedRetriever` constructor gained new optional keyword
        # arguments (``llm`` and ``use_intelligent_queries``).  We only care
        # that the mandatory arguments are forwarded correctly, so we loosen
        # the expectation here.

        call_args, call_kwargs = mock_retriever_class.call_args
        assert call_kwargs["vector_store"] == mock_store_class.return_value
        assert call_kwargs["embeddings"] == mock_embeddings
        assert call_kwargs["decay_rate"] == 0.01
        assert call_kwargs["k"] == 5


@pytest.mark.unit
class TestMemoryManagerIngestion:
    """Test MemoryManager document ingestion functionality."""

    @patch("app.core.memory.to_document")
    def test_add_chunks(
        self, mock_to_document, memory_manager_no_time_weight, mock_store
    ):
        """Test add_chunks converts and upserts documents."""
        chunks = [
            {"id": "1", "content": "First chunk", "metadata": {"source": "test"}},
            {"id": "2", "content": "Second chunk", "metadata": {"source": "test"}},
        ]
        mock_docs = [MagicMock(), MagicMock()]
        mock_to_document.side_effect = mock_docs

        memory_manager_no_time_weight.add_chunks(chunks)

        expected_calls = [call(chunk) for chunk in chunks]
        mock_to_document.assert_has_calls(expected_calls)
        mock_store.upsert.assert_called_once_with(mock_docs)

    @patch("app.core.memory.to_document")
    def test_add_chunks_empty_list(
        self, mock_to_document, memory_manager_no_time_weight, mock_store
    ):
        """Test add_chunks handles empty chunk list."""
        memory_manager_no_time_weight.add_chunks([])
        mock_to_document.assert_not_called()
        mock_store.upsert.assert_called_once_with([])


@pytest.mark.unit
class TestMemoryManagerRetrieval:
    """Test MemoryManager retrieval logic and scoring."""

    def test_search_with_time_weighting_enabled(self, memory_manager, mock_retriever):
        """Test search uses time-weighted retrieval when enabled."""
        expected_results = [{"id": "1", "content": "test", "score": 0.9}]
        mock_retriever.search.return_value = expected_results
        results = memory_manager.search("test query", k=3)
        assert mock_retriever.k == 3
        mock_retriever.search.assert_called_once_with("test query")
        assert results == expected_results

    def test_search_with_time_weighting_disabled(
        self, memory_manager_no_time_weight, mock_store
    ):
        """Test search uses basic similarity search when time weighting disabled."""
        mock_docs = [
            Document(
                page_content="test content", metadata={"id": "1", "source": "test"}
            )
        ]
        mock_store.similarity_search.return_value = mock_docs
        results = memory_manager_no_time_weight.search("test query", k=3)
        mock_store.similarity_search.assert_called_once_with("test query", k=3)
        expected_results = [{"id": "1", "source": "test", "content": "test content"}]
        assert results == expected_results

    def test_search_override_time_weighting(
        self, memory_manager, mock_store, mock_retriever
    ):
        """Test search can override time weighting setting per query."""
        mock_docs = [Document(page_content="test content", metadata={"id": "1"})]
        mock_store.similarity_search.return_value = mock_docs
        memory_manager.search("test query", k=3, use_time_weighting=False)
        mock_store.similarity_search.assert_called_once()
        mock_retriever.search.assert_not_called()

    def test_search_with_time_range(self, memory_manager_no_time_weight):
        """Test search_with_time_range currently delegates to regular search."""
        with patch.object(memory_manager_no_time_weight, "search") as mock_search:
            mock_search.return_value = [{"id": "1", "content": "test"}]
            memory_manager_no_time_weight.search_with_time_range(
                "test query", start_date="2023-01-01", end_date="2023-12-31", k=3
            )
            mock_search.assert_called_once_with("test query", k=3)
