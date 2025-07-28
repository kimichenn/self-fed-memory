from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.core.memory import MemoryManager
from app.core.vector_store.mock import MockVectorStore


@pytest.fixture
def mock_embeddings():
    """Fixture for a mock embeddings object."""
    return MagicMock()


@pytest.fixture
def mock_vector_store(mock_embeddings) -> MockVectorStore:
    """Fixture for a MockVectorStore instance."""
    return MockVectorStore(embeddings=mock_embeddings)


@pytest.fixture
def mock_retriever_class(mocker):
    """Fixture for patching TimeWeightedRetriever."""
    return mocker.patch("app.core.memory.TimeWeightedRetriever")


@pytest.fixture
def mock_store_class(mocker):
    """Fixture for patching PineconeVectorStore."""
    return mocker.patch("app.core.memory.PineconeVectorStore")


@pytest.fixture
def mock_retriever(mock_retriever_class):
    """Fixture for a mock retriever instance."""
    return mock_retriever_class.return_value


@pytest.fixture
def mock_store(mock_store_class):
    """Fixture for a mock vector store instance."""
    return mock_store_class.return_value


@pytest.fixture
def memory_manager(mock_embeddings, mock_store, mock_retriever):
    """Fixture for a MemoryManager instance with time weighting enabled."""
    return MemoryManager(embeddings=mock_embeddings, use_time_weighting=True)


@pytest.fixture
def memory_manager_no_time_weight(mock_embeddings, mock_store):
    """Fixture for a MemoryManager instance with time weighting disabled."""
    return MemoryManager(embeddings=mock_embeddings, use_time_weighting=False)
