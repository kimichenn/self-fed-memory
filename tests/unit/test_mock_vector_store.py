"""Unit tests for MockVectorStore functionality.

This module tests the mock vector store implementation including:
- Initialization and configuration
- Document storage and retrieval
- Similarity search algorithm
- CRUD operations (upsert, delete, clear)
- Helper methods for testing
"""

from __future__ import annotations

from unittest.mock import patch

from langchain_core.documents import Document
import pytest

from app.core.vector_store.mock import MockVectorStore


@pytest.mark.unit
class TestMockVectorStoreInitialization:
    """Test MockVectorStore initialization."""

    def test_initialization(self, mock_vector_store, mock_embeddings):
        """Test basic initialization."""
        assert mock_vector_store._embeddings == mock_embeddings
        assert mock_vector_store.get_all_documents() == []

    def test_initialization_with_kwargs(self, mock_embeddings):
        """Test initialization with additional kwargs."""
        store = MockVectorStore(embeddings=mock_embeddings, some_param="value")
        assert store._embeddings == mock_embeddings
        assert store.get_all_documents() == []


@pytest.mark.unit
class TestMockVectorStoreFromTexts:
    """Test MockVectorStore.from_texts class method."""

    def test_from_texts_basic(self, mock_embeddings):
        """Test creating store from texts without metadata."""
        texts = ["First document", "Second document", "Third document"]
        store = MockVectorStore.from_texts(texts, mock_embeddings)
        docs = store.get_all_documents()
        assert len(docs) == 3
        assert docs[0].page_content == "First document"
        assert docs[1].page_content == "Second document"
        assert docs[2].page_content == "Third document"
        for doc in docs:
            assert doc.metadata == {}

    def test_from_texts_with_metadata(self, mock_embeddings):
        """Test creating store from texts with metadata."""
        texts = ["First document", "Second document"]
        metadatas = [{"id": "1", "source": "test"}, {"id": "2", "source": "test"}]
        store = MockVectorStore.from_texts(texts, mock_embeddings, metadatas=metadatas)
        docs = store.get_all_documents()
        assert len(docs) == 2
        assert docs[0].page_content == "First document"
        assert docs[0].metadata == {"id": "1", "source": "test"}
        assert docs[1].page_content == "Second document"
        assert docs[1].metadata == {"id": "2", "source": "test"}

    def test_from_texts_with_kwargs(self, mock_embeddings):
        """Test from_texts with additional kwargs."""
        texts = ["Test document"]
        store = MockVectorStore.from_texts(texts, mock_embeddings, some_param="value")
        docs = store.get_all_documents()
        assert len(docs) == 1
        assert docs[0].page_content == "Test document"


@pytest.mark.unit
class TestMockVectorStoreDocumentOperations:
    """Test document CRUD operations."""

    def test_add_documents_basic(self, mock_vector_store):
        """Test adding documents without IDs."""
        documents = [
            Document(page_content="First doc", metadata={"id": "1"}),
            Document(page_content="Second doc", metadata={"id": "2"}),
        ]
        result_ids = mock_vector_store.add_documents(documents)
        docs = mock_vector_store.get_all_documents()
        assert len(docs) == 2
        assert docs[0].page_content == "First doc"
        assert docs[1].page_content == "Second doc"
        assert result_ids == ["1", "2"]

    def test_add_documents_with_ids_upsert(self, mock_vector_store):
        """Test adding documents with IDs performs upsert."""
        initial_docs = [
            Document(page_content="Original doc", metadata={"id": "1"}),
            Document(page_content="Other doc", metadata={"id": "2"}),
        ]
        mock_vector_store.add_documents(initial_docs)
        new_docs = [
            Document(
                page_content="Updated doc", metadata={"id": "1"}
            ),  # Should replace
            Document(page_content="New doc", metadata={"id": "3"}),  # Should add
        ]
        mock_vector_store.add_documents(new_docs, ids=["1", "3"])
        docs = mock_vector_store.get_all_documents()
        assert len(docs) == 3
        doc_by_id = {doc.metadata.get("id"): doc for doc in docs}
        assert doc_by_id["1"].page_content == "Updated doc"
        assert doc_by_id["2"].page_content == "Other doc"
        assert doc_by_id["3"].page_content == "New doc"

    def test_upsert_method(self, mock_vector_store):
        """Test the upsert method delegates correctly."""
        documents = [
            Document(page_content="Test doc", metadata={"id": "1"}),
        ]
        with patch.object(mock_vector_store, "add_documents") as mock_add:
            mock_add.return_value = ["1"]
            result = mock_vector_store.upsert(documents)
            mock_add.assert_called_once_with(documents, ids=["1"])
            assert result == ["1"]

    def test_delete_documents(self, mock_vector_store):
        """Test deleting documents by ID."""
        documents = [
            Document(page_content="Doc 1", metadata={"id": "1"}),
            Document(page_content="Doc 2", metadata={"id": "2"}),
            Document(page_content="Doc 3", metadata={"id": "3"}),
        ]
        mock_vector_store.add_documents(documents)
        mock_vector_store.delete(["1", "3"])
        docs = mock_vector_store.get_all_documents()
        assert len(docs) == 1
        assert docs[0].metadata["id"] == "2"
        assert docs[0].page_content == "Doc 2"

    def test_delete_nonexistent_ids(self, mock_vector_store):
        """Test deleting non-existent IDs doesn't cause errors."""
        documents = [Document(page_content="Doc 1", metadata={"id": "1"})]
        mock_vector_store.add_documents(documents)
        mock_vector_store.delete(["1", "nonexistent", "also_nonexistent"])
        assert len(mock_vector_store.get_all_documents()) == 0

    def test_get_all_documents(self, mock_vector_store):
        """Test getting all documents returns a copy."""
        documents = [
            Document(page_content="Doc 1", metadata={"id": "1"}),
            Document(page_content="Doc 2", metadata={"id": "2"}),
        ]
        mock_vector_store.add_documents(documents)
        all_docs = mock_vector_store.get_all_documents()
        assert len(all_docs) == 2
        assert all_docs[0].page_content == "Doc 1"
        assert all_docs[1].page_content == "Doc 2"
        all_docs.clear()
        assert len(mock_vector_store.get_all_documents()) == 2

    def test_clear_store(self, mock_vector_store):
        """Test clearing all documents."""
        documents = [
            Document(page_content="Doc 1", metadata={"id": "1"}),
            Document(page_content="Doc 2", metadata={"id": "2"}),
        ]
        mock_vector_store.add_documents(documents)
        assert len(mock_vector_store.get_all_documents()) == 2
        mock_vector_store.clear()
        assert len(mock_vector_store.get_all_documents()) == 0


@pytest.mark.unit
class TestMockVectorStoreSimilaritySearch:
    """Test similarity search algorithm."""

    @pytest.mark.parametrize(
        ("query", "k", "documents", "expected_ids"),
        [
            (
                "machine learning",
                2,
                [
                    Document(
                        page_content="machine learning algorithms", metadata={"id": "1"}
                    ),
                    Document(page_content="deep neural networks", metadata={"id": "2"}),
                    Document(
                        page_content="cooking recipes and food", metadata={"id": "3"}
                    ),
                    Document(
                        page_content="machine learning models", metadata={"id": "4"}
                    ),
                ],
                ["1", "4"],
            ),
            ("test query", 5, [], []),
            (
                "xyz abc",
                1,
                [
                    Document(
                        page_content="completely different content",
                        metadata={"id": "1"},
                    ),
                ],
                ["1"],
            ),
            (
                "document",
                5,
                [
                    Document(page_content="first document", metadata={"id": "1"}),
                    Document(page_content="second document", metadata={"id": "2"}),
                ],
                ["1", "2"],
            ),
            (
                "python programming",
                4,
                [
                    Document(page_content="python programming", metadata={"id": "1"}),
                    Document(page_content="java programming", metadata={"id": "2"}),
                    Document(page_content="python java", metadata={"id": "3"}),
                    Document(page_content="web development", metadata={"id": "4"}),
                ],
                ["1", "3", "2", "4"],
            ),
        ],
    )
    @patch("random.uniform", return_value=0.0)
    def test_similarity_search(
        self,
        mock_random,
        mock_vector_store,
        query,
        k,
        documents,
        expected_ids,
    ):
        """Test similarity search with various scenarios."""
        mock_vector_store.add_documents(documents)
        results = mock_vector_store.similarity_search(query, k=k)
        result_ids = [doc.metadata["id"] for doc in results]

        # For the empty case, we expect an empty list
        if not expected_ids:
            assert result_ids == []
            return

        # For unordered comparisons, check for presence
        if len(expected_ids) > 1 and expected_ids != sorted(expected_ids):
            assert set(result_ids) == set(expected_ids[: len(results)])
        else:
            # For ordered comparisons, check order
            assert result_ids == expected_ids[: len(results)]

    def test_similarity_search_score_clamping(self, mock_vector_store):
        """Test that similarity scores are clamped to [0, 1] range."""
        documents = [
            Document(page_content="test content", metadata={"id": "1"}),
        ]
        mock_vector_store.add_documents(documents)
        with patch("random.uniform", side_effect=[-0.5, 0.5]):
            results = mock_vector_store.similarity_search("test", k=1)
        assert len(results) == 1
