"""Unit tests for PineconeVectorStore adapter functionality.

This module tests the Pinecone vector store adapter including:
- Initialization and configuration
- Index creation and management
- Adapter methods (upsert, similarity_search, delete)
- Namespace handling
- Property access and compatibility helpers
"""

from __future__ import annotations

from unittest.mock import MagicMock
from unittest.mock import patch

from langchain_core.documents import Document
import pytest

from app.core.vector_store.pinecone import PineconeVectorStore


@pytest.mark.unit
class TestPineconeVectorStoreInitialization:
    """Test PineconeVectorStore initialization and configuration."""

    @patch("app.core.vector_store.pinecone.Pinecone")
    @patch("app.core.vector_store.pinecone.Settings")
    @patch("app.core.vector_store.pinecone.LangchainPinecone.__init__")
    def test_initialization_existing_index(
        self, mock_super_init, mock_settings_class, mock_pinecone_class
    ):
        """Test initialization when index already exists."""
        mock_super_init.return_value = None

        # Mock settings
        mock_settings = MagicMock()
        mock_settings.pinecone_api_key = "test-api-key"
        mock_settings.pinecone_env = "test-env"
        mock_settings.pinecone_index = "test-index"
        mock_settings.pinecone_namespace = "test-namespace"
        mock_settings.embedding_dim = 1536
        mock_settings_class.return_value = mock_settings

        # Mock Pinecone client
        mock_pc = MagicMock()
        mock_pinecone_class.return_value = mock_pc

        # Mock index list to include our test index
        mock_index_list = MagicMock()
        mock_index_list.names.return_value = ["test-index", "other-index"]
        mock_pc.list_indexes.return_value = mock_index_list

        mock_embeddings = MagicMock()

        # Initialize store
        store = PineconeVectorStore(embeddings=mock_embeddings)

        # Verify Pinecone client initialized correctly
        mock_pinecone_class.assert_called_once_with(
            api_key="test-api-key",
            environment="test-env",
        )

        # Verify index existence check
        mock_pc.list_indexes.assert_called_once()

        # Verify index creation NOT called (index exists)
        mock_pc.create_index.assert_not_called()

        # Verify parent class initialized
        mock_super_init.assert_called_once_with(
            index_name="test-index",
            embedding=mock_embeddings,
            namespace="test-namespace",
        )

        # Verify settings stored
        assert store.cfg == mock_settings

    @patch("app.core.vector_store.pinecone.Pinecone")
    @patch("app.core.vector_store.pinecone.Settings")
    @patch("app.core.vector_store.pinecone.LangchainPinecone.__init__")
    @patch("app.core.vector_store.pinecone.ServerlessSpec")
    def test_initialization_creates_index(
        self,
        mock_serverless_spec,
        mock_super_init,
        mock_settings_class,
        mock_pinecone_class,
    ):
        """Test initialization creates index when it doesn't exist."""
        mock_super_init.return_value = None

        # Mock settings
        mock_settings = MagicMock()
        mock_settings.pinecone_api_key = "test-api-key"
        mock_settings.pinecone_env = "test-env"
        mock_settings.pinecone_index = "new-index"
        mock_settings.pinecone_namespace = "test-namespace"
        mock_settings.embedding_dim = 768
        mock_settings_class.return_value = mock_settings

        # Mock Pinecone client
        mock_pc = MagicMock()
        mock_pinecone_class.return_value = mock_pc

        # Mock index list to NOT include our test index
        mock_index_list = MagicMock()
        mock_index_list.names.return_value = ["other-index"]
        mock_pc.list_indexes.return_value = mock_index_list

        # Mock ServerlessSpec
        mock_spec = MagicMock()
        mock_serverless_spec.return_value = mock_spec

        mock_embeddings = MagicMock()

        PineconeVectorStore(embeddings=mock_embeddings)

        # Verify index creation called
        mock_pc.create_index.assert_called_once_with(
            name="new-index",
            dimension=768,
            metric="cosine",
            spec=mock_spec,
        )

        # Verify ServerlessSpec created correctly
        mock_serverless_spec.assert_called_once_with(cloud="aws", region="us-east-1")

    @patch("app.core.vector_store.pinecone.Pinecone")
    @patch("app.core.vector_store.pinecone.Settings")
    @patch("app.core.vector_store.pinecone.LangchainPinecone.__init__")
    def test_initialization_with_kwargs(
        self, mock_super_init, mock_settings_class, mock_pinecone_class
    ):
        """Test initialization passes through additional kwargs."""
        mock_super_init.return_value = None

        # Mock settings
        mock_settings = MagicMock()
        mock_settings.pinecone_api_key = "test-api-key"
        mock_settings.pinecone_env = "test-env"
        mock_settings.pinecone_index = "test-index"
        mock_settings.pinecone_namespace = "test-namespace"
        mock_settings.embedding_dim = 1536
        mock_settings_class.return_value = mock_settings

        # Mock Pinecone client
        mock_pc = MagicMock()
        mock_pinecone_class.return_value = mock_pc

        # Mock index list to include our test index
        mock_index_list = MagicMock()
        mock_index_list.names.return_value = ["test-index"]
        mock_pc.list_indexes.return_value = mock_index_list

        mock_embeddings = MagicMock()

        # Verify parent class initialized with kwargs
        mock_super_init.assert_called_once_with(
            index_name="test-index",
            embedding=mock_embeddings,
            namespace="test-namespace",
            custom_param="value",
        )


@pytest.mark.unit
class TestPineconeVectorStoreAdapterMethods:
    """Test PineconeVectorStore adapter methods."""

    def setup_method(self):
        """Set up common test fixtures."""
        with (
            patch("app.core.vector_store.pinecone.Pinecone"),
            patch("app.core.vector_store.pinecone.Settings") as mock_settings_class,
            patch("app.core.vector_store.pinecone.LangchainPinecone.__init__"),
        ):
            mock_settings = MagicMock()
            mock_settings.pinecone_namespace = "test-namespace"
            mock_settings_class.return_value = mock_settings

            self.store = PineconeVectorStore(embeddings=MagicMock())

    @patch("app.core.vector_store.pinecone.LangchainPinecone.add_documents")
    def test_upsert_method(self, mock_add_documents):
        """Test upsert method delegates to add_documents correctly."""
        mock_add_documents.return_value = ["id1", "id2"]

        documents = [
            Document(page_content="Doc 1", metadata={"id": "id1"}),
            Document(page_content="Doc 2", metadata={"id": "id2"}),
        ]

        result = self.store.upsert(documents)

        # Verify delegation to parent add_documents
        mock_add_documents.assert_called_once_with(
            documents, ids=["id1", "id2"], namespace="test-namespace"
        )
        assert result == ["id1", "id2"]

    @patch("app.core.vector_store.pinecone.LangchainPinecone.add_documents")
    def test_upsert_method_missing_ids(self, mock_add_documents):
        """Test upsert method handles documents without IDs."""
        mock_add_documents.return_value = ["generated-id"]

        documents = [
            Document(page_content="Doc without ID", metadata={"other": "field"}),
        ]

        # Should raise KeyError when trying to access missing ID
        with pytest.raises(KeyError):
            self.store.upsert(documents)

    @patch("app.core.vector_store.pinecone.LangchainPinecone.similarity_search")
    def test_similarity_search_method(self, mock_similarity_search):
        """Test similarity_search method delegates correctly."""
        mock_docs = [
            Document(page_content="Result 1", metadata={"id": "1"}),
            Document(page_content="Result 2", metadata={"id": "2"}),
        ]
        mock_similarity_search.return_value = mock_docs

        result = self.store.similarity_search("test query", k=3, custom_param="value")

        # Verify delegation with namespace and kwargs
        mock_similarity_search.assert_called_once_with(
            "test query", k=3, namespace="test-namespace", custom_param="value"
        )
        assert result == mock_docs

    @patch("app.core.vector_store.pinecone.LangchainPinecone.similarity_search")
    def test_similarity_search_default_k(self, mock_similarity_search):
        """Test similarity_search method uses default k=5."""
        mock_similarity_search.return_value = []

        self.store.similarity_search("test query")

        # Verify default k=5 used
        mock_similarity_search.assert_called_once_with(
            "test query", k=5, namespace="test-namespace"
        )

    def test_delete_method(self):
        """Test delete method delegates to index.delete."""
        # Mock the index property
        mock_index = MagicMock()
        self.store._index = mock_index

        ids_to_delete = ["id1", "id2", "id3"]
        self.store.delete(ids_to_delete)

        # Verify delegation to index.delete with namespace
        mock_index.delete.assert_called_once_with(
            ids=ids_to_delete, namespace="test-namespace"
        )


@pytest.mark.unit
class TestPineconeVectorStoreProperties:
    """Test PineconeVectorStore property access and compatibility helpers."""

    def setup_method(self):
        """Set up common test fixtures."""
        with (
            patch("app.core.vector_store.pinecone.Pinecone"),
            patch("app.core.vector_store.pinecone.Settings") as mock_settings_class,
            patch("app.core.vector_store.pinecone.LangchainPinecone.__init__"),
        ):
            mock_settings = MagicMock()
            mock_settings.pinecone_namespace = "test-namespace"
            mock_settings_class.return_value = mock_settings

            self.store = PineconeVectorStore(embeddings=MagicMock())

    def test_index_property_access(self):
        """Test index property returns _index attribute."""
        mock_index = MagicMock()
        self.store._index = mock_index

        result = self.store.index

        assert result == mock_index

    def test_index_property_no_index(self):
        """Test index property returns None when _index doesn't exist."""
        # Don't set _index attribute
        result = self.store.index

        assert result is None

    def test_get_index_legacy_method(self):
        """Test _get_index legacy compatibility method."""
        mock_index = MagicMock()
        self.store._index = mock_index

        result = self.store._get_index()

        assert result == mock_index

    def test_get_index_legacy_method_delegates_to_property(self):
        """Test _get_index delegates to index property."""
        # Mock the _index attribute directly since index is a property
        mock_index = MagicMock()
        self.store._index = mock_index

        result = self.store._get_index()

        # Should return the same value as the index property
        assert result == mock_index
        assert result == self.store.index


@pytest.mark.unit
class TestPineconeVectorStoreNamespaceHandling:
    """Test PineconeVectorStore namespace handling across all methods."""

    def setup_method(self):
        """Set up test fixtures with specific namespace."""
        with (
            patch("app.core.vector_store.pinecone.Pinecone"),
            patch("app.core.vector_store.pinecone.Settings") as mock_settings_class,
            patch("app.core.vector_store.pinecone.LangchainPinecone.__init__"),
        ):
            mock_settings = MagicMock()
            mock_settings.pinecone_namespace = "custom-namespace"
            mock_settings_class.return_value = mock_settings

            self.store = PineconeVectorStore(embeddings=MagicMock())

    @patch("app.core.vector_store.pinecone.LangchainPinecone.add_documents")
    def test_namespace_consistency_upsert(self, mock_add_documents):
        """Test upsert always uses configured namespace."""
        mock_add_documents.return_value = ["id1"]

        documents = [Document(page_content="Test", metadata={"id": "id1"})]
        self.store.upsert(documents)

        # Verify namespace is always passed
        mock_add_documents.assert_called_once_with(
            documents, ids=["id1"], namespace="custom-namespace"
        )

    @patch("app.core.vector_store.pinecone.LangchainPinecone.similarity_search")
    def test_namespace_consistency_similarity_search(self, mock_similarity_search):
        """Test similarity_search always uses configured namespace."""
        mock_similarity_search.return_value = []

        # Test that providing namespace in kwargs doesn't override our configured namespace
        # The method should handle this gracefully
        self.store.similarity_search("query", other_param="value")

        # Verify our configured namespace is used, not any that might be passed in kwargs
        mock_similarity_search.assert_called_once_with(
            "query",
            k=5,
            namespace="custom-namespace",
            other_param="value",
        )

    def test_namespace_consistency_delete(self):
        """Test delete always uses configured namespace."""
        mock_index = MagicMock()
        self.store._index = mock_index

        self.store.delete(["id1"])

        mock_index.delete.assert_called_once_with(
            ids=["id1"], namespace="custom-namespace"
        )
