from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


class MockVectorStore(VectorStore):
    """A mock vector store for testing purposes."""

    def __init__(self, embeddings: Embeddings, **kwargs: Any):
        self._embeddings = embeddings
        self._store: list[Document] = []

    def add_documents(self, documents: list[Document], **kwargs: Any) -> list[str]:
        """Add documents to the mock store."""
        self._store.extend(documents)
        return [doc.metadata.get("id", "") for doc in documents]

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[Document]:
        """Perform a mock similarity search."""
        # This is a simplistic mock. A real implementation might use
        # a simple distance metric on the embeddings.
        return self._store[:k]

    def upsert(self, documents: list[Document]) -> list[str]:
        """Insert or update documents."""
        return self.add_documents(documents)
