from __future__ import annotations

import random
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


class MockVectorStore(VectorStore):
    """A mock vector store for testing purposes with enhanced similarity simulation."""

    def __init__(self, embeddings: Embeddings, **kwargs: Any):
        self._embeddings = embeddings
        self._store: list[Document] = []

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict] | None = None,
        **kwargs: Any,
    ) -> MockVectorStore:
        """Create a MockVectorStore from a list of texts."""
        instance = cls(embedding, **kwargs)

        if metadatas is None:
            metadatas = [{}] * len(texts)

        documents = [
            Document(page_content=text, metadata=metadata)
            for text, metadata in zip(texts, metadatas)
        ]

        instance.add_documents(documents)
        return instance

    def add_documents(self, documents: list[Document], **kwargs: Any) -> list[str]:
        """Add documents to the mock store."""
        ids = kwargs.get("ids", [])

        # If IDs provided, this might be an upsert operation
        # Sanity-check: reject duplicate IDs within the same request – this
        # usually indicates an error in the caller logic and helps surface
        # issues early.
        if ids and len(ids) != len(set(ids)):
            raise ValueError("Duplicate document IDs detected in upsert operation.")

        if ids:
            # Remove existing documents with same IDs
            existing_ids = set(ids)
            self._store = [
                doc for doc in self._store if doc.metadata.get("id") not in existing_ids
            ]

        self._store.extend(documents)
        return [doc.metadata.get("id", "") for doc in documents]

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[Document]:
        """Perform a mock similarity search with basic text matching."""
        if not self._store:
            return []

        # Simple text-based similarity for testing
        query_lower = query.lower()
        scored_docs = []

        for doc in self._store:
            content_lower = doc.page_content.lower()

            # Calculate a simple similarity score based on shared words
            query_words = set(query_lower.split())
            content_words = set(content_lower.split())

            if query_words and content_words:
                intersection = query_words.intersection(content_words)
                union = query_words.union(content_words)
                similarity = len(intersection) / len(union) if union else 0
            else:
                similarity = 0

            # Add some randomness to simulate vector similarity
            similarity += random.uniform(-0.1, 0.1)
            similarity = max(0, min(1, similarity))  # Clamp to [0, 1]

            scored_docs.append((similarity, doc))

        # Sort by similarity score and return top k
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:k]]

    def upsert(self, documents: list[Document]) -> list[str]:
        """Insert or update documents."""
        ids = [doc.metadata.get("id") for doc in documents if doc.metadata.get("id")]
        return self.add_documents(documents, ids=ids)

    def delete(self, ids: list[str]) -> None:
        """Delete documents by ID."""
        if not ids:
            raise ValueError("No document IDs provided for deletion.")

        ids_set = set(ids)
        self._store = [
            doc for doc in self._store if doc.metadata.get("id") not in ids_set
        ]

    def get_all_documents(self) -> list[Document]:
        """Get all documents in the store (for testing)."""
        return self._store.copy()

    def clear(self) -> None:
        """Clear all documents (for testing)."""
        self._store.clear()

    def delete_all(self) -> None:
        """Delete all documents in the store."""
        self._store.clear()
