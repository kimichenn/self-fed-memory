"""Facade that ties *loaders* + *vector store* together."""

from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings

from app.core.types import to_document
from app.core.vector_store.pinecone import PineconeVectorStore


class MemoryManager:
    """High-level API used by CLI, tests, and future web routes."""

    def __init__(self, embeddings: Embeddings):
        self.store = PineconeVectorStore(embeddings)

    # -------- Ingestion --------------------------------------------------
    def add_chunks(self, chunks: list[dict[str, Any]]) -> None:
        documents = [to_document(chunk) for chunk in chunks]
        self.store.upsert(documents)

    # -------- Query ------------------------------------------------------
    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Similarity search that returns metadata dicts."""
        # Pinecone indexing can be eventually consistent; newly-upserted
        # vectors may not be immediately queryable even after the index
        # statistics show them. We add a short retry loop to improve test
        # stability without materially affecting latency in production.

        attempts = 5
        backoff = 2  # seconds

        for attempt in range(attempts):
            docs = self.store.similarity_search(query, k=k)
            if docs:
                return [{**doc.metadata, "content": doc.page_content} for doc in docs]

            if attempt < attempts - 1:
                import time

                time.sleep(backoff)

        # If still no results, return empty list for caller to handle.
        return []
