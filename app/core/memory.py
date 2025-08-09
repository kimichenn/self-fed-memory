"""Facade that ties *loaders* + *vector store* together."""

from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings

from app.core.config import Settings
from app.core.retriever import TimeWeightedRetriever
from app.core.types import to_document
from app.core.vector_store.pinecone import PineconeVectorStore


class MemoryManager:
    """High-level API used by CLI, tests, and future web routes."""

    def __init__(
        self,
        embeddings: Embeddings,
        use_time_weighting: bool = True,
        decay_rate: float = 0.01,
        use_intelligent_queries: bool = True,
        llm=None,
        cfg: Settings | None = None,
    ):
        """Initialize MemoryManager.

        Args:
            embeddings: Embeddings model for vector operations
            use_time_weighting: Whether to use time-weighted retrieval (default: True)
            decay_rate: Time decay rate for time-weighted retrieval (default: 0.01)
            use_intelligent_queries: Whether to use LLM-powered query analysis (default: True)
            llm: Language model for query analysis (optional)
        """
        self.embeddings = embeddings
        self.store = PineconeVectorStore(embeddings, cfg=cfg)
        self.use_time_weighting = use_time_weighting

        if use_time_weighting:
            self.retriever: TimeWeightedRetriever | None = TimeWeightedRetriever(
                vector_store=self.store,
                embeddings=embeddings,
                decay_rate=decay_rate,
                k=5,  # Default k, can be overridden in search
                llm=llm,
                use_intelligent_queries=use_intelligent_queries,
            )
        else:
            self.retriever = None

    # -------- Ingestion --------------------------------------------------
    def add_chunks(self, chunks: list[dict[str, Any]]) -> None:
        documents = [to_document(chunk) for chunk in chunks]
        self.store.upsert(documents)

    # -------- Query ------------------------------------------------------
    def search(
        self, query: str, k: int = 5, use_time_weighting: bool | None = None
    ) -> list[dict[str, Any]]:
        """Similarity search that returns metadata dicts.

        Args:
            query: Search query
            k: Number of results to return
            use_time_weighting: Override default time weighting setting for this search
        """
        # Determine whether to use time weighting for this search
        should_use_time_weighting = (
            use_time_weighting
            if use_time_weighting is not None
            else self.use_time_weighting
        )

        if should_use_time_weighting and self.retriever:
            # Use time-weighted retrieval
            self.retriever.k = k  # Update k for this search
            return self.retriever.search(query)
        else:
            # Use basic similarity search (original behavior)
            return self._basic_similarity_search(query, k)

    def _basic_similarity_search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Basic similarity search without time weighting (backward compatibility)."""
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

    def search_with_time_range(
        self, query: str, start_date: str = None, end_date: str = None, k: int = 5
    ) -> list[dict[str, Any]]:
        """Search with optional date filtering (future enhancement).

        Args:
            query: Search query
            start_date: ISO format date string (optional)
            end_date: ISO format date string (optional)
            k: Number of results to return

        Note: This is a placeholder for future date filtering functionality.
        Currently just performs regular search.
        """
        # For now, just use regular search
        # TODO: Implement date filtering in vector store query
        return self.search(query, k=k)
