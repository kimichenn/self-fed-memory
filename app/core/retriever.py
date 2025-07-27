"""Time-weighted retriever that combines semantic similarity with recency bias.

This retriever scores memories by both semantic similarity AND recency,
using a decay function: score = similarity + (1 - decay_rate) ^ hours_since_last_access

Key features:
- Prioritizes recent memories while preserving older ones
- Updates last_accessed_at timestamps when memories are retrieved
- Handles missing timestamps gracefully
- Compatible with our existing vector store interface
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from app.core.types import to_document


class TimeWeightedRetriever:
    """A retriever that scores documents by similarity + time decay."""

    def __init__(
        self,
        vector_store: VectorStore,
        embeddings: Embeddings,
        decay_rate: float = 0.01,
        k: int = 5,
    ):
        """Initialize the time-weighted retriever.

        Args:
            vector_store: The underlying vector store
            embeddings: Embeddings model for query encoding
            decay_rate: Controls how quickly old memories decay (0-1, lower = slower decay)
            k: Number of documents to retrieve
        """
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.decay_rate = decay_rate
        self.k = k

    def get_relevant_documents(self, query: str, **kwargs: Any) -> list[Document]:
        """Retrieve documents with time-weighted scoring.

        Returns documents sorted by combined similarity + recency score.
        Updates last_accessed_at timestamps for retrieved documents.
        """
        # Get more candidates than needed for re-ranking
        candidates_k = min(self.k * 3, 20)  # Get 3x as many candidates for re-ranking

        # Get initial similarity-based results
        docs = self.vector_store.similarity_search(query, k=candidates_k)

        if not docs:
            return []

        # Calculate time-weighted scores
        now = datetime.utcnow()
        scored_docs = []

        for doc in docs:
            # Get similarity score (approximate from position, since Pinecone doesn't return scores by default)
            similarity_score = self._estimate_similarity_score(
                docs.index(doc), len(docs)
            )

            # Get time decay score
            time_score = self._calculate_time_score(doc, now)

            # Combined score
            combined_score = similarity_score + time_score
            scored_docs.append((combined_score, doc))

        # Sort by combined score and take top k
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        top_docs = [doc for _, doc in scored_docs[: self.k]]

        # Update last_accessed_at timestamps
        self._update_access_timestamps(top_docs, now)

        return top_docs

    def _estimate_similarity_score(self, position: int, total: int) -> float:
        """Estimate similarity score based on position in results.

        Since Pinecone similarity_search doesn't return scores by default,
        we estimate based on position. First result gets highest score.
        """
        if total <= 1:
            return 1.0
        # Linear decay from 1.0 to 0.1 based on position
        score = 1.0 - (position * 0.9 / (total - 1))
        return round(score, 10)  # Round to avoid floating point precision issues

    def _calculate_time_score(self, doc: Document, now: datetime) -> float:
        """Calculate time decay score for a document."""
        try:
            # Try last_accessed_at first, then fall back to created_at
            timestamp_str = doc.metadata.get("last_accessed_at") or doc.metadata.get(
                "created_at"
            )

            if not timestamp_str:
                # No timestamp available, give neutral time score
                return 0.0

            # Parse timestamp (handle both ISO format and other common formats)
            if isinstance(timestamp_str, str):
                # Handle Z suffix by replacing with +00:00
                clean_timestamp = timestamp_str.replace("Z", "+00:00")

                # Parse the timestamp
                timestamp = datetime.fromisoformat(clean_timestamp)

                # Convert to naive UTC if it's timezone-aware
                if timestamp.tzinfo is not None:
                    timestamp = timestamp.replace(tzinfo=None)
            else:
                timestamp = timestamp_str
                # Ensure it's naive
                if hasattr(timestamp, "tzinfo") and timestamp.tzinfo is not None:
                    timestamp = timestamp.replace(tzinfo=None)

            # Calculate hours since last access/creation
            hours_since = (now - timestamp).total_seconds() / 3600

            # Apply decay function: (1 - decay_rate) ^ hours_since
            # This gives higher scores to more recent documents
            time_score = math.pow(1 - self.decay_rate, hours_since)

            return time_score

        except (ValueError, TypeError) as e:
            # If timestamp parsing fails, give neutral score
            return 0.0

    def _update_access_timestamps(self, docs: list[Document], accessed_at: datetime):
        """Update last_accessed_at timestamps for retrieved documents."""
        if not docs:
            return

        # Update metadata
        timestamp_str = accessed_at.isoformat()
        updated_docs = []

        for doc in docs:
            # Create updated document with new timestamp
            new_metadata = doc.metadata.copy()
            new_metadata["last_accessed_at"] = timestamp_str

            updated_doc = Document(page_content=doc.page_content, metadata=new_metadata)
            updated_docs.append(updated_doc)

        # Upsert back to vector store to persist the timestamp updates
        try:
            if hasattr(self.vector_store, "upsert"):
                self.vector_store.upsert(updated_docs)
            elif hasattr(self.vector_store, "add_documents"):
                # For stores that don't have upsert, we'll try add_documents with the same IDs
                ids = [
                    doc.metadata.get("id")
                    for doc in updated_docs
                    if doc.metadata.get("id")
                ]
                if ids:
                    self.vector_store.add_documents(updated_docs, ids=ids)
        except Exception as e:
            # If timestamp update fails, log but don't break retrieval
            # In production, you might want to use proper logging here
            pass

    def search(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        """Convenience method that returns results as dictionaries."""
        docs = self.get_relevant_documents(query, **kwargs)
        return [{**doc.metadata, "content": doc.page_content} for doc in docs]
