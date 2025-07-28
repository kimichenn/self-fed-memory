from __future__ import annotations

from unittest.mock import MagicMock

from langchain_core.documents import Document
import pytest

from app.core.vector_store.mock import MockVectorStore


@pytest.fixture
def store() -> MockVectorStore:
    """A fresh mock store for each test."""
    return MockVectorStore(embeddings=MagicMock())


# ---------------------------------------------------------------------------
# Upsert failure scenarios
# ---------------------------------------------------------------------------


def test_upsert_with_duplicate_ids_raises_error(store: MockVectorStore):
    """Providing duplicate IDs in a single upsert call should raise ValueError."""
    documents = [
        Document(page_content="doc 1", metadata={"id": "dup"}),
        Document(page_content="doc 1 again", metadata={"id": "dup"}),
    ]

    with pytest.raises(ValueError, match="Duplicate document IDs"):
        store.upsert(documents)


# ---------------------------------------------------------------------------
# Delete failure scenarios
# ---------------------------------------------------------------------------


def test_delete_with_empty_list_raises_error(store: MockVectorStore):
    """Calling delete with an empty list should raise ValueError."""
    with pytest.raises(ValueError, match="No document IDs"):
        store.delete([])
