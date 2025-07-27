from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from app.core.embeddings import get_embeddings
from app.core.memory import MemoryManager
from app.ingestion.markdown_loader import parse_markdown_file


@pytest.fixture
def tmp_file(tmp_path: Path) -> Path:
    """Create a temporary markdown file for testing."""
    p = tmp_path / "test.md"
    p.write_text("# Test\n\nThis is a test document.")
    return p


@pytest.mark.unit
@patch("app.core.memory.PineconeVectorStore")
def test_ingestion_pipeline(mock_pinecone_store, tmp_file: Path):
    """Test the full ingestion pipeline with a mock vector store."""
    mock_store_instance = mock_pinecone_store.return_value
    embeddings = get_embeddings()
    manager = MemoryManager(embeddings)

    # Replace the real store with the mock
    manager.store = mock_store_instance

    chunks = parse_markdown_file(tmp_file)
    manager.add_chunks(chunks)

    mock_store_instance.upsert.assert_called_once()
    args, _ = mock_store_instance.upsert.call_args
    documents = args[0]

    assert len(documents) == 1
    assert documents[0].page_content == "# Test\n\nThis is a test document."
    assert documents[0].metadata["source"] == str(tmp_file)
