from __future__ import annotations

from pathlib import Path

import pytest

from app.ingestion.markdown_loader import parse_markdown_file


@pytest.mark.integration
def test_parsing_of_real_notes():
    """Test parsing against real files in `personal_notes` directory."""
    personal_notes_dir = Path("personal_notes")
    if not personal_notes_dir.exists() or not personal_notes_dir.is_dir():
        pytest.skip("`personal_notes` directory not found for testing.")

    markdown_files = list(personal_notes_dir.glob("*.md"))
    if not markdown_files:
        pytest.skip("No markdown files found in personal_notes/ to test parsing.")

    # Test with the first file found
    test_file = markdown_files[0]
    chunks = parse_markdown_file(test_file)

    assert len(chunks) > 0, f"No chunks parsed from {test_file}"

    for chunk in chunks:
        assert "id" in chunk
        assert "content" in chunk
        assert "source" in chunk
        assert "created_at" in chunk
        assert len(chunk["content"].strip()) > 0, "Chunk content is empty"
        assert chunk["source"] == str(test_file)
