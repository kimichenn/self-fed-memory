from __future__ import annotations

from pathlib import Path

import pytest

from app.ingestion.markdown_loader import parse_markdown_file


@pytest.fixture
def tmp_file(tmp_path: Path) -> Path:
    """Create a temporary markdown file for testing."""
    p = tmp_path / "test.md"
    p.write_text(
        "---\ncreated: Jun 11, 2024 at 9:40 AM\n---\n# Title\n\nThis is a test."
    )
    return p


@pytest.mark.unit
def test_parse_markdown_file(tmp_file: Path):
    """Check that a markdown file is parsed correctly."""
    chunks = parse_markdown_file(tmp_file)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk["content"] == "# Title\n\nThis is a test."
    assert chunk["source"] == str(tmp_file)
    assert chunk["created_at"] == "2024-06-11T09:40:00"
