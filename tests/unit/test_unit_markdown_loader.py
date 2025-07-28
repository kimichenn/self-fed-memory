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


@pytest.mark.unit
def test_frontmatter_metadata_preservation(tmp_path: Path):
    """Check that frontmatter metadata is preserved in chunks."""
    # Create a markdown file with rich frontmatter
    p = tmp_path / "test_rich_frontmatter.md"
    p.write_text(
        "---\n"
        "created: Jun 11, 2024 at 9:40 AM\n"
        "title: My Important Note\n"
        "tags: [productivity, planning, work]\n"
        "category: journal\n"
        "priority: high\n"
        "author: John Doe\n"
        "---\n"
        "# Title\n\n"
        "This is a test with rich frontmatter.\n\n"
        "## Section 2\n\n"
        "More content here to test chunking."
    )

    chunks = parse_markdown_file(p)
    assert len(chunks) > 0

    # Check first chunk for frontmatter metadata
    chunk = chunks[0]
    assert chunk["content"].startswith("# Title")
    assert chunk["source"] == str(p)
    assert chunk["created_at"] == "2024-06-11T09:40:00"

    # Verify all frontmatter fields are preserved (except 'created')
    assert chunk["title"] == "My Important Note"
    assert chunk["tags"] == ["productivity", "planning", "work"]
    assert chunk["category"] == "journal"
    assert chunk["priority"] == "high"
    assert chunk["author"] == "John Doe"

    # Verify 'created' is not in metadata (handled separately as 'created_at')
    assert "created" not in chunk

    # Check that all chunks have the same frontmatter metadata
    for chunk in chunks:
        assert chunk["title"] == "My Important Note"
        assert chunk["tags"] == ["productivity", "planning", "work"]
        assert chunk["category"] == "journal"
        assert chunk["priority"] == "high"
        assert chunk["author"] == "John Doe"
        assert chunk["created_at"] == "2024-06-11T09:40:00"
