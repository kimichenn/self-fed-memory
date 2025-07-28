from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from scripts.ingest_folder import main as ingest_main


@pytest.mark.unit
def test_ingest_folder_script(tmp_path: Path):
    """Test the ingest_folder.py script functionality."""
    # 1. Setup - Create a temporary directory with a markdown file
    mock_memory_manager = MagicMock()

    temp_dir = tmp_path / "notes"
    temp_dir.mkdir()
    p = temp_dir / "test.md"
    p.write_text(
        "---\ncreated: Jun 11, 2024 at 9:40 AM\n---\n# Title\n\nThis is a test."
    )

    # 2. Run the script callback directly (bypass Click's CLI parsing entirely)
    ingest_main.callback(
        directory=temp_dir, dry_run=False, memory_manager=mock_memory_manager
    )

    # 3. Assertion - Verify that add_chunks was called with the correct data
    assert mock_memory_manager.add_chunks.call_count == 1
    call_args, _ = mock_memory_manager.add_chunks.call_args
    chunks = call_args[0]
    assert len(chunks) == 1
    assert chunks[0]["content"] == "# Title\n\nThis is a test."
    assert chunks[0]["source"] == str(p)
