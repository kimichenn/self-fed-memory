from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.core.memory_router import MemoryRouter

pytestmark = pytest.mark.unit


def _fake_memory_manager():
    mm = MagicMock()
    # store has upsert/delete
    mm.store = MagicMock()
    return mm


def test_upsert_routes_core_to_supabase_and_pinecone():
    mm = _fake_memory_manager()
    supa = MagicMock()
    router = MemoryRouter(memory_manager=mm, supabase_store=supa)

    items = [
        {"id": "a", "content": "likes sushi", "type": "preference", "source": "ui"},
        {"id": "b", "content": "random doc", "type": "document"},
    ]

    summary = router.upsert_items(items)

    # MemoryManager.add_chunks should be invoked once with two items
    assert mm.add_chunks.called
    args, _ = mm.add_chunks.call_args
    assert len(args[0]) == 2
    # Supabase receives core memory upsert for the preference item
    supa.upsert_permanent_memory.assert_called_once()
    assert summary["pinecone_upserts"] == 2
    assert summary["supabase_upserts"] == 1


def test_delete_routes_to_targets():
    mm = _fake_memory_manager()
    supa = MagicMock()
    router = MemoryRouter(memory_manager=mm, supabase_store=supa)

    # Delete from both
    summary = router.delete_items(["x", "y"], target=None)
    mm.store.delete.assert_called_once()
    assert supa.delete_permanent_memory.call_count == 2
    assert summary["pinecone_deletes"] == 2
    assert summary["supabase_deletes"] == 2

    # Delete Pinecone only
    mm = _fake_memory_manager()
    supa = MagicMock()
    router = MemoryRouter(memory_manager=mm, supabase_store=supa)
    router.delete_items(["x"], target="pinecone")
    mm.store.delete.assert_called_once()
    supa.delete_permanent_memory.assert_not_called()

    # Delete Supabase only
    mm = _fake_memory_manager()
    supa = MagicMock()
    router = MemoryRouter(memory_manager=mm, supabase_store=supa)
    router.delete_items(["x"], target="supabase")
    mm.store.delete.assert_not_called()
    supa.delete_permanent_memory.assert_called_once()


def test_search_merges_supabase_and_vector_results():
    mm = _fake_memory_manager()
    # Vector results
    mm.search.return_value = [
        {"id": "1", "content": "hello", "type": "document"},
        {"id": "2", "content": "world", "type": "document"},
    ]
    supa = MagicMock()
    supa.list_permanent_memories.return_value = [
        {"id": "2", "content": "world core"},  # duplicate id should be de-duplicated
        {"id": "3", "content": "related to world topic"},
    ]

    router = MemoryRouter(memory_manager=mm, supabase_store=supa)
    result = router.search("world", k=5)
    assert "combined" in result
    ids = [d["id"] for d in result["combined"]]
    assert "1" in ids and "2" in ids and "3" in ids
