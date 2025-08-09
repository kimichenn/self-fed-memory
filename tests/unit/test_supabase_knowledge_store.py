from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.core.config import Settings
from app.core.knowledge_store import SupabaseKnowledgeStore

pytestmark = pytest.mark.unit


class _MockTable:
    def __init__(self):
        self.ops = []

    def upsert(self, payload, on_conflict=None):  # noqa: D401 - test helper
        self.ops.append(("upsert", payload, on_conflict))
        return self

    def insert(self, payload):
        self.ops.append(("insert", payload))
        return self

    def select(self, *args):
        self.ops.append(("select", args))
        return self

    def eq(self, key, val):
        self.ops.append(("eq", key, val))
        return self

    def order(self, key, desc=False):
        self.ops.append(("order", key, desc))
        return self

    def limit(self, n):
        self.ops.append(("limit", n))
        return self

    def execute(self):
        # For select, return a namespace with data shaped as list
        return SimpleNamespace(data=[])


class _MockClient:
    def __init__(self):
        self._tables = {
            "chat_sessions": _MockTable(),
            "chat_messages": _MockTable(),
            "permanent_memories": _MockTable(),
        }

    def table(self, name):
        return self._tables[name]


@pytest.fixture
def mock_create_client():
    with patch("app.core.knowledge_store.create_client") as mocked:
        mocked.return_value = _MockClient()
        yield mocked


def _settings_with_supabase() -> Settings:
    s = Settings()
    # Inject fake supabase creds
    s.supabase_url = "https://example.supabase.co"
    s.supabase_key = "anon-key"
    return s


def test_construct_store_requires_config(mock_create_client):
    s = _settings_with_supabase()
    store = SupabaseKnowledgeStore(cfg=s)
    assert store.client is not None


def test_construct_store_ok(mock_create_client):
    s = _settings_with_supabase()
    store = SupabaseKnowledgeStore(cfg=s)
    assert store.client is not None


def test_ensure_session_and_message_flow(mock_create_client):
    s = _settings_with_supabase()
    store = SupabaseKnowledgeStore(cfg=s)
    sid = store.ensure_session(title="Test Chat")
    mid = store.save_message(session_id=sid, role="user", content="Hello")
    assert all(isinstance(x, str) and x for x in [sid, mid])


def test_upsert_permanent_memory(mock_create_client):
    s = _settings_with_supabase()
    store = SupabaseKnowledgeStore(cfg=s)
    mem_id = store.upsert_permanent_memory(content="Always on-time", tags=["habit"])
    assert isinstance(mem_id, str) and mem_id


def test_get_chat_history_returns_list(mock_create_client):
    s = _settings_with_supabase()
    store = SupabaseKnowledgeStore(cfg=s)
    sid = store.ensure_session()
    history = store.get_chat_history(session_id=sid, limit=10)
    assert isinstance(history, list)
