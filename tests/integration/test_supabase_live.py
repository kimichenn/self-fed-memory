from __future__ import annotations

import os

from fastapi.testclient import TestClient
import pytest

from app.api.main import create_app

pytestmark = pytest.mark.integration


def _have_supabase_env() -> bool:
    return bool(os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_KEY"))


skip_if_no_supabase = pytest.mark.skipif(
    not _have_supabase_env(), reason="SUPABASE_URL and SUPABASE_KEY not set"
)


@skip_if_no_supabase
def test_health_reports_supabase_configured(monkeypatch):
    client = TestClient(create_app())
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["supabase_configured"] is True


@skip_if_no_supabase
def test_permanent_memory_upsert_and_chat_history_with_supabase(monkeypatch):
    # Ensure requests use test-prefixed tables and enforce API auth
    monkeypatch.setenv(
        "TEST_SUPABASE_TABLE_PREFIX",
        os.environ.get("TEST_SUPABASE_TABLE_PREFIX", "test_"),
    )
    monkeypatch.setenv("API_AUTH_KEY", "test-api-key")

    client = TestClient(create_app())

    # Upsert a permanent memory
    pm = {
        "content": "Always on time.",
        "tags": ["habit"],
        "source": "tests",
        "use_test_supabase": True,
    }
    r = client.post(
        "/permanent_memories/upsert", json=pm, headers={"x-api-key": "test-api-key"}
    )
    assert r.status_code == 200
    mem_id = r.json().get("id")
    assert isinstance(mem_id, str) and len(mem_id) > 0

    # Chat with persistence to create a session + messages in test tables
    chat_payload = {
        "question": "Ping?",
        "intelligent": False,
        "store_chat": True,
        "use_test_supabase": True,
    }
    r = client.post("/chat", json=chat_payload, headers={"x-api-key": "test-api-key"})
    assert r.status_code == 200
    sid = r.json().get("session_id")
    assert isinstance(sid, str) and len(sid) > 0

    # Read chat history from test tables
    r = client.get(
        "/chat/history",
        params={"session_id": sid, "limit": 10, "use_test_supabase": True},
        headers={"x-api-key": "test-api-key"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["session_id"] == sid
    assert isinstance(data.get("messages"), list)

    # Cleanup: best-effort delete test data
    client.post(
        "/memories/delete_all",
        json={"use_test_supabase": True, "target": "supabase"},
    )
