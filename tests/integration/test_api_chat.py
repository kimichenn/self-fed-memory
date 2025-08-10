from fastapi.testclient import TestClient
import pytest

from app.api.main import create_app
from app.core.config import Settings

pytestmark = pytest.mark.integration


def test_health_endpoint():
    app = create_app()
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_models_endpoint():
    app = create_app()
    client = TestClient(app)
    r = client.get("/v1/models")
    assert r.status_code == 200
    data = r.json()
    assert "data" in data
    assert any(m["id"] for m in data["data"])  # simple shape check


def test_chat_rejects_empty_question(monkeypatch):
    app = create_app()
    client = TestClient(app)

    r = client.post("/chat", json={"question": ""})
    assert r.status_code == 422 or r.status_code == 400


def test_chat_with_store_chat_gracefully_works_without_supabase(monkeypatch):
    """When Supabase isn't configured, store_chat should be ignored (no crash)."""
    app = create_app()
    client = TestClient(app)

    payload = {
        "question": "Hello there",
        "name": "Tester",
        "store_chat": True,
        "intelligent": False,
    }
    r = client.post("/chat", json=payload)
    # Should still succeed and return an answer/result even if Supabase is missing
    assert r.status_code in (200, 201)
    data = r.json()
    assert "answer" in data or "result" in data


def test_permanent_memory_upsert_requires_config_or_auth(monkeypatch):
    app = create_app()
    client = TestClient(app)
    payload = {"content": "Test permanent memory"}
    r = client.post("/permanent_memories/upsert", json=payload)
    # When Supabase is not configured, endpoint should reject
    if not Settings().supabase_url or not Settings().supabase_key:
        assert r.status_code == 400
        assert r.json()["detail"] == "Supabase is not configured"
    else:
        # When Supabase is configured, it should require auth if API_AUTH_KEY is set
        monkeypatch.setenv("API_AUTH_KEY", "test-key")
        app2 = create_app()
        client2 = TestClient(app2)
        r2 = client2.post("/permanent_memories/upsert", json=payload)
        assert r2.status_code == 401


def test_chat_history_requires_supabase_config():
    app = create_app()
    client = TestClient(app)
    r = client.get("/chat/history", params={"session_id": "abc", "limit": 5})
    assert r.status_code == 400
    assert r.json()["detail"] == "Supabase is not configured"
