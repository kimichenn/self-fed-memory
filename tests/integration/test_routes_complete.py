from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient
import pytest

from app.api.main import create_app

pytestmark = pytest.mark.integration


def _client() -> TestClient:
    app = create_app()
    return TestClient(app)


def test_openai_compatible_chat_basic():
    client = _client()
    payload = {
        "model": "gpt-4.1",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Say hello"},
        ],
    }
    r = client.post("/v1/chat/completions", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["role"] == "assistant"


def test_memories_upsert_search_delete_flow():
    client = _client()

    # Upsert two items: one core (routes to supabase when configured), one document
    payload = {
        "items": [
            {
                "id": "11111111-1111-1111-1111-111111111111",
                "content": "user likes sushi",
                "source": "test",
                "type": "preference",
                "category": "food",
                "route_to_vector": True,
            },
            {
                "id": "t-integ-2",
                "content": "some unrelated document",
                "source": "test",
                "type": "document",
                "route_to_vector": True,
            },
        ],
        "use_test_supabase": True,
    }
    r = client.post("/memories/upsert", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["inserted"] == 2
    assert set(data["routing"]).issuperset(
        {"pinecone_upserts", "supabase_upserts", "pinecone_deletes", "supabase_deletes"}
    )

    # Deterministic similarity for mock store
    with patch("random.uniform", return_value=0.0):
        r = client.get(
            "/memories/search",
            params={"query": "likes sushi", "k": 5, "use_test_index": True},
        )
        assert r.status_code == 200
        results = r.json()
        assert "combined" in results

    # Delete by IDs (both backends best-effort)
    r = client.post(
        "/memories/delete",
        json={"ids": ["t-integ-1", "t-integ-2"], "use_test_index": True},
    )
    assert r.status_code == 200
    d = r.json()
    assert set(d["routing"]).issuperset(
        {"pinecone_upserts", "supabase_upserts", "pinecone_deletes", "supabase_deletes"}
    )


def test_delete_all_targets_no_crash():
    client = _client()
    # Pinecone only
    r = client.post(
        "/memories/delete_all",
        json={"use_test_index": True, "target": "pinecone"},
    )
    assert r.status_code == 200
    # Supabase only (silently ignored when not configured)
    r = client.post(
        "/memories/delete_all",
        json={"use_test_supabase": True, "target": "supabase"},
    )
    assert r.status_code == 200


def test_chat_basic_and_intelligent_no_supabase():
    client = _client()

    # Basic chain
    r = client.post("/chat", json={"question": "Hello?", "intelligent": False})
    assert r.status_code == 200
    assert "answer" in r.json() or "result" in r.json()

    # Intelligent chain
    r = client.post("/chat", json={"question": "Any tips?", "intelligent": True})
    assert r.status_code == 200
    assert "answer" in r.json()
