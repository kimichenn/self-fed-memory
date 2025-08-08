import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app


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
