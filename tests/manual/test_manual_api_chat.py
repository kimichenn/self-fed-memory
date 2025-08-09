import os
import time

import pytest
import requests

pytestmark = pytest.mark.manual


def test_manual_chat_flow(monkeypatch_module):
    """
    Manual verification:
    - Start the API server separately: `make api-dev`
    - Ensure `.env` contains your OpenAI and Pinecone keys
    - This test will call the live API and print responses for you to check
    """

    base = os.environ.get("SELF_MEMORY_API", "http://localhost:8000")

    # Health check
    r = requests.get(f"{base}/health", timeout=10)
    assert r.status_code == 200

    # Upsert a small memory
    payload = {
        "items": [
            {
                "id": f"manual-{int(time.time())}",
                "content": "I enjoy quiet sushi places and dislike loud music.",
                "source": "manual-test",
                "type": "preference",
            }
        ]
    }
    r = requests.post(f"{base}/memories/upsert", json=payload, timeout=30)
    assert r.status_code == 200

    # Ask a question
    chat = {
        "question": "Where should I go for dinner tonight?",
        "name": "Tester",
        "intelligent": True,
        "k": 5,
    }
    r = requests.post(f"{base}/chat", json=chat, timeout=120)
    assert r.status_code == 200
    data = r.json()
    print("\n--- MANUAL RESPONSE ---\n", data.get("answer", data))
