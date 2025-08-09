from pydantic import BaseModel
import pytest

pytestmark = pytest.mark.unit


def test_chat_request_model_shape():
    # Import lazily to avoid importing FastAPI app during test collection
    from app.api.routes_chat import ChatRequest

    class _X(ChatRequest):
        pass

    assert issubclass(_X, BaseModel)
    # Required field
    assert "question" in ChatRequest.model_fields
    # Optional fields exist
    for f in [
        "name",
        "k",
        "intelligent",
        "conversation_history",
        "store_chat",
        "session_id",
    ]:
        assert f in ChatRequest.model_fields
