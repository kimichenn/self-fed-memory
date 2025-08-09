from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_default_session_id_env_override(monkeypatch):
    # Import lazily to avoid side effects
    from app.api.routes_chat import _default_session_id
    from app.core.config import Settings

    # Ensure override not set → expect built-in default
    monkeypatch.delenv("SUPABASE_DEFAULT_SESSION_ID", raising=False)
    cfg = Settings()
    assert _default_session_id(cfg) == "00000000-0000-0000-0000-000000000001"

    # Set override and expect it to be used
    custom = "11111111-2222-3333-4444-555555555555"
    monkeypatch.setenv("SUPABASE_DEFAULT_SESSION_ID", custom)
    cfg2 = Settings()
    assert _default_session_id(cfg2) == custom
