from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.integration


def _import_frontend_module() -> ModuleType:
    """Import the Streamlit frontend module in a headless/non-running context.

    This ensures that the module can be imported without starting a UI server
    and that its top-level helpers can be referenced.
    """
    # Ensure repository root is importable
    repo_root = Path(__file__).resolve().parents[2]
    import sys

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Import the module
    return importlib.import_module("frontend.app")


def test_frontend_imports_headlessly(monkeypatch):
    import streamlit  # noqa: F401

    # Pretend no backend is reachable so any probe short-circuits
    monkeypatch.setenv("SELF_MEMORY_API", "http://127.0.0.1:9")
    mod = _import_frontend_module()
    assert hasattr(mod, "call_chat_api")
