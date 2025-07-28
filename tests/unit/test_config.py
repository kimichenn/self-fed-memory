import pytest

from app.core.config import Settings
from tests.helpers import get_test_settings


@pytest.mark.unit
def test_test_settings_use_test_env_vars(monkeypatch):
    """Test that test settings properly use TEST_* environment variables."""
    # Set test-specific environment variables
    monkeypatch.setenv("PINECONE_INDEX", "production-index")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("TEST_PINECONE_INDEX", "test-index")
    monkeypatch.setenv("TEST_EMBEDDING_MODEL", "text-embedding-3-large")

    # Get test settings
    test_settings = get_test_settings()

    # Should use the TEST_* values, not the regular ones
    assert test_settings.pinecone_index == "test-index"
    assert test_settings.embedding_model == "text-embedding-3-large"


@pytest.mark.unit
def test_test_settings_different_from_regular_settings():
    """Test that test settings differ from regular settings when TEST_* vars are set."""
    # Get regular and test settings
    regular_settings = Settings()
    test_settings = get_test_settings()

    # If TEST_* variables are set, the test settings should be different
    if regular_settings.test_pinecone_index:
        assert test_settings.pinecone_index == regular_settings.test_pinecone_index

        if regular_settings.pinecone_index != regular_settings.test_pinecone_index:
            assert test_settings.pinecone_index != regular_settings.pinecone_index

    if regular_settings.test_embedding_model:
        assert test_settings.embedding_model == regular_settings.test_embedding_model
