"""Global configuration (12-factor style).

Environment variables expected (see `.env.example`):

* ``OPENAI_API_KEY``   - required
* ``PINECONE_API_KEY`` - required
* ``PINECONE_ENV``     - default: ``"us-east-1"``
* ``PINECONE_INDEX``   - default: ``"self-memory"``
* ``EMBEDDING_MODEL``  - default: ``"text-embedding-3-large"``

Test environment variables:
* ``TEST_PINECONE_INDEX``   - default: uses PINECONE_INDEX value
* ``TEST_EMBEDDING_MODEL``  - default: uses EMBEDDING_MODEL value

Usage:

    from app.core.config import Settings
    settings = Settings()  # auto-loads & validates env vars

    # For tests:
    test_settings = Settings.for_testing()
"""

from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict


class Settings(BaseSettings):
    """Typed view over process environment - cached at import time."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str
    pinecone_api_key: str

    pinecone_env: str = "us-east-1"
    pinecone_index: str = "self-memory"
    pinecone_namespace: str = "self-memory-namespace"

    embedding_model: str = "text-embedding-3-large"
    embedding_dim: int = 3072

    # Test-specific environment variables
    test_pinecone_index: str | None = None
    test_embedding_model: str | None = None

    @classmethod
    def for_testing(cls) -> "Settings":
        """Returns a Settings instance configured for testing.

        Uses TEST_* environment variables when available, falling back to
        regular values if not set.
        """
        settings = cls()

        # Override with test-specific values if they exist
        if settings.test_pinecone_index:
            settings.pinecone_index = settings.test_pinecone_index
        if settings.test_embedding_model:
            settings.embedding_model = settings.test_embedding_model

        return settings
