from app.core.config import Settings


def get_test_settings() -> Settings:
    """Returns a Settings instance for testing.

    Automatically uses TEST_PINECONE_INDEX and TEST_EMBEDDING_MODEL
    environment variables when available.
    """
    return Settings.for_testing()
