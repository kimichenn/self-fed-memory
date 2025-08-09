"""Wrapper around OpenAI text embeddings.

*Keeps everything behind a thin, testable abstraction so you can swap
models later (e.g. Azure, local models) without rewiring callers.*
"""

from __future__ import annotations

from functools import lru_cache

from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

from app.core.config import Settings


@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings:
    """Singleton OpenAIEmbeddings instance (thread-safe).

    Cached so that repeated calls don't re-instantiate network clients.
    """
    cfg = Settings()
    return OpenAIEmbeddings(
        model=cfg.embedding_model,
        openai_api_key=SecretStr(cfg.openai_api_key),
        chunk_size=1000,  # Match OpenAI API limit
    )
