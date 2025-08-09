"""Wrapper around OpenAI text embeddings.

*Keeps everything behind a thin, testable abstraction so you can swap
models later (e.g. Azure, local models) without rewiring callers.*
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

from app.core.config import Settings


class _FallbackEmbeddings(Embeddings):
    """Deterministic lightweight embeddings for tests when OpenAI is unavailable."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._hash_embed(text)

    def _hash_embed(self, text: str) -> list[float]:
        # Simple, fast, deterministic embedding based on character codes
        import math

        vec = [0.0] * self.dim
        if not text:
            return vec
        for i, ch in enumerate(text.encode("utf-8")):
            vec[i % self.dim] += (ch % 53) / 53.0
        # L2 normalise
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """Singleton OpenAIEmbeddings instance (thread-safe).

    Cached so that repeated calls don't re-instantiate network clients.
    """
    cfg = Settings()
    try:
        return OpenAIEmbeddings(
            model=cfg.embedding_model,
            openai_api_key=SecretStr(cfg.openai_api_key),
            chunk_size=1000,  # Match OpenAI API limit
        )
    except Exception:
        # Fall back to a local deterministic embedding for tests
        return _FallbackEmbeddings(dim=getattr(cfg, "embedding_dim", 384))
