"""Consistent chunking strategy shared by loaders & LLM chains."""

from __future__ import annotations

from langchain.text_splitter import RecursiveCharacterTextSplitter

__all__ = ["split_markdown"]


_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,  # tokens ≈ 2‑3× chars → keep ≤ ~200 tokens
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""],
)


def split_markdown(text: str) -> list[str]:
    """Return a list of *overlapping* chunks suitable for embedding."""
    return _splitter.split_text(text)
