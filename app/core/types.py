"""Domain models shared across the package."""

from __future__ import annotations

from langchain_core.documents import Document


def to_document(data: dict) -> Document:
    """Convert a dictionary to a Langchain Document.

    The `page_content` will be the `content` field, and the rest of
    the fields will be stored in the `metadata` attribute.
    """
    # Create a copy to avoid mutating the original dictionary
    data_copy = data.copy()
    return Document(page_content=data_copy.pop("content"), metadata=data_copy)
