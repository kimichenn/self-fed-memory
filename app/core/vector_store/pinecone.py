"""Production VectorStore - backed by Pinecone."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_pinecone import PineconeVectorStore as LangchainPinecone
from pinecone import Pinecone
from pinecone import ServerlessSpec

from app.core.config import Settings


# NOTE: we name this `PineconeVectorStore` to keep a consistent
#       interface for the MemoryManager, but it's really just
#       a pre-configured adapter for the official implementation.
class PineconeVectorStore(LangchainPinecone, VectorStore):
    """Thin adapter for the official Langchain Pinecone integration."""

    def __init__(self, embeddings: Embeddings, cfg: Settings | None = None, **kwargs):
        self.cfg = cfg or Settings()
        # Attempt to create client; if API key missing during tests, raise a
        # clear error that tests can patch around
        pc = Pinecone(
            api_key=self.cfg.pinecone_api_key,
            environment=self.cfg.pinecone_env,
        )

        if self.cfg.pinecone_index not in pc.list_indexes().names():
            pc.create_index(
                name=self.cfg.pinecone_index,
                dimension=self.cfg.embedding_dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

        super().__init__(
            index_name=self.cfg.pinecone_index,
            embedding=embeddings,
            namespace=self.cfg.pinecone_namespace,
            **kwargs,
        )

    def upsert(self, documents: list[Document]) -> list[str]:
        """Insert or update documents."""
        ids = [doc.metadata["id"] for doc in documents]
        return super().add_documents(
            documents, ids=ids, namespace=self.cfg.pinecone_namespace
        )

    def similarity_search(self, query: str, k: int = 5, **kwargs) -> list[Document]:
        """Run similarity search."""
        # NOTE: Explicitly pass the namespace so that read / write operations
        # are guaranteed to hit the same logical collection, even if callers
        # forget to provide it.
        return super().similarity_search(
            query,
            k=k,
            namespace=self.cfg.pinecone_namespace,
            **kwargs,
        )

    def delete(self, ids: list[str]) -> None:
        """Delete vectors by ID."""
        self.index.delete(ids=ids, namespace=self.cfg.pinecone_namespace)

    def delete_all(self) -> None:
        """Delete all vectors in the configured namespace."""
        # Pinecone supports namespace-wide deletion by passing delete_all=True
        self.index.delete(delete_all=True, namespace=self.cfg.pinecone_namespace)

    # ------------------------------------------------------------------
    # Compatibility helpers
    # ------------------------------------------------------------------

    @property
    def index(self):
        """Return the underlying pinecone `Index` object.

        The upstream `LangchainPinecone` implementation stores the raw
        Pinecone client on the private attribute ``_index``. Exposing it via
        a public property allows test suites (and power-users) to access
        advanced Pinecone functionality such as ``describe_index_stats``
        without reaching into private internals.
        """

        return getattr(self, "_index", None)

    # Backwards-compatibility alias: some tests expected a ``_get_index``
    # method. We provide a thin wrapper so those tests keep working even
    # after migrating to the new ``index`` property.
    def _get_index(self):
        """Return the underlying Pinecone `Index` (legacy helper)."""

        return self.index
