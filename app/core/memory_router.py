from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app.core.knowledge_store import SupabaseKnowledgeStore
from app.core.memory import MemoryManager

CORE_TYPES = {"preference", "fact", "profile", "user_core"}


@dataclass
class RouteSummary:
    pinecone_upserts: int = 0
    supabase_upserts: int = 0
    pinecone_deletes: int = 0
    supabase_deletes: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "pinecone_upserts": self.pinecone_upserts,
            "supabase_upserts": self.supabase_upserts,
            "pinecone_deletes": self.pinecone_deletes,
            "supabase_deletes": self.supabase_deletes,
        }


class MemoryRouter:
    """Routes memory items between Supabase and Pinecone (single-user).

    Routing rules (simple and explicit):
    - Items with `type` in {preference, fact, profile, user_core} are considered
      core and will be stored in Supabase permanent_memories.
    - All items are stored in Pinecone for semantic retrieval unless explicitly
      disabled via `route_to_vector=False` in the item.
    - Single-user mode: no user profile management is required.
    """

    def __init__(
        self,
        memory_manager: MemoryManager,
        supabase_store: SupabaseKnowledgeStore | None = None,
    ) -> None:
        self.memory_manager = memory_manager
        self.supabase = supabase_store

    # ------------------------- Public API ---------------------------------
    def upsert_items(self, items: list[dict[str, Any]]) -> dict[str, int]:
        summary = RouteSummary()

        # Normalize and split by route
        pinecone_chunks: list[dict[str, Any]] = []

        for raw in items:
            item = self._normalize_item(raw)
            item_type = (item.get("type") or "").lower()
            is_core = item_type in CORE_TYPES

            # Route to Supabase if core and store available
            if is_core and self.supabase is not None:
                tags: list[str] = [item_type]
                category = item.get("category")
                if isinstance(category, str) and category:
                    tags.append(category)
                source = item.get("source") or "router"
                self.supabase.upsert_permanent_memory(
                    content=item.get("content", ""),
                    tags=tags,
                    source=source,
                    memory_id=item.get("id"),
                )
                summary.supabase_upserts += 1

            # Single-user: no user profile management required

            # Route to Pinecone unless explicitly disabled
            if item.get("route_to_vector", True):
                pinecone_chunks.append(item)

        if pinecone_chunks:
            self.memory_manager.add_chunks(pinecone_chunks)
            summary.pinecone_upserts += len(pinecone_chunks)

        return summary.as_dict()

    def delete_items(self, ids: list[str], target: str | None = None) -> dict[str, int]:
        """Delete items from one or both backends.

        Args:
          ids: document IDs (Pinecone metadata.id or Supabase memory id)
          target: "pinecone", "supabase", or None for both
        """
        summary = RouteSummary()

        if target in (None, "pinecone"):
            try:
                self.memory_manager.store.delete(ids)
                summary.pinecone_deletes += len(ids)
            except Exception:
                # Non-fatal: deleting non-existent IDs is fine
                pass

        if target in (None, "supabase") and self.supabase is not None:
            for mid in ids:
                try:
                    self.supabase.delete_permanent_memory(mid)
                    summary.supabase_deletes += 1
                except Exception:
                    pass

        return summary.as_dict()

    def delete_all(self, target: str | None = None) -> dict[str, int]:
        """Delete all items from one or both backends.

        Args:
          target: "pinecone", "supabase", or None for both
        """
        summary = RouteSummary()

        if target in (None, "pinecone"):
            try:
                # Prefer vector store API if available
                delete_all_fn = getattr(self.memory_manager.store, "delete_all", None)
                if callable(delete_all_fn):
                    delete_all_fn()
                else:
                    # Fallback: if no bulk delete, we cannot list IDs generically here
                    # so we treat as best-effort no-op
                    pass
            except Exception:
                pass

        if target in (None, "supabase") and self.supabase is not None:
            try:
                delete_all_pm = getattr(
                    self.supabase, "delete_all_permanent_memories", None
                )
                if callable(delete_all_pm):
                    delete_all_pm()
            except Exception:
                pass

        return summary.as_dict()

    def search(self, query: str, k: int = 5) -> dict[str, Any]:
        """Aggregate search across Pinecone (semantic) and Supabase (core facts).

        Supabase results are filtered to likely relevant core items using a
        simple substring match on content (best-effort, non-semantic).
        """
        vector_results = self.memory_manager.search(query, k=k)

        supabase_results: list[dict[str, Any]] = []
        if self.supabase is not None:
            try:
                # Pull recent core items; lightweight client-side filter
                candidates = self.supabase.list_permanent_memories(
                    tags=["preference", "fact"], limit=100
                )
                q = query.lower()
                supabase_results = [
                    m for m in candidates if q in (m.get("content", "").lower())
                ][:k]
            except Exception:
                supabase_results = []

        # Merge, preferring vector results order
        seen = set()
        combined: list[dict[str, Any]] = []
        for d in vector_results + supabase_results:
            doc_id = d.get("id") or d.get("metadata", {}).get("id")
            if doc_id and doc_id in seen:
                continue
            seen.add(doc_id)
            combined.append(d)

        return {
            "vector_results": vector_results,
            "supabase_results": supabase_results,
            "combined": combined[: k * 2],
        }

    # ------------------------- Internals ----------------------------------
    def _normalize_item(self, item: dict[str, Any]) -> dict[str, Any]:
        """Ensure required fields exist for Pinecone ingestion."""
        norm = dict(item)
        if not norm.get("content"):
            norm["content"] = ""
        if not norm.get("id"):
            # Time-based id; Pinecone can accept arbitrary deterministic IDs
            norm["id"] = f"mem_{int(datetime.utcnow().timestamp() * 1000)}"
        if not norm.get("created_at"):
            norm["created_at"] = datetime.utcnow().isoformat()
        if not norm.get("source"):
            norm["source"] = "manual"
        return norm
