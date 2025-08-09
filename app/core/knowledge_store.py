"""Supabase-backed persistence for chat history and permanent memories (single-user).

This module provides a thin wrapper around the Supabase Python client to
persist and retrieve:

- Chat sessions and messages
- Permanent memories (authoritative record)

Notes:
- Vector retrieval remains powered by the configured VectorStore (Pinecone),
  and you should still add/index relevant content there via `MemoryManager`.
  This store is the system-of-record for non-vector data.
"""

from __future__ import annotations

import contextlib
from datetime import datetime
from typing import Any, cast
import uuid

from app.core.config import Settings

try:
    # Import lazily to keep import-time errors out of unit tests when the
    # dependency is not needed. Tests can patch these symbols.
    from supabase import create_client
except Exception:  # pragma: no cover - handled in tests via monkeypatch
    create_client = None  # type: ignore


class SupabaseTables:
    chat_sessions: str
    chat_messages: str
    permanent_memories: str

    def __init__(self) -> None:
        self.chat_sessions = "chat_sessions"
        self.chat_messages = "chat_messages"
        self.permanent_memories = "permanent_memories"

    @classmethod
    def with_prefix(cls, prefix: str) -> SupabaseTables:
        prefix = prefix or ""
        inst = cls()
        inst.chat_sessions = f"{prefix}chat_sessions"
        inst.chat_messages = f"{prefix}chat_messages"
        inst.permanent_memories = f"{prefix}permanent_memories"
        return inst


class SupabaseKnowledgeStore:
    """Persistence layer backed by Supabase Postgres.

    The schema is intentionally simple and can map to the following SQL:

        -- chat sessions
        create table if not exists chat_sessions (
          id uuid primary key,
          title text,
          created_at timestamptz default now(),
          updated_at timestamptz default now()
        );

        -- chat messages
        create table if not exists chat_messages (
          id uuid primary key,
          session_id uuid references chat_sessions(id) on delete cascade,
          role text check (role in ('user','assistant','system')) not null,
          content text not null,
          created_at timestamptz default now()
        );

        -- permanent memories
        create table if not exists permanent_memories (
          id uuid primary key,
          content text not null,
          tags jsonb default '[]'::jsonb,
          source text,
          created_at timestamptz default now()
        );
    """

    def __init__(
        self, cfg: Settings | None = None, tables: SupabaseTables | None = None
    ):
        self.cfg = cfg or Settings()
        self.tables = tables or SupabaseTables()

        if not self.cfg.supabase_url or not self.cfg.supabase_key:
            raise RuntimeError(
                "Supabase configuration missing. Set SUPABASE_URL and SUPABASE_KEY."
            )

        if create_client is None:
            raise RuntimeError(
                "Supabase client is unavailable. Ensure 'supabase' package is installed."
            )

        # Create client
        self.client = create_client(self.cfg.supabase_url, self.cfg.supabase_key)

    # --------------------------- Chat sessions ---------------------------
    def ensure_session(
        self, session_id: str | None = None, title: str | None = None
    ) -> str:
        """Ensure a chat session exists; create if needed; return session_id."""
        final_id = session_id or str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        payload = {
            "id": final_id,
            "title": title or "Chat",
            "updated_at": now,
        }
        (
            self.client.table(self.tables.chat_sessions)
            .upsert(payload, on_conflict="id")
            .execute()
        )
        return final_id

    def save_message(self, session_id: str, role: str, content: str) -> str:
        """Persist a single chat message; returns message id."""
        if role not in {"user", "assistant", "system"}:
            raise ValueError("role must be 'user', 'assistant', or 'system'")

        message_id = str(uuid.uuid4())
        payload = {
            "id": message_id,
            "session_id": session_id,
            "role": role,
            "content": content,
        }
        (self.client.table(self.tables.chat_messages).insert(payload).execute())

        # Touch session updated_at
        (
            self.client.table(self.tables.chat_sessions)
            .upsert(
                {"id": session_id, "updated_at": datetime.utcnow().isoformat()},
                on_conflict="id",
            )
            .execute()
        )

        return message_id

    def get_chat_history(
        self, session_id: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Fetch chat history for a given session, ordered by created_at ascending."""
        res = (
            self.client.table(self.tables.chat_messages)
            .select("id, role, content, created_at")
            .eq("session_id", session_id)
            .order("created_at", desc=False)
            .limit(limit)
            .execute()
        )
        # The supabase client returns an object; duck-type data attr
        data_any = getattr(res, "data", res)
        # Best-effort typing for mypy: coerce to expected list of dicts
        data_list = cast("list[dict[str, Any]] | None", data_any)
        return data_list or []

    # ------------------------ Permanent memories ------------------------
    def upsert_permanent_memory(
        self,
        content: str,
        tags: list[str] | None = None,
        source: str | None = None,
        memory_id: str | None = None,
    ) -> str:
        """Create or update a permanent memory; returns memory id."""
        final_id = memory_id or str(uuid.uuid4())
        payload = {
            "id": final_id,
            "content": content,
            "tags": tags or [],
            "source": source or "manual",
        }
        (
            self.client.table(self.tables.permanent_memories)
            .upsert(payload, on_conflict="id")
            .execute()
        )
        return final_id

    def list_permanent_memories(
        self,
        tags: list[str] | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List permanent memories for a user, optionally filtered by tags.

        Filtering is best-effort; if the client lacks JSONB helpers, we fall back
        to returning the most recent items without tag filtering.
        """
        q = (
            self.client.table(self.tables.permanent_memories)
            .select("id, content, tags, source, created_at")
            .order("created_at", desc=True)
            .limit(limit)
        )
        # Attempt tag filter if provided
        if tags:
            # Supabase Python client supports .contains for JSONB arrays
            with contextlib.suppress(Exception):
                q = q.contains("tags", tags)

        res = q.execute()
        data_any = getattr(res, "data", res)
        data_list = cast("list[dict[str, Any]] | None", data_any)
        return data_list or []

    def delete_permanent_memory(self, memory_id: str) -> bool:
        """Delete a permanent memory by ID. Returns True if deletion attempted."""
        try:
            (
                self.client.table(self.tables.permanent_memories)
                .delete()
                .eq("id", memory_id)
                .execute()
            )
            return True
        except Exception:
            return False
