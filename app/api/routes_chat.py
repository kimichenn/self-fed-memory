from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter
from fastapi import Header
from fastapi import HTTPException
from pydantic import BaseModel
from pydantic import Field

from app.core.chains.intelligent_qa_chain import IntelligentQAChain
from app.core.chains.qa_chain import IntegratedQAChain
from app.core.config import Settings
from app.core.embeddings import get_embeddings
from app.core.knowledge_store import SupabaseKnowledgeStore
from app.core.memory import MemoryManager
from app.core.memory_router import MemoryRouter

router = APIRouter()


class ChatRequest(BaseModel):
    question: str = Field(..., description="User question or message")
    name: str = Field(default="User", description="User name for personalization")
    k: int = Field(default=5, ge=1, le=15, description="Number of memories to use")
    intelligent: bool = Field(
        default=True, description="Use intelligent chain with preferences"
    )
    conversation_history: str | None = Field(
        default=None, description="Optional raw conversation transcript"
    )
    use_test_index: bool = Field(
        default=False,
        description="Use test Pinecone index/namespace instead of production",
    )
    use_test_supabase: bool = Field(
        default=False,
        description="Use test Supabase tables (prefix) instead of production",
    )
    # Optional Supabase persistence fields
    store_chat: bool = Field(
        default=False, description="Persist chat to Supabase (sessions/messages)"
    )
    session_id: str | None = Field(
        default=None, description="Existing chat session UUID"
    )


class ChatChoice(BaseModel):
    index: int
    message: dict[str, Any]
    finish_reason: str = "stop"


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    choices: list[ChatChoice]
    model: str = "gpt-4.1"


def _settings_for_request(
    use_test_index: bool, use_test_supabase: bool = False
) -> Settings:
    s = Settings.for_testing() if use_test_index else Settings()
    # Apply test Supabase prefix only when requested
    if use_test_supabase and s.test_supabase_table_prefix:
        s.supabase_table_prefix = s.test_supabase_table_prefix
    return s


def _default_session_id(cfg: Settings) -> str:
    """Return the single-user default session id.

    If ``SUPABASE_DEFAULT_SESSION_ID`` is set, use it; otherwise a stable
    built-in UUID is returned. This enables single-user deployments to
    always resume the same chat history without passing a session id.
    """
    if cfg.supabase_default_session_id:
        return cfg.supabase_default_session_id
    return "00000000-0000-0000-0000-000000000001"


def get_memory_manager(cfg: Settings | None = None) -> MemoryManager:
    embeddings = get_embeddings()
    return MemoryManager(embeddings=embeddings, use_time_weighting=True, cfg=cfg)


def get_supabase_store(cfg: Settings) -> SupabaseKnowledgeStore | None:
    """Create a SupabaseKnowledgeStore if configured, else return None.

    We avoid raising on missing config so regular non-persistent flows keep working.
    """
    try:
        # During automated tests, treat Supabase as unavailable to keep tests
        # hermetic and independent of developer/local env configuration.
        if os.environ.get("PYTEST_CURRENT_TEST"):
            return None
        if not cfg.supabase_url or not cfg.supabase_key:
            return None
        # Build table names with optional prefix (for test/production separation)
        tables = None
        if cfg.supabase_table_prefix:
            from app.core.knowledge_store import SupabaseTables

            tables = SupabaseTables.with_prefix(cfg.supabase_table_prefix)
        return SupabaseKnowledgeStore(cfg=cfg, tables=tables)
    except Exception:
        # If Supabase is not available or misconfigured, proceed without persistence
        return None


def _require_api_key_if_configured(cfg: Settings, x_api_key: str | None) -> None:
    """Require x-api-key header if API auth key is configured.

    If no api_auth_key is set in configuration, this is a no-op.
    """
    if cfg.api_auth_key and x_api_key != cfg.api_auth_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@router.post("/chat")
def chat(
    req: ChatRequest,
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
) -> dict[str, Any]:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    cfg = _settings_for_request(req.use_test_index, req.use_test_supabase)
    memory = get_memory_manager(cfg)
    supa = get_supabase_store(cfg)

    # Optional: persist session and the user message (single-user)
    # Default to single-user session when persisting and no session id provided
    session_id: str | None = req.session_id
    if req.store_chat and supa is not None:
        _require_api_key_if_configured(cfg, x_api_key)
        # Use a short title derived from question if no title exists
        title = (req.question[:60] + "…") if len(req.question) > 60 else req.question
        session_id = supa.ensure_session(
            session_id=session_id or _default_session_id(cfg), title=title
        )
        supa.save_message(session_id=session_id, role="user", content=req.question)

    if req.intelligent:
        chain: IntelligentQAChain | IntegratedQAChain
        chain = IntelligentQAChain(memory, k=req.k, name=req.name)
        result = chain.invoke(
            {
                "question": req.question,
                "conversation_history": req.conversation_history or "",
            }
        )
        # Persist assistant message if requested
        if req.store_chat and supa is not None and session_id:
            _require_api_key_if_configured(cfg, x_api_key)
            supa.save_message(
                session_id=session_id,
                role="assistant",
                content=result.get("answer", ""),
            )
        # Route extracted preferences/facts from conversation into Supabase+Pinecone
        # when persistence is enabled
        if req.store_chat and supa is not None:
            from app.core.memory_router import MemoryRouter

            router = MemoryRouter(memory_manager=memory, supabase_store=supa)
            extraction = result.get("extraction_results") or {}
            items = extraction.get("items") or []
            if items:
                router.upsert_items(items)
        if session_id:
            result["session_id"] = session_id
        return result
    else:
        chain = IntegratedQAChain(memory, k=req.k, name=req.name)
        basic = chain.invoke({"question": req.question})
        if req.store_chat and supa is not None and session_id:
            _require_api_key_if_configured(cfg, x_api_key)
            supa.save_message(
                session_id=session_id, role="assistant", content=basic.get("answer", "")
            )
        if session_id:
            basic["session_id"] = session_id
        return basic


@router.get("/v1/models")
def list_models() -> dict[str, Any]:
    return {"data": [{"id": "gpt-4.1"}, {"id": "gpt-4o"}]}


class OpenAIChatMessage(BaseModel):
    role: str
    content: str


class OpenAIChatRequest(BaseModel):
    model: str
    messages: list[OpenAIChatMessage]
    temperature: float | None = None
    top_p: float | None = None


@router.post("/v1/chat/completions")
def openai_compatible_chat(
    req: OpenAIChatRequest,
) -> ChatResponse:
    # Extract last user question
    question = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
    if not question:
        raise HTTPException(status_code=400, detail="No user message provided")

    cfg = _settings_for_request(False, False)
    memory = get_memory_manager(cfg)
    chain = IntelligentQAChain(memory, k=5, name="User")
    result = chain.invoke({"question": question})

    return ChatResponse(
        id="cmpl-self-memory-1",
        choices=[
            ChatChoice(
                index=0, message={"role": "assistant", "content": result["answer"]}
            )
        ],
    )


class UpsertMemoryItem(BaseModel):
    id: str
    content: str
    source: str | None = None
    created_at: str | None = None
    type: str | None = None
    category: str | None = None
    route_to_vector: bool | None = None


class UpsertMemoriesRequest(BaseModel):
    items: list[UpsertMemoryItem]
    use_test_index: bool = False
    use_test_supabase: bool = False


@router.post("/memories/upsert")
def upsert_memories(
    req: UpsertMemoriesRequest,
) -> dict[str, Any]:
    cfg = _settings_for_request(req.use_test_index, req.use_test_supabase)
    memory = get_memory_manager(cfg)
    supa = get_supabase_store(cfg)
    router = MemoryRouter(memory_manager=memory, supabase_store=supa)

    items: list[dict[str, Any]] = []
    for item in req.items:
        items.append(
            {
                "id": item.id,
                "content": item.content,
                "source": item.source or "api",
                "created_at": item.created_at,
                "type": item.type or "document",
                **({"category": item.category} if item.category else {}),
                **(
                    {"route_to_vector": item.route_to_vector}
                    if item.route_to_vector is not None
                    else {}
                ),
            }
        )

    summary = router.upsert_items(items)
    return {"inserted": len(items), "routing": summary}


class DeleteMemoriesRequest(BaseModel):
    ids: list[str]
    use_test_index: bool = False
    use_test_supabase: bool = False
    target: str | None = Field(
        default=None, description='"pinecone", "supabase", or null for both'
    )


@router.post("/memories/delete")
def delete_memories(req: DeleteMemoriesRequest) -> dict[str, Any]:
    cfg = _settings_for_request(req.use_test_index, req.use_test_supabase)
    memory = get_memory_manager(cfg)
    supa = get_supabase_store(cfg)
    router = MemoryRouter(memory_manager=memory, supabase_store=supa)
    summary = router.delete_items(req.ids, target=req.target)
    return {"deleted": req.ids, "routing": summary}


@router.get("/memories/search")
def search_memories(
    query: str,
    k: int = 5,
    use_test_index: bool = False,
    use_test_supabase: bool = False,
) -> dict[str, Any]:
    cfg = _settings_for_request(use_test_index, use_test_supabase)
    memory = get_memory_manager(cfg)
    supa = get_supabase_store(cfg)
    router = MemoryRouter(memory_manager=memory, supabase_store=supa)
    return router.search(query, k=k)


class PermanentMemoryUpsertRequest(BaseModel):
    content: str = Field(..., description="Permanent memory content")
    tags: list[str] | None = Field(default=None, description="Optional tags")
    source: str | None = Field(default=None, description="Source descriptor")
    id: str | None = Field(
        default=None, description="Optional explicit memory id (UUID)"
    )
    use_test_supabase: bool = False


@router.post("/permanent_memories/upsert")
def upsert_permanent_memory(
    req: PermanentMemoryUpsertRequest,
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
) -> dict[str, Any]:
    cfg = _settings_for_request(False, req.use_test_supabase)
    _require_api_key_if_configured(cfg, x_api_key)
    supa = get_supabase_store(cfg)
    if supa is None:
        raise HTTPException(status_code=400, detail="Supabase is not configured")

    memory_id = supa.upsert_permanent_memory(
        content=req.content,
        tags=req.tags,
        source=req.source,
        memory_id=req.id,
    )
    return {"id": memory_id}


@router.get("/chat/history")
def get_chat_history(
    session_id: str | None = None,
    limit: int = 100,
    use_test_supabase: bool = False,
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
) -> dict[str, Any]:
    cfg = _settings_for_request(False, use_test_supabase)
    _require_api_key_if_configured(cfg, x_api_key)
    supa = get_supabase_store(cfg)
    if supa is None:
        raise HTTPException(status_code=400, detail="Supabase is not configured")
    sid = session_id or _default_session_id(cfg)
    history = supa.get_chat_history(session_id=sid, limit=limit)
    return {"session_id": sid, "messages": history}
