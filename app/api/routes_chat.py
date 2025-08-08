from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.chains.intelligent_qa_chain import IntelligentQAChain
from app.core.chains.qa_chain import IntegratedQAChain
from app.core.embeddings import get_embeddings
from app.core.memory import MemoryManager
from app.core.config import Settings


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


class ChatChoice(BaseModel):
    index: int
    message: dict[str, Any]
    finish_reason: str = "stop"


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    choices: list[ChatChoice]
    model: str = "gpt-4.1"


def _settings_for_request(use_test_index: bool) -> Settings:
    if use_test_index:
        return Settings.for_testing()
    return Settings()


def get_memory_manager(cfg: Settings | None = None) -> MemoryManager:
    embeddings = get_embeddings()
    return MemoryManager(embeddings=embeddings, use_time_weighting=True, cfg=cfg)


@router.post("/chat")
def chat(
    req: ChatRequest,
) -> dict[str, Any]:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    cfg = _settings_for_request(req.use_test_index)
    memory = get_memory_manager(cfg)

    if req.intelligent:
        chain = IntelligentQAChain(memory, k=req.k, name=req.name)
        result = chain.invoke(
            {
                "question": req.question,
                "conversation_history": req.conversation_history or "",
            }
        )
        return result
    else:
        chain = IntegratedQAChain(memory, k=req.k, name=req.name)
        return chain.invoke({"question": req.question})


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

    cfg = _settings_for_request(False)
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


class UpsertMemoriesRequest(BaseModel):
    items: list[UpsertMemoryItem]


@router.post("/memories/upsert")
def upsert_memories(
    req: UpsertMemoriesRequest, memory: MemoryManager = Depends(get_memory_manager)
) -> dict[str, Any]:
    # Convert to chunk dicts as expected by MemoryManager.add_chunks
    chunks: list[dict[str, Any]] = []
    for item in req.items:
        chunk = {
            "id": item.id,
            "content": item.content,
            "source": item.source or "api",
            "created_at": item.created_at,
            "type": item.type or "document",
        }
        chunks.append(chunk)

    memory.add_chunks(chunks)
    return {"inserted": len(chunks)}
