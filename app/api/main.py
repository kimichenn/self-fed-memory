from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes_chat import router as chat_router
from app.core.config import Settings
from app.core.knowledge_store import SupabaseKnowledgeStore


def create_app() -> FastAPI:
    app = FastAPI(title="Self-Fed Memory API", version="0.1.0")

    # CORS for local dev UIs (Streamlit, Open WebUI, etc.)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, object]:
        settings = Settings()
        supabase_configured = bool(settings.supabase_url and settings.supabase_key)
        api_auth_required = bool(settings.api_auth_key)

        supabase_can_query = False
        supabase_error: str | None = None
        if supabase_configured:
            try:
                supa = SupabaseKnowledgeStore(cfg=settings)
                # Lightweight probe: attempt to list 1 permanent memory
                supa.list_permanent_memories(limit=1)
                supabase_can_query = True
            except Exception as e:  # avoid leaking secrets; return brief reason
                supabase_can_query = False
                supabase_error = str(e)[:200]

        return {
            "status": "ok",
            "supabase_configured": supabase_configured,
            "supabase_can_query": supabase_can_query,
            "supabase_error": supabase_error,
            "api_auth_required": api_auth_required,
        }

    app.include_router(chat_router)
    return app


app = create_app()
