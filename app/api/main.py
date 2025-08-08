from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes_chat import router as chat_router


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
    def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(chat_router)
    return app


app = create_app()
