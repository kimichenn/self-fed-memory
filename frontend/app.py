from __future__ import annotations

from contextlib import suppress
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Literal, TypedDict, cast
import uuid

# Ensure PyYAML uses the pure-Python implementation to avoid platform-specific C-extension issues
os.environ.setdefault("YAML_CEXT_DISABLED", "1")
with suppress(Exception):
    import yaml as _yaml  # noqa: F401

import requests
import streamlit as st


def _load_parse_markdown_file():
    """Load the shared markdown parser while avoiding the 'app' name collision.

    When this file is named app.py, absolute import 'app.ingestion...' can fail
    with 'app is not a package'. We first try normal import; if it fails,
    we ensure the repo root is on sys.path and remove any non-package 'app'
    module from sys.modules, then retry. Finally, fall back to loading by file path.
    """
    # 1) Try the regular package import first
    try:
        from app.ingestion.markdown_loader import parse_markdown_file as fn

        return fn
    except Exception:
        pass

    # 2) Ensure proper import environment and retry
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # If a non-package module named 'app' was loaded (e.g., this frontend/app.py),
    # remove it so the real package at repo_root/app/ can be imported.
    mod_app = sys.modules.get("app")
    if mod_app is not None and not hasattr(mod_app, "__path__"):
        del sys.modules["app"]

    try:
        from app.ingestion.markdown_loader import parse_markdown_file as fn

        return fn
    except Exception as err:
        # 3) Fallback: load by absolute file path (avoids 'app' name collision)
        loader_path = repo_root / "app" / "ingestion" / "markdown_loader.py"
        spec = importlib.util.spec_from_file_location(
            "self_memory_markdown_loader", str(loader_path)
        )
        if spec is None or spec.loader is None:
            raise ImportError("Unable to load markdown_loader module") from err
        module = importlib.util.module_from_spec(spec)
        # Make sure absolute imports inside the module work
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module.parse_markdown_file


# Lazy-load the shared markdown parser to avoid any startup import issues
_parse_markdown_file_cached = None


def get_parse_markdown_file():
    global _parse_markdown_file_cached
    if _parse_markdown_file_cached is None:
        _parse_markdown_file_cached = _load_parse_markdown_file()
    return _parse_markdown_file_cached


# --- Page config
st.set_page_config(page_title="Self-Fed Memory", page_icon="🧠", layout="wide")


# --- Defaults & session state
API_BASE_DEFAULT = os.environ.get("SELF_MEMORY_API", "http://localhost:8000")


def _probe_backend_health(base: str) -> dict[str, Any] | None:
    try:
        r = requests.get(f"{base}/health", timeout=3)
        if r.status_code == 200:
            data = cast(dict[str, Any], r.json())
            return data
    except Exception:
        return None
    return None


# Settings defaults
st.session_state.setdefault("api_base", API_BASE_DEFAULT)
st.session_state.setdefault("backend_health", _probe_backend_health(API_BASE_DEFAULT))
st.session_state.setdefault("intelligent", True)
st.session_state.setdefault("k", 5)
st.session_state.setdefault(
    "dev_mode", False
)  # single toggle for test Pinecone + test Supabase
st.session_state.setdefault("store_chat", True)
st.session_state.setdefault("api_auth_key", os.environ.get("API_AUTH_KEY", ""))
st.session_state.setdefault("show_memory_inline", False)
st.session_state.setdefault("show_settings_inline", False)


class ChatMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class Conversation(TypedDict, total=False):
    key: str
    title: str
    history: list[ChatMessage]
    session_id: str
    remote_loaded: bool
    dev_mode: bool


# Conversations state
def _new_conversation(
    title: str = "New chat", generate_session_id: bool = False
) -> Conversation:
    return {
        "key": f"chat-{uuid.uuid4().hex[:8]}",
        "title": title,
        "history": [],
        # For a brand-new chat created by the user, generate a unique backend session id
        # so it does not load the default single-user history. For the very first
        # conversation on app load, leave this empty so we can resume the default session.
        "session_id": (
            str(uuid.uuid4())
            if (generate_session_id and st.session_state.get("store_chat"))
            else ""
        ),
        "remote_loaded": False,
        "dev_mode": cast(bool, st.session_state.get("dev_mode", False)),
    }


st.session_state.setdefault("conversations", {})
st.session_state.setdefault("current_conv_key", "")

if not st.session_state["conversations"]:
    conv = _new_conversation()
    st.session_state.conversations[conv["key"]] = conv
    st.session_state.current_conv_key = conv["key"]


def _get_current_conversation() -> Conversation:
    key = cast(str, st.session_state.get("current_conv_key", ""))
    existing = st.session_state.conversations.get(key)
    if existing is None:
        # Fallback: create one
        conv = _new_conversation()
        st.session_state.conversations[conv["key"]] = conv
        st.session_state.current_conv_key = conv["key"]
    else:
        conv = cast(Conversation, existing)
    # Backfill missing keys for older conversations
    if "dev_mode" not in conv:
        conv["dev_mode"] = cast(bool, st.session_state.get("dev_mode", False))
    return conv


# --- Helpers
def call_chat_api(
    question: str,
    conversation_history: str | None,
    session_id: str | None,
    dev_mode: bool,
) -> dict[str, Any]:
    url = f"{st.session_state.api_base}/chat"
    payload = {
        "question": question,
        "k": st.session_state.k,
        "intelligent": st.session_state.intelligent,
        "conversation_history": conversation_history or "",
        "use_test_index": bool(dev_mode),
        "use_test_supabase": bool(dev_mode),
        "store_chat": st.session_state.store_chat,
        "session_id": session_id or None,
    }
    headers = {}
    if st.session_state.api_auth_key:
        headers["x-api-key"] = st.session_state.api_auth_key
    resp = requests.post(url, json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    data = cast(dict[str, Any], resp.json())
    return data


def _shorten_title(text: str, max_len: int = 50) -> str:
    first_line = (text or "").strip().splitlines()[0]
    return (first_line[:max_len] + "...") if len(first_line) > max_len else first_line


def render_settings_body():
    tabs = st.tabs(["General", "Backend", "Persistence", "Advanced"])

    with tabs[0]:
        st.toggle(
            "Intelligent mode",
            value=st.session_state.intelligent,
            key="intelligent",
            help=(
                "Uses preference-aware, multi-query search with time-weighted reranking. "
                "Also auto-extracts new preferences/facts from chats and routes them to storage when persistence is enabled."
            ),
        )
        st.slider(
            "Memories to retrieve (k)",
            min_value=1,
            max_value=15,
            value=int(st.session_state.k),
            key="k",
        )

    with tabs[1]:
        st.text_input(
            "API base URL",
            value=st.session_state.api_base,
            key="api_base",
            help="Base URL for the backend API (defaults from SELF_MEMORY_API env).",
        )
        health = _probe_backend_health(st.session_state.api_base)
        if health:
            ok = health.get("status") == "ok"
            sb = health.get("supabase_configured")
            auth = health.get("api_auth_required")
            st.caption(
                f"Backend health: {'OK' if ok else 'ERROR'} • Supabase: {'ON' if sb else 'OFF'} • API auth: {'ON' if auth else 'OFF'}"
            )
        st.text_input(
            "API auth key (x-api-key)",
            value=st.session_state.api_auth_key,
            key="api_auth_key",
            type="password",
            help="Optional; required if backend sets API_AUTH_KEY for protected endpoints.",
        )
        st.toggle(
            "Developer mode (use TEST Pinecone + TEST Supabase)",
            value=st.session_state.dev_mode,
            key="dev_mode",
            help="When enabled, all requests use the test Pinecone index/namespace and test Supabase tables.",
        )

    with tabs[2]:
        bc = st.session_state.get("backend_health") or _probe_backend_health(
            st.session_state.api_base
        )
        supa_ok = bool(bc and bc.get("supabase_configured"))
        st.toggle(
            "Store chat history (Supabase)",
            value=st.session_state.store_chat,
            key="store_chat",
            disabled=not supa_ok,
            help=(
                "When enabled and Supabase is configured, chat sessions/messages are persisted. "
                "Core items extracted from conversations are routed to Supabase permanent memories."
            ),
        )
        if not supa_ok:
            st.caption(
                "Supabase not configured on backend. Set SUPABASE_URL and SUPABASE_KEY in .env."
            )
        # Single-user design: no explicit Session ID needed

    with tabs[3]:
        st.caption("No advanced options yet.")

    st.divider()
    cols = st.columns([1, 1, 2])
    with cols[0]:
        if st.button("Done", type="primary", key="settings_done_btn"):
            st.rerun()
    with cols[1]:
        if st.button("Reset to defaults", type="secondary", key="settings_reset_btn"):
            st.session_state.api_base = API_BASE_DEFAULT
            st.session_state.intelligent = True
            st.session_state.k = 5
            st.session_state.dev_mode = False
            st.session_state.store_chat = True
            st.session_state.session_id = ""
            st.session_state.api_auth_key = os.environ.get("API_AUTH_KEY", "")
            st.session_state.remote_loaded = False
            st.rerun()


def _load_remote_history(
    session_id: str | None, dev_mode: bool
) -> tuple[str | None, list[ChatMessage]]:
    if not st.session_state.store_chat:
        return (None, [])
    try:
        headers = {}
        if st.session_state.api_auth_key:
            headers["x-api-key"] = st.session_state.api_auth_key
        params: dict[str, object] = {
            "limit": 100,
            "use_test_supabase": bool(dev_mode),
        }
        if session_id:
            params["session_id"] = session_id
        r = requests.get(
            f"{st.session_state.api_base}/chat/history",
            params=params,
            headers=headers,
            timeout=5,
        )
        r.raise_for_status()
        data = r.json()
        sid = data.get("session_id")
        msgs = data.get("messages") or []
        converted: list[ChatMessage] = []
        for m in msgs:
            role = m.get("role", "user")
            if role not in ("user", "assistant"):
                continue
            # At this point role is guaranteed to be one of the expected literals
            role_literal = cast(Literal["user", "assistant"], role)
            converted.append({"role": role_literal, "content": m.get("content", "")})
        return (sid, converted)
    except Exception:
        return (None, [])


def _ensure_supabase_available() -> None:
    # Only probe once per session
    if st.session_state.get("supabase_available") is not None:
        return
    try:
        headers = {}
        if st.session_state.api_auth_key:
            headers["x-api-key"] = st.session_state.api_auth_key
        params: dict[str, object] = {
            "limit": 1,
            "use_test_supabase": bool(st.session_state.get("dev_mode", False)),
        }
        r = requests.get(
            f"{st.session_state.api_base}/chat/history",
            params=params,
            headers=headers,
            timeout=3,
        )
        if r.status_code == 200:
            st.session_state.supabase_available = True
        else:
            # Treat 400 "Supabase is not configured" as unavailable; other errors as unavailable
            # 401 may indicate API auth required; Supabase still available, but history will be blocked without a key
            try:
                msg = r.json().get("detail")
            except Exception:
                msg = None
            if r.status_code == 400 and msg == "Supabase is not configured":
                st.session_state.supabase_available = False
            else:
                st.session_state.supabase_available = False
    except Exception:
        st.session_state.supabase_available = False


def render_memory_manager_body(in_dialog: bool = False):
    with st.expander("Upload Markdown Files", expanded=True):
        uploads = st.file_uploader(
            "Upload one or more Markdown files (.md)",
            type=["md", "markdown"],
            accept_multiple_files=True,
            key="md_uploads",
            help=(
                "Files are parsed with the same markdown loader used by the backend: "
                "front-matter and filename dates are respected; documents are chunked for retrieval."
            ),
        )
        colu1, colu2 = st.columns([1, 1])
        with colu1:
            upload_route_to_vector = st.toggle(
                "Index for retrieval (Pinecone)",
                value=True,
                key="upload_route_to_vector",
                help=(
                    "Adds uploaded chunks to the vector store for semantic search. "
                    "Core items are additionally stored in Supabase when configured."
                ),
            )
        with colu2:
            upload_type = st.selectbox(
                "Type for uploaded chunks",
                ["document", "preference", "fact", "profile"],
                index=0,
                key="upload_mem_type",
                help=(
                    "Controls routing: core types (preference/fact/profile) are also written to Supabase permanent memories when available."
                ),
            )
        if st.button(
            "Ingest Files", type="primary", disabled=not uploads, key="mem_ingest_btn"
        ):
            total_items = 0
            try:
                batched: list[dict] = []
                for uf in uploads:
                    # Persist to a temp file to reuse the canonical parser
                    suffix = (
                        ".md"
                        if uf.name and not str(uf.name).lower().endswith(".markdown")
                        else ".markdown"
                    )
                    with tempfile.NamedTemporaryFile(
                        "wb", suffix=suffix, delete=False
                    ) as tmp:
                        tmp.write(uf.read())
                        tmp_path = tmp.name
                    # Parse and map to API items (lazy-load the parser)
                    for chunk in get_parse_markdown_file()(tmp_path):
                        item = {
                            "id": chunk.get("id", ""),
                            "content": chunk.get("content", ""),
                            "source": chunk.get("source") or uf.name,
                            "created_at": chunk.get("created_at"),
                            "type": upload_type,
                            "route_to_vector": upload_route_to_vector,
                        }
                        # Preserve optional front‑matter category if present
                        if isinstance(chunk.get("category"), str) and chunk.get(
                            "category"
                        ):
                            item["category"] = chunk["category"]
                        batched.append(item)
                    # Cleanup temp file
                    from contextlib import suppress

                    with suppress(Exception):
                        Path(tmp_path).unlink(missing_ok=True)
                if batched:
                    payload = {
                        "items": batched,
                        "use_test_index": bool(st.session_state.dev_mode),
                        "use_test_supabase": bool(st.session_state.dev_mode),
                    }
                    r = requests.post(
                        f"{st.session_state.api_base}/memories/upsert",
                        json=payload,
                        timeout=20,
                    )
                    r.raise_for_status()
                    total_items = len(batched)
                    st.success(
                        f"Ingested {total_items} chunks from {len(uploads)} file(s). Routing: {r.json().get('routing')}"
                    )
                else:
                    st.info("No content found to ingest.")
            except Exception as e:
                st.error(f"Upload failed: {e}")

    with st.expander("Add/Update Memories", expanded=True):
        mem_text = st.text_area(
            "Content",
            height=140,
            placeholder="Write a fact, preference, or note…",
            key="mem_text",
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            mem_type = st.selectbox(
                "Type",
                ["document", "preference", "fact", "profile"],
                index=0,
                key="mem_type",
            )
        with col2:
            mem_category = st.text_input("Category", value="", key="mem_category")
        with col3:
            route_to_vector = st.toggle(
                "Index for retrieval (Pinecone)",
                value=True,
                key="mem_route_to_vector",
                help=(
                    "Adds this item to the vector store for semantic search. "
                    "Core types (preference/fact/profile) are also stored in Supabase when configured."
                ),
            )

        mem_id = st.text_input("ID (optional)", value="", key="mem_id")
        mem_source = st.text_input("Source", value="ui", key="mem_source")
        created_at = st.text_input("Created at (ISO)", value="", key="mem_created_at")

        if st.button("Upsert Memory", type="primary", key="mem_upsert_btn"):
            try:
                payload = {
                    "items": [
                        {
                            "id": mem_id or "",
                            "content": mem_text,
                            "source": mem_source,
                            "created_at": created_at or None,
                            "type": mem_type,
                            "category": mem_category or None,
                            "route_to_vector": route_to_vector,
                        }
                    ],
                    "use_test_index": bool(st.session_state.dev_mode),
                    "use_test_supabase": bool(st.session_state.dev_mode),
                }
                r = requests.post(
                    f"{st.session_state.api_base}/memories/upsert",
                    json=payload,
                    timeout=15,
                )
                r.raise_for_status()
                st.success(f"Upserted. Routing: {r.json().get('routing')}")
            except Exception as e:
                st.error(f"Error: {e}")

    with st.expander("Search Memories"):
        q = st.text_input("Query", value="", key="mem_search_q")
        kk = st.slider("k", 1, 20, 5, key="mem_search_k")
        if st.button("Search", key="mem_search_btn"):
            try:
                params = {
                    "query": q,
                    "k": kk,
                    "use_test_index": bool(st.session_state.dev_mode),
                    "use_test_supabase": bool(st.session_state.dev_mode),
                }
                r = requests.get(
                    f"{st.session_state.api_base}/memories/search",
                    params=params,
                    timeout=10,
                )
                r.raise_for_status()
                data = r.json()
                st.write("Results:")
                st.json(data)
            except Exception as e:
                st.error(f"Error: {e}")

    with st.expander("Delete Memories"):
        ids_csv = st.text_input("IDs (comma-separated)", value="", key="mem_delete_ids")
        target = st.selectbox(
            "Target", ["both", "pinecone", "supabase"], index=0, key="mem_delete_target"
        )
        col_del_ids, col_del_all = st.columns([1, 1])
        with col_del_ids:
            if st.button("Delete by IDs", type="secondary", key="mem_delete_btn"):
                try:
                    ids = [s.strip() for s in ids_csv.split(",") if s.strip()]
                    if not ids:
                        st.warning("Provide at least one ID to delete.")
                    else:
                        payload = {
                            "ids": ids,
                            "use_test_index": bool(st.session_state.dev_mode),
                            "use_test_supabase": bool(st.session_state.dev_mode),
                            "target": None if target == "both" else target,
                        }
                        r = requests.post(
                            f"{st.session_state.api_base}/memories/delete",
                            json=payload,
                            timeout=10,
                        )
                        r.raise_for_status()
                        st.success(
                            f"Deleted: {len(ids)}. Routing: {r.json().get('routing')}"
                        )
                except Exception as e:
                    st.error(f"Error: {e}")
        with col_del_all:
            if st.button(
                "Delete ALL (with confirm)", type="secondary", key="mem_delete_all_btn"
            ):
                # Show confirmation modal/dialog
                def _confirm_delete_all_body():
                    st.warning(
                        "This will permanently delete ALL memories in the selected target(s)."
                    )
                    st.write(
                        f"Mode: {'DEV (test Pinecone + test Supabase)' if st.session_state.dev_mode else 'PROD'}"
                    )
                    target_choice = st.selectbox(
                        "Target to wipe",
                        ["both", "pinecone", "supabase"],
                        index=["both", "pinecone", "supabase"].index(target),
                        key="mem_delete_all_target_confirm",
                    )
                    cols = st.columns([1, 1])
                    with cols[0]:
                        if st.button(
                            "Confirm delete ALL",
                            type="primary",
                            key="confirm_delete_all_yes",
                        ):
                            try:
                                payload = {
                                    "use_test_index": bool(st.session_state.dev_mode),
                                    "use_test_supabase": bool(
                                        st.session_state.dev_mode
                                    ),
                                    "target": (
                                        None
                                        if target_choice == "both"
                                        else target_choice
                                    ),
                                }
                                r = requests.post(
                                    f"{st.session_state.api_base}/memories/delete_all",
                                    json=payload,
                                    timeout=20,
                                )
                                r.raise_for_status()
                                st.success(
                                    f"Wipe complete. Routing: {r.json().get('routing')}"
                                )
                            except Exception as e:
                                st.error(f"Error: {e}")
                    with cols[1]:
                        st.button("Cancel", key="confirm_delete_all_no")

                if dialog_decorator and not in_dialog:
                    # Use modal dialog for confirmation
                    @dialog_decorator("Confirm delete ALL memories")
                    def _confirm_delete_all_dialog():
                        _confirm_delete_all_body()

                    _confirm_delete_all_dialog()
                else:
                    with st.expander("Confirm delete ALL", expanded=True):
                        _confirm_delete_all_body()


# --- Dialogs (use modal if available; fallback to inline display)
# Prefer the stable API; fall back to experimental if needed
HAS_DIALOG = hasattr(st, "dialog") or hasattr(st, "experimental_dialog")

dialog_decorator = getattr(st, "dialog", None) or getattr(
    st, "experimental_dialog", None
)

if dialog_decorator:

    @dialog_decorator("Memory Manager")
    def memory_manager_dialog():
        render_memory_manager_body(in_dialog=True)

    @dialog_decorator("Settings")
    def settings_dialog():
        render_settings_body()


# --- Header & actions
left, right = st.columns([3, 2])
with left:
    st.title("🧠 Self-Fed Memory")
    st.caption("Personal AI with long-term memory")
with right:
    b1, b2 = st.columns(2)
    with b1:
        if st.button(
            "🧠 Memory Manager", use_container_width=True, key="open_memory_manager_btn"
        ):
            if HAS_DIALOG:
                memory_manager_dialog()
            else:
                st.session_state.show_memory_inline = (
                    not st.session_state.show_memory_inline
                )
    with b2:
        if st.button("⚙ Settings", use_container_width=True, key="open_settings_btn"):
            if HAS_DIALOG:
                settings_dialog()
            else:
                st.session_state.show_settings_inline = (
                    not st.session_state.show_settings_inline
                )


# --- Sync global dev_mode to current conversation when toggled
st.session_state.setdefault("last_dev_mode", st.session_state.dev_mode)
if st.session_state.last_dev_mode != st.session_state.dev_mode:
    curr_sync = _get_current_conversation()
    curr_sync["dev_mode"] = st.session_state.dev_mode
    st.session_state.last_dev_mode = st.session_state.dev_mode


# --- Sidebar: Conversations
with st.sidebar:
    st.header("Conversations")

    # Build a stable list of (key, title)
    conv_items = list(st.session_state.conversations.items())
    conv_keys = [key for key, _ in conv_items]
    conv_titles = [
        (
            f"🧪 DEV • {v.get('title') or key}"
            if v.get("dev_mode")
            else (v.get("title") or key)
        )
        for key, v in conv_items
    ]

    # Current index
    try:
        current_index = conv_keys.index(st.session_state.current_conv_key)
    except ValueError:
        current_index = 0

    selected_title = st.radio(
        "Select a chat",
        options=conv_titles,
        index=current_index,
        key="conv_radio",
    )

    # Update current key if selection changed
    selected_idx = conv_titles.index(selected_title) if conv_titles else 0
    st.session_state.current_conv_key = conv_keys[selected_idx]

    colc1, colc2 = st.columns(2)
    with colc1:
        if st.button("New chat", key="new_chat_btn", use_container_width=True):
            # Ensure a fresh backend session id so history does not copy from the default session
            conv = _new_conversation(generate_session_id=True)
            st.session_state.conversations[conv["key"]] = conv
            st.session_state.current_conv_key = conv["key"]
            st.rerun()
    with colc2:
        if st.button("Delete", key="delete_chat_btn", use_container_width=True):
            if len(st.session_state.conversations) > 1:
                del st.session_state.conversations[st.session_state.current_conv_key]
                # Select another existing conversation
                next_key = next(iter(st.session_state.conversations.keys()))
                st.session_state.current_conv_key = next_key
            else:
                # Reset to a fresh conversation if last one
                st.session_state.conversations = {}
                conv = _new_conversation()
                st.session_state.conversations[conv["key"]] = conv
                st.session_state.current_conv_key = conv["key"]
            st.rerun()

    # Actions for current conversation
    curr = _get_current_conversation()
    if st.button("Clear messages", key="clear_messages_btn"):
        curr["history"] = []
        st.rerun()

    # Detect Supabase availability once and only load remote history when available
    if st.session_state.store_chat:
        _ensure_supabase_available()
    if (
        st.session_state.store_chat
        and st.session_state.get("supabase_available") is True
        and not curr.get("remote_loaded")
    ):
        sid, msgs = _load_remote_history(
            curr.get("session_id"),
            curr.get("dev_mode", st.session_state.get("dev_mode", False)),
        )
        title_updated = False
        if msgs:
            curr["history"] = msgs
            # Set title to first user message if still default
            if not curr.get("title") or curr.get("title") == "New chat":
                first_user = next(
                    (m for m in msgs if m.get("role") == "user"),
                    None,
                )
                if first_user and isinstance(first_user.get("content"), str):
                    curr["title"] = _shorten_title(first_user["content"])
                    title_updated = True
        if sid:
            curr["session_id"] = sid
        curr["remote_loaded"] = True
        if title_updated:
            st.rerun()


# --- Inline fallbacks
if not HAS_DIALOG and st.session_state.show_settings_inline:
    with st.expander("Settings", expanded=True):
        render_settings_body()

if not HAS_DIALOG and st.session_state.show_memory_inline:
    st.subheader("Memory Manager")
    render_memory_manager_body()


# --- Chat area (per selected conversation)
curr = _get_current_conversation()

# Header badge for dev/prod per conversation
mode_badge = "🧪 DEV MODE" if curr.get("dev_mode") else "PROD"
st.caption(f"Mode: {mode_badge}")

# Display messages
for msg in curr["history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask anything…")
if prompt:
    # Append user message locally
    if not curr["history"]:
        curr["title"] = _shorten_title(prompt)
    curr["history"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build conversation text for preference extraction
    conversation_text = "\n".join(
        f"{m['role'].capitalize()}: {m['content']}" for m in curr["history"][-6:]
    )

    with st.chat_message("assistant"), st.spinner("Thinking..."):
        try:
            data = call_chat_api(
                prompt,
                conversation_text,
                session_id=(curr.get("session_id") or None),
                dev_mode=bool(
                    curr.get("dev_mode", st.session_state.get("dev_mode", False))
                ),
            )
            answer = data.get("answer") or data.get("result") or "(no answer)"
            # Persist returned session id to this conversation
            if data.get("session_id"):
                curr["session_id"] = data["session_id"]
        except Exception as e:
            answer = f"Error: {e}"
        st.markdown(answer)
    curr["history"].append({"role": "assistant", "content": answer})
