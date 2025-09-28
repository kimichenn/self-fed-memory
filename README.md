# Self-Fed Memory: Personal AI Assistant

A personalized AI assistant with long-term memory that learns from your notes, conversations, and experiences. Built with retrieval-augmented generation (RAG) to provide personalized responses grounded in your own knowledge base.

⚠️ Work in Progress: This repo is still under development. The UI and per-chat saves are partially implemented; expect bugs.

## Overview

This system ingests your personal notes and documents, converts them into semantic embeddings stored in a vector database, and uses GPT-4.1 to answer questions with full context of your life, preferences, and experiences. Think of it as "ChatGPT with perfect memory of who you are."

**Key Features:**

-   🧠 **Long-term Memory**: Remembers everything you tell it across sessions
-   🎯 **Intelligent Retrieval**: Advanced multi-query retrieval that understands context and implied questions
-   🤖 **Preference Learning**: Automatically extracts and applies your preferences from conversations
-   📝 **Note Ingestion**: Import Markdown files, journals, and personal documents
-   🔍 **Semantic Search**: Find relevant memories using meaning, not just keywords
-   ⏰ **Time-Aware**: Weights recent information higher while preserving old memories
-   💬 **Contextual Responses**: Tailored advice based on your preferences, habits, and history
-   🔄 **Continuous Learning**: Learns new facts and preferences from every conversation

## Architecture

```
📁 Personal Notes/Documents
    ↓ (Markdown Loader & Chunker)
🔢 Vector Embeddings (OpenAI)
    ↓ (Store in Vector DB)
🗄️ Pinecone Vector Store
    ↓ (Intelligent Multi-Query Retrieval)
🔍 Intelligent Retriever + Preference Tracker
    ↓ (Context-Aware RAG)
🤖 GPT-4.1 + Personal Context + User Preferences
    ↓ (Auto-Extract New Preferences)
💬 Personalized Response + Learning
```

## Quick Start

### 1. Prerequisites

-   Python 3.11+
-   OpenAI API key
-   Pinecone API key (for vector storage)

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/self-fed-memory.git
cd self-fed-memory

# Option A: Local venv (recommended for most users)
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[ui]"

# Option B: Conda (optional, if you prefer)
conda create -n self-mem python=3.11 -y
conda activate self-mem
pip install -e ".[ui]"

# Option C: Docker (no local Python needed)
docker build -t self-fed-memory:latest .

```

#### Pinned install (recommended)

-   Use the lockfile for fast, deterministic installs:

```bash
# with your venv/conda env activated
make dev           # installs from requirements-dev.txt and sets up git hooks
# or
pip install -r requirements-dev.txt
```

To update dependencies (and refresh the lock):

```bash
make lock-upgrade  # regenerates requirements-dev.txt from pyproject extras
```

### 3. Environment Setup

Create a `.env` file in the root directory:

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

```env
# Mandatory
OPENAI_API_KEY=sk-your-openai-key-here
PINECONE_API_KEY=pcn-your-pinecone-key-here

# Optional tracing and monitoring
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2-your-langsmith-key-here
LANGSMITH_PROJECT=self_memory

# Optional (with defaults)
PINECONE_ENV=us-east-1
PINECONE_INDEX=self-memory
EMBEDDING_MODEL=text-embedding-3-large

# Testing (recommended for development)
TEST_PINECONE_INDEX=self-memory-test
TEST_EMBEDDING_MODEL=text-embedding-3-large

# Optional: Supabase (for chat history, users, permanent memories)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-secret-or-service-role-key
```

### 4. Initialize Vector Store (automatic)

No manual step is required. The Pinecone index and namespace are created on first use if missing. Proceed directly to ingestion.

### 5. Ingest Your Personal Data

```bash
# Ingest a folder of Markdown files
python scripts/ingest_folder.py /path/to/your/notes
```

### 6. Start the System

#### Option A: Simple CLI Chat

```bash
python -c "
from app.core.embeddings import get_embeddings
from app.core.memory import MemoryManager
from app.core.chains.qa_chain import IntegratedQAChain
mem = MemoryManager(get_embeddings())
chain = IntegratedQAChain(mem, k=5, name='Your Name')
while True:
    q = input('You: ')
    if q.lower() in ['quit', 'exit']: break
    res = chain.invoke({'question': q})
    print('AI:', res['answer'])
"

# Or use the intelligent QA chain
python -c "
from app.core.embeddings import get_embeddings
from app.core.memory import MemoryManager
from app.core.chains.intelligent_qa_chain import IntelligentQAChain
mem = MemoryManager(get_embeddings())
chain = IntelligentQAChain(mem, name='Your Name', k=8)
while True:
    q = input('You: ')
    if q.lower() in ['quit', 'exit']: break
    res = chain.invoke({'question': q, 'conversation_history': ''})
    print('AI:', res['answer'])
"
```

#### Option B: Web Interface

```bash
# First ensure dependencies are installed in your active env
# (either of the following):
#   make dev                  # installs from the lockfile (recommended)
#   pip install -r requirements-dev.txt
#   # or if you prefer extras (slower resolver): pip install -e ".[ui]"

# Then start API and UI (requires venv/conda activated)
make run

# Or run via Docker Compose (no local Python needed)
docker compose up --build
```

Once the UI is running (Streamlit at http://localhost:8501), you can:

-   Open the Settings (⚙) to configure Intelligent mode, API base URL, and persistence. "Store chat history (Supabase)" is enabled by default; the backend will simply ignore it if Supabase is not configured.
-   Use the Memory Manager (🧠) → "Upload Markdown Files" to ingest `.md` files. The loader honors front‑matter and filename dates and chunks documents for retrieval. Uploaded items are indexed for vector search (Pinecone) and, for core types (preference/fact/profile), also persisted to Supabase when configured.
-   In Memory Manager → "Delete Memories", you can delete by IDs or use "Delete ALL (with confirm)" which opens a confirmation dialog and lets you select target: both, Pinecone only, or Supabase only. The action respects the conversation's DEV/PROD mode toggle.

### 7. Supabase Persistence (Optional)

If you want to persist chat history and permanent memories, set `SUPABASE_URL` and `SUPABASE_KEY` in your `.env`. Use the Project API “Secret” key (new) or the legacy `service_role` key. Do not use the JWT signing secret. A suggested schema is provided in `app/core/knowledge_store.py` docstring. When enabled:

-   Single-user by default: chats are stored under a single default session. You don’t need to manage session IDs.
-   You can override the default with `SUPABASE_DEFAULT_SESSION_ID` if desired.
-   `/chat` accepts `store_chat` and will persist messages automatically. It also returns the session id used (for debugging/ops), but the UI no longer requires it.
-   `POST /permanent_memories/upsert` stores permanent memories (single-user)
-   `GET /chat/history?session_id=...` reads a session's message history; the UI auto-loads this history on start when a `sid` is present in the URL or a `Session ID` is set in Settings.
-   Test vs Production tables: set `TEST_SUPABASE_TABLE_PREFIX` (defaults to `test_`). Toggle per-request with `use_test_supabase` (see below). The UI now provides a single **Developer mode** toggle that switches both Pinecone and Supabase to test resources for the current conversation.

Vector retrieval remains in Pinecone; store any retrievable content via `/memories/upsert` as before.

#### Memory Router

-   The system uses a rules-based router to send core items (types: `preference`, `fact`, `profile`, `user_core`) to Supabase permanent memories, and all items to Pinecone for semantic retrieval (unless `route_to_vector=false`).
-   From conversations, the Preference Extractor labels items with `type`, and the router persists them accordingly when `store_chat=true`.
-   API supports:
    -   `POST /memories/upsert` with fields `type`, `category`, `route_to_vector`.
    -   `POST /memories/delete` with `target=pinecone|supabase|both`.
    -   `POST /memories/delete_all` to wipe all memories in selected backend(s); respects `use_test_index` and `use_test_supabase` for DEV vs PROD separation.
    -   `GET /memories/search` merging Pinecone + Supabase (substring) results.

#### Test/Prod toggles

-   Pinecone: `use_test_index` (and `TEST_PINECONE_INDEX` env) switch to test index/namespace.
-   Supabase: `use_test_supabase` switches to test table prefix (`TEST_SUPABASE_TABLE_PREFIX`, default `test_`).
    -   Frontend now uses a single **Developer mode** toggle that sets both flags for all requests issued by that conversation. The sidebar shows "🧪 DEV" next to chats in developer mode, and the chat header shows a "Mode: 🧪 DEV MODE" badge.
    -   Note: the default session id is the same across prefixes unless you override with `SUPABASE_DEFAULT_SESSION_ID`.

#### Security

-   Use the Supabase SERVICE-ROLE key only on the server (never expose to the browser). Keep it in environment secrets.
-   If you must use the anon/publishable key in the browser, enforce Row Level Security (RLS) policies that strictly scope access by user/session and restrict all writes.
-   Protect write/read endpoints with an API key: set `API_AUTH_KEY` in backend env and include header `x-api-key: <API_AUTH_KEY>` when calling:
    -   `POST /chat` (only enforced when `store_chat=true`)
    -   `POST /permanent_memories/upsert`
    -   `GET /chat/history`
-   Always use HTTPS and rotate keys regularly.

## Project Structure

```
self-fed-memory/
├── app/                          # Main Python package
│   ├── api/
│   │   ├── main.py               # FastAPI app factory and CORS
│   │   └── routes_chat.py        # /chat, /memories/*, OpenAI-compatible endpoints
│   ├── core/                     # Core business logic
│   │   ├── chains/
│   │   │   ├── intelligent_qa_chain.py # Preference-aware, contextual QA
│   │   │   ├── memory_chain.py         # (present, reserved)
│   │   │   └── qa_chain.py             # Basic integrated QA
│   │   ├── embeddings.py         # Embedding engine (OpenAI with offline fallback)
│   │   ├── knowledge_store.py    # Supabase-backed persistence (sessions, memories)
│   │   ├── memory.py             # MemoryManager (Pinecone/Mock + retriever)
│   │   ├── memory_router.py      # Routes items to Pinecone and Supabase
│   │   ├── preference_tracker.py # Extracts and queries user preferences/facts
│   │   ├── retriever.py          # Time-weighted + intelligent multi-query retrieval
│   │   ├── types.py              # Helpers (dict → Document)
│   │   └── vector_store/
│   │       ├── mock.py           # In-memory VectorStore for tests/offline
│   │       └── pinecone.py       # Pinecone VectorStore adapter
│   ├── ingestion/
│   │   └── markdown_loader.py    # Markdown parsing + timestamping + chunking
│   └── utils/
│       └── text_splitter.py      # Shared chunking strategy
├── frontend/
│   └── app.py                    # Streamlit UI (chat + memory manager + settings)
├── scripts/
│   └── ingest_folder.py          # CLI: bulk-ingest a directory of .md files
├── tests/                        # Unit, integration, and manual tests
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── README.md
├── tests/README.md
└── design_doc.md
```

## Usage Examples

### Intelligent Q&A with Preference Learning

```python
from app.core.embeddings import get_embeddings
from app.core.memory import MemoryManager
from app.core.chains.intelligent_qa_chain import IntelligentQAChain

memory = MemoryManager(get_embeddings(), use_time_weighting=True)
chain = IntelligentQAChain(memory, name="Alex", k=8)

result = chain.invoke({
    "question": "Based on what you know about me, which restaurant would I enjoy more: a fancy French bistro or a simple sushi place?"
})

print(result["answer"])
print(f"Applied {result['user_preferences_found']} preferences and {result['user_facts_found']} facts")
```

### Basic Q&A

```python
from app.core.embeddings import get_embeddings
from app.core.memory import MemoryManager
from app.core.chains.qa_chain import IntegratedQAChain

mm = MemoryManager(get_embeddings())
chain = IntegratedQAChain(mm, k=5, name="User")
response = chain.invoke({"question": "What are my core values?"})
print(response["answer"])
```

### Adding New Memories

```python
from app.core.embeddings import get_embeddings
from app.core.memory import MemoryManager

memory = MemoryManager(get_embeddings())
memory.add_chunks([
    {
        "id": "mem_2024_01_15_spanish",
        "content": "I started learning Spanish today using Duolingo.",
        "source": "conversation",
        "created_at": "2024-01-15T12:00:00",
        "type": "fact",
        "category": "learning"
    }
])
```

### Manual Preference Extraction

```python
from app.core.embeddings import get_embeddings
from app.core.memory import MemoryManager
from app.core.preference_tracker import PreferenceTracker

memory_manager = MemoryManager(get_embeddings())
tracker = PreferenceTracker(memory_manager)

# Extract preferences from a conversation
conversation = """
User: I really enjoyed that small tapas place, but the loud music was annoying.
I prefer quieter atmospheres where I can actually have a conversation.
Assistant: That's great feedback! For quieter dining, you might enjoy...
"""

result = tracker.extract_and_store_preferences(conversation)
print(f"Extracted {result['preferences']} preferences and {result['facts']} facts")

# Get all stored preferences
preferences = tracker.get_user_preferences()
print(f"Total stored preferences: {len(preferences)}")
```

### Time-Based Queries

```python
# The system automatically weights recent memories higher
response = chain.invoke({"query": "What have I been working on lately?"})
```

## Development

### Running Tests

```bash
# Install test dependencies
pip install -e ".[test]"

# Run automated unit tests (fast, no API keys required)
make test

# Run with coverage
make test-cov

# Run manual verification tests (requires API keys + human evaluation)
make test-manual

# List available manual tests
make test-manual-list
```

**Testing Philosophy**: Fast, reliable automation for logic verification + comprehensive manual testing for quality assurance.

### Code Quality

### Git hooks (pre-commit)

We use `pre-commit` to automatically run checks locally.

Setup (once):

```bash
# If you used `make dev`, hooks are already installed.
# Otherwise, install dev/test deps and hooks:
pip install -e ".[dev,test]"
pre-commit install
# Optional: run on all files once
pre-commit run --all-files
```

What runs when:

-   **Commit**:
    -   ruff auto-fix and format
    -   mypy (uses config in `pyproject.toml`)
    -   unit + integration tests: `pytest -q -m "unit or integration" --maxfail=1` (with coverage; same thresholds as CI)
-   **Push**:
    -   no additional local checks (to avoid duplication); CI repeats the same suite

Notes:

-   Manual tests are not run by hooks (they require real API keys and human review).
-   The commit hook selects tests by marker; any "deselected" count you see in output just reflects excluded markers (e.g., `manual`).
-   Use any environment you prefer (venv, Conda, Docker). Hooks run in your active shell environment.

```bash
# Format code
ruff format .

# Lint code
ruff check .

# Type checking
mypy app/
```

### Adding New Vector Stores

To add support for a new vector database:

1. Create a new file in `app/core/vector_store/`
2. Extend the LangChain `VectorStore` base class
3. Implement required methods like `similarity_search`, `add_documents`, etc.
4. Update the factory logic in `MemoryManager` to use your new store

Example:

```python
# app/core/vector_store/chroma.py
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document

class ChromaVectorStore(VectorStore):
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        # Implementation here
        pass

    def add_documents(self, documents: List[Document]) -> List[str]:
        # Implementation here
        pass
```

## Configuration

All configuration is handled through environment variables and the `Settings` class in `app/core/config.py`.

### Key Settings

| Variable               | Default                  | Description                               |
| ---------------------- | ------------------------ | ----------------------------------------- |
| `OPENAI_API_KEY`       | -                        | **Required** OpenAI API key               |
| `PINECONE_API_KEY`     | -                        | **Required** Pinecone API key             |
| `PINECONE_INDEX`       | `self-memory`            | Pinecone index name                       |
| `PINECONE_NAMESPACE`   | `self-memory-namespace`  | Pinecone namespace                        |
| `EMBEDDING_MODEL`      | `text-embedding-3-large` | OpenAI embedding model                    |
| `LANGSMITH_TRACING`    | `false`                  | Enable LangSmith tracing                  |
| `LANGSMITH_API_KEY`    | -                        | LangSmith API key (optional)              |
| `LANGSMITH_PROJECT`    | `self_memory`            | LangSmith project name                    |
| `TEST_PINECONE_INDEX`  | `self-memory-test`       | Separate index for testing                |
| `TEST_EMBEDDING_MODEL` | `text-embedding-3-large` | Embedding model for tests                 |
| `API_AUTH_KEY`         | -                        | Optional API auth for protected endpoints |

## Roadmap

### ✅ MVP (Current)

-   [x] Markdown ingestion and chunking
-   [x] Vector embeddings with OpenAI
-   [x] Pinecone vector storage
-   [x] Time-weighted retrieval
-   [x] Basic Q&A chain with LangChain
-   [x] Configuration management
-   [x] **Intelligent retrieval system with multi-query support**
-   [x] **Automatic preference extraction and learning**
-   [x] **Context-aware response generation**
-   [x] **Conversation memory with preference application**

### 🚧 Phase 2 (In Progress)

-   [x] FastAPI backend with `/chat`, OpenAI-compatible endpoints, and memory upsert
-   [x] Streamlit web interface (chat UI)
-   [ ] Enhanced preference management and editing
-   [ ] Improved chunking strategies

### 🔮 Phase 3 (Planned)

-   [ ] Knowledge graph layer (Neo4j)
-   [ ] Multiple data source connectors
-   [ ] Advanced UI with React/Next.js
-   [ ] Mobile app
-   [ ] Voice interface

### 🌟 Future Ideas

-   [ ] Email/calendar integration
-   [ ] Social media ingestion
-   [ ] Goal tracking and reminders
-   [ ] Memory visualization
-   [ ] Local LLM support for privacy

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests (unit tests for logic, manual tests for user experience)
4. Run the automated test suite: `make test`
5. For changes affecting user experience, run manual verification: `make test-manual`
6. Submit a pull request

## Privacy & Security

-   **Local Control**: Your data stays in your chosen vector database
-   **API Usage**: OpenAI processes queries but doesn't store them (with proper settings)
-   **Encryption**: Consider encrypting sensitive data at rest

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

-   Built with [LangChain](https://langchain.com/) for LLM orchestration
-   Uses [Pinecone](https://pinecone.io/) for vector storage
-   Inspired by the "Second Brain" methodology
-   OpenAI GPT-4.1 for language understanding

---

**"Your AI assistant that truly knows you."**
