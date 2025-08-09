# Self-Fed Memory: Personal AI Assistant

A personalized AI assistant with long-term memory that learns from your notes, conversations, and experiences. Built with retrieval-augmented generation (RAG) to provide personalized responses grounded in your own knowledge base.

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
```

### 4. Initialize Vector Store

Set up your Pinecone index:

```bash
# This will create the index if it doesn't exist
python -c "from app.core.vector_store.pinecone import PineconeVectorStore; PineconeVectorStore().initialize_index()"
```

### 5. Ingest Your Personal Data

```bash
# Ingest a folder of Markdown files
python scripts/ingest_folder.py /path/to/your/notes

# Or ingest individual files
python -m app.ingestion.markdown_loader --file /path/to/your/journal.md
```

### 6. Start the System

#### Option A: Simple CLI Chat

```bash
python -c "
from app.core.chains.qa_chain import get_qa_chain
chain = get_qa_chain()
while True:
    q = input('You: ')
    if q.lower() in ['quit', 'exit']: break
    print('AI:', chain.invoke({'query': q})['result'])
"

# Or use the intelligent QA chain
python -c "
from app.core.chains.intelligent_qa_chain import get_intelligent_qa_chain
from app.core.memory import MemoryManager
memory = MemoryManager()
chain = get_intelligent_qa_chain(memory, name='Your Name')
while True:
    q = input('You: ')
    if q.lower() in ['quit', 'exit']: break
    result = chain.invoke({'question': q})
    print('AI:', result['answer'])
"
```

#### Option B: Web Interface

```bash
# One command (starts API and UI; requires venv/conda activated)
make run

# Or run via Docker Compose (no local Python needed)
docker compose up --build
```

## Project Structure

```
self-fed-memory/
├── app/                          # Main Python package
│   ├── core/                     # Core business logic
│   │   ├── config.py            # Environment configuration
│   │   ├── types.py             # Pydantic models & types
│   │   ├── embeddings.py        # Embedding engine
│   │   ├── llm.py               # LLM client wrapper
│   │   ├── memory.py            # Memory manager
│   │   ├── preference_tracker.py # Preference extraction & intelligent retrieval
│   │   ├── retriever.py         # Time-weighted retriever
│   │   ├── vector_store/        # Vector database adapters
│   │   │   ├── pinecone.py     # Pinecone implementation
│   │   │   └── mock.py         # In-memory testing
│   │   └── chains/              # LangChain orchestration
│   │       ├── qa_chain.py           # Basic Q&A chain
│   │       ├── intelligent_qa_chain.py # Advanced intelligent Q&A
│   │       └── memory_chain.py       # Memory management
│   ├── ingestion/               # Data ingestion
│   │   └── markdown_loader.py  # Markdown file processor
│   ├── api/                     # Web API (FastAPI)
│   │   ├── main.py             # FastAPI app
│   │   └── routes_chat.py      # Chat endpoints
│   └── utils/                   # Utilities
│       └── text_splitter.py    # Text chunking
├── scripts/                     # CLI tools
│   └── ingest_folder.py        # Bulk ingestion
├── tests/                       # Test suite
├── frontend/                    # Web UI (future)
└── docs/                        # Documentation
```

## Usage Examples

### Intelligent Q&A with Preference Learning

```python
from app.core.chains.intelligent_qa_chain import get_intelligent_qa_chain
from app.core.memory import MemoryManager

# Setup with intelligent retrieval enabled
memory = MemoryManager(use_intelligent_queries=True)
chain = get_intelligent_qa_chain(memory, name="Alex", auto_extract_preferences=True)

# The system understands context and applies preferences automatically
response = chain.invoke({
    "question": "Based on what you know about me, which restaurant would I enjoy more: a fancy French bistro or a simple sushi place?"
})

print(response["answer"])
print(f"Applied {response['user_preferences_found']} preferences and {response['user_facts_found']} facts")
print(f"Learned {response['extraction_results'].get('extracted_count', 0)} new preferences")
```

### Basic Q&A

```python
from app.core.chains.qa_chain import get_qa_chain

chain = get_qa_chain()
response = chain.invoke({"query": "What are my core values?"})
print(response["result"])
```

### Adding New Memories

```python
from app.core.memory import MemoryManager

memory = MemoryManager()
memory.add_memory(
    content="I started learning Spanish today using Duolingo.",
    source="conversation",
    timestamp="2024-01-15"
)
```

### Manual Preference Extraction

```python
from app.core.preference_tracker import PreferenceTracker

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

We use `pre-commit` to automatically run fast checks locally.

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

-   Commit:
    -   ruff auto-fix and format
    -   mypy (uses config in `pyproject.toml`)
    -   unit tests: `pytest -q -m "unit" --maxfail=1`
-   Push:
    -   full check via `make check` (ruff lint, mypy type-check, unit + integration tests)

Notes:

-   Manual tests are not run by hooks (they require real API keys and human review).
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

| Variable               | Default                  | Description                   |
| ---------------------- | ------------------------ | ----------------------------- |
| `OPENAI_API_KEY`       | -                        | **Required** OpenAI API key   |
| `PINECONE_API_KEY`     | -                        | **Required** Pinecone API key |
| `PINECONE_INDEX`       | `self-memory`            | Pinecone index name           |
| `EMBEDDING_MODEL`      | `text-embedding-3-large` | OpenAI embedding model        |
| `PINECONE_NAMESPACE`   | `self-memory-namespace`  | Pinecone namespace            |
| `LANGSMITH_TRACING`    | `false`                  | Enable LangSmith tracing      |
| `LANGSMITH_API_KEY`    | -                        | LangSmith API key (optional)  |
| `LANGSMITH_PROJECT`    | `self_memory`            | LangSmith project name        |
| `TEST_PINECONE_INDEX`  | `self-memory-test`       | Separate index for testing    |
| `TEST_EMBEDDING_MODEL` | `text-embedding-3-large` | Embedding model for tests     |

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
