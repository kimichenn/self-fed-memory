# Self-Fed Memory: Personal AI Assistant

A personalized AI assistant with long-term memory that learns from your notes, conversations, and experiences. Built with retrieval-augmented generation (RAG) to provide personalized responses grounded in your own knowledge base.

## Overview

This system ingests your personal notes and documents, converts them into semantic embeddings stored in a vector database, and uses GPT-4.1 to answer questions with full context of your life, preferences, and experiences. Think of it as "ChatGPT with perfect memory of who you are."

**Key Features:**

-   🧠 **Long-term Memory**: Remembers everything you tell it across sessions
-   📝 **Note Ingestion**: Import Markdown files, journals, and personal documents
-   🔍 **Semantic Search**: Find relevant memories using meaning, not just keywords
-   ⏰ **Time-Aware**: Weights recent information higher while preserving old memories
-   🎯 **Personalized Responses**: Tailored advice based on your preferences and history
-   🔄 **Continuous Learning**: Learns new facts from conversations

## Architecture

```
📁 Personal Notes/Documents
    ↓ (Markdown Loader & Chunker)
🔢 Vector Embeddings (OpenAI)
    ↓ (Store in Vector DB)
🗄️ Pinecone Vector Store
    ↓ (Semantic Search + Time Weighting)
🔍 Retriever (LangChain)
    ↓ (Retrieval-Augmented Generation)
🤖 GPT-4.1 + Personal Context
    ↓
💬 Personalized Response
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

# Install dependencies
pip install -e .

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
from app.core.retriever import get_retriever

chain = get_qa_chain()
while True:
    q = input('You: ')
    if q.lower() in ['quit', 'exit']: break
    print('AI:', chain.invoke({'query': q})['result'])
"
```

#### Option B: Web Interface (Coming Soon)

```bash
# Note: API and frontend are placeholder files currently
# Start the FastAPI backend (when implemented)
uvicorn app.api.main:app --reload

# Start the Streamlit frontend (when implemented)
streamlit run frontend/app.py
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
│   │   ├── retriever.py         # Time-weighted retriever
│   │   ├── vector_store/        # Vector database adapters
│   │   │   ├── pinecone.py     # Pinecone implementation
│   │   │   └── mock.py         # In-memory testing
│   │   └── chains/              # LangChain orchestration
│   │       ├── qa_chain.py     # Q&A chain
│   │       └── memory_chain.py # Memory management
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

### Time-Based Queries

```python
# The system automatically weights recent memories higher
response = chain.invoke({"query": "What have I been working on lately?"})
```

## Development

### Running Tests

Our test suite includes **14 automated unit tests** (fast, deterministic) and **9 manual verification tests** (require human evaluation with real APIs).

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

### 🚧 Phase 2 (In Progress)

-   [ ] FastAPI backend with `/chat` endpoint (placeholder files exist)
-   [ ] Streamlit web interface (directory structure ready)
-   [ ] Conversation memory (learns from chats)
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
