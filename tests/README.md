# Testing

This directory contains all the tests for the Self-Fed Memory system. The test suite is organized into **unit**, **integration**, and **manual** tests, managed by `pytest` markers.

## Test Philosophy

**"Test behavior, not implementation. Isolate logic for unit tests, verify component interactions for integration tests, and use manual tests for qualitative assessment."**

-   **Unit Tests (`test_unit_*.py`):** These are fast, deterministic tests that verify a single, isolated piece of logic (e.g., a function or a class). All external dependencies (APIs, databases, other classes) are mocked. This includes comprehensive tests for:
    -   **MemoryManager**: Retrieval logic, scoring mechanisms, ingestion, time weighting vs basic similarity search
    -   **Vector Store Adapters**: MockVectorStore and PineconeVectorStore functionality, CRUD operations, configuration handling
    -   **Configuration**: Environment variable loading and validation
    -   **Time-Weighted Retriever**: Mathematical scoring functions and document ranking
    -   **QA Chains**: Basic and intelligent question-answering logic and response formatting
    -   **Preference Tracking**: Preference extraction and intelligent retrieval components
-   **Integration Tests (`test_integration_*.py`):** These tests ensure that multiple components of the application work together correctly. They may use mocks for external services (like OpenAI or Pinecone APIs) but test the real interaction between internal modules.
-   **Manual Tests (`test_manual_*.py`):** These are end-to-end tests that use **real APIs** and require a human to evaluate the qualitative aspects of the system, such as the relevance and coherence of an AI's response.

## Test Structure & Markers

```
tests/
├── README.md                           # This file
├── conftest.py                         # Shared test fixtures, markers assignment, env guards
├── helpers.py                          # Helper utilities for tests
├── unit/
│   ├── conftest.py                     # Unit test specific fixtures
│   ├── test_api_contracts.py           # Validates API request/response schemas
│   ├── test_config.py                  # Settings & env resolution
│   ├── test_ingest_folder.py           # CLI ingestion script
│   ├── test_ingestion.py               # Markdown loader logic
│   ├── test_memory_manager.py          # MemoryManager behavior
│   ├── test_memory_router.py           # Routing to Pinecone/Supabase
│   ├── test_mock_vector_store.py       # Mock vector store functionality
│   ├── test_mock_vector_store_failures.py # Failure paths in mock store
│   ├── test_pinecone_vector_store.py   # Pinecone adapter (mocked)
│   ├── test_qa_chain.py                # Basic QA chain
│   ├── test_single_session_default.py  # Default single-session behavior
│   ├── test_supabase_knowledge_store.py# Supabase persistence wrapper (mocked)
│   ├── test_time_weighted_retriever.py # Time-weighted scoring & ranking
│   ├── test_unit_intelligent_qa_chain.py # Intelligent QA chain components
│   └── test_unit_markdown_loader.py    # Markdown loader specifics
├── integration/
│   ├── test_api_chat.py                # API end-to-end with mocks
│   ├── test_end_to_end_qa.py           # End-to-end QA flow (MockVectorStore)
│   ├── test_integration_intelligent_qa_chain.py # Intelligent QA integration
│   └── test_integration_markdown_loader.py      # Markdown parsing integration
└── manual/
    ├── test_manual_api_chat.py         # Live API manual verification
    ├── test_manual_qa_chain.py         # Basic QA with real APIs
    └── test_manual_intelligent_qa.py   # Intelligent QA with real APIs
```

## API Usage Summary

### Tests Using REAL APIs (Require API Keys)

-   `tests/manual/test_manual_api_chat.py` - Calls the running FastAPI server (`make api-dev`) and exercises `/memories/upsert`, `/memories/delete`, `/memories/search`, `/chat`, and `/chat/history` end-to-end. Prints responses for manual review
-   `tests/manual/test_manual_qa_chain.py` - Uses real Pinecone + OpenAI for basic QA
-   `tests/manual/test_manual_intelligent_qa.py` - Uses real APIs for intelligent QA with preference extraction

### Tests Using OFFLINE Mocks/Fakes

-   **`tests/unit/test_pinecone_vector_store.py`** - Extensively mocks Pinecone API using `@patch` decorators (no real API calls)
-   **`tests/unit/test_mock_vector_store.py`** - Tests the MockVectorStore implementation (pure in-memory, no APIs)
-   **`tests/unit/test_unit_intelligent_qa_chain.py`** - Tests intelligent QA chain components with mocked dependencies
-   **`tests/integration/test_integration_intelligent_qa_chain.py`** - Integration test for intelligent QA with MockVectorStore (no API calls)
-   **All other unit and integration tests** - Use either MockVectorStore or patch PineconeVectorStore (no real API calls)

## Running Tests

### 1. Install Dependencies

Prefer the pinned lock for deterministic CI/dev installs:

```bash
make dev                  # installs from requirements-dev.txt
# or
pip install -r requirements-dev.txt
```

### 2. Set Up Environment Variables

For manual tests, you must provide API keys. Copy the example environment file and add your credentials:

```bash
cp .env.example .env
```

Edit `.env` to include your `OPENAI_API_KEY` and `PINECONE_API_KEY`.

### 3. Execute Test Suites

The `Makefile` provides convenient shortcuts for running different test categories.

#### Automated Tests (Unit + Integration)

These tests are fast, deterministic, and safe to run at any time. They are executed in CI/CD pipelines and **do not make real API calls**.

```bash
# Run all unit and integration tests (fast, hermetic)
make test
```

#### Manual Verification Tests

**⚠️ Warning:** These tests use **real, paid APIs** and may persist data to your configured Supabase test tables. They require your manual review of the output.

```bash
# Run all manual tests
make test-manual

# Or run only the API manual test (ensure API is running in another terminal)
pytest tests/manual/test_manual_api_chat.py -s -m manual
```

### Running the API locally for manual tests

```bash
make api-dev

# Or via Docker (terminal 1)
docker compose up --build api
```

Then, in a separate terminal, run the manual tests as shown above.

#### Running Specific Tests with Pytest

You can also use `pytest` directly for more granular control:

```bash
# Run only unit tests
pytest -m "unit"

# Run only integration tests
pytest -m "integration"

# Run a specific test file
pytest tests/unit/test_time_weighted_retriever.py
```

## Live Supabase Integration Tests (Automatic when configured)

Integration tests in `tests/integration/test_supabase_live.py` will run automatically when both `SUPABASE_URL` and `SUPABASE_KEY` are set. No extra opt-in flag is required.

Requirements:

-   `SUPABASE_URL` and `SUPABASE_KEY` set in your environment (e.g., via `.env`)
-   Recommended: `TEST_SUPABASE_TABLE_PREFIX` (defaults to `test_`) to ensure isolation
-   Recommended: `API_AUTH_KEY` for protected endpoints like `/permanent_memories/upsert` and `/chat/history`

Example run:

```bash
export SUPABASE_URL=your-url
export SUPABASE_KEY=your-key
export TEST_SUPABASE_TABLE_PREFIX=test_
export API_AUTH_KEY=test-api-key
pytest -m "integration" -k supabase_live -q
```

These tests target the test-prefixed tables by passing `use_test_supabase=true` and use `x-api-key: $API_AUTH_KEY` for endpoints that require it. They perform best‑effort cleanup of test data at the end.

## UI dependencies are mandatory

The Streamlit frontend is part of the test surface. If you installed via the lockfile or `make dev`, UI deps are already included. Otherwise:

```bash
pip install -e ".[ui]"
```

## Manual Verification Explained

The manual tests in `test_manual_qa_chain.py` and `test_manual_intelligent_qa.py` are essential for evaluating the aspects of the system that cannot be easily asserted in code. When you run `make test-manual`, the scripts will:

1.  Connect to your real Pinecone and OpenAI accounts.
2.  Ingest a small, controlled set of test memories.
3.  Run a series of queries against both the basic and intelligent QA systems.
4.  Print the AI-generated answers, retrieved context, preference application, and guided prompts for your evaluation.

Your role is to manually inspect this output to assess:

-   Response quality and relevance
-   Preference extraction and application accuracy
-   Contextual understanding in intelligent QA
-   Overall system behavior and personality consistency

Both tests automatically clean up the data from your Pinecone index after the run.
