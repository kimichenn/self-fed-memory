# Testing

This directory contains all the tests for the Self-Fed Memory system. The test suite is organized into **unit**, **integration**, and **manual** tests, managed by `pytest` markers.

## Test Philosophy

**"Test behavior, not implementation. Isolate logic for unit tests, verify component interactions for integration tests, and use manual tests for qualitative assessment."**

-   **Unit Tests (`test_unit_*.py`):** These are fast, deterministic tests that verify a single, isolated piece of logic (e.g., a function or a class). All external dependencies (APIs, databases, other classes) are mocked. This includes comprehensive tests for:
    -   **MemoryManager**: Retrieval logic, scoring mechanisms, ingestion, time weighting vs basic similarity search
    -   **Vector Store Adapters**: MockVectorStore and PineconeVectorStore functionality, CRUD operations, configuration handling
    -   **Configuration**: Environment variable loading and validation
    -   **Time-Weighted Retriever**: Mathematical scoring functions and document ranking
    -   **QA Chain**: Question-answering logic and response formatting
-   **Integration Tests (`test_integration_*.py`):** These tests ensure that multiple components of the application work together correctly. They may use mocks for external services (like OpenAI or Pinecone APIs) but test the real interaction between internal modules.
-   **Manual Tests (`test_manual_*.py`):** These are end-to-end tests that use **real APIs** and require a human to evaluate the qualitative aspects of the system, such as the relevance and coherence of an AI's response.

## Test Structure & Markers

```
tests/
├── README.md                           # This file
├── conftest.py                         # Shared test fixtures and configuration
├── helpers.py                          # Helper utilities for tests
├── unit/
│   ├── conftest.py                     # Unit test specific fixtures
│   ├── test_config.py                  # Unit tests for configuration loading
│   ├── test_memory_manager.py          # Unit tests for MemoryManager (uses mocked PineconeVectorStore)
│   ├── test_mock_vector_store.py       # Unit tests for MockVectorStore functionality
│   ├── test_pinecone_vector_store.py   # Unit tests for PineconeVectorStore (fully mocked, no real API calls)
│   ├── test_time_weighted_retriever.py # Unit tests for time-weighted scoring logic
│   ├── test_qa_chain.py                # Unit tests for the QA chain's internal logic
│   ├── test_end_to_end_qa.py           # End-to-end QA flow test (uses MockVectorStore)
│   ├── test_ingestion.py               # Unit tests for ingestion logic
│   ├── test_ingest_folder.py           # Unit tests for folder ingestion script
│   └── test_unit_markdown_loader.py    # Unit tests for Markdown document loading
├── integration/
│   └── test_integration_markdown_loader.py # Integration test for Markdown parsing (filesystem I/O)
└── manual/
    └── test_manual_qa_chain.py         # Manual verification tests (uses REAL Pinecone & OpenAI APIs)
```

## API Usage Summary

### Tests Using REAL APIs (Require API Keys)

-   **`tests/manual/test_manual_qa_chain.py`** - Uses real PineconeVectorStore instance that makes actual API calls to your Pinecone test index and OpenAI

### Tests Using OFFLINE Mocks/Fakes

-   **`tests/unit/test_pinecone_vector_store.py`** - Extensively mocks Pinecone API using `@patch` decorators (no real API calls)
-   **`tests/unit/test_mock_vector_store.py`** - Tests the MockVectorStore implementation (pure in-memory, no APIs)
-   **All other unit tests** - Use either MockVectorStore or patch PineconeVectorStore (no real API calls)

## Running Tests

### 1. Install Dependencies

First, ensure you have the necessary testing packages installed:

```bash
pip install -e ".[test]"
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
# Run all unit and integration tests
make test
```

#### Manual Verification Tests

**⚠️ Warning:** These tests use **real, paid APIs** and require your manual review of the output.

```bash
# Run all manual tests
make test-manual
```

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

## Manual Verification Explained

The manual tests in `test_manual_qa_chain.py` are essential for evaluating the aspects of the system that cannot be easily asserted in code. When you run `make test-manual`, the script will:

1.  Connect to your real Pinecone and OpenAI accounts.
2.  Ingest a small, controlled set of test memories.
3.  Run a series of queries against the system.
4.  Print the AI-generated answers, the retrieved context, and guided prompts for your evaluation.

Your role is to manually inspect this output to assess response quality, retrieval relevance, and overall system behavior. The test automatically cleans up the data from your Pinecone index after the run.
