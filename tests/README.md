# Testing

This directory contains all the tests for the Self-Fed Memory system. The test suite includes unit tests, integration tests, and end-to-end testing for the personal AI memory system.

## Test Structure

```
tests/
├── README.md                        # This file
├── test_config.py                   # Configuration tests
├── test_chains.py                   # LangChain integration tests
├── test_markdown_loader.py          # Document ingestion tests
├── test_vector_store_integration.py # Vector database tests
├── test_ingestion_integration.py    # End-to-end ingestion tests (mocked)
└── test_end_to_end_ingestion.py     # Complete pipeline tests (real files)
```

## Setup

### 1. Install Test Dependencies

```bash
# Install the package with test dependencies
pip install -e ".[test]"

# Or install everything (recommended for development)
pip install -e ".[dev,test]"
```

### 2. Environment Variables

Tests require the same environment variables as the main application. Create a `.env` file in the root directory:

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

```env
# Required for tests
OPENAI_API_KEY=sk-your-openai-key-here
PINECONE_API_KEY=pcn-your-pinecone-key-here

# Test-specific settings (optional)
PINECONE_INDEX=self-memory-test  # Use separate index for testing
PINECONE_NAMESPACE=test-namespace
```

**⚠️ Important**: Use a separate Pinecone index for testing to avoid polluting your production data.

## Running Tests

### Basic Testing

```bash
# Run all tests
make test

# Or directly with pytest
pytest
```

### Coverage Reports

```bash
# Run tests with coverage
make test-cov

# View HTML coverage report
open htmlcov/index.html
```

### Test Categories

```bash
# Run only unit tests (fast)
make test-unit
pytest -m "not integration"

# Run only integration tests (slower, requires API keys)
make test-integration
pytest -m integration

# Run specific test file
pytest tests/test_config.py

# Run specific test function
pytest tests/test_chains.py::test_get_qa_chain

# Run end-to-end pipeline tests
pytest tests/test_end_to_end_ingestion.py
```

### Verbose Output

```bash
# Run with verbose output
pytest -v

# Run with extra verbose output and show print statements
pytest -vv -s
```

## Test Categories

### End-to-End Pipeline Tests (`test_end_to_end_ingestion.py`)

This new test suite provides comprehensive testing of the complete ingestion pipeline using real markdown files from the `personal_notes/` folder:

```bash
# Run all end-to-end tests (requires PINECONE_API_KEY)
pytest tests/test_end_to_end_ingestion.py

# Run only the dry-run test (no API key needed)
pytest tests/test_end_to_end_ingestion.py::test_dry_run_functionality

# Run the complete pipeline test with real Pinecone
pytest tests/test_end_to_end_ingestion.py::test_end_to_end_personal_notes_ingestion

# Test the ingest_folder.py script functionality
pytest tests/test_end_to_end_ingestion.py::test_ingest_folder_script_functionality
```

**What it tests:**

-   Complete pipeline: markdown files → parsing → embedding → Pinecone storage → retrieval
-   Real file processing from `personal_notes/` directory
-   Semantic search functionality with actual embedded content
-   The core logic that `scripts/ingest_folder.py` uses
-   Both dry-run (parsing only) and full pipeline modes

**Data Management:**

-   **End-to-end test**: Clears test index before starting, preserves data after completion for manual inspection
-   **Script functionality test**: Cleans up its own test data after completion
-   **Mini integration tests**: Clean up their own test vectors after completion

### Unit Tests

-   **Fast**: No external API calls
-   **Isolated**: Use mocks for external dependencies
-   **Examples**: Configuration parsing, text splitting, data models

### Integration Tests

-   **API Dependent**: Require OpenAI and Pinecone API keys
-   **Slower**: Make actual network calls
-   **Examples**: Vector store operations, embedding generation, full Q&A pipeline

Mark integration tests with the `@pytest.mark.integration` decorator:

```python
import pytest

@pytest.mark.integration
def test_pinecone_integration():
    # Test requires real Pinecone connection
    pass
```

## Writing Tests

### Test File Structure

```python
# tests/test_example.py
import pytest
from unittest.mock import MagicMock, patch

from app.core.example import ExampleClass


class TestExampleClass:
    """Test suite for ExampleClass."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.example = ExampleClass()

    def test_basic_functionality(self):
        """Test basic functionality."""
        result = self.example.do_something()
        assert result == expected_value

    @pytest.mark.integration
    def test_api_integration(self):
        """Test with real API calls."""
        # This test requires API keys and network access
        pass

    @patch('app.core.example.external_service')
    def test_with_mocked_dependency(self, mock_service):
        """Test with mocked external dependency."""
        mock_service.return_value = "mocked_response"
        result = self.example.method_using_service()
        assert result == "expected_result"
```

### Test Fixtures

Use `conftest.py` for shared test fixtures:

```python
# tests/conftest.py
import pytest
from app.core.config import Settings

@pytest.fixture
def test_settings():
    """Provide test configuration."""
    return Settings(
        openai_api_key="test-key",
        pinecone_api_key="test-key",
        pinecone_index="test-index"
    )

@pytest.fixture
def mock_vector_store():
    """Provide a mocked vector store."""
    with patch('app.core.vector_store.pinecone.PineconeVectorStore') as mock:
        yield mock
```

### Testing Best Practices

1. **Isolate External Dependencies**: Mock API calls for unit tests
2. **Use Descriptive Names**: Test function names should describe what's being tested
3. **Test Edge Cases**: Include tests for error conditions and boundary cases
4. **Keep Tests Fast**: Unit tests should run in milliseconds
5. **Clean Up**: Integration tests should clean up any data they create

### Example Test Patterns

#### Testing Configuration

```python
def test_settings_from_env(monkeypatch):
    """Test that settings load correctly from environment variables."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("PINECONE_API_KEY", "test-pinecone-key")

    settings = Settings()
    assert settings.openai_api_key == "test-key"
    assert settings.pinecone_api_key == "test-pinecone-key"
```

#### Testing with Async Functions

```python
@pytest.mark.asyncio
async def test_async_function():
    """Test async functionality."""
    result = await some_async_function()
    assert result is not None
```

#### Testing Error Handling

```python
def test_invalid_input_raises_error():
    """Test that invalid input raises appropriate error."""
    with pytest.raises(ValueError, match="Invalid input"):
        function_that_should_fail("invalid_input")
```

## Continuous Integration

The test suite runs automatically on:

-   Pull requests
-   Pushes to main branch

### GitHub Actions Workflow

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
    test:
        runs-on: ubuntu-latest
        steps:
            - uses: actions/checkout@v4
            - uses: actions/setup-python@v4
              with:
                  python-version: "3.11"
            - run: pip install -e ".[test]"
            - run: pytest --cov=app --cov-report=xml
            - uses: codecov/codecov-action@v3
```

## Debugging Tests

### Running Tests in Debug Mode

```bash
# Run with Python debugger
pytest --pdb

# Run specific test with extra output
pytest -vv -s tests/test_specific.py::test_function

# Run with custom markers
pytest -m "not slow"
```

### Common Issues

1. **Missing API Keys**: Ensure `.env` file is properly configured
2. **Network Issues**: Integration tests may fail if APIs are down
3. **Index Conflicts**: Use separate Pinecone index for testing
4. **Import Errors**: Ensure `PYTHONPATH` includes project root

### Test Environment Variables

```bash
# Skip integration tests if no API keys
pytest -m "not integration"

# Run only fast tests
pytest -m "not slow"

# Custom test database
PINECONE_INDEX=test-index pytest
```

## Code Coverage

The project aims for >80% test coverage. Current coverage can be viewed after running:

```bash
make test-cov
open htmlcov/index.html
```

Coverage excludes:

-   Abstract methods
-   Debug code
-   Exception handling for edge cases
-   CLI entry points

## Performance Testing

For performance-sensitive components:

```python
import time
import pytest

def test_performance_requirement():
    """Test that function completes within time limit."""
    start_time = time.time()

    result = expensive_function()

    duration = time.time() - start_time
    assert duration < 1.0  # Should complete within 1 second
    assert result is not None
```

## Contributing Tests

When contributing:

1. Add tests for new functionality
2. Maintain or improve test coverage
3. Ensure all tests pass before submitting PR
4. Use appropriate test markers (`@pytest.mark.integration`, etc.)
5. Update this README if adding new test patterns

---
