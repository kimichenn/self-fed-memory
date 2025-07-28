"""
Unit Tests for the IntegratedQAChain

Purpose:
- To test the internal logic of the IntegratedQAChain in isolation.
- To ensure the chain correctly processes inputs, interacts with its components
  (like the memory manager and LLM), and formats its output.
- To run fast, deterministic tests without any external API calls.
"""

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from app.core.chains.qa_chain import IntegratedQAChain


@pytest.fixture
def mock_memory_manager():
    """Fixture to create a mock MemoryManager."""
    manager = MagicMock()
    # Simulate a search result with a single document
    manager.search.return_value = [
        {
            "id": "doc1",
            "content": "This is a test document.",
            "source": "test.md",
            "created_at": "2024-01-01T00:00:00",
        }
    ]
    return manager


@pytest.fixture
def mock_llm():
    """Fixture to create a mock LLM."""
    llm = MagicMock()
    # The actual chain returns a string, so we mock the entire chain's output
    llm.invoke.return_value = "This is a mock LLM response."
    return llm


@pytest.fixture
def mock_qa_chain(mock_llm):
    """Fixture to create a mock QA chain that simulates the real one's output."""
    chain = MagicMock()
    # When the chain is invoked, it should return a string response.
    # This simulates the final output of `prompt | llm | StrOutputParser()`.
    chain.invoke.return_value = mock_llm.invoke.return_value
    return chain


@pytest.mark.unit
def test_qa_chain_initialization(mock_memory_manager, mock_llm):
    """
    Test that the IntegratedQAChain initializes correctly with its components.
    """
    chain = IntegratedQAChain(
        memory_manager=mock_memory_manager, llm=mock_llm, k=5, name="TestUser"
    )
    assert chain.memory_manager == mock_memory_manager
    assert chain.llm == mock_llm
    assert chain.k == 5
    assert chain.name == "TestUser"


@pytest.mark.unit
@patch("app.core.chains.qa_chain.get_qa_chain")
def test_qa_chain_invocation_flow(
    mock_get_qa_chain, mock_memory_manager, mock_qa_chain, mock_llm
):
    """
    Test the complete invocation flow of the QA chain, mocking all external
    dependencies.
    """
    # 1. Setup: Use the mock QA chain
    mock_get_qa_chain.return_value = mock_qa_chain
    chain = IntegratedQAChain(memory_manager=mock_memory_manager, llm=mock_llm, k=3)

    # 2. Define the input question
    question = "What is the test document about?"

    # 3. Invoke the chain
    result = chain.invoke({"question": question})

    # 4. Assert that the memory manager was called correctly
    mock_memory_manager.search.assert_called_once_with(query=question, k=3)

    # 5. Assert that the underlying QA chain was called with the correct context
    retrieved_docs = mock_memory_manager.search.return_value
    expected_context = chain._format_context(retrieved_docs)
    mock_qa_chain.invoke.assert_called_once_with(
        {"name": "User", "context": expected_context, "question": question}
    )

    # 6. Assert that the final output is structured correctly
    assert result["question"] == question
    assert result["answer"] == mock_qa_chain.invoke.return_value
    assert len(result["source_documents"]) == 1
    assert result["source_documents"][0]["id"] == "doc1"


@pytest.mark.unit
@patch("app.core.chains.qa_chain.get_qa_chain")
def test_qa_chain_with_no_retrieved_documents(
    mock_get_qa_chain, mock_memory_manager, mock_qa_chain, mock_llm
):
    """
    Test how the QA chain behaves when the memory manager returns no documents.
    """
    # 1. Setup: Simulate no documents found and use the mock chain
    mock_memory_manager.search.return_value = []
    mock_get_qa_chain.return_value = mock_qa_chain
    chain = IntegratedQAChain(memory_manager=mock_memory_manager, llm=mock_llm, k=5)
    question = "A question with no relevant memories"

    # 2. Invoke the chain
    result = chain.invoke({"question": question})

    # 3. Assert that the context passed to the chain is the "no memories" message
    expected_context = "No relevant memories found for this question."
    mock_qa_chain.invoke.assert_called_once_with(
        {"name": "User", "context": expected_context, "question": question}
    )

    # 4. Assert that the final answer is the mock response
    assert result["answer"] == mock_qa_chain.invoke.return_value
    assert len(result["source_documents"]) == 0
