from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from app.core.chains.intelligent_qa_chain import IntelligentQAChain
from app.core.memory import MemoryManager
from app.core.preference_tracker import IntelligentRetriever
from app.core.preference_tracker import PreferenceTracker


@pytest.fixture
def mock_memory_manager():
    """Fixture for a mocked MemoryManager."""
    return MagicMock(spec=MemoryManager)


@pytest.fixture
def mock_llm():
    """Fixture for a mocked LLM."""
    return MagicMock()


@pytest.mark.unit
def test_intelligent_qa_chain_initialization(mock_memory_manager, mock_llm):
    """Test that the IntelligentQAChain initializes correctly."""
    chain = IntelligentQAChain(memory_manager=mock_memory_manager, llm=mock_llm)
    assert chain.memory_manager == mock_memory_manager
    assert chain.llm == mock_llm
    assert isinstance(chain.preference_tracker, PreferenceTracker)
    assert isinstance(chain.intelligent_retriever, IntelligentRetriever)


@pytest.mark.unit
@patch("app.core.chains.intelligent_qa_chain.PreferenceTracker")
@patch("app.core.chains.intelligent_qa_chain.IntelligentRetriever")
def test_intelligent_qa_chain_invoke(
    MockIntelligentRetriever, MockPreferenceTracker, mock_memory_manager, mock_llm
):
    """Test the invoke method of IntelligentQAChain."""
    # Arrange
    mock_retriever_instance = MockIntelligentRetriever.return_value
    mock_retriever_instance.retrieve_with_context.return_value = {
        "main_results": [{"content": "main context", "source": "test.md"}],
        "context_results": [
            {
                "content": "user preference",
                "source": "pref.md",
                "type": "preference",
                "preference": "user preference",
            }
        ],
        "user_preferences_found": [],
        "user_facts_found": [],
    }

    mock_preference_tracker_instance = MockPreferenceTracker.return_value
    mock_preference_tracker_instance.extract_and_store_preferences.return_value = {
        "new_preferences": 1
    }

    chain = IntelligentQAChain(
        memory_manager=mock_memory_manager, llm=mock_llm, auto_extract_preferences=True
    )
    # We need to re-assign the mocked instances after initialization
    chain.intelligent_retriever = mock_retriever_instance
    chain.preference_tracker = mock_preference_tracker_instance

    # Mock the entire qa_chain to return a simple string
    chain.qa_chain = MagicMock()
    chain.qa_chain.invoke.return_value = "A smart answer."

    # Act
    result = chain.invoke(
        {"question": "test question", "conversation_history": "previous message"}
    )

    # Assert
    mock_retriever_instance.retrieve_with_context.assert_called_once_with(
        "test question", k=8
    )
    chain.qa_chain.invoke.assert_called_once()
    mock_preference_tracker_instance.extract_and_store_preferences.assert_called_once()
    assert result["answer"] == "A smart answer."
    assert "main context" in result["context"]
    assert "user preference" in result["user_context"]
