from unittest.mock import MagicMock

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from app.core.chains.memory_chain import MEMORY_TEMPLATE
from app.core.chains.memory_chain import get_memory_chain
from app.core.chains.qa_chain import get_qa_chain


# Test case for the QA chain
def test_get_qa_chain():
    """
    Tests that the QA chain is created correctly and is runnable.
    """
    # 1. Create the QA chain
    qa_chain = get_qa_chain()

    # 2. Assert that the chain is a Runnable instance
    assert isinstance(qa_chain, Runnable)

    # 3. Test the prompt structure (optional but good practice)
    # Get the prompt from the chain; this might vary based on langchain updates
    prompt = qa_chain.steps[0]

    # Verify that the template matches the expected structure
    assert "context" in prompt.input_schema.model_json_schema()["properties"]
    assert "question" in prompt.input_schema.model_json_schema()["properties"]


# Test case for the Memory chain
def test_get_memory_chain():
    """
    Tests that the Memory chain is created correctly and is runnable.
    """
    # 1. Create the Memory chain
    memory_chain = get_memory_chain()

    # 2. Assert that the chain is a Runnable instance
    assert isinstance(memory_chain, Runnable)

    # 3. Test the prompt structure
    prompt = memory_chain.steps[0]
    assert "query" in prompt.input_schema.model_json_schema()["properties"]


# Test case for invoking the QA chain with a mock LLM
def test_qa_chain_invocation():
    """
    Tests the invocation of the QA chain with a mock LLM to ensure it processes inputs correctly.
    """
    # 1. Create a mock LLM
    mock_llm = MagicMock(spec=BaseChatModel)
    mock_llm.invoke.return_value = "The answer is 42."

    # 2. Build the chain with the mock LLM
    qa_chain = get_qa_chain(llm=mock_llm)

    # 3. Invoke the chain with test data
    response = qa_chain.invoke(
        {
            "name": "Kimi",
            "context": "The context is irrelevant for this test.",
            "question": "What is the meaning of life?",
        }
    )

    # 4. Assert that the mock LLM was called with the correct input
    mock_llm.invoke.assert_called_once()
    # The input to the LLM is a ChatPromptValue, not a simple dict.
    # We can inspect the call arguments to be more precise.
    call_args = mock_llm.invoke.call_args[0][0]
    assert "The context is irrelevant" in str(
        call_args
    )  # Check if context is in the formatted prompt
    assert "What is the meaning of life?" in str(
        call_args
    )  # Check if question is in the formatted prompt

    # 5. Assert the response from the chain
    assert response == "The answer is 42."


# Test case for invoking the Memory chain with a mock LLM
def get_test_memory_chain(llm: BaseChatModel = None) -> Runnable:
    """Create a chain that determines if a user wants to save a memory."""

    llm = llm or ChatOpenAI(model="gpt-4.1-2025-04-14", temperature=0)
    prompt = ChatPromptTemplate.from_template(MEMORY_TEMPLATE)
    return prompt | llm


def test_memory_chain_invocation():
    """
    Tests the invocation of the Memory chain with a mock LLM.
    """
    # 1. Create a mock LLM
    mock_llm = MagicMock(spec=BaseChatModel)
    mock_llm.invoke.return_value = "User wants to know about personal memories."

    # 2. Build the chain with the mock LLM
    memory_chain = get_test_memory_chain(llm=mock_llm)

    # 3. Invoke the chain
    response = memory_chain.invoke({"query": "Tell me about my childhood."})

    # 4. Assert the mock LLM was called correctly
    mock_llm.invoke.assert_called_once()
    call_args = mock_llm.invoke.call_args[0][0]
    assert "Tell me about my childhood." in str(call_args)

    # 5. Assert the response
    assert response == "User wants to know about personal memories."
