from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.core.chains.qa_chain import IntegratedQAChain
from app.core.memory import MemoryManager
from app.core.vector_store.mock import MockVectorStore


@pytest.mark.integration
@patch("app.core.chains.qa_chain.get_qa_chain")
def test_end_to_end_qa_flow(mock_get_qa_chain):
    """Functional smoke test for the end-to-end question-answering flow."""
    # 1. Setup - Mock LLM, Embeddings, and Vector Store
    mock_qa_chain = MagicMock()
    mock_qa_chain.invoke.return_value = (
        "The answer is based on documents with IDs: 1, 2"
    )
    mock_get_qa_chain.return_value = mock_qa_chain

    mock_embeddings = MagicMock()
    mock_vector_store = MockVectorStore(embeddings=mock_embeddings)

    # 2. Ingestion - Add documents to the memory manager
    memory_manager = MemoryManager(embeddings=mock_embeddings, use_time_weighting=False)
    memory_manager.store = mock_vector_store
    documents = [
        Document(page_content="This is a test document.", metadata={"id": "1"}),
        Document(page_content="This is another test document.", metadata={"id": "2"}),
    ]
    memory_manager.store.add_documents(documents)

    # 3. Retrieval & QA - Run the QA chain
    qa_chain = IntegratedQAChain(llm=MagicMock(), memory_manager=memory_manager)
    result = qa_chain.invoke({"question": "What is the answer?"})

    # 4. Assertion - Verify the answer contains expected memory IDs
    assert "1" in result["answer"]
    assert "2" in result["answer"]
    mock_qa_chain.invoke.assert_called_once()
