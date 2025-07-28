import os

import pytest

from app.core.chains.intelligent_qa_chain import IntelligentQAChain
from app.core.config import Settings
from app.core.embeddings import get_embeddings
from app.core.memory import MemoryManager
from app.core.vector_store.pinecone import PineconeVectorStore


@pytest.mark.manual
def test_manual_intelligent_qa():
    """
    Manual test for the IntelligentQAChain using real APIs.
    This test requires manual verification of the output.
    """
    settings = Settings.for_testing()
    pinecone_api_key = settings.pinecone_api_key
    openai_api_key = settings.openai_api_key

    if not pinecone_api_key or not openai_api_key or not os.getenv("RUN_MANUAL_TESTS"):
        pytest.skip(
            "Skipping manual Intelligent QA test: manual tests disabled or API keys not set."
        )

    # The `PineconeVectorStore` now reads its configuration from `Settings`, so
    # we no longer need to (and indeed must not) pass `api_key` / `environment`
    # parameters that it doesn't accept.

    vector_store = PineconeVectorStore(embeddings=get_embeddings())
    index_name = settings.pinecone_index
    memory_manager = MemoryManager(
        vector_store=vector_store, embeddings=get_embeddings()
    )

    # Clean up index before test
    vector_store.delete(index_name=index_name, delete_all=True)

    # Ingest sample data
    memory_manager.add_memory(
        "User enjoys hiking on weekends.", metadata={"source": "conversation"}
    )
    memory_manager.add_memory(
        "User is a software engineer.", metadata={"source": "profile"}
    )
    memory_manager.add_memory(
        "User prefers concise and direct answers.", metadata={"source": "feedback"}
    )

    chain = IntelligentQAChain(memory_manager=memory_manager, name="TestUser")

    question = "What are some weekend activity suggestions for me?"

    print("\n--- Running Manual Test: Intelligent QA Chain ---")
    print(f"Question: {question}")

    # Run the chain
    result = chain.invoke({"question": question})

    print("\n--- Full Response ---")
    print(result)

    print("\n--- Manual Verification Steps ---")
    print(
        "1. Review the 'answer'. Does it suggest hiking or related outdoor activities?"
    )
    print(
        "2. Check if the response style is concise and direct, as per the user's preference."
    )
    print("3. Examine the 'context_used'. Does it include the ingested memories?")
    print(
        "4. Verify that the 'preferences_applied' section reflects the preference for concise answers."
    )

    # Clean up index after test
    vector_store.delete(index_name=index_name, delete_all=True)

    # For manual verification we avoid strict assertions that may fail due to
    # LLM variability.  Human reviewers should inspect the printed output
    # following the steps above.
