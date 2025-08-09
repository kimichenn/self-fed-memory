import os
from pathlib import Path

import pytest

from app.core.chains.intelligent_qa_chain import IntelligentQAChain
from app.core.config import Settings
from app.core.embeddings import get_embeddings
from app.core.memory import MemoryManager
from app.core.vector_store.pinecone import PineconeVectorStore
from app.ingestion.markdown_loader import parse_markdown_file

# Skip when real API keys are not available (same pattern as other manual tests)
requires_real_apis = pytest.mark.skipif(
    not (os.getenv("OPENAI_API_KEY") and os.getenv("PINECONE_API_KEY")),
    reason="Requires OPENAI_API_KEY and PINECONE_API_KEY for manual verification",
)


def load_actual_notes():
    """Load actual notes from the personal_notes directory."""
    personal_notes_dir = Path("personal_notes")
    if not personal_notes_dir.exists() or not personal_notes_dir.is_dir():
        return []

    markdown_files = list(personal_notes_dir.glob("*.md"))
    if not markdown_files:
        return []

    all_chunks = []
    for md_file in markdown_files:
        chunks = parse_markdown_file(md_file)
        all_chunks.extend(chunks)

    return all_chunks


@pytest.mark.manual
@requires_real_apis
def test_manual_intelligent_qa():
    """
    Manual test for the IntelligentQAChain using real APIs and actual notes.
    This test requires manual verification of the output.
    """
    Settings.for_testing()

    vector_store = PineconeVectorStore(embeddings=get_embeddings())

    memory_manager = MemoryManager(
        embeddings=get_embeddings(), use_time_weighting=False
    )
    memory_manager.store = vector_store

    # Clean up namespace before test run
    vector_store.index.delete(
        delete_all=True, namespace=vector_store.cfg.pinecone_namespace
    )

    # Load actual notes from personal_notes directory
    real_chunks = load_actual_notes()

    if not real_chunks:
        pytest.skip(
            "No actual notes found in personal_notes/ directory. Cannot run test with real data."
        )

    print(f"\nFound {len(real_chunks)} chunks from actual notes")

    # Show sample of what we're working with
    print("\nSample chunks from actual notes:")
    for i, chunk in enumerate(real_chunks[:3]):  # Show first 3 chunks
        content_preview = (
            chunk["content"][:100] + "..."
            if len(chunk["content"]) > 100
            else chunk["content"]
        )
        print(f"  {i + 1}. [{chunk.get('source', 'unknown')}] {content_preview}")

    # Ingest the actual note chunks
    memory_manager.add_chunks(real_chunks)

    chain = IntelligentQAChain(memory_manager=memory_manager, name="TestUser")

    # Use a more general question that should work with various types of notes
    question = "What can you tell me about my recent thoughts, activities, or projects based on my notes?"

    print("\n--- Running Manual Test: Intelligent QA Chain with Real Notes ---")
    print(f"Question: {question}")

    # Run the chain
    result = chain.invoke({"question": question})

    print("\n--- Full Response ---")
    print(result)

    print("\n--- Manual Verification Steps ---")
    print(
        "1. Review the 'answer'. Does it accurately reflect content from your actual notes?"
    )
    print(
        "2. Check if the response style and tone seem appropriate for your personal data."
    )
    print(
        "3. Examine the 'context_used'. Do you recognize these as actual excerpts from your notes?"
    )
    print(
        "4. Verify that the system is successfully retrieving and synthesizing your real information."
    )
    print(
        "5. Consider asking follow-up questions about specific topics you know are in your notes."
    )

    # Clean up namespace after test
    vector_store.index.delete(
        delete_all=True, namespace=vector_store.cfg.pinecone_namespace
    )

    # For manual verification we avoid strict assertions that may fail due to
    # LLM variability.  Human reviewers should inspect the printed output
    # following the steps above.
