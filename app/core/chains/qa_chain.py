from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

from app.core.memory import MemoryManager

QA_TEMPLATE = """You are a personal AI assistant for {name}.
You have access to the user's personal notes and memories.
Use the following information to answer the question.

{context}

User's question: {question}
Answer in a helpful manner based on the above context and the user's preferences.
"""


def get_qa_chain(llm: BaseChatModel = None) -> Runnable:
    """Build a basic chain that answers questions based on provided context.

    Note: This chain expects context to be provided in the input.
    For automatic context retrieval, use get_integrated_qa_chain() instead.
    """
    prompt = ChatPromptTemplate.from_template(QA_TEMPLATE)
    # Try to initialize LLM; if unavailable (e.g., no API key), fall back to a
    # simple deterministic responder to keep tests and offline environments working.
    if llm is None:
        try:
            llm = ChatOpenAI(model="gpt-4.1-2025-04-14", temperature=0)
            return prompt | llm | StrOutputParser()
        except Exception:
            pass

    if llm is not None:
        return prompt | llm | StrOutputParser()

    # Fallback Runnable: return a basic answer using provided context
    def _fallback(inputs: dict[str, Any]) -> str:
        context = inputs.get("context", "")
        question = inputs.get("question", "")
        name = inputs.get("name", "User")
        if context:
            return f"(offline) {name}, based on your context, here's a helpful reply to: {question}"
        return f"(offline) {name}, I cannot access a model right now, but you asked: {question}"

    return RunnableLambda(_fallback)


def get_integrated_qa_chain(
    memory_manager: MemoryManager,
    llm: BaseChatModel = None,
    k: int = 5,
    name: str = "User",
) -> Runnable:
    """Build a chain that automatically retrieves context and answers questions.

    This is the recommended chain for production use as it integrates
    with your time-weighted retriever to automatically get relevant context.

    Args:
        memory_manager: MemoryManager instance with configured retriever
        llm: Language model to use (defaults to GPT-4)
        k: Number of relevant documents to retrieve for context
        name: User's name to personalize responses

    Returns:
        A Runnable chain that accepts {"question": str} and returns a string answer
    """
    llm = llm or ChatOpenAI(model="gpt-4.1-2025-04-14", temperature=0)

    def retrieve_and_answer(inputs: dict[str, Any]) -> str:
        """Retrieve relevant context and answer the question."""
        question = inputs.get("question", "")

        if not question:
            return "Please provide a question."

        # Retrieve relevant documents using the time-weighted retriever
        relevant_docs = memory_manager.search(query=question, k=k)

        # Format the context from retrieved documents
        if relevant_docs:
            context_parts = []
            for i, doc in enumerate(relevant_docs, 1):
                content = doc.get("content", "")
                source = doc.get("source", "unknown")
                created_at = doc.get("created_at", "")

                # Create a readable context entry
                entry = f"[Memory {i}]"
                if created_at:
                    entry += f" (from {created_at})"
                if source and source != "unknown":
                    entry += f" [Source: {source}]"
                entry += f"\n{content}"

                context_parts.append(entry)

            context = "\n\n".join(context_parts)
        else:
            context = "No relevant memories found for this question."

        # Use the basic QA chain with the retrieved context
        qa_chain = get_qa_chain(llm=llm)
        response = qa_chain.invoke(
            {"name": name, "context": context, "question": question}
        )

        # Ensure we return a string as declared
        return str(response)

    # Wrap the callable as a Runnable for proper typing
    return RunnableLambda(retrieve_and_answer)


class IntegratedQAChain:
    """A more structured approach to the integrated QA chain.

    This class provides better control over the retrieval and generation process,
    and is easier to test and extend.
    """

    def __init__(
        self,
        memory_manager: MemoryManager,
        llm: BaseChatModel = None,
        k: int = 5,
        name: str = "User",
    ):
        self.memory_manager = memory_manager
        # Lazy/defensive LLM init to avoid hard dependency in tests
        self.llm = llm
        if self.llm is None:
            try:
                self.llm = ChatOpenAI(model="gpt-4.1-2025-04-14", temperature=0)
            except Exception:
                self.llm = None
        self.k = k
        self.name = name
        self.qa_chain = get_qa_chain(llm=self.llm)

    def invoke(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Process a question and return detailed results.

        Args:
            inputs: Dictionary containing "question" key

        Returns:
            Dictionary with "answer", "context", and "source_documents" keys
        """
        question = inputs.get("question", "")

        if not question:
            return {
                "question": question,
                "answer": "Please provide a question.",
                "context": "",
                "source_documents": [],
            }

        # Retrieve relevant documents
        relevant_docs = self.memory_manager.search(query=question, k=self.k)

        # Format context and prepare response
        context = self._format_context(relevant_docs)

        # Generate answer
        answer = self.qa_chain.invoke(
            {"name": self.name, "context": context, "question": question}
        )

        return {
            "question": question,
            "answer": answer,
            "context": context,
            "source_documents": relevant_docs,
        }

    def _format_context(self, docs: list[dict[str, Any]]) -> str:
        """Format retrieved documents into readable context."""
        if not docs:
            return "No relevant memories found for this question."

        context_parts = []
        for i, doc in enumerate(docs, 1):
            content = doc.get("content", "")
            source = doc.get("source", "unknown")
            created_at = doc.get("created_at", "")

            entry = f"[Memory {i}]"
            if created_at:
                entry += f" (from {created_at})"
            if source and source != "unknown":
                entry += f" [Source: {source}]"
            entry += f"\n{content}"

            context_parts.append(entry)

        return "\n\n".join(context_parts)

    def search_only(self, question: str) -> list[dict[str, Any]]:
        """Just retrieve relevant documents without generating an answer."""
        return self.memory_manager.search(query=question, k=self.k)
