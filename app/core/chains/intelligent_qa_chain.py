"""Intelligent QA Chain that provides contextual, preference-aware responses.

This chain integrates multiple advanced features:
- LLM-powered query analysis for comprehensive retrieval
- Automatic preference extraction and storage
- Context-aware response generation
- Memory of user patterns and preferences across conversations
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.core.memory import MemoryManager
from app.core.preference_tracker import IntelligentRetriever
from app.core.preference_tracker import PreferenceTracker

INTELLIGENT_QA_TEMPLATE = """You are an intelligent personal AI assistant for {name}.

You have access to comprehensive information about the user, including:
- Their personal notes, documents, and memories
- Their preferences, habits, and decision patterns
- Their communication style and learning preferences
- Important facts and context about their life

RETRIEVED CONTEXT:
{context}

USER PREFERENCES & CONTEXT:
{user_context}

CONVERSATION HISTORY:
{conversation_history}

Current Question: {question}

Instructions:
1. Use ALL available information to provide the most helpful response
2. Apply the user's known preferences without them having to restate them
3. Reference relevant past conversations or decisions when helpful
4. Adapt your response style to match their preferences
5. If this is a choice/decision question, consider their past patterns
6. Be proactive - offer insights based on what you know about them

Provide a comprehensive, personalized response that demonstrates you understand and remember the user's context.
"""


class IntelligentQAChain:
    """Advanced QA chain with preference awareness and contextual intelligence."""

    def __init__(
        self,
        memory_manager: MemoryManager,
        llm: BaseChatModel = None,
        k: int = 8,
        name: str = "User",
        auto_extract_preferences: bool = True,
    ):
        """Initialize the intelligent QA chain.

        Args:
            memory_manager: MemoryManager with intelligent retrieval enabled
            llm: Language model for generation and analysis
            k: Number of documents to retrieve for context
            name: User's name for personalization
            auto_extract_preferences: Whether to automatically extract preferences from conversations
        """
        self.memory_manager = memory_manager
        self.llm = llm or ChatOpenAI(model="gpt-4.1-2025-04-14", temperature=0.1)
        self.k = k
        self.name = name
        self.auto_extract_preferences = auto_extract_preferences

        # Initialize advanced components
        self.preference_tracker = PreferenceTracker(memory_manager, llm=self.llm)
        self.intelligent_retriever = IntelligentRetriever(
            memory_manager, self.preference_tracker
        )

        # Setup prompt and chain
        self.prompt = ChatPromptTemplate.from_template(INTELLIGENT_QA_TEMPLATE)
        self.qa_chain = self.prompt | self.llm | StrOutputParser()

        # Store conversation history for context
        self.conversation_history: list[dict[str, str]] = []

    def invoke(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Process a question with full contextual intelligence."""
        question = inputs.get("question", "")
        conversation_context = inputs.get("conversation_history", "")

        if not question:
            return {
                "question": question,
                "answer": "Please provide a question.",
                "context_used": [],
                "preferences_applied": [],
                "extraction_results": {},
            }

        # Step 1: Intelligent retrieval with context
        retrieval_results = self.intelligent_retriever.retrieve_with_context(
            question, k=self.k
        )

        # Step 2: Get user preferences relevant to this question
        user_context = self._format_user_context(retrieval_results["context_results"])

        # Step 3: Format main context
        main_context = self._format_main_context(retrieval_results["main_results"])

        # Step 4: Prepare conversation history
        formatted_history = self._format_conversation_history()

        # Step 5: Generate contextual response
        answer = self.qa_chain.invoke(
            {
                "name": self.name,
                "question": question,
                "context": main_context,
                "user_context": user_context,
                "conversation_history": formatted_history,
            }
        )

        # Step 6: Extract preferences from this conversation (if enabled)
        extraction_results = {}
        if self.auto_extract_preferences and conversation_context:
            full_conversation = f"User: {question}\nAssistant: {answer}"
            if conversation_context:
                full_conversation = f"{conversation_context}\n{full_conversation}"

            extraction_results = self.preference_tracker.extract_and_store_preferences(
                full_conversation
            )

        # Step 7: Update conversation history
        self.conversation_history.append(
            {
                "question": question,
                "answer": answer,
                "timestamp": str(datetime.utcnow()),
            }
        )

        # Keep only last 10 exchanges in memory
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]

        return {
            "question": question,
            "answer": answer,
            "context": main_context,
            "user_context": user_context,
            "context_used": retrieval_results["main_results"],
            "preferences_applied": retrieval_results["context_results"],
            "user_preferences_found": retrieval_results["user_preferences_found"],
            "user_facts_found": retrieval_results["user_facts_found"],
            "extraction_results": extraction_results,
            "retrieval_strategy": "intelligent_multi_query",
        }

    def _format_main_context(self, docs: list[dict[str, Any]]) -> str:
        """Format main retrieved documents into readable context."""
        if not docs:
            return "No directly relevant information found."

        context_parts = []
        for i, doc in enumerate(docs, 1):
            content = doc.get("content", "")
            source = doc.get("source", "unknown")
            created_at = doc.get("created_at", "")
            doc_type = doc.get("type", "document")

            if doc_type in ["preference", "fact"]:
                # Skip preferences/facts here - they go in user_context
                continue

            entry = f"[Memory {i}]"
            if created_at:
                entry += f" (from {created_at})"
            if source and source != "unknown":
                entry += f" [Source: {source}]"
            entry += f"\n{content}"

            context_parts.append(entry)

        return (
            "\n\n".join(context_parts)
            if context_parts
            else "No content documents found."
        )

    def _format_user_context(self, context_docs: list[dict[str, Any]]) -> str:
        """Format user preferences and facts into readable context."""
        if not context_docs:
            return "No specific user preferences or context found for this topic."

        preferences = []
        facts = []

        for doc in context_docs:
            doc_type = doc.get("type", "unknown")
            if doc_type == "preference":
                pref_text = doc.get("preference", doc.get("content", ""))
                context = doc.get("context", "")
                if context:
                    preferences.append(f"• {pref_text} (Context: {context})")
                else:
                    preferences.append(f"• {pref_text}")
            elif doc_type == "fact":
                fact_text = doc.get("fact", doc.get("content", ""))
                facts.append(f"• {fact_text}")

        result_parts = []
        if preferences:
            result_parts.append("USER PREFERENCES:\n" + "\n".join(preferences))
        if facts:
            result_parts.append("IMPORTANT USER FACTS:\n" + "\n".join(facts))

        return (
            "\n\n".join(result_parts)
            if result_parts
            else "No specific user context found."
        )

    def _format_conversation_history(self) -> str:
        """Format recent conversation history for context."""
        if not self.conversation_history:
            return "No recent conversation history."

        # Show last 3 exchanges
        recent = self.conversation_history[-3:]
        formatted = []

        for exchange in recent:
            formatted.append(f"User: {exchange['question']}")
            formatted.append(
                f"Assistant: {exchange['answer'][:200]}..."
            )  # Truncate for context

        return "\n".join(formatted)

    def get_user_preferences_summary(self) -> dict[str, Any]:
        """Get a summary of all stored user preferences."""
        all_preferences = self.preference_tracker.get_user_preferences()

        # Group by category
        by_category = {}
        for pref in all_preferences:
            category = pref.get("category", "general")
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(pref)

        return {
            "total_preferences": len(all_preferences),
            "categories": list(by_category.keys()),
            "by_category": by_category,
            "most_recent": sorted(
                all_preferences, key=lambda x: x.get("created_at", ""), reverse=True
            )[:5],
        }

    def clear_conversation_history(self):
        """Clear the conversation history (for new sessions)."""
        self.conversation_history = []

    def simulate_preference_extraction(self, conversation: str) -> dict[str, Any]:
        """Manually extract preferences from a conversation (for testing/manual use)."""
        return self.preference_tracker.extract_and_store_preferences(conversation)


# Convenience function for backward compatibility
def get_intelligent_qa_chain(
    memory_manager: MemoryManager,
    llm: BaseChatModel = None,
    k: int = 8,
    name: str = "User",
    auto_extract_preferences: bool = True,
) -> IntelligentQAChain:
    """Create an intelligent QA chain with all advanced features enabled."""
    return IntelligentQAChain(
        memory_manager=memory_manager,
        llm=llm,
        k=k,
        name=name,
        auto_extract_preferences=auto_extract_preferences,
    )
