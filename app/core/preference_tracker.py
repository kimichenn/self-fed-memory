"""Preference and context tracking system for intelligent memory retrieval.

This system automatically identifies and stores user preferences, habits, and context
from conversations to enable more intelligent retrieval and responses.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from app.core.memory import MemoryManager


PREFERENCE_EXTRACTION_PROMPT = """You are an intelligent preference extraction system for a personal AI assistant.

Your job is to identify and extract user preferences, habits, and contextual information from conversations.

Look for:
1. **Explicit preferences**: "I prefer...", "I like...", "I always..."
2. **Decision patterns**: Choices the user makes and why
3. **Work/learning styles**: How they like information presented
4. **Personal context**: Important facts about their life, work, interests
5. **Constraints**: Allergies, limitations, requirements
6. **Meta-preferences**: How they like the AI to behave

Conversation: {conversation}

Extract any preferences, patterns, or important context. Format as JSON:
{
  "preferences": [
    {
      "category": "food/learning/work/communication/etc",
      "preference": "clear description of preference",
      "context": "when/where this applies",
      "confidence": 0.8,
      "examples": ["specific examples from conversation"]
    }
  ],
  "facts": [
    {
      "category": "personal/work/hobby/etc", 
      "fact": "important factual information",
      "confidence": 0.9
    }
  ]
}

Only extract clear, useful information. Skip vague or uncertain items.
Return ONLY the JSON, no other text.
"""


class PreferenceTracker:
    """Tracks and manages user preferences and context for intelligent retrieval."""

    def __init__(self, memory_manager: MemoryManager, llm: BaseChatModel = None):
        self.memory_manager = memory_manager
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

        self.extraction_prompt = ChatPromptTemplate.from_template(
            PREFERENCE_EXTRACTION_PROMPT
        )
        self.extraction_chain = self.extraction_prompt | self.llm | StrOutputParser()

    def extract_and_store_preferences(self, conversation: str) -> Dict[str, Any]:
        """Extract preferences from a conversation and store them."""
        try:
            # Extract preferences using LLM
            response = self.extraction_chain.invoke({"conversation": conversation})

            # Parse JSON response
            extracted = json.loads(response.strip())

            # Store preferences as special documents
            timestamp = datetime.utcnow().isoformat()
            stored_items = []

            # Store preferences
            for pref in extracted.get("preferences", []):
                doc_data = {
                    "id": f"preference_{hash(pref['preference'])}_{int(datetime.utcnow().timestamp())}",
                    "content": f"User preference: {pref['preference']}. Context: {pref.get('context', '')}",
                    "type": "preference",
                    "category": pref.get("category", "general"),
                    "preference": pref["preference"],
                    "context": pref.get("context", ""),
                    "confidence": pref.get("confidence", 0.8),
                    "examples": pref.get("examples", []),
                    "created_at": timestamp,
                    "source": "auto_extracted",
                }
                stored_items.append(doc_data)

            # Store facts
            for fact in extracted.get("facts", []):
                doc_data = {
                    "id": f"fact_{hash(fact['fact'])}_{int(datetime.utcnow().timestamp())}",
                    "content": f"User fact: {fact['fact']}",
                    "type": "fact",
                    "category": fact.get("category", "general"),
                    "fact": fact["fact"],
                    "confidence": fact.get("confidence", 0.9),
                    "created_at": timestamp,
                    "source": "auto_extracted",
                }
                stored_items.append(doc_data)

            # Store in memory system
            if stored_items:
                self.memory_manager.add_chunks(stored_items)

            return {
                "extracted_count": len(stored_items),
                "preferences": len(extracted.get("preferences", [])),
                "facts": len(extracted.get("facts", [])),
                "items": stored_items,
            }

        except Exception as e:
            # If extraction fails, return empty result
            return {"extracted_count": 0, "preferences": 0, "facts": 0, "error": str(e)}

    def get_user_preferences(self, category: str = None) -> List[Dict[str, Any]]:
        """Retrieve stored user preferences, optionally filtered by category."""
        if category:
            query = f"user preferences {category}"
        else:
            query = "user preferences"

        # Search for preference documents
        results = self.memory_manager.search(query, k=20, use_time_weighting=False)

        # Filter to only preference-type documents
        preferences = [doc for doc in results if doc.get("type") == "preference"]

        return preferences

    def get_user_context(self, topic: str) -> List[Dict[str, Any]]:
        """Get relevant user context for a specific topic."""
        # Search for both preferences and facts related to the topic
        query = f"user context preferences facts {topic}"

        results = self.memory_manager.search(query, k=15, use_time_weighting=True)

        # Filter to context-relevant documents
        context = [
            doc
            for doc in results
            if doc.get("type") in ["preference", "fact"]
            or any(
                keyword in doc.get("content", "").lower()
                for keyword in ["prefer", "like", "always", "never", "important"]
            )
        ]

        return context

    def update_preference(self, preference_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing preference (for manual corrections)."""
        # This would require implementing update functionality in the vector store
        # For now, we'll add a new version and mark the old one as superseded
        try:
            # Get existing preference
            existing = self.memory_manager.search(f"id:{preference_id}", k=1)
            if not existing:
                return False

            # Create updated version
            old_pref = existing[0]
            new_pref = old_pref.copy()
            new_pref.update(updates)
            new_pref["id"] = (
                f"{preference_id}_updated_{int(datetime.utcnow().timestamp())}"
            )
            new_pref["supersedes"] = preference_id
            new_pref["updated_at"] = datetime.utcnow().isoformat()

            # Store updated version
            self.memory_manager.add_chunks([new_pref])

            return True

        except Exception:
            return False


class IntelligentRetriever:
    """Enhanced retriever that incorporates user preferences and context."""

    def __init__(
        self,
        memory_manager: MemoryManager,
        preference_tracker: PreferenceTracker = None,
    ):
        self.memory_manager = memory_manager
        self.preference_tracker = preference_tracker or PreferenceTracker(
            memory_manager
        )

    def retrieve_with_context(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Retrieve documents with automatic preference and context enhancement."""

        # Get main search results
        main_results = self.memory_manager.search(query, k=k)

        # Get relevant user context
        context_results = self.preference_tracker.get_user_context(query)

        # Combine and deduplicate
        all_results = main_results.copy()
        seen_ids = {doc.get("id") for doc in main_results}

        for context_doc in context_results:
            if context_doc.get("id") not in seen_ids:
                all_results.append(context_doc)
                if len(all_results) >= k * 2:  # Limit total results
                    break

        return {
            "query": query,
            "main_results": main_results,
            "context_results": context_results,
            "combined_results": all_results[: k * 2],
            "user_preferences_found": len(
                [d for d in context_results if d.get("type") == "preference"]
            ),
            "user_facts_found": len(
                [d for d in context_results if d.get("type") == "fact"]
            ),
        }
