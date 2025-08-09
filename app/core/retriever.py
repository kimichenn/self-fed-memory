"""Time-weighted retriever that combines semantic similarity with recency bias.

This retriever scores memories by both semantic similarity AND recency,
using a decay function: score = similarity + (1 - decay_rate) ^ hours_since_last_access

Key features:
- Prioritizes recent memories while preserving older ones
- Updates last_accessed_at timestamps when memories are retrieved
- Handles missing timestamps gracefully
- Compatible with our existing vector store interface
- Intelligent query analysis for implied and contextual retrieval
"""

from __future__ import annotations

from datetime import datetime
import math
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI

QUERY_ANALYSIS_PROMPT = """You are an intelligent query analyzer for a personal memory system.
Your job is to understand what information the user really needs and generate effective search queries.

The user has a personal knowledge base containing:
- Notes, documents, conversations
- Preferences, opinions, past decisions
- Learning materials, projects, ideas
- Personal context and background info
- Future goals

User Query: {user_query}

Analyze this query and generate 3-5 search queries that would help retrieve ALL relevant information, including:
1. Direct semantic matches
2. Background context the user might need
3. Related preferences or past decisions
4. Implied information needs

Examples:
- User: "Which restaurant should I choose?"
  Searches: ["restaurant preferences", "food allergies dietary restrictions", "recent restaurant experiences", "favorite cuisines types", "budget dining preferences"]

- User: "Help me solve this calculus problem"
  Searches: ["calculus problem solving", "mathematics learning style", "preferred math explanation format", "calculus notes examples", "math learning preferences"]

- User: "Based on what you know about me, which choice would I enjoy?"
  Searches: ["personal preferences hobbies", "past decisions choices", "personality traits values", "interests activities", "decision-making patterns"]

- User: "What should I do to best prepare for my upcoming interview?"
  Searches: ["previous interviews", "interview tips", "skills to highlight", "interests activities", "strengths weaknesses", "past projects"]

Output format: Return ONLY a JSON list of search query strings, no other text.
["query1", "query2", "query3", "query4", "query5"]
"""


class IntelligentQueryAnalyzer:
    """Analyzes user queries to generate comprehensive search strategies."""

    def __init__(self, llm: BaseChatModel = None):
        self.llm = llm or ChatOpenAI(model="gpt-4.1-04-14", temperature=0.1)

        self.prompt = ChatPromptTemplate.from_template(QUERY_ANALYSIS_PROMPT)
        self.chain = self.prompt | self.llm | StrOutputParser()

    def analyze_query(self, user_query: str) -> list[str]:
        """Analyze user query and generate multiple search queries."""
        try:
            response = self.chain.invoke({"user_query": user_query})

            # Parse JSON response
            import json

            queries = json.loads(response.strip())

            # Ensure we have a list of strings
            if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
                return queries[:5]  # Limit to 5 queries max
            else:
                # Fallback to original query if parsing fails
                return [user_query]

        except Exception:
            # Fallback to original query if analysis fails
            return [user_query]


class TimeWeightedRetriever:
    """A retriever that scores documents by similarity + time decay."""

    def __init__(
        self,
        vector_store: VectorStore,
        embeddings: Embeddings,
        decay_rate: float = 0.01,
        k: int = 5,
        llm: BaseChatModel = None,
        use_intelligent_queries: bool = True,
    ):
        """Initialize the time-weighted retriever.

        Args:
            vector_store: The underlying vector store
            embeddings: Embeddings model for query encoding
            decay_rate: Controls how quickly old memories decay (0-1, lower = slower decay)
            k: Number of documents to retrieve
            llm: Language model for query analysis
            use_intelligent_queries: Whether to use LLM-powered query analysis
        """
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.decay_rate = decay_rate
        self.k = k
        self.use_intelligent_queries = use_intelligent_queries

        # Query analyzer is optional based on configuration
        self.query_analyzer: IntelligentQueryAnalyzer | None
        if use_intelligent_queries:
            self.query_analyzer = IntelligentQueryAnalyzer(llm)
        else:
            self.query_analyzer = None

    def get_relevant_documents(self, query: str, **kwargs: Any) -> list[Document]:
        """Retrieve documents with time-weighted scoring.

        Returns documents sorted by combined similarity + recency score.
        Updates last_accessed_at timestamps for retrieved documents.
        """
        if self.use_intelligent_queries and self.query_analyzer:
            return self._intelligent_retrieval(query, **kwargs)
        else:
            return self._basic_retrieval(query, **kwargs)

    def _intelligent_retrieval(self, user_query: str, **kwargs: Any) -> list[Document]:
        """Advanced retrieval using LLM-generated queries for comprehensive coverage."""
        # Ensure analyzer is available (mypy: narrow Optional)
        if self.query_analyzer is None:
            return self._basic_retrieval(user_query, **kwargs)

        # Generate multiple search queries
        search_queries = self.query_analyzer.analyze_query(user_query)

        # Collect candidates from all queries
        all_candidates: dict[
            str, tuple[Document, float]
        ] = {}  # doc_id -> (doc, max_score)
        now = datetime.utcnow()

        for i, query in enumerate(search_queries):
            # Get more candidates per query for better coverage
            candidates_k = min(self.k * 2, 15)
            docs = self.vector_store.similarity_search(query, k=candidates_k)

            for doc in docs:
                doc_id = doc.metadata.get("id", doc.page_content[:50])

                # Calculate scores for this document with this query
                similarity_score = self._estimate_similarity_score(
                    docs.index(doc), len(docs)
                )
                time_score = self._calculate_time_score(doc, now)

                # Boost score slightly for earlier queries (more direct matches)
                query_boost = 1.0 - (
                    i * 0.1
                )  # First query gets full boost, later queries slightly less
                combined_score = (similarity_score + time_score) * query_boost

                # Keep the highest score for each document
                if (
                    doc_id not in all_candidates
                    or combined_score > all_candidates[doc_id][1]
                ):
                    all_candidates[doc_id] = (doc, combined_score)

        if not all_candidates:
            return []

        # Sort all candidates by score and take top k
        scored_docs = list(all_candidates.values())
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, _ in scored_docs[: self.k]]

        # Update last_accessed_at timestamps
        self._update_access_timestamps(top_docs, now)

        return top_docs

    def _basic_retrieval(self, query: str, **kwargs: Any) -> list[Document]:
        """Basic retrieval using single query (fallback/backward compatibility)."""
        # Get more candidates than needed for re-ranking
        candidates_k = min(self.k * 3, 20)  # Get 3x as many candidates for re-ranking

        # Get initial similarity-based results
        docs = self.vector_store.similarity_search(query, k=candidates_k)

        if not docs:
            return []

        # Calculate time-weighted scores
        now = datetime.utcnow()
        scored_docs = []

        for doc in docs:
            # Get similarity score (approximate from position, since Pinecone doesn't return scores by default)
            similarity_score = self._estimate_similarity_score(
                docs.index(doc), len(docs)
            )

            # Get time decay score
            time_score = self._calculate_time_score(doc, now)

            # Combined score – favour recency by down-weighting similarity.
            # Using ``decay_rate`` as the weight ensures that similarity has
            # *less* influence when a slow decay is configured (typical use
            # case) and slightly more influence for fast-decaying setups.
            combined_score = time_score + (self.decay_rate * similarity_score)
            scored_docs.append((combined_score, doc))

        # Sort by combined score and take top k
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        top_docs = [doc for _, doc in scored_docs[: self.k]]

        # Update last_accessed_at timestamps
        self._update_access_timestamps(top_docs, now)

        return top_docs

    def _estimate_similarity_score(self, position: int, total: int) -> float:
        """Estimate similarity score based on position in results.

        Since Pinecone similarity_search doesn't return scores by default,
        we estimate based on position. First result gets highest score.
        """
        if total <= 1:
            return 1.0
        # Linear decay from 1.0 to 0.1 based on position
        score = 1.0 - (position * 0.9 / (total - 1))
        return round(score, 10)  # Round to avoid floating point precision issues

    def _calculate_time_score(self, doc: Document, now: datetime) -> float:
        """Calculate time decay score for a document."""
        try:
            # Try last_accessed_at first, then fall back to created_at
            timestamp_str = doc.metadata.get("last_accessed_at") or doc.metadata.get(
                "created_at"
            )

            if not timestamp_str:
                # No timestamp available, give neutral time score
                return 0.0

            # Parse timestamp (handle both ISO format and other common formats)
            if isinstance(timestamp_str, str):
                # Handle Z suffix by replacing with +00:00
                clean_timestamp = timestamp_str.replace("Z", "+00:00")

                # Parse the timestamp
                timestamp = datetime.fromisoformat(clean_timestamp)

                # Convert to naive UTC if it's timezone-aware
                if timestamp.tzinfo is not None:
                    timestamp = timestamp.replace(tzinfo=None)
            else:
                timestamp = timestamp_str
                # Ensure it's naive
                if hasattr(timestamp, "tzinfo") and timestamp.tzinfo is not None:
                    timestamp = timestamp.replace(tzinfo=None)

            # Calculate hours since last access/creation
            hours_since = (now - timestamp).total_seconds() / 3600

            # Apply decay function: (1 - decay_rate) ^ hours_since
            # This gives higher scores to more recent documents
            time_score = math.pow(1 - self.decay_rate, hours_since)

            return time_score

        except (ValueError, TypeError):
            # If timestamp parsing fails, give neutral score
            return 0.0

    def _update_access_timestamps(self, docs: list[Document], accessed_at: datetime):
        """Update last_accessed_at timestamps for retrieved documents."""
        if not docs:
            return

        # Update metadata
        timestamp_str = accessed_at.isoformat()
        updated_docs = []

        for idx, doc in enumerate(docs):
            # Create updated document with new timestamp
            new_metadata = doc.metadata.copy()
            new_metadata["last_accessed_at"] = timestamp_str

            updated_doc = Document(page_content=doc.page_content, metadata=new_metadata)
            updated_docs.append(updated_doc)

            # Replace the original reference so callers see updated metadata
            docs[idx] = updated_doc

        # Upsert back to vector store to persist the timestamp updates
        try:
            if hasattr(self.vector_store, "upsert"):
                self.vector_store.upsert(updated_docs)
            elif hasattr(self.vector_store, "add_documents"):
                # For stores that don't have upsert, we'll try add_documents with the same IDs
                ids = [
                    doc.metadata.get("id")
                    for doc in updated_docs
                    if doc.metadata.get("id")
                ]
                if ids:
                    self.vector_store.add_documents(updated_docs, ids=ids)
        except Exception:
            # If timestamp update fails, log but don't break retrieval
            # In production, you might want to use proper logging here
            pass

    def search(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        """Convenience method that returns results as dictionaries."""
        docs = self.get_relevant_documents(query, **kwargs)
        return [{**doc.metadata, "content": doc.page_content} for doc in docs]
