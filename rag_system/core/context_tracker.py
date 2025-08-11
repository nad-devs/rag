"""
Conversation Context Tracker for RAG System

Tracks conversation state across queries within a session to enable:
- Follow-up query detection
- Context-aware document retrieval  
- Prevention of document repetition
- Conversation memory for enhanced responses
"""

import time
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class QueryContext:
    """Represents a single query and its context within a conversation"""
    query: str
    embedding: np.ndarray
    timestamp: float
    document_ids: List[str]
    response_summary: str
    entities_mentioned: List[str]
    # New fields for semantic followup detection
    result_content: List[str] = None  # Key content returned to user
    key_takeaways: List[str] = None  # Structured takeaways from results


class ConversationTracker:
    """
    Manages conversation context across multiple queries in a RAG session.
    
    Features:
    - Tracks last 10 queries with embeddings and results
    - Detects follow-up queries using embedding similarity
    - Prevents document repetition across conversation
    - Provides context for enhanced response generation
    """
    
    def __init__(self, max_history: int = 10, follow_up_threshold: float = 0.7):
        """
        Initialize conversation tracker.
        
        Args:
            max_history: Maximum number of queries to keep in memory (default: 10)
            follow_up_threshold: Cosine similarity threshold for follow-up detection (default: 0.7)
        """
        self.max_history = max_history
        self.follow_up_threshold = follow_up_threshold
        self.query_history: List[QueryContext] = []
        self.session_start_time = time.time()
        
    def add_query_context(
        self, 
        query: str, 
        embedding: np.ndarray, 
        document_ids: List[str],
        response_summary: str = "",
        entities_mentioned: List[str] = None,
        result_content: List[str] = None,
        key_takeaways: List[str] = None
    ) -> None:
        """
        Add a new query and its results to conversation history.
        
        Args:
            query: The user's query text
            embedding: Query embedding vector
            document_ids: List of document IDs returned for this query
            response_summary: Brief summary of the response given
            entities_mentioned: Entities/topics discussed in this query
            result_content: Key content that was returned to the user
            key_takeaways: Structured takeaways from the results
        """
        if entities_mentioned is None:
            entities_mentioned = []
        if result_content is None:
            result_content = []
        if key_takeaways is None:
            key_takeaways = []
            
        context = QueryContext(
            query=query,
            embedding=embedding,
            timestamp=time.time(),
            document_ids=document_ids,
            response_summary=response_summary,
            entities_mentioned=entities_mentioned,
            result_content=result_content,
            key_takeaways=key_takeaways
        )
        
        self.query_history.append(context)
        
        # Maintain max history limit
        if len(self.query_history) > self.max_history:
            self.query_history.pop(0)
    
    def is_follow_up_query(self, current_query_embedding: np.ndarray) -> Tuple[bool, float, Optional[QueryContext]]:
        """
        Determine if current query is a follow-up to a recent query.
        
        Args:
            current_query_embedding: Embedding of the current query
            
        Returns:
            Tuple of (is_follow_up, max_similarity, most_similar_context)
        """
        if not self.query_history:
            return False, 0.0, None
        
        # Compare with recent queries (last 3 for follow-up detection)
        recent_contexts = self.query_history[-3:]
        max_similarity = 0.0
        most_similar_context = None
        
        for context in recent_contexts:
            # Calculate cosine similarity between embeddings
            similarity = cosine_similarity(
                current_query_embedding.reshape(1, -1),
                context.embedding.reshape(1, -1)
            )[0][0]
            
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_context = context
        
        is_follow_up = max_similarity >= self.follow_up_threshold
        return is_follow_up, max_similarity, most_similar_context
    
    def get_recent_context(self, limit: int = 3) -> List[QueryContext]:
        """
        Get the most recent query contexts.
        
        Args:
            limit: Number of recent contexts to return
            
        Returns:
            List of recent QueryContext objects
        """
        return self.query_history[-limit:] if self.query_history else []
    
    def get_used_document_ids(self, recent_only: bool = True, for_followup: bool = False) -> Set[str]:
        """
        Get set of document IDs that have been shown in this conversation.
        
        Args:
            recent_only: If True, only return docs from last N queries
            for_followup: If True, only block docs from immediate last query (less aggressive)
            
        Returns:
            Set of document IDs previously shown to user
        """
        if for_followup:
            # For follow-ups, only block documents from the immediate previous query
            # This allows deeper exploration of the same topic with different documents
            contexts_to_check = self.query_history[-1:] if self.query_history else []
        else:
            # For new topics, block documents from last 3 queries (reduced from 5)
            contexts_to_check = (
                self.query_history[-3:] if recent_only and len(self.query_history) > 3 
                else self.query_history
            )
        
        used_docs = set()
        for context in contexts_to_check:
            used_docs.update(context.document_ids)
        
        return used_docs
    
    def get_conversation_summary(self) -> Dict:
        """
        Generate a summary of the current conversation for context-aware responses.
        
        Returns:
            Dictionary with conversation metadata
        """
        if not self.query_history:
            return {
                "total_queries": 0,
                "session_duration_minutes": 0,
                "topics_discussed": [],
                "recent_entities": []
            }
        
        # Calculate session duration
        duration_seconds = time.time() - self.session_start_time
        duration_minutes = round(duration_seconds / 60, 1)
        
        # Extract topics and entities from recent queries
        recent_entities = []
        topics_discussed = []
        
        for context in self.query_history[-3:]:  # Last 3 queries
            recent_entities.extend(context.entities_mentioned)
            if context.response_summary:
                topics_discussed.append(context.response_summary[:100])  # Truncate long summaries
        
        return {
            "total_queries": len(self.query_history),
            "session_duration_minutes": duration_minutes,
            "topics_discussed": topics_discussed,
            "recent_entities": list(set(recent_entities)),  # Deduplicate
            "last_query": self.query_history[-1].query if self.query_history else None
        }
    
    def clear_session(self) -> None:
        """Reset conversation context (called on new session/page refresh)"""
        self.query_history.clear()
        self.session_start_time = time.time()
    
    def get_context_for_response_generation(self) -> Dict:
        """
        Get formatted context for LLM response generation.
        
        Returns:
            Dictionary with context information for prompt construction
        """
        if not self.query_history:
            return {"has_context": False}
        
        recent_contexts = self.get_recent_context(limit=2)
        
        context_info = {
            "has_context": True,
            "previous_queries": [ctx.query for ctx in recent_contexts],
            "previous_topics": [ctx.response_summary for ctx in recent_contexts if ctx.response_summary],
            "mentioned_entities": [],
            "conversation_depth": len(self.query_history)
        }
        
        # Collect mentioned entities from recent context
        for ctx in recent_contexts:
            context_info["mentioned_entities"].extend(ctx.entities_mentioned)
        
        # Deduplicate entities
        context_info["mentioned_entities"] = list(set(context_info["mentioned_entities"]))
        
        return context_info
    
    def debug_conversation_state(self) -> str:
        """
        Generate debug output showing current conversation state.
        
        Returns:
            Formatted string with conversation state information
        """
        if not self.query_history:
            return "DEBUG: No conversation history"
        
        debug_lines = [
            f"DEBUG: Conversation State",
            f"  - Total queries: {len(self.query_history)}",
            f"  - Session duration: {round((time.time() - self.session_start_time) / 60, 1)} minutes",
            f"  - Used documents: {len(self.get_used_document_ids())}",
            f"  - Recent queries:"
        ]
        
        for i, ctx in enumerate(self.query_history[-3:], 1):
            query_preview = ctx.query[:50] + "..." if len(ctx.query) > 50 else ctx.query
            debug_lines.append(f"    {i}. \"{query_preview}\" ({len(ctx.document_ids)} docs)")
        
        return "\n".join(debug_lines)