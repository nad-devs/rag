"""
E5-Large-V2 Similarity-Based Follow-up Detection

Uses semantic similarity with intelligent contextual analysis
to determine conversation continuation vs new topics.
"""

import time
import re
import numpy as np
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()


class E5FollowupDetector:
    """
    Smart follow-up detection using E5-Large-V2 embeddings with contextual analysis.
    
    Features:
    - Semantic similarity with adaptive thresholds
    - Pronoun reference detection
    - Entity overlap analysis
    - Query type classification
    - Context-aware decision making
    """
    
    def __init__(self):
        """Initialize E5-Large-V2 follow-up detector."""
        print("🚀 Initializing E5-Large-V2 follow-up detection...")
        
        self.available = False
        
        try:
            print("📥 Loading E5-Large-V2 embedding model...")
            self.embedding_model = SentenceTransformer('intfloat/e5-large-v2', device='cpu')
            self.embedding_cache = {}
            self.available = True
            print("✅ E5-Large-V2 model loaded successfully")
            print("💾 Model size: ~1.3GB, Embedding dimension: 1024")
        except Exception as e:
            print(f"❌ E5-Large-V2 initialization failed: {e}")
            self.embedding_model = None
    
    def is_follow_up_query(
        self,
        current_query: str,
        conversation_history: List[Dict],
        previous_entities: List[str] = None,
        similarity_score: float = 0.0  # We'll compute our own
    ) -> Tuple[bool, float, str, Dict]:
        """
        Smart follow-up detection using E5 embeddings with contextual analysis.
        
        Args:
            current_query: The user's current query
            conversation_history: List of previous queries with context
            previous_entities: Entities mentioned in recent conversation
            similarity_score: Ignored - we compute our own
            
        Returns:
            Tuple of (is_follow_up, confidence, reasoning, debug_info)
        """
        if not conversation_history:
            return False, 1.0, "No previous conversation context", {
                "processing_time_ms": 0,
                "method": "no_history"
            }
        
        if not self.available:
            return False, 0.0, "E5 model not available", {
                "processing_time_ms": 0,
                "method": "e5_unavailable"
            }
        
        start_time = time.time()
        recent_context = conversation_history[-1]
        previous_query = recent_context.get("query", "")
        
        # Level 1: Strong pronoun detection (highest confidence)
        pronoun_result = self._detect_pronouns(current_query, previous_query)
        if pronoun_result["is_pronoun_reference"]:
            return True, 0.95, pronoun_result["reasoning"], {
                "processing_time_ms": int((time.time() - start_time) * 1000),
                "method": "pronoun_detection",
                "pronouns_found": pronoun_result["pronouns"]
            }
        
        # Level 2: Semantic similarity analysis
        try:
            similarity_result = self._analyze_similarity(current_query, previous_query)
            entity_result = self._analyze_entities(current_query, previous_query, previous_entities or [])
            query_type_result = self._analyze_query_types(current_query, previous_query)
            
            # Level 3: Intelligent decision making
            decision_result = self._make_intelligent_decision(
                similarity_result, entity_result, query_type_result, 
                current_query, previous_query
            )
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return decision_result["is_follow_up"], decision_result["confidence"], decision_result["reasoning"], {
                "processing_time_ms": processing_time,
                "method": "e5_intelligent_analysis",
                "similarity_score": similarity_result["score"],
                "entity_overlap": entity_result["overlap_ratio"],
                "query_types": f"{query_type_result['previous']} → {query_type_result['current']}"
            }
            
        except Exception as e:
            return False, 0.5, f"E5 analysis failed: {str(e)}", {
                "processing_time_ms": int((time.time() - start_time) * 1000),
                "method": "error",
                "error": str(e)
            }
    
    def _detect_pronouns(self, current_query: str, previous_query: str) -> Dict:
        """Detect clear pronoun references indicating follow-up."""
        pronouns = ["it", "that", "this", "them", "they", "those", "these"]
        current_words = current_query.lower().split()
        
        found_pronouns = [p for p in pronouns if p in current_words]
        
        if not found_pronouns:
            return {"is_pronoun_reference": False}
        
        # Check if pronoun is likely referring to previous content
        # Stronger signal if pronoun is early in the query
        pronoun_positions = [current_words.index(p) for p in found_pronouns]
        early_pronoun = any(pos < 3 for pos in pronoun_positions)
        
        if early_pronoun:
            return {
                "is_pronoun_reference": True,
                "pronouns": found_pronouns,
                "reasoning": f"Clear pronoun reference: '{', '.join(found_pronouns)}' likely referring to previous topic"
            }
        
        return {"is_pronoun_reference": False}
    
    def _get_cached_embedding(self, text: str) -> np.ndarray:
        """Get embedding with caching for performance."""
        text_hash = hash(text)
        if text_hash not in self.embedding_cache:
            # E5 models work better with instruction prefixes
            prefixed_text = f"query: {text}"
            self.embedding_cache[text_hash] = self.embedding_model.encode(prefixed_text)
        return self.embedding_cache[text_hash]
    
    def _analyze_similarity(self, current_query: str, previous_query: str) -> Dict:
        """Analyze semantic similarity between queries."""
        current_embedding = self._get_cached_embedding(current_query)
        previous_embedding = self._get_cached_embedding(previous_query)
        
        similarity = cosine_similarity(
            current_embedding.reshape(1, -1),
            previous_embedding.reshape(1, -1)
        )[0][0]
        
        return {
            "score": float(similarity),
            "level": self._classify_similarity(similarity)
        }
    
    def _classify_similarity(self, score: float) -> str:
        """Classify similarity level."""
        if score > 0.85:
            return "very_high"
        elif score > 0.75:
            return "high"
        elif score > 0.65:
            return "moderate"
        elif score > 0.5:
            return "low"
        else:
            return "very_low"
    
    def _analyze_entities(self, current_query: str, previous_query: str, previous_entities: List[str]) -> Dict:
        """Analyze entity overlap between queries."""
        # Extract key entities/terms from both queries
        current_entities = self._extract_key_terms(current_query)
        previous_query_entities = self._extract_key_terms(previous_query)
        
        # Combine with provided previous entities
        all_previous_entities = set(previous_query_entities + [e.lower() for e in previous_entities])
        current_entity_set = set(current_entities)
        
        overlap = current_entity_set & all_previous_entities
        overlap_ratio = len(overlap) / max(len(current_entity_set), 1)
        
        return {
            "current_entities": current_entities,
            "previous_entities": list(all_previous_entities),
            "overlap": list(overlap),
            "overlap_ratio": overlap_ratio
        }
    
    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms/entities from query."""
        # Common tech terms and tools
        tech_terms = [
            "claude code", "claude", "gemini cli", "gemini", "rag", "vector database",
            "qdrant", "pinecone", "weaviate", "llm", "ai", "vibe coding", "backend",
            "frontend", "developer", "programming", "coding", "automation", "docker",
            "github", "copilot", "cursor", "windsurf", "fine-tuning", "prompt engineering"
        ]
        
        query_lower = query.lower()
        found_terms = []
        
        # Find tech terms
        for term in tech_terms:
            if term in query_lower:
                found_terms.append(term)
        
        # Extract other potential entities (capitalized words, quoted terms)
        words = query.split()
        for word in words:
            if len(word) > 3 and word[0].isupper():
                found_terms.append(word.lower())
        
        return list(set(found_terms))
    
    def _analyze_query_types(self, current_query: str, previous_query: str) -> Dict:
        """Analyze the types of queries to understand intent."""
        def classify_query(query: str) -> str:
            query_lower = query.lower()
            
            # Question types
            if any(query_lower.startswith(q) for q in ["what is", "what are", "define"]):
                return "definition"
            elif any(query_lower.startswith(q) for q in ["how do", "how to", "how can"]):
                return "instruction"
            elif any(query_lower.startswith(q) for q in ["why", "why does", "why should"]):
                return "explanation"
            elif any(query_lower.startswith(q) for q in ["should", "can", "is it"]):
                return "recommendation"
            elif any(query_lower.startswith(q) for q in ["what happens", "what about"]):
                return "consequence"
            elif "vs" in query_lower or "difference" in query_lower or "compare" in query_lower:
                return "comparison"
            elif any(word in query_lower for word in ["tips", "advice", "best"]):
                return "advice"
            else:
                return "general"
        
        return {
            "current": classify_query(current_query),
            "previous": classify_query(previous_query)
        }
    
    def _make_intelligent_decision(
        self, 
        similarity: Dict, 
        entities: Dict, 
        query_types: Dict,
        current_query: str,
        previous_query: str
    ) -> Dict:
        """Make intelligent follow-up decision based on all factors."""
        
        sim_score = similarity["score"]
        entity_overlap = entities["overlap_ratio"]
        current_type = query_types["current"]
        previous_type = query_types["previous"]
        
        # DEBUG: Print analysis details (COMMENTED OUT FOR CLEAN OUTPUT)
        # print(f"🔍 E5 Intelligent Analysis:")
        # print(f"  Similarity: {sim_score:.3f} ({similarity['level']})")
        # print(f"  Entity overlap: {entity_overlap:.3f} ({len(entities['overlap'])}/{len(entities['current_entities'])})")
        # print(f"  Query types: {previous_type} → {current_type}")
        # print(f"  Shared entities: {entities['overlap']}")
        
        # Decision rules based on combinations
        
        # Rule 1: High similarity + entity overlap = likely follow-up
        if sim_score > 0.75 and entity_overlap > 0.5:
            return {
                "is_follow_up": True,
                "confidence": min(0.9, sim_score + entity_overlap * 0.3),
                "reasoning": f"High similarity ({sim_score:.2f}) + strong entity overlap ({entity_overlap:.2f})"
            }
        
        # Rule 2: Same domain, different questions = NOT follow-up (STRENGTHENED)
        if (current_type != previous_type and sim_score < 0.9):
            return {
                "is_follow_up": False,
                "confidence": 0.9,
                "reasoning": f"Different question types: {previous_type} → {current_type} (sim: {sim_score:.2f})"
            }
        
        # Rule 3: Clarification patterns = follow-up
        clarification_words = ["which", "what about", "how about", "what if"]
        if any(word in current_query.lower() for word in clarification_words) and sim_score > 0.6:
            return {
                "is_follow_up": True,
                "confidence": 0.85,
                "reasoning": f"Clarification question with moderate similarity ({sim_score:.2f})"
            }
        
        # Rule 4: Sequential instructions = follow-up
        if (previous_type in ["definition", "explanation"] and 
            current_type in ["instruction", "consequence"] and sim_score > 0.65):
            return {
                "is_follow_up": True,
                "confidence": 0.8,
                "reasoning": f"Natural progression: {previous_type} → {current_type}"
            }
        
        # Rule 5: Different domains = NOT follow-up
        if sim_score < 0.5 and entity_overlap < 0.2:
            return {
                "is_follow_up": False,
                "confidence": 0.9,
                "reasoning": f"Different domains: low similarity ({sim_score:.2f}) + no entity overlap"
            }
        
        # Rule 6: Very high threshold for unclear cases (MUCH MORE CONSERVATIVE)
        threshold = 0.92  # Much higher threshold - require near-identical semantic meaning
        is_followup = sim_score > threshold
        
        return {
            "is_follow_up": is_followup,
            "confidence": 0.8,
            "reasoning": f"Conservative fallback: similarity {sim_score:.2f} {'above' if is_followup else 'below'} strict threshold {threshold:.2f}"
        }
    
    def print_debug_analysis(self, current_query: str, is_follow_up: bool, confidence: float, reasoning: str, debug_info: Dict, **kwargs):
        """Print debug analysis for compatibility with existing code."""
        print(f"🔍 E5 Follow-up Analysis:")
        print(f"  Query: {current_query}")
        print(f"  Decision: {'FOLLOW-UP' if is_follow_up else 'NEW-TOPIC'}")
        print(f"  Confidence: {confidence:.2f}")
        print(f"  Method: {debug_info.get('method', 'unknown')}")
        print(f"  Reasoning: {reasoning}")
        if debug_info.get("processing_time_ms"):
            print(f"  Processing time: {debug_info['processing_time_ms']}ms")
        if debug_info.get("similarity_score"):
            print(f"  Similarity: {debug_info['similarity_score']:.3f}")
        if debug_info.get("entity_overlap"):
            print(f"  Entity overlap: {debug_info['entity_overlap']:.3f}")


def test_e5_detector():
    """Test function for standalone testing."""
    print("🧪 Testing E5 Follow-up Detector")
    print("=" * 60)
    
    try:
        detector = E5FollowupDetector()
        
        if not detector.available:
            print("❌ E5 model not available for testing")
            return
        
        # Test cases based on your actual scenarios
        test_cases = [
            {
                "name": "Clear Pronoun Follow-up",
                "previous": "What is Claude Code?",
                "current": "How do I install it?",
                "expected": True
            },
            {
                "name": "Different Backend Questions (Should be NEW)",
                "previous": "Should backend developers use vibe coding?",
                "current": "AI coding tips for backend developers",
                "expected": False
            },
            {
                "name": "Comparison vs General Question (Should be NEW)",
                "previous": "What is the difference between Gemini CLI and Claude Code?",
                "current": "5 AI topics every software engineer should know",
                "expected": False
            },
            {
                "name": "Tool Clarification (Might be follow-up)",
                "previous": "Ed discusses vector databases",
                "current": "Which one should I choose?",
                "expected": True
            },
            {
                "name": "Completely Different Topics",
                "previous": "What is Claude Code?",
                "current": "What's the weather today?",
                "expected": False
            }
        ]
        
        correct_predictions = 0
        
        for i, test in enumerate(test_cases, 1):
            print(f"\n📝 Test Case {i}: {test['name']}")
            print(f"Previous: {test['previous']}")
            print(f"Current: {test['current']}")
            print(f"Expected: {'FOLLOW-UP' if test['expected'] else 'NEW TOPIC'}")
            
            conversation_history = [{"query": test["previous"]}]
            result = detector.is_follow_up_query(
                test["current"], 
                conversation_history
            )
            
            is_follow_up, confidence, reasoning, debug_info = result
            correct = (is_follow_up == test["expected"])
            
            if correct:
                correct_predictions += 1
            
            print(f"Result: {'✅ CORRECT' if correct else '❌ WRONG'}")
            print(f"Prediction: {'FOLLOW-UP' if is_follow_up else 'NEW TOPIC'} (confidence: {confidence:.2f})")
            print(f"Time: {debug_info.get('processing_time_ms', 0)}ms")
        
        accuracy = (correct_predictions / len(test_cases)) * 100
        print(f"\n🎯 RESULTS:")
        print(f"   Accuracy: {accuracy:.1f}% ({correct_predictions}/{len(test_cases)})")
        
        if accuracy >= 80:
            print(f"✅ E5 detector ready for production!")
        else:
            print(f"⚠️ Consider rule tuning to improve accuracy")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_e5_detector()