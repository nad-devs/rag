"""
Ollama-based Follow-up Detection for RAG System

Uses Ollama Llama 3.1:8B for intelligent follow-up detection.
Based on battle test results showing 78% accuracy vs Claude Haiku.
"""

import time
import json
import re
from typing import List, Dict, Optional, Tuple
import requests
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()


class OllamaFollowupDetector:
    """
    Enhanced follow-up detection using DeepSeek-R1:8B reasoning model.
    
    Features:
    - Uses DeepSeek-R1's natural thinking process
    - Leverages step-by-step reasoning in <think> blocks
    - 100% free local inference
    - No rate limits or API costs
    - Superior reasoning capabilities
    """
    
    def __init__(self, model_name: str = "deepseek-r1:8b"):
        """
        Initialize Ollama follow-up detector.
        
        Args:
            model_name: Ollama model to use (default: llama3.1:8b)
        """
        print(f"🧠 Initializing DeepSeek-R1 follow-up detection ({model_name})...")
        
        self.model_name = model_name
        self.ollama_url = "http://localhost:11434"
        self.available = False
        
        # Test Ollama connectivity
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                available_models = [model['name'] for model in response.json().get('models', [])]
                
                if model_name in available_models:
                    print(f"✅ Ollama model {model_name} ready")
                    self.available = True
                else:
                    print(f"⚠️ Model {model_name} not found. Available: {available_models}")
                    print(f"💡 Run: ollama pull {model_name}")
                    self.available = False
            else:
                print("❌ Ollama not responding")
                self.available = False
                
        except Exception as e:
            print(f"❌ Ollama initialization failed: {e}")
            self.available = False
    
    def is_follow_up_query(
        self,
        current_query: str,
        conversation_history: List[Dict],
        previous_entities: List[str] = None,
        similarity_score: float = 0.0
    ) -> Tuple[bool, float, str, Dict]:
        """
        Enhanced follow-up detection using Ollama Llama 3.1.
        
        Args:
            current_query: The user's current query
            conversation_history: List of previous queries with context
            previous_entities: Entities mentioned in recent conversation
            similarity_score: Embedding similarity score (for comparison)
            
        Returns:
            Tuple of (is_follow_up, confidence, reasoning, debug_info)
        """
        if not conversation_history:
            return False, 1.0, "No previous conversation context", {
                "processing_time_ms": 0,
                "method": "no_history"
            }
        
        if not self.available:
            return False, 0.0, "Ollama not available", {
                "processing_time_ms": 0,
                "method": "ollama_unavailable",
                "error": "Model not loaded"
            }
        
        start_time = time.time()
        recent_context = conversation_history[-1]
        previous_query = recent_context.get("query", "")
        
        # Level 1: Quick pronoun detection (same as Claude system)
        pronouns = ["it", "that", "this", "them", "they", "those", "these"]
        has_pronouns = any(pronoun in current_query.lower().split() for pronoun in pronouns)
        if has_pronouns and similarity_score > 0.8:
            return True, 0.95, "Clear pronoun reference with high similarity", {
                "processing_time_ms": int((time.time() - start_time) * 1000),
                "method": "pronoun_detection",
                "pronouns_found": [p for p in pronouns if p in current_query.lower()]
            }
        
        # Level 2: Ollama analysis for complex cases
        try:
            ollama_result = self._ollama_analysis(current_query, previous_query, previous_entities or [], similarity_score)
            processing_time = int((time.time() - start_time) * 1000)
            return ollama_result["is_follow_up"], ollama_result["confidence"], ollama_result["reasoning"], {
                "processing_time_ms": processing_time,
                "method": "ollama_analysis",
                "model": self.model_name
            }
            
        except Exception as e:
            return False, 0.5, f"Ollama analysis failed: {str(e)}", {
                "processing_time_ms": int((time.time() - start_time) * 1000),
                "method": "error",
                "error": str(e)
            }
    
    def _ollama_analysis(
        self,
        current_query: str,
        previous_query: str,
        previous_entities: List[str],
        similarity_score: float
    ) -> Dict:
        """Run DeepSeek-R1 natural reasoning analysis."""
        
        # Extract entities from current query
        current_entities = self._extract_entities(current_query)
        
        # Focus on question structure patterns, not content knowledge
        prompt = f"""Previous: "{previous_query}"
Current: "{current_query}"

IGNORE what these tools/topics are. Focus ONLY on question patterns:

SAME-TOPIC-FOLLOWUP patterns:
- Asks for more details about SAME thing: "What is X?" → "How does X work?"
- Continues same subject: "Explain X" → "What types of X are there?"
- Simple pronoun reference WITHOUT new entities: "What is X?" → "How does it work?"

NEW-TOPIC patterns:
- Asks about different things: "What is X?" → "What is Y?"
- Introduces new entity for comparison: "What is X?" → "How does it compare to Y?"
- Different domains: tech → cooking, AI → sports

Answer only: SAME-TOPIC-FOLLOWUP or NEW-TOPIC"""
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 50
                    },
                    "stream": False
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                raw_response = result.get('response', '').strip()
                
                # DEBUG: Print raw response to understand what DeepSeek is saying
                print(f"🔍 DEBUG RAW RESPONSE: {raw_response[:200]}...")
                
                return self._parse_ollama_response(raw_response)
            else:
                raise Exception(f"HTTP {response.status_code}")
                
        except Exception as e:
            raise Exception(f"Ollama request failed: {str(e)}")
    
    def _parse_ollama_response(self, response: str) -> Dict:
        """Parse DeepSeek-R1's natural reasoning response."""
        
        # Extract thinking content if present
        thinking_content = ""
        final_answer = response
        
        if "<think>" in response:
            think_start = response.find("<think>")
            think_end = response.find("</think>")
            if think_start != -1 and think_end != -1:
                thinking_content = response[think_start + 7:think_end].strip()
                final_answer = response[think_end + 8:].strip()
            elif think_start != -1:
                thinking_content = response[think_start + 7:].strip()
                final_answer = ""
        
        # Look for SAME-TOPIC-FOLLOWUP or NEW-TOPIC in the response
        response_upper = final_answer.upper()
        
        if "SAME-TOPIC-FOLLOWUP" in response_upper or "SAME TOPIC FOLLOWUP" in response_upper:
            confidence = 0.95 if thinking_content else 0.9
            reasoning = "DeepSeek-R1: Same topic follow-up"
            return {"is_follow_up": True, "confidence": confidence, "reasoning": reasoning}
        
        elif "NEW-TOPIC" in response_upper or "NEW TOPIC" in response_upper:
            confidence = 0.95 if thinking_content else 0.9
            reasoning = "DeepSeek-R1: New topic (not same-topic follow-up)"
            return {"is_follow_up": False, "confidence": confidence, "reasoning": reasoning}
        
        # Analyze thinking content for reasoning indicators
        if thinking_content:
            thinking_upper = thinking_content.upper()
            
            # Look for follow-up indicators in thinking
            follow_indicators = ["FOLLOW-UP", "CONTINUATION", "RELATED", "BUILDS", "PRONOUN", "REFERS", "CONNECTED"]
            new_indicators = ["NEW TOPIC", "DIFFERENT", "SEPARATE", "INDEPENDENT", "UNRELATED"]
            
            follow_count = sum(1 for indicator in follow_indicators if indicator in thinking_upper)
            new_count = sum(1 for indicator in new_indicators if indicator in thinking_upper)
            
            if follow_count > new_count:
                return {"is_follow_up": True, "confidence": 0.8, "reasoning": f"DeepSeek thinking analysis: {follow_count} follow-up indicators"}
            elif new_count > follow_count:
                return {"is_follow_up": False, "confidence": 0.8, "reasoning": f"DeepSeek thinking analysis: {new_count} new-topic indicators"}
        
        # Enhanced fallback analysis with broader patterns
        full_response_upper = (thinking_content + " " + final_answer).upper()
        
        # Look for follow-up patterns
        if any(pattern in full_response_upper for pattern in [
            'SAME', 'CONTINUE', 'FOLLOW', 'ELABORATE', 'CLARIFY', 'BUILD', 'EXPAND'
        ]):
            return {"is_follow_up": True, "confidence": 0.75, "reasoning": "Follow-up patterns detected"}
            
        # Look for new-topic patterns  
        elif any(pattern in full_response_upper for pattern in [
            'NEW', 'DIFFERENT', 'SEPARATE', 'DISTINCT', 'UNRELATED', 'SWITCH'
        ]):
            return {"is_follow_up": False, "confidence": 0.75, "reasoning": "New-topic patterns detected"}
            
        # Removed pronoun rule - too aggressive and causes false positives
            
        else:
            return {"is_follow_up": False, "confidence": 0.6, "reasoning": "Unclear response, defaulting to new topic"}
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract key entities from text."""
        entities = []
        text_lower = text.lower()
        
        # AI/ML terms
        ai_terms = ["claude", "gpt", "llama", "ai", "rag", "fine-tuning", "embedding", "vector", "openai", "anthropic"]
        # Tools/platforms
        tools = ["qdrant", "pinecone", "weaviate", "selenium", "docker", "github", "copilot"]
        # Concepts
        concepts = ["automation", "agile", "scrum", "productivity", "startup", "funding"]
        
        all_terms = ai_terms + tools + concepts
        
        for term in all_terms:
            if term in text_lower:
                entities.append(term)
        
        return entities
    
    def print_debug_analysis(self, current_query: str, is_follow_up: bool, confidence: float, reasoning: str, debug_info: Dict, **kwargs):
        """Print debug analysis for compatibility with existing code."""
        print(f"🔍 Follow-up Analysis:")
        print(f"  Query: {current_query}")
        print(f"  Decision: {'FOLLOW-UP' if is_follow_up else 'NEW-TOPIC'}")
        print(f"  Confidence: {confidence:.2f}")
        print(f"  Method: {debug_info.get('method', 'unknown')}")
        print(f"  Reasoning: {reasoning}")
        if debug_info.get("processing_time_ms"):
            print(f"  Processing time: {debug_info['processing_time_ms']}ms")
        if debug_info.get("model"):
            print(f"  Model: {debug_info['model']}")


def test_ollama_detector():
    """Test function for standalone testing."""
    print("🧪 Testing Ollama Follow-up Detector")
    print("=" * 50)
    
    try:
        detector = OllamaFollowupDetector()
        
        # Test cases from battle tests
        test_cases = [
            {
                "previous": "Ed explains supervised fine-tuning for breaking into AI",
                "current": "What datasets should I use for fine-tuning?",
                "expected": True,
                "similarity": 0.7
            },
            {
                "previous": "Ed discusses his DeepSeek R1 distilled LAMA3 8B parameter model",
                "current": "How should I hire AI engineers for my startup?",
                "expected": False,
                "similarity": 0.1
            },
            {
                "previous": "Ed's RAG system uses Qdrant for vector storage",
                "current": "Why not Pinecone or Weaviate?",
                "expected": True,
                "similarity": 0.6
            }
        ]
        
        for i, test in enumerate(test_cases, 1):
            print(f"\n📝 Test Case {i}:")
            print(f"Previous: {test['previous']}")
            print(f"Current: {test['current']}")
            print(f"Expected: {'FOLLOW-UP' if test['expected'] else 'NEW TOPIC'}")
            
            conversation_history = [{"query": test["previous"]}]
            result = detector.is_follow_up_query(
                test["current"], 
                conversation_history,
                similarity_score=test["similarity"]
            )
            
            is_follow_up, confidence, reasoning, debug_info = result
            correct = (is_follow_up == test["expected"])
            
            print(f"Result: {'✅ CORRECT' if correct else '❌ WRONG'}")
            print(f"Prediction: {'FOLLOW-UP' if is_follow_up else 'NEW TOPIC'} (confidence: {confidence:.2f})")
            print(f"Reasoning: {reasoning}")
            print(f"Time: {debug_info.get('processing_time_ms', 0)}ms")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    test_ollama_detector()