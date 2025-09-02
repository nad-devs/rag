#!/usr/bin/env python3
"""
Local Model Synthesizer for RAG System
Replaces Claude API calls with local Mistral 7B model for document synthesis
"""

import json
import requests
import re
from typing import Dict, List, Any
import time


class LocalMistralSynthesizer:
    """
    Local Mistral 7B model interface for document synthesis and question answering.
    
    Designed to replace Claude API calls in the RAG system with local inference
    while maintaining the same JSON response format and quality.
    """
    
    def __init__(self, model_name: str = "mistral:7b-instruct", base_url: str = "http://localhost:11434"):
        """
        Initialize local model synthesizer
        
        Args:
            model_name: Ollama model name (default: mistral:7b-instruct)
            base_url: Ollama API base URL (default: http://localhost:11434)
        """
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        
        # Test connection on initialization
        self._test_connection()
    
    def _test_connection(self):
        """Test connection to Ollama API"""
        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model_name,
                    "prompt": "Test connection",
                    "stream": False
                },
                timeout=60
            )
            if response.status_code == 200:
                print(f"✅ Connected to local model: {self.model_name}")
            else:
                print(f"⚠️ Connection issue with Ollama: {response.status_code}")
        except Exception as e:
            print(f"❌ Failed to connect to Ollama: {e}")
            raise
    
    def synthesize_documents(
        self, 
        prompt: str, 
        temperature: float = 0.1, 
        max_tokens: int = 2500,
        timeout: float = 90.0
    ) -> str:
        """
        Generate response using local Mistral model
        
        Args:
            prompt: The complete prompt for document synthesis
            temperature: Generation temperature (0.1 for consistent results)
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            
        Returns:
            Raw text response from the model
        """
        print(f"🤖 MISTRAL SYNTHESIS: Processing prompt ({len(prompt)} chars)")
        start_time = time.time()
        
        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                        "top_p": 0.9,
                        "repeat_penalty": 1.1
                    }
                },
                timeout=timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                raw_content = result.get("response", "")
                
                duration = time.time() - start_time
                print(f"✅ MISTRAL RESPONSE: {len(raw_content)} chars in {duration:.1f}s")
                
                return raw_content
            else:
                print(f"❌ Mistral API error: {response.status_code}")
                raise Exception(f"Mistral API returned {response.status_code}: {response.text}")
                
        except requests.exceptions.Timeout:
            print(f"⏰ Mistral request timed out after {timeout}s")
            raise
        except Exception as e:
            print(f"❌ Mistral synthesis error: {e}")
            raise
    
    def extract_with_context_aware_prompt(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2500,
        timeout: float = 60.0
    ) -> Dict[str, Any]:
        """
        Context-aware document synthesis that mimics Claude's JSON response format
        
        This method replaces the Claude API call in _extract_with_context_aware_prompt
        and returns the same JSON structure expected by the RAG system.
        
        Args:
            prompt: Complete synthesis prompt with documents and instructions
            temperature: Generation temperature  
            max_tokens: Maximum response tokens
            timeout: Request timeout
            
        Returns:
            Dict with same structure as Claude response:
            {
                'answer': str,
                'confidence': float,
                'response_type': str,
                'reasoning': str,
                'follow_up_questions': List[str]
            }
        """
        try:
            # Get raw response from Mistral
            raw_content = self.synthesize_documents(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout
            )
            
            print(f"🧠 MISTRAL PROCESSING: {len(raw_content)} characters")
            
            # Try to parse as JSON first (Mistral should follow instructions)
            try:
                result = json.loads(raw_content)
                
                # Validate required fields and add defaults if missing
                answer = result.get('answer', raw_content)
                confidence = result.get('confidence', 0.85)  # Mistral is quite good
                response_type = result.get('response_type', 'local_synthesis')
                reasoning = result.get('reasoning', 'Local Mistral document synthesis')
                follow_up_questions = result.get('follow_up_questions', [])
                
                print(f"✅ MISTRAL JSON: Parsed structured response")
                print(f"   Confidence: {confidence}")
                print(f"   Type: {response_type}")
                
                return {
                    'answer': answer,
                    'confidence': confidence,
                    'response_type': response_type,
                    'reasoning': reasoning,
                    'follow_up_questions': follow_up_questions
                }
                
            except json.JSONDecodeError:
                print(f"🔧 MISTRAL FALLBACK: Parsing non-JSON response")
                
                # Fallback: treat entire response as answer
                # This happens when Mistral doesn't follow JSON format perfectly
                answer = raw_content.strip()
                
                # Try to extract confidence if mentioned in text  
                confidence_match = re.search(r'confidence[:\s]*([0-9]+\.?[0-9]*)', raw_content.lower())
                confidence = float(confidence_match.group(1)) if confidence_match else 0.85
                
                # Generate appropriate follow-up questions based on content
                follow_up_questions = self._generate_followup_questions(answer)
                
                return {
                    'answer': answer,
                    'confidence': confidence,
                    'response_type': 'local_synthesis_fallback',
                    'reasoning': 'Local Mistral synthesis (fallback parsing)',
                    'follow_up_questions': follow_up_questions
                }
                
        except Exception as e:
            print(f"❌ MISTRAL EXTRACTION ERROR: {e}")
            
            # Emergency fallback
            return {
                'answer': f"I apologize, but I encountered an error processing your request: {str(e)}",
                'confidence': 0.1,
                'response_type': 'error_fallback',
                'reasoning': 'Local synthesis error - emergency fallback',
                'follow_up_questions': ["Please try rephrasing your question"]
            }
    
    def _generate_followup_questions(self, answer: str) -> List[str]:
        """
        Generate follow-up questions based on the answer content
        
        Args:
            answer: The generated answer text
            
        Returns:
            List of relevant follow-up questions
        """
        # Simple heuristic-based follow-up generation
        follow_ups = []
        
        # Look for mentions of tools/technologies
        tools = re.findall(r'\b(?:Claude Code|GitHub|Git|Docker|Python|JavaScript|AI|LLM)\b', answer, re.IGNORECASE)
        if tools:
            unique_tools = list(set(tools))[:2]  # Max 2
            for tool in unique_tools:
                follow_ups.append(f"Tell me more about {tool}")
        
        # Look for process-related content
        if any(word in answer.lower() for word in ['setup', 'install', 'configure', 'start']):
            follow_ups.append("What are the next steps?")
        
        # Look for comparison content
        if any(word in answer.lower() for word in ['vs', 'versus', 'compare', 'difference']):
            follow_ups.append("Which option is better for beginners?")
        
        # Default follow-ups if none generated
        if not follow_ups:
            follow_ups = [
                "Can you give me more examples?",
                "What are the common pitfalls to avoid?"
            ]
        
        return follow_ups[:3]  # Max 3 follow-ups
    
    def rank_documents(
        self, 
        query: str, 
        doc_summaries: List[Dict[str, Any]],
        max_docs: int = 20,
        timeout: float = 15.0
    ) -> List[Dict[str, Any]]:
        """
        Rank documents using local Mistral model for contextual relevance
        
        Replaces the Claude document ranking functionality with local inference.
        
        Args:
            query: User's query for ranking context
            doc_summaries: List of document summaries with metadata
            max_docs: Maximum documents to return
            timeout: Request timeout
            
        Returns:
            List of ranked documents with scores
        """
        print(f"🎯 MISTRAL RANKING: Evaluating {len(doc_summaries)} documents")
        
        # Create ranking prompt
        ranking_prompt = self._create_ranking_prompt(query, doc_summaries)
        
        try:
            raw_response = self.synthesize_documents(
                prompt=ranking_prompt,
                temperature=0.1,
                max_tokens=1000,
                timeout=timeout
            )
            
            # Parse ranking scores
            ranked_docs = self._parse_ranking_response(raw_response, doc_summaries)
            
            # Return top N documents
            return ranked_docs[:max_docs]
            
        except Exception as e:
            print(f"⚠️ MISTRAL RANKING ERROR: {e}, falling back to original order")
            # Fallback: return documents in original order
            return doc_summaries[:max_docs]
    
    def _create_ranking_prompt(self, query: str, doc_summaries: List[Dict[str, Any]]) -> str:
        """Create prompt for document ranking"""
        
        docs_text = ""
        for i, doc in enumerate(doc_summaries, 1):
            title = doc.get('title', f"Document {i}")
            summary = doc.get('preview', '')[:200]  # First 200 chars
            main_lesson = doc.get('main_lesson', '')
            
            docs_text += f"""
Document {i}: {doc.get('doc_id', f'doc_{i}')}
Title: {title}
Main Lesson: {main_lesson}
Preview: {summary}...

"""
        
        return f"""You are ranking documents for relevance to a user's query.

USER QUERY: "{query}"

DOCUMENTS TO RANK:
{docs_text}

INSTRUCTIONS:
1. Score each document 1-10 based on relevance to the query
2. Consider both direct topic match AND contextual usefulness  
3. Higher scores = more relevant to answering the user's question
4. Return ONLY a JSON array with format: [{{"doc_id": "doc_1", "score": 9}}, {{"doc_id": "doc_2", "score": 7}}]

JSON Response:"""
    
    def _parse_ranking_response(
        self, 
        raw_response: str, 
        original_docs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Parse ranking response and sort documents by score"""
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', raw_response, re.DOTALL)
            if json_match:
                rankings = json.loads(json_match.group())
                
                # Create doc_id to score mapping
                scores = {item['doc_id']: item['score'] for item in rankings}
                
                # Sort original docs by score (highest first)
                sorted_docs = sorted(
                    original_docs,
                    key=lambda doc: scores.get(doc.get('doc_id', ''), 0),
                    reverse=True
                )
                
                print(f"✅ MISTRAL RANKING: Successfully ranked {len(sorted_docs)} documents")
                return sorted_docs
                
        except Exception as e:
            print(f"⚠️ Ranking parse error: {e}")
        
        # Fallback: return original order
        return original_docs


# Convenience function for easy integration
def create_local_synthesizer() -> LocalMistralSynthesizer:
    """Create and return a LocalMistralSynthesizer instance"""
    return LocalMistralSynthesizer()


if __name__ == "__main__":
    # Test the synthesizer
    synthesizer = LocalMistralSynthesizer()
    
    test_prompt = """You are analyzing Ed Honour's Instagram content to answer user questions.

DOCUMENTS TO ANALYZE:
Document 1: Test document about Claude Code setup
Content: Create a .ClaudeCode directory in your project root and configure rules...

USER QUESTION: How do I get started with Claude Code?

INSTRUCTIONS:
1. Provide a comprehensive answer based on the documents
2. Format as clear sections with bullet points for complex answers
3. Return response as JSON: {"answer": "...", "confidence": 0.9, "response_type": "synthesis", "reasoning": "...", "follow_up_questions": [...]}

RESPONSE:"""
    
    result = synthesizer.extract_with_context_aware_prompt(test_prompt)
    print(f"\n🧪 TEST RESULT:")
    print(f"Answer: {result['answer'][:100]}...")
    print(f"Confidence: {result['confidence']}")
    print(f"Type: {result['response_type']}")
