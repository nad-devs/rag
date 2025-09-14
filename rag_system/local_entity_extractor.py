#!/usr/bin/env python3
"""
Local Entity Extractor
======================

Integrates local models for entity recognition alongside Claude for side-by-side comparison.
Uses the EXACT same prompt as Claude to ensure fair comparison.
"""

import requests
import time
from typing import Dict, Optional, Tuple
import os

class LocalEntityExtractor:
    """Extract entities using local models with Claude fallback"""
    
    def __init__(self, ollama_base_url: str = "http://localhost:11434", preload_model: bool = True):
        """Initialize local entity extractor"""
        self.ollama_base_url = ollama_base_url
        self.available = self._check_ollama_connection()

        if self.available:
            print("✅ Local entity extractor ready")
            if preload_model:
                self.warmup_model()
        else:
            print("⚠️ Local models unavailable - Claude-only mode")
    
    def _check_ollama_connection(self) -> bool:
        """Check if Ollama is available"""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/version", timeout=5)
            return response.status_code == 200
        except:
            return False

    def warmup_model(self) -> None:
        """Preload Mistral model into GPU memory with a dummy request"""
        try:
            print("🔥 Preloading Mistral 7B into GPU memory...")
            start_time = time.time()

            # Send a simple warmup request
            warmup_prompt = "Extract entity from: test query. Reply with entity:"
            payload = {
                "model": "mistral:7b-instruct",
                "prompt": warmup_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 10
                }
            }

            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                load_time = time.time() - start_time
                print(f"✅ Mistral 7B loaded into GPU in {load_time:.2f}s - Ready for fast inference!")
            else:
                print(f"⚠️ Model preload failed: {response.status_code}")

        except Exception as e:
            print(f"⚠️ Could not preload model: {e}")
    
    def extract_entity(self, query: str) -> Dict[str, any]:
        """
        Extract entity using Mistral
        
        Args:
            query: User's query
            
        Returns:
            Dict with extraction result
        """
        print(f"🔍 LOCAL EXTRACTOR INTERNAL DEBUG:")
        print(f"   Query: '{query}'")
        print(f"   Ollama available: {self.available}")
        print(f"   Ollama URL: {self.ollama_base_url}")
        
        # Get Mistral result
        start_time = time.time()
        if self.available:
            try:
                print(f"   Starting Mistral extraction...")
                extraction_start = time.time()
                entity = self._extract_with_mistral(query)
                extraction_duration = time.time() - extraction_start
                print(f"   Mistral extraction completed: {extraction_duration:.2f}s")
                
                process_time = (time.time() - start_time) * 1000
                success = True
            except Exception as e:
                print(f"   Mistral extraction failed: {e}")
                entity = f"ERROR: {str(e)}"
                process_time = (time.time() - start_time) * 1000
                success = False
        else:
            entity = "LOCAL_UNAVAILABLE"
            process_time = 0
            success = False
        
        return {
            "primary_entity": entity,
            "local_entity": entity,
            "local_time_ms": process_time,
            "local_success": success
        }
    
    def _extract_with_mistral(self, query: str) -> str:
        """Extract entity using Mistral with focused, concise extraction"""
        
        # Time prompt construction
        prompt_start = time.time()
        prompt = f"""[INST] Extract the main topic/entity from this user question: "{query}"

Rules:
1. Return ONLY the main topic/entity name (2-4 words MAX)
2. Use lowercase
3. Focus on the PRIMARY subject, ignore action words like "start", "create", "manage"
4. For tool questions, return just the tool name
5. For comparison questions, use "vs" between items
6. Be as concise as possible - prefer 2 words over 4

Examples:
- "What is Claude Code?" → "claude code"
- "How do I start a project in Claude Code?" → "claude code"
- "Tell me about Claude Code setup" → "claude code"
- "Claude Code vs Cursor comparison" → "claude code vs cursor"
- "What's the unicycle model Ed uses?" → "unicycle model"
- "How to setup Proxmox Server?" → "proxmox server"
- "Best practices for Python coding?" → "python coding"

Your response (topic only): [/INST]"""
        print(f"   Prompt construction: {time.time() - prompt_start:.2f}s")
        print(f"   Prompt length: {len(prompt)} chars")
        
        # Time payload preparation
        payload_start = time.time()
        payload = {
            "model": "mistral:7b-instruct",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 20,
                "stop": ["\n", ".", "?", "!"]
            }
        }
        print(f"   Payload preparation: {time.time() - payload_start:.2f}s")
        print(f"   Model: {payload['model']}")
        print(f"   Stream: {payload['stream']}")
        print(f"   Timeout: 90s")
        
        # Time Ollama API call
        api_start = time.time()
        print(f"   Making POST request to: {self.ollama_base_url}/api/generate")
        response = requests.post(
            f"{self.ollama_base_url}/api/generate",
            json=payload,
            timeout=90
        )
        api_duration = time.time() - api_start
        print(f"   Ollama API call: {api_duration:.2f}s")
        print(f"   Response status: {response.status_code}")
        
        # Time response processing
        process_start = time.time()
        response.raise_for_status()
        result = response.json()
        print(f"   JSON parsing: {time.time() - process_start:.2f}s")
        print(f"   Response keys: {list(result.keys())}")
        
        # Time result extraction and cleaning
        extract_start = time.time()
        entity = result.get("response", "").strip().lower()
        
        # Clean up quotes, extra whitespace, and backslashes
        entity = entity.strip('"\'').strip()
        entity = entity.replace('\\', '')  # Remove backslashes
        print(f"   Result extraction/cleaning: {time.time() - extract_start:.2f}s")
        print(f"   Final entity: '{entity}'")
        
        return entity
    
    def _analyze_comparison(self, claude_entity: str, local_entity: str, query: str) -> Dict[str, any]:
        """Analyze how similar the two extractions are"""
        
        if not claude_entity or not local_entity:
            return {"similarity": 0.0, "verdict": "ERROR", "analysis": "One extraction failed"}
        
        claude_words = set(claude_entity.lower().split())
        local_words = set(local_entity.lower().split())
        
        if not claude_words or not local_words:
            return {"similarity": 0.0, "verdict": "ERROR", "analysis": "Empty extraction"}
        
        # Calculate Jaccard similarity
        intersection = claude_words.intersection(local_words)
        union = claude_words.union(local_words)
        
        similarity = len(intersection) / len(union) if union else 0.0
        
        # Determine verdict
        if similarity >= 0.8:
            verdict = "EXCELLENT"
        elif similarity >= 0.6:
            verdict = "GOOD"
        elif similarity >= 0.4:
            verdict = "ACCEPTABLE"
        else:
            verdict = "POOR"
        
        # Analysis details
        shared_words = list(intersection)
        claude_only = list(claude_words - local_words)
        local_only = list(local_words - claude_words)
        
        return {
            "similarity": similarity,
            "verdict": verdict,
            "shared_words": shared_words,
            "claude_only": claude_only,
            "local_only": local_only,
            "analysis": f"{len(shared_words)}/{len(union)} words match"
        }
    
    def _print_comparison(self, query: str, claude_entity: str, local_entity: str, 
                         claude_time: float, local_time: float, comparison: Dict):
        """Print side-by-side comparison"""
        
        print(f"\n🔍 ENTITY EXTRACTION COMPARISON")
        print(f"Query: '{query}'")
        print(f"{'─' * 80}")
        print(f"🤖 Claude:  '{claude_entity}' ({claude_time:.0f}ms)")
        print(f"🦙 Mistral: '{local_entity}' ({local_time:.0f}ms)")
        print(f"{'─' * 80}")
        
        verdict = comparison.get("verdict", "UNKNOWN")
        similarity = comparison.get("similarity", 0.0)
        
        if verdict == "EXCELLENT":
            print(f"✅ {verdict}: {similarity:.2f} similarity - Local model matches Claude!")
        elif verdict == "GOOD":
            print(f"🟡 {verdict}: {similarity:.2f} similarity - Very close, acceptable")
        elif verdict == "ACCEPTABLE":
            print(f"⚠️ {verdict}: {similarity:.2f} similarity - Some differences")
        else:
            print(f"❌ {verdict}: {similarity:.2f} similarity - Significant differences")
        
        # Show word analysis
        shared = comparison.get("shared_words", [])
        claude_only = comparison.get("claude_only", [])
        local_only = comparison.get("local_only", [])
        
        if shared:
            print(f"   Shared: {', '.join(shared)}")
        if claude_only:
            print(f"   Claude only: {', '.join(claude_only)}")
        if local_only:
            print(f"   Mistral only: {', '.join(local_only)}")
        
        print(f"{'─' * 80}")


# Integration helper for existing system
def create_hybrid_entity_extractor():
    """Create hybrid extractor for integration into existing system"""
    return LocalEntityExtractor() 
