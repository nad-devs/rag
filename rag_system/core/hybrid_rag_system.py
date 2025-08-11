"""
Hybrid RAG System with Intelligent Model Selection

Uses Ollama as primary, Claude as fallback, with smart routing based on
query complexity, confidence scores, and performance monitoring.
"""

import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import threading

# Import components
# Disabled for CrossEncoder-only testing:
# from .ollama_followup_detector import OllamaFollowupDetector
# from .claude_followup_detector import ClaudeFollowupDetector
# from .ab_testing_rag import ABTestingRAG

# Only import what we're using:
# from .crossencoder_followup_detector import CrossEncoderFollowupDetector
from .e5_followup_detector import E5FollowupDetector


class HybridRAGSystem:
    """
    Production-ready hybrid RAG system with intelligent model routing.
    
    Strategy:
    1. Use Ollama Llama 3.1 for 95% of queries (78% accuracy, free)
    2. Claude fallback for complex cases or when Ollama fails
    3. Smart routing based on query complexity and confidence
    4. Performance monitoring and auto-adjustment
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize hybrid RAG system.
        
        Args:
            config_file: Optional path to configuration file
        """
        print("🚀 Initializing Hybrid RAG System...")
        
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Initialize components - ONLY E5 for testing
        self.e5_detector = E5FollowupDetector()
        
        # Disabled for testing
        # self.ollama_detector = OllamaFollowupDetector()
        # self.claude_detector = ClaudeFollowupDetector()
        # self.ab_tester = ABTestingRAG()
        
        # Performance tracking - E5 only
        self.stats = {
            "total_queries": 0,
            "e5_usage": 0,
            "error_count": 0,
            "avg_response_time": 0.0
        }
        
        # Status tracking - E5 only
        self.e5_available = self.e5_detector.available
        
        self._print_status()
    
    def _load_config(self, config_file: Optional[str]) -> Dict:
        """Load system configuration."""
        default_config = {
            # Model selection strategy
            "primary_model": "e5",  # E5 only mode for testing
            "fallback_enabled": True,
            "auto_fallback_threshold": 0.5,  # Minimum confidence to trust Ollama
            
            # Performance thresholds
            "max_ollama_response_time": 10.0,  # seconds
            "max_claude_response_time": 15.0,
            
            # Quality monitoring
            "confidence_threshold": 0.6,
            "monitor_agreement_rate": True,
            "min_agreement_rate": 0.7,  # If below this, increase Claude usage
            
            # Cost optimization
            "max_claude_queries_per_hour": 100,
            "cost_optimization_mode": True,
            
            # Routing rules
            "complex_query_keywords": ["explain", "detailed", "elaborate", "comprehensive"],
            "force_claude_keywords": ["complex", "nuanced", "detailed analysis"],
            
            # Monitoring
            "log_all_decisions": True,
            "performance_window_hours": 24
        }
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                print(f"⚠️ Failed to load config file: {e}")
        
        return default_config
    
    def _print_status(self):
        """Print system status - E5 only mode."""
        print(f"📊 E5 Intelligent System Status:")
        print(f"  🎯 E5-Large-V2: {'✅ Ready' if self.e5_available else '❌ Unavailable'}")
        print(f"  🔄 Strategy: E5 semantic similarity + intelligent rules")
        print(f"  💰 Cost: 100% free local inference")
    
    def detect_followup(
        self,
        current_query: str,
        conversation_history: List[Dict],
        previous_entities: List[str] = None,
        similarity_score: float = 0.0,
        force_model: Optional[str] = None
    ) -> Tuple[bool, float, str, Dict]:
        """Main follow-up detection method."""
        return self._detect_followup_internal(current_query, conversation_history, previous_entities, similarity_score, force_model)
    
    def is_follow_up_query(
        self,
        current_query: str,
        conversation_history: List[Dict],
        previous_entities: List[str] = None,
        similarity_score: float = 0.0
    ) -> Tuple[bool, float, str, Dict]:
        """Compatibility method for existing RAG system integration."""
        return self._detect_followup_internal(current_query, conversation_history, previous_entities, similarity_score, None)
    
    def _detect_followup_internal(
        self,
        current_query: str,
        conversation_history: List[Dict],
        previous_entities: List[str] = None,
        similarity_score: float = 0.0,
        force_model: Optional[str] = None
    ) -> Tuple[bool, float, str, Dict]:
        """
        Main follow-up detection with intelligent model routing.
        
        Args:
            current_query: User's current query
            conversation_history: Previous conversation context
            previous_entities: Entities from previous queries
            similarity_score: Embedding similarity score
            force_model: Force specific model ('ollama' or 'claude')
            
        Returns:
            Tuple of (is_follow_up, confidence, reasoning, debug_info)
        """
        start_time = time.time()
        self.stats["total_queries"] += 1
        
        # Enhanced debug info
        debug_info = {
            "system": "hybrid",
            "timestamp": datetime.now().isoformat(),
            "query_length": len(current_query),
            "has_history": len(conversation_history) > 0
        }
        
        try:
            # SIMPLIFIED: Use only E5 for clean testing
            selected_model = "e5"
            debug_info["selected_model"] = selected_model
            debug_info["selection_reason"] = "e5_intelligent_testing_mode"
            
            # Execute E5 detection only
            result = self._run_e5_detection(current_query, conversation_history, previous_entities, similarity_score)
            self.stats["e5_usage"] = self.stats.get("e5_usage", 0) + 1
            
            is_follow_up, confidence, reasoning, model_debug = result
            
            # No fallback needed - E5 only mode
            
            # Update debug info
            debug_info.update(model_debug)
            debug_info["response_time_ms"] = int((time.time() - start_time) * 1000)
            debug_info["final_confidence"] = confidence
            
            # Log decision if enabled
            if self.config["log_all_decisions"]:
                self._log_decision(current_query, is_follow_up, confidence, debug_info)
            
            # Update performance stats
            self._update_stats(time.time() - start_time)
            
            return is_follow_up, confidence, reasoning, debug_info
            
        except Exception as e:
            self.stats["error_count"] += 1
            error_debug = debug_info.copy()
            error_debug.update({
                "error": str(e),
                "response_time_ms": int((time.time() - start_time) * 1000),
                "method": "error_fallback"
            })
            
            print(f"❌ Hybrid system error: {e}")
            
            # Emergency fallback
            if self.claude_available:
                try:
                    return self._run_claude_detection(current_query, conversation_history, previous_entities, similarity_score)
                except:
                    pass
            
            return False, 0.0, f"System error: {str(e)}", error_debug
    
    def _select_model(self, current_query: str, force_model: Optional[str]) -> str:
        """Intelligent model selection based on query characteristics."""
        if force_model in ["ollama", "claude"]:
            return force_model
        
        # Check availability
        if not self.ollama_available and self.claude_available:
            return "claude"
        elif self.ollama_available and not self.claude_available:
            return "ollama"
        elif not self.ollama_available and not self.claude_available:
            return "ollama"  # Will fail gracefully
        
        # Both available - make intelligent choice
        query_lower = current_query.lower()
        
        # Force Claude for complex queries
        if any(keyword in query_lower for keyword in self.config["force_claude_keywords"]):
            return "claude"
        
        # Use primary model for most queries
        if self.config["primary_model"] == "claude":
            return "claude"
        elif self.config["primary_model"] == "hybrid":
            return "hybrid"
        else:  # ollama (default)
            return "ollama"
    
    def _get_selection_reason(self, current_query: str, selected_model: str) -> str:
        """Get human-readable reason for model selection."""
        query_lower = current_query.lower()
        
        if not self.ollama_available:
            return "ollama_unavailable"
        elif not self.claude_available:
            return "claude_unavailable"
        elif any(keyword in query_lower for keyword in self.config["force_claude_keywords"]):
            return "complex_query_keywords"
        elif selected_model == self.config["primary_model"]:
            return "primary_model_preference"
        else:
            return "default_selection"
    
    # REMOVED: Ollama and Claude detection methods - E5 only mode
    
    def _run_e5_detection(self, current_query, conversation_history, previous_entities, similarity_score):
        """Run E5 detection with error handling."""
        try:
            return self.e5_detector.is_follow_up_query(
                current_query, conversation_history, previous_entities, similarity_score
            )
        except Exception as e:
            return False, 0.0, f"E5 error: {str(e)}", {"method": "e5_error", "error": str(e)}
    
    # REMOVED: Hybrid detection method - E5 only mode
    
    def _log_decision(self, query: str, is_follow_up: bool, confidence: float, debug_info: Dict):
        """Log decision for analysis."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "is_follow_up": is_follow_up,
            "confidence": confidence,
            "debug_info": debug_info
        }
        
        log_dir = Path("hybrid_rag_logs")
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"decisions_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        try:
            with open(log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            print(f"⚠️ Failed to log decision: {e}")
    
    def _update_stats(self, response_time: float):
        """Update performance statistics."""
        self.stats["avg_response_time"] = (
            (self.stats["avg_response_time"] * (self.stats["total_queries"] - 1) + response_time) / 
            self.stats["total_queries"]
        )
    
    def get_performance_stats(self) -> Dict:
        """Get current performance statistics."""
        total_queries = max(self.stats["total_queries"], 1)
        
        return {
            "total_queries": self.stats["total_queries"],
            "model_usage": {
                "ollama_percentage": (self.stats["ollama_usage"] / total_queries) * 100,
                "claude_percentage": (self.stats["claude_usage"] / total_queries) * 100
            },
            "performance": {
                "avg_response_time_ms": self.stats["avg_response_time"] * 1000,
                "error_rate": (self.stats["error_count"] / total_queries) * 100,
                "fallback_rate": (self.stats["fallback_triggers"] / total_queries) * 100
            },
            "availability": {
                "ollama": self.ollama_available,
                "claude": self.claude_available
            }
        }
    
    def print_performance_stats(self):
        """Print formatted performance statistics."""
        stats = self.get_performance_stats()
        
        print(f"\n📊 Hybrid RAG Performance Stats")
        print("=" * 40)
        print(f"Total Queries: {stats['total_queries']}")
        print(f"Model Usage:")
        print(f"  🦙 Ollama: {stats['model_usage']['ollama_percentage']:.1f}%")
        print(f"  🤖 Claude: {stats['model_usage']['claude_percentage']:.1f}%")
        print(f"Performance:")
        print(f"  ⚡ Avg Response: {stats['performance']['avg_response_time_ms']:.0f}ms")
        print(f"  ❌ Error Rate: {stats['performance']['error_rate']:.1f}%")
        print(f"  🔄 Fallback Rate: {stats['performance']['fallback_rate']:.1f}%")
    
    def update_config(self, **kwargs):
        """Update system configuration."""
        for key, value in kwargs.items():
            if key in self.config:
                old_value = self.config[key]
                self.config[key] = value
                print(f"⚙️ Updated {key}: {old_value} → {value}")
            else:
                print(f"⚠️ Unknown config key: {key}")
    
    def health_check(self) -> Dict:
        """Perform system health check."""
        health = {
            "status": "healthy",
            "components": {
                "ollama": self.ollama_available,
                "claude": self.claude_available
            },
            "issues": []
        }
        
        # Check component availability
        if not self.ollama_available and not self.claude_available:
            health["status"] = "critical"
            health["issues"].append("Both models unavailable")
        elif not self.ollama_available:
            health["status"] = "degraded"
            health["issues"].append("Ollama unavailable, using Claude fallback")
        elif not self.claude_available:
            health["status"] = "degraded"
            health["issues"].append("Claude unavailable, no fallback")
        
        # Check performance
        stats = self.get_performance_stats()
        if stats["performance"]["error_rate"] > 10:
            health["status"] = "degraded"
            health["issues"].append(f"High error rate: {stats['performance']['error_rate']:.1f}%")
        
        return health
    
    def print_debug_analysis(self, current_query: str, is_follow_up: bool, confidence: float, reasoning: str, debug_info: Dict, **kwargs):
        """Print debug analysis for compatibility with existing RAG system."""
        print(f"🔍 Follow-up Analysis:")
        print(f"  Query: {current_query}")
        print(f"  Decision: {'FOLLOW-UP' if is_follow_up else 'NEW-TOPIC'}")
        print(f"  Confidence: {confidence:.2f}")
        print(f"  Method: {debug_info.get('method', debug_info.get('selected_model', 'unknown'))}")
        print(f"  Reasoning: {reasoning}")
        if debug_info.get("response_time_ms"):
            print(f"  Processing time: {debug_info['response_time_ms']}ms")
        if debug_info.get("selected_model"):
            print(f"  Model used: {debug_info['selected_model']}")
        if debug_info.get("fallback_used"):
            print(f"  🔄 Fallback triggered: {debug_info.get('fallback_reason', 'unknown')}")


def test_hybrid_system():
    """Test the hybrid RAG system."""
    print("🧪 Testing Hybrid RAG System")
    print("=" * 50)
    
    try:
        hybrid = HybridRAGSystem()
        
        # Test cases
        test_cases = [
            {
                "query": "What datasets should I use for fine-tuning?",
                "history": [{"query": "Ed explains supervised fine-tuning for breaking into AI"}],
                "expected": True
            },
            {
                "query": "How do I hire engineers?",
                "history": [{"query": "Ed discusses Claude vs GPT for AI tasks"}],
                "expected": False
            }
        ]
        
        print(f"\n🧪 Running test cases...")
        
        for i, test in enumerate(test_cases, 1):
            print(f"\n📝 Test {i}: {test['query'][:50]}...")
            
            result = hybrid.detect_followup(
                test["query"],
                test["history"],
                similarity_score=0.6
            )
            
            is_follow_up, confidence, reasoning, debug_info = result
            correct = (is_follow_up == test["expected"])
            
            print(f"Result: {'✅ CORRECT' if correct else '❌ WRONG'}")
            print(f"Prediction: {'FOLLOW-UP' if is_follow_up else 'NEW TOPIC'} (confidence: {confidence:.2f})")
            print(f"Model: {debug_info.get('selected_model', 'unknown')}")
            print(f"Time: {debug_info.get('response_time_ms', 0)}ms")
        
        # Show performance stats
        hybrid.print_performance_stats()
        
        # Health check
        health = hybrid.health_check()
        print(f"\n🏥 System Health: {health['status'].upper()}")
        if health['issues']:
            for issue in health['issues']:
                print(f"  ⚠️ {issue}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    test_hybrid_system()