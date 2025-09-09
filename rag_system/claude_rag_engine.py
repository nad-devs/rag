#!/usr/bin/env python3
"""
Claude 3.5 Powered Smart Extraction Engine
==========================================

This engine replaces complex hardcoded pattern matching with Claude 3.5's natural language understanding.

PROCESS FLOW:
1. User asks a question
2. Claude 3.5 analyzes the intent and extracts the main entity/topic
3. Vector search finds relevant content from Ed's captions
4. Claude 3.5 creates an intelligent response based on the content and query type
5. System tracks conversation context to avoid repetition

MAIN COMPONENTS:
- RAGExtractionResult: Data structure for results
- QueryIntent: Enum for different types of questions
- ClaudeRAGEngine: Main engine class (uses Claude 3.5)

KEY METHODS:
- extract_smart_answer(): Main public API - answers any question intelligently

"""

import json
import re
import os
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Import extracted components
from core.document_searcher import DocumentSearcher
from core.metadata_scorer import MetadataScorer
from core.entity_detector import EntityDetector
from core.claude_ranker import ClaudeRanker

# Ensure Qdrant URL is set for Docker instance if not already set
if not os.getenv('QDRANT_URL'):
    os.environ['QDRANT_URL'] = 'http://localhost:6333'
from dataclasses import dataclass
from enum import Enum
import anthropic
from dotenv import load_dotenv
import qdrant_client
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, Range, MatchValue
from sentence_transformers import SentenceTransformer
import uuid
from datetime import datetime
from core.context_tracker import ConversationTracker
from core.hybrid_rag_system import HybridRAGSystem
from local_entity_extractor import LocalEntityExtractor
from local_mistral_synthesizer import LocalMistralSynthesizer


# Claude returns valid JSON, no need for special parsing

# Load environment variables from current directory
load_dotenv()
# Also try loading from parent directory as backup
load_dotenv(dotenv_path=Path(__file__).parent / '.env')
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')

# ==========================================
# DATA STRUCTURES
# ==========================================

@dataclass
class RAGExtractionResult:
    """Result from GPT-4 extraction"""
    primary_answer: str
    confidence: float
    source_type: str  # "factual", "subjective", "mixed", "intelligent", etc.
    supporting_details: List[str]
    follow_up_suggestions: List[str]
    documents_used: List[str]
    extraction_reasoning: str  # Why GPT-4 chose this answer
    temporal_info: Optional[str] = None  # Track opinion evolution if detected

# Removed QueryIntent enum - GPT-4 handles intent naturally in the prompt

# ==========================================
# MAIN ENGINE CLASS
# ==========================================

class ClaudeRAGEngine:
    """
    Complete GPT-4 extraction with vector embeddings, full data indexing, and conversation context
    
    This engine uses GPT-4's natural language understanding to:
    1. Understand what users are asking for
    2. Find relevant content from Ed's captions  
    3. Generate intelligent, contextual responses
    4. Track conversation to avoid repetition
    """
    
    def __init__(self, enhanced_data_path: str, use_vector_search: bool = True, use_local_synthesis: bool = False):
        """Initialize the GPT-4 extraction engine"""
        self.enhanced_data_path = enhanced_data_path
        self.use_vector_search = use_vector_search
        self.use_local_synthesis = use_local_synthesis
        
        # Initialize local Qwen synthesizer if requested
        if use_local_synthesis:
            try:
                print("🤖 Initializing Local Qwen Synthesizer...")
                self.local_synthesizer = LocalMistralSynthesizer()
                print("✅ Local Qwen synthesis enabled - using local model for final response generation")
            except Exception as e:
                print(f"❌ Failed to initialize local synthesizer: {e}")
                print("⚠️ Falling back to API-based synthesis")
                self.use_local_synthesis = False
                self.local_synthesizer = None
        else:
            self.local_synthesizer = None
        
        # Anthropic setup (preferred) with OpenAI fallback  
        # Still needed for document ranking and entity extraction
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        
        if anthropic_key:
            print(f"✅ Using Anthropic Claude API: sk-ant-...{anthropic_key[-10:]}")
            self.client = anthropic.Anthropic(api_key=anthropic_key)
            self.use_anthropic = True
        else:
            if not use_local_synthesis:
                print("❌ No ANTHROPIC_API_KEY found!")
                raise ValueError("ANTHROPIC_API_KEY is required")
            else:
                print("⚠️ No ANTHROPIC_API_KEY found - using local synthesis only")
                self.client = None
                self.use_anthropic = False
        
        # Vector search setup
        if use_vector_search:
            try:
                self.embedding_model = SentenceTransformer('intfloat/e5-large-v2', device='cpu')
                print(f"🧠 Initializing E5-Large-V2 embeddings ({getattr(self.embedding_model, 'max_seq_length', 'N/A')} token context)...")
                print(f"📐 E5-Large-V2 embedding dimension: {self.embedding_model.get_sentence_embedding_dimension()}")
                
                # Qdrant setup with error handling
                print(f"🔍 GPT4_ENGINE DEBUG: QDRANT_URL env = {os.getenv('QDRANT_URL')}")
                print(f"🔍 GPT4_ENGINE DEBUG: Connecting to localhost:6333")
                self.qdrant_client = QdrantClient(host="localhost", port=6333)
                # Just use existing collections - NO CREATION OR INDEXING
                # Map collection keys to full names for DocumentSearcher compatibility
                self.collections = {
                    'content': 'instagram_content_vectors',
                    'qa': 'instagram_qa_vectors', 
                    'tech': 'instagram_tech_vectors',
                    'context': 'instagram_context_vectors'
                }
                print("✅ Vector search initialized (using existing collections)")
            except Exception as e:
                print(f"⚠️ Vector search initialization failed: {e}")
                print("🔄 Continuing with text-only search...")
                self.use_vector_search = False
        
        # Simple topic tracking for display purposes
        self.current_topic = None
        
        # Add missing attributes to prevent old method errors
        self.conversation_depth = 0
        self.used_documents = set()
        self.conversation_context = {}
        self.shared_content = {}
        self._last_claude_scores = []  # Store Claude scores for series filtering
        
        # Initialize conversation tracking
        self.conversation_tracker = ConversationTracker(
            max_history=10, 
            follow_up_threshold=0.7  # Configurable for tuning
        )
        print("🔄 ConversationTracker initialized (history=10, threshold=0.7)")
        
        # Initialize Hybrid RAG follow-up detector (Ollama + Claude fallback)
        try:
            self.phi3_detector = HybridRAGSystem()
            print("✅ Hybrid RAG follow-up detector initialized (Ollama + Claude fallback)")
        except Exception as e:
            print(f"⚠️ Hybrid follow-up detector initialization failed, falling back to similarity detection: {e}")
            self.phi3_detector = None
        
        # Initialize Local Entity Extractor for side-by-side comparison
        try:
            self.local_entity_extractor = LocalEntityExtractor()
            print("✅ Local entity extractor initialized for comparison testing")
        except Exception as e:
            print(f"⚠️ Local entity extractor failed: {e}")
            self.local_entity_extractor = None
        
        # Load and index all documents
        self._load_all_documents()
        
        # Initialize extracted components after documents are loaded
        print("🔧 Initializing extracted components...")
        try:
            # Initialize document searcher
            if self.use_vector_search:
                self.document_searcher = DocumentSearcher(
                    qdrant_client=self.qdrant_client,
                    collections=self.collections,
                    enhanced_data=self.documents,
                    embedding_model=self.embedding_model
                )
                print("✅ DocumentSearcher initialized")
            else:
                self.document_searcher = None
            
            # Initialize metadata scorer
            self.metadata_scorer = MetadataScorer(
                enhanced_content_index=self.enhanced_content_index
            )
            print("✅ MetadataScorer initialized")
            
            # Initialize entity detector
            self.entity_detector = EntityDetector(
                client=self.client,
                use_mistral=not self.use_anthropic,  # Use Mistral when not using Anthropic
                local_entity_extractor=self.local_entity_extractor
            )
            print("✅ EntityDetector initialized")
            
            # Initialize claude ranker
            self.claude_ranker = ClaudeRanker(
                client=self.client,
                enhanced_content_index=self.enhanced_content_index
            )
            print("✅ ClaudeRanker initialized")
            
            print("🎉 All components initialized successfully!")
            
        except Exception as e:
            print(f"❌ Component initialization failed: {e}")
            print("⚠️ Falling back to monolithic methods")
            self.document_searcher = None
            self.metadata_scorer = None
            self.entity_detector = None
            self.claude_ranker = None
        # NOTE: Vector indexing now handled by separate rebuild_vector_index.py script
    

    

    
    def _extract_vector_specific_content(self, mention: Dict[str, Any]) -> str:
        """Extract content directly from vector payload (no reconstruction needed)"""
        vector_type = mention.get('vector_type', 'instagram_content_vectors')
        
        # Vector type debug removed for cleaner output
        
        # Fields are directly in mention from DocumentSearcher, not in metadata
        # Use the fields that were actually stored in this vector's payload
        content_parts = []
        
        # Map actual collection names to focused content extraction
        if vector_type == 'instagram_content_vectors':
            # Primary content: main lesson, key takeaways, actionable insights (no redundant full text)
            content_parts = [
                mention.get('main_lesson', ''),
                ' '.join(mention.get('key_takeaways', [])),
                ' '.join(mention.get('actionable_insights', []))
            ]
            
        elif vector_type == 'instagram_tech_vectors':
            # Technical content: technologies, tools, and platforms + FULL CONTENT
            tech_parts = []
            for tech in mention.get('technologies_mentioned', []):
                if isinstance(tech, dict):
                    tech_parts.append(f"{tech.get('name', '')}: {tech.get('context', '')}")
                else:
                    tech_parts.append(str(tech))
            
            for tool in mention.get('tools_and_platforms', []):
                if isinstance(tool, dict):
                    tech_parts.append(f"{tool.get('name', '')}: {tool.get('use_case', '')}")
                else:
                    tech_parts.append(str(tool))
                    
            content_parts = tech_parts + [mention.get('main_lesson', '')]  # Removed redundant content_text
            
        elif vector_type == 'instagram_qa_vectors':
            # Q&A content: natural questions, search scenarios, main lesson + FULL CONTENT for safety
            content_parts = [
                ' '.join(mention.get('natural_questions', [])),
                ' '.join(mention.get('search_scenarios', [])),
                mention.get('main_lesson', '')
                # Removed redundant content_text - all key details in structured fields
            ]
            
        elif vector_type == 'instagram_context_vectors':
            # Contextual/practical content: examples, applications, prerequisites + FULL CONTENT
            content_parts = [
                ' '.join(mention.get('specific_examples', [])),
                ' '.join(mention.get('practical_applications', [])),
                ' '.join(mention.get('prerequisites', [])),
                ' '.join(mention.get('related_topics', [])),
                mention.get('main_lesson', '')
                # Removed redundant content_text - all key details in structured fields
            ]
            
        # Legacy support for old collection names
        elif vector_type == 'primary':
            content_parts = [
                mention.get('main_lesson', ''),
                mention.get('content_text', ''),
                ' '.join(mention.get('key_takeaways', [])),
                ' '.join(mention.get('actionable_insights', []))
            ]
            
        elif vector_type == 'insights':
            tech_parts = []
            for tech in mention.get('technologies_mentioned', []):
                if isinstance(tech, dict):
                    tech_parts.append(f"{tech.get('name', '')}: {tech.get('context', '')}")
                else:
                    tech_parts.append(str(tech))
            
            for tool in mention.get('tools_and_platforms', []):
                if isinstance(tool, dict):
                    tech_parts.append(f"{tool.get('name', '')}: {tool.get('use_case', '')}")
                else:
                    tech_parts.append(str(tool))
                    
            content_parts = tech_parts + [mention.get('main_lesson', '')]
            
        elif vector_type == 'technical':
            content_parts = [
                ' '.join(mention.get('natural_questions', [])),
                ' '.join(mention.get('search_scenarios', [])),
                mention.get('main_lesson', '')
            ]
            
        elif vector_type == 'practical':
            content_parts = [
                ' '.join(mention.get('specific_examples', [])),
                ' '.join(mention.get('practical_applications', [])),
                ' '.join(mention.get('prerequisites', [])),
                ' '.join(mention.get('related_topics', [])),
                mention.get('main_lesson', '')
            ]
        
        # Combine all non-empty parts
        final_content = ' '.join(filter(None, content_parts))
        
        # Fallback to content_text if no vector-specific content found
        if not final_content.strip():
            final_content = mention.get('content_text', '')
            if final_content:
                print(f"   ⚠️ No vector-specific content found, using fallback")
        
        return final_content
    
    # ==========================================
    # MAIN PUBLIC API
    # ==========================================
    
    def extract_smart_answer(self, user_query: str) -> RAGExtractionResult:
        """
        Simplified main extraction using only working methods
        
        PROCESS:
        1. Check for follow-up query context
        2. Extract entity from query (with conversation context)
        3. Find relevant content using working vector/text search
        4. Use intelligent GPT-4 extraction to generate response
        5. Store query context for future follow-ups
        
        Args:
            user_query: Natural language question from user
            
        Returns:
            RAGExtractionResult with answer, confidence, and metadata
        """
        import time
        
        total_start = time.time()
        timings = {}
        
        # STEP 1: Check for follow-up context using Claude API + conversation tracker
        step_start = time.time()
        query_embedding = None
        is_followup = False
        similarity = 0.0
        related_context = None
        
        if self.use_vector_search:
            query_embedding = self.embedding_model.encode(user_query)
            
            # Get old similarity-based detection for comparison
            old_is_followup, similarity, related_context = self.conversation_tracker.is_follow_up_query(query_embedding)
            
            # Use Claude API for intelligent follow-up detection
            if self.phi3_detector:
                # Prepare conversation history for Claude
                conversation_history = []
                recent_contexts = self.conversation_tracker.get_recent_context(limit=3)
                for ctx in recent_contexts:
                    conversation_history.append({
                        "query": ctx.query,
                        "entities": ctx.entities_mentioned,
                        "timestamp": ctx.timestamp
                    })
                
                # Get previous entities
                previous_entities = []
                if recent_contexts:
                    for ctx in recent_contexts:
                        previous_entities.extend(ctx.entities_mentioned)
                    previous_entities = list(set(previous_entities))  # Deduplicate
                
                # Use Claude for intelligent analysis
                is_followup, confidence, reasoning, debug_info = self.phi3_detector.is_follow_up_query(
                    current_query=user_query,
                    conversation_history=conversation_history,
                    previous_entities=previous_entities,
                    similarity_score=similarity
                )
                
                # Print detailed debug output
                self.phi3_detector.print_debug_analysis(
                    current_query=user_query,
                    is_follow_up=is_followup,
                    confidence=confidence,
                    reasoning=reasoning,
                    debug_info=debug_info,
                    similarity_score=similarity
                )
                
            else:
                # Fallback to old similarity-based method
                is_followup = old_is_followup
                print(f"🔍 FALLBACK FOLLOW-UP DEBUG (Claude unavailable):")
                print(f"   Query: '{user_query}'")
                print(f"   Is follow-up: {is_followup}")
                print(f"   Similarity: {similarity:.3f} (threshold: {self.conversation_tracker.follow_up_threshold})")
                if related_context:
                    print(f"   Related to: '{related_context.query}'")
                    print(f"   Previous entities: {related_context.entities_mentioned}")
                else:
                    print(f"   No previous context")
        
        timings['follow_up_detection'] = time.time() - step_start
        
        # STEP 2: Get conversation context first
        step_start = time.time()
        conversation_context = self.conversation_tracker.get_context_for_response_generation()
        
        # Extract key entity/topic from the user's question using GPT-4
        if self.entity_detector:
            if is_followup:
                entity = self.entity_detector.extract_entity_from_query(user_query, conversation_context)
            else:
                entity = self.entity_detector.extract_entity_from_query(user_query)
        else:
            entity = self._extract_entity_from_query(user_query)  # Fallback to monolithic
        
        print(f"🎯 Entity Detected: '{entity}'")
        
        timings['entity_extraction'] = time.time() - step_start
        
        # CONTEXT-AWARE SEARCH: Find relevant content using conversation context
        step_start = time.time()
        used_document_ids = self.conversation_tracker.get_used_document_ids(recent_only=True, for_followup=is_followup)
        
        # Context-aware entity content search
        if self.document_searcher:
            entity_mentions = self.document_searcher.vector_search_entity_with_context(
                entity, 
                conversation_context, 
                used_document_ids, 
                is_followup
            )
        else:
            # Fallback to basic vector search if component not available
            entity_mentions = self.vector_search_entity(entity)
        
        # METADATA SCORING INTEGRATION: Add metadata-enhanced results
        if self.metadata_scorer and entity_mentions:
            print(f"   📊 Applying metadata scoring to {len(entity_mentions)} vector results...")
            
            # Get metadata-enhanced results for the user query
            metadata_results = self.metadata_scorer.search_metadata_fields(user_query)
            
            # ENHANCED MERGING: Preserve vector similarity scores while adding metadata scores
            combined_results = []
            
            # Create lookup for metadata results by doc_id
            metadata_lookup = {meta['doc_id']: meta for meta in metadata_results if meta.get('doc_id')}
            
            # Enhance vector results with metadata scores
            for vector_result in entity_mentions:
                doc_id = vector_result.get('doc_id', vector_result.get('post_id', ''))
                
                # Start with vector result (preserves similarity_score)
                enhanced_result = dict(vector_result)
                
                # Ensure similarity_score is preserved from vector search
                if 'similarity_score' not in enhanced_result and 'score' in enhanced_result:
                    enhanced_result['similarity_score'] = enhanced_result['score']
                
                # Add metadata scoring if available
                if doc_id in metadata_lookup:
                    meta_data = metadata_lookup[doc_id]
                    enhanced_result['metadata_score'] = meta_data.get('metadata_score', 0)
                    enhanced_result['match_reasons'] = meta_data.get('match_reasons', [])
                    enhanced_result['search_type'] = 'vector_with_metadata'  # Hybrid type
                else:
                    enhanced_result['metadata_score'] = 0
                    enhanced_result['match_reasons'] = []
                
                combined_results.append(enhanced_result)
            
            # Add metadata-only results (docs not found by vector search)
            vector_doc_ids = {r.get('doc_id', r.get('post_id', '')) for r in entity_mentions}
            for meta_result in metadata_results:
                meta_doc_id = meta_result.get('doc_id', '')
                if meta_doc_id and meta_doc_id not in vector_doc_ids:
                    combined_results.append(meta_result)
            
            # Apply context scoring to all results
            combined_results = self.metadata_scorer.apply_context_scoring(combined_results, conversation_context)
            
            # Use context-aware deduplication and ranking
            entity_mentions = self.metadata_scorer.deduplicate_and_rank_with_context(
                combined_results, 
                entity, 
                conversation_context, 
                used_document_ids, 
                is_followup
            )
            
            print(f"   🎯 Final ranked results: {len(entity_mentions)} documents")
        
        timings['vector_search'] = time.time() - step_start
        
        # Claude ranking preparation step
        step_start = time.time()
        
        if not entity_mentions:
            return RAGExtractionResult(
                primary_answer=f"I don't have information about '{entity}' in the CEO's content.",
                confidence=0.0,
                source_type="missing",
                supporting_details=[],
                follow_up_suggestions=[],
                documents_used=[],
                extraction_reasoning=f"No content found mentioning '{entity}'"
            )
        
        timings['claude_ranking'] = time.time() - step_start
        
        # Final synthesis step  
        step_start = time.time()
        result = self._extract_with_context_aware_prompt(
            entity, 
            entity_mentions, 
            user_query, 
            conversation_context, 
            is_followup
        )
        
        timings['final_synthesis'] = time.time() - step_start
        
        # STEP 5: Store query context for future follow-ups
        if result and result.confidence > 0.3 and query_embedding is not None:
            # Extract entities mentioned for better context tracking
            entities_mentioned = [entity]  # Primary entity
            if result.documents_used:
                # Add document topics as additional context
                for doc_id in result.documents_used[:3]:  # Limit to avoid noise
                    if doc_id in self.shared_content:
                        doc_data = self.shared_content[doc_id]
                        if 'main_lesson' in doc_data:
                            # Extract key terms from main lesson for entity tracking
                            lesson_words = doc_data['main_lesson'].split()[:5]
                            entities_mentioned.extend([w.strip('.,!?') for w in lesson_words if len(w) > 3])
            
            # Extract result content for semantic followup detection
            result_content = []
            key_takeaways = []
            
            for doc_id in result.documents_used:
                doc_path = Path(self.enhanced_data_path) / f"enhanced_{doc_id}.json"
                if doc_path.exists():
                    try:
                        with open(doc_path, 'r', encoding='utf-8') as f:
                            doc_data = json.load(f)
                            
                        # Extract key structured content
                        if 'key_takeaways' in doc_data:
                            key_takeaways.extend(doc_data['key_takeaways'][:3])  # Top 3 per document
                        
                        if 'actionable_insights' in doc_data:
                            result_content.extend(doc_data['actionable_insights'][:2])  # Top 2 per document
                            
                        if 'main_lesson' in doc_data:
                            result_content.append(doc_data['main_lesson'])
                            
                    except Exception as e:
                        print(f"⚠️ Failed to extract content from {doc_id}: {e}")
            
            # Store conversation context with enhanced data
            self.conversation_tracker.add_query_context(
                query=user_query,
                embedding=query_embedding,
                document_ids=result.documents_used,
                response_summary=result.primary_answer[:100] + "..." if len(result.primary_answer) > 100 else result.primary_answer,
                entities_mentioned=entities_mentioned[:10],  # Limit to avoid bloat
                result_content=result_content[:5],  # Top 5 content items
                key_takeaways=key_takeaways[:5]  # Top 5 takeaways
            )
            
            print(f"💾 Stored query context: {len(entities_mentioned)} entities, {len(result.documents_used)} documents")
        
        # Track conversation (simple increment for old method compatibility)
        if result and result.confidence > 0.3:
            self.conversation_depth += 1
        
        # Add total timing and print results
        timings['total_time'] = time.time() - total_start
        print(f"🔍 PERFORMANCE PROFILE:")
        for step, duration in timings.items():
            print(f"   {step}: {duration:.2f}s")
        
        return result
    
    # ==========================================
    # CORE EXTRACTION LOGIC
    # ==========================================
    
    def _normalize_json_input(self, raw_content: str) -> str:
        """
        Minimal preprocessing to handle common LLM JSON formatting issues.
        Does not change the core logic, just normalizes input.
        """
        # Basic strip (existing behavior)
        normalized = raw_content.strip()
        
        # Remove invisible Unicode characters that break JSON parsing
        normalized = re.sub(r'[\u200B-\u200D\u2060\uFEFF\u00A0\u180E]', '', normalized)
        
        # Remove any leading whitespace inside braces (common LLM issue)
        normalized = re.sub(r'^{\s+', '{', normalized)
        
        return normalized

    def _extract_answer_from_broken_json(self, raw_content: str) -> str:
        """
        Extract answer field from malformed JSON using regex patterns.
        This handles cases where JSON parsing fails due to control characters.
        """
        try:
            # Pattern 1: Try to find "answer": "content" pattern
            answer_pattern = r'"answer"\s*:\s*"([^"]*(?:\\.[^"]*)*)"'
            match = re.search(answer_pattern, raw_content, re.DOTALL)
            if match:
                answer = match.group(1)
                # Unescape common JSON escape sequences
                answer = answer.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
                return answer
            
            # Pattern 2: Try to find answer without quotes (multiline)
            answer_pattern_multiline = r'"answer"\s*:\s*"([^"]*(?:\\.[^"]*)*(?:\n[^"]*)*)"'
            match = re.search(answer_pattern_multiline, raw_content, re.DOTALL | re.MULTILINE)
            if match:
                answer = match.group(1)
                answer = answer.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
                return answer
            
            # Pattern 3: Extract everything between first { and last } and try to find answer
            json_like = re.search(r'\{.*\}', raw_content, re.DOTALL)
            if json_like:
                json_content = json_like.group(0)
                # Try to find answer field more loosely
                answer_loose = re.search(r'"answer"\s*:\s*"([^"]+[^"}]*)', json_content, re.DOTALL)
                if answer_loose:
                    return answer_loose.group(1).strip()
            
            # Fallback: Clean the raw content and return it
            clean_content = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', raw_content)
            clean_content = re.sub(r'^[^a-zA-Z]*', '', clean_content)  # Remove leading non-letters
            return clean_content[:1000] + "..." if len(clean_content) > 1000 else clean_content
            
        except Exception as e:
            print(f"⚠️ Fallback extraction failed: {e}")
            # Last resort: just return cleaned raw content
            return re.sub(r'[\x00-\x1f\x7f-\x9f]', '', raw_content)[:500]

    def _extract_with_context_aware_prompt(
        self, 
        entity: str, 
        mentions: List[Dict], 
        query: str, 
        conversation_context: Dict,
        is_followup: bool
    ) -> RAGExtractionResult:
        """
        Context-aware GPT-4 extraction that incorporates conversation history
        
        This enhances the intelligent extraction with conversation awareness:
        - References previous queries and topics when appropriate
        - Adjusts response style for follow-up vs new queries
        - Resolves anaphora ("it", "that", "them") using conversation context
        - Provides continuity in multi-turn conversations
        
        Args:
            entity: Main topic/entity being asked about
            mentions: Relevant content from Ed's captions
            query: Original user question
            conversation_context: Previous conversation state
            is_followup: Whether this is a follow-up to previous query
            
        Returns:
            RAGExtractionResult with context-aware response
        """
        
        # Prepare comprehensive data from mentions
        print(f"DEBUG CLAUDE: Processing {len(mentions)} mentions for entity '{entity}'")
        all_content = []
        all_structured_data = []
        doc_ids = []
        
        # OPTIMIZED: Evidence-based document selection strategy
        # Based on comprehensive analysis of 446 documents and 8 diverse query types
        # Achieves 89% of theoretical maximum performance (62.5% vs 70% ceiling)
        query_complexity = self._assess_query_complexity(query)
        # Use component or fallback
        if self.entity_detector:
            entity_category = self.entity_detector.categorize_entity_type(entity, query)
        else:
            # Simple fallback categorization
            entity_category = "general"
        
        # Optimized category-based limits for speed and accuracy balance
        # NOTE: Reduced limits for faster performance while maintaining accuracy
        category_limits = {
            'tool_definition': 8,      # Comparisons + features + alternatives
            'comparison': 10,          # Multiple perspectives + differences
            'best_practices': 8,       # Diverse approaches + examples
            'numbered_list': 8,        # Complete lists + context
            'technical': 8,            # Comprehensive explanations
            'concept_definition': 8,   # Thorough explanation + examples
            'howto': 10               # Step-by-step + alternatives
        }
        
        # Base document count with category awareness  
        base_docs = category_limits.get(entity_category, 8)  # Default 8 for speed
        
        # Quality bonus: increase if high-quality results available
        # This is estimated based on the fact we have search results
        quality_bonus = 0
        if query_complexity == "comprehensive":
            quality_bonus = 5  # Comprehensive queries need much more context
        elif len(mentions) > 20:  # Many high-quality results available
            quality_bonus = 3
        elif len(mentions) > 10:  # Some good results available
            quality_bonus = 2
            
        max_docs = min(base_docs + quality_bonus, 20)  # Cap at 20 to allow contextual selection
            
        # STAGE 1: Series Detection and Boosting (ALWAYS runs)
        # NEW: Detect and boost series BEFORE ranking decisions
        # Use component or fallback
        if self.claude_ranker:
            mentions_with_series = self.claude_ranker.detect_and_boost_series(query, mentions)
        else:
            # Fallback: no series boosting
            mentions_with_series = mentions
        
        # STAGE 2: Claude-based document ranking for better contextual selection
        # Determine if we should use two-stage ranking based on query complexity and document count
        if self.claude_ranker:
            use_claude_ranking = self.claude_ranker.should_use_claude_ranking(query, len(mentions_with_series), query_complexity)
        else:
            use_claude_ranking = False  # Fallback if component not available
        
        if use_claude_ranking and len(mentions_with_series) > max_docs:
            print(f"🎯 CLAUDE RANKING: Using two-stage selection for better relevance")
            # Get more candidates for Claude to rank
            ranking_candidates = min(len(mentions_with_series), max_docs * 2)  # 2x candidates for ranking
            
            if self.claude_ranker:
                ranked_mentions = self.claude_ranker.claude_rank_documents(query, mentions_with_series[:ranking_candidates])
            else:
                ranked_mentions = mentions_with_series[:ranking_candidates]  # Fallback
            
            # NEW: Apply Claude threshold filtering for series parts
            if self.claude_ranker:
                selected_mentions = self.claude_ranker.apply_series_threshold_filter(query, ranked_mentions, mentions[:ranking_candidates])[:max_docs]
            else:
                selected_mentions = ranked_mentions[:max_docs]  # Fallback
            
            print(f"📊 CLAUDE SELECTION: Ranked {len(mentions_with_series)} docs (with series) → selected top {len(selected_mentions)}")
        else:
            selected_mentions = mentions_with_series[:max_docs]
            print(f"📊 STANDARD SELECTION: {entity_category} → {max_docs} docs (with series boost)")
        
        final_doc_count = len(selected_mentions)  # Update to actual selection size
        
        # Group mentions by document ID to aggregate multiple vector types
        doc_mentions = {}
        for mention in selected_mentions:
            doc_id = mention.get('doc_id', '')
            if doc_id not in doc_mentions:
                doc_mentions[doc_id] = []
            doc_mentions[doc_id].append(mention)
        
        for i, (doc_id, mentions_for_doc) in enumerate(doc_mentions.items()):
            
            # Aggregate content from all vector types for this document
            doc_vector_contents = []
            vector_types_found = []
            
            for mention in mentions_for_doc:
                vector_specific_content = self._extract_vector_specific_content(mention)
                if vector_specific_content.strip():  # Only add non-empty content
                    doc_vector_contents.append(vector_specific_content)
                    vector_types_found.append(mention.get('vector_type', 'unknown'))
            
            # Combine all vector-specific content for this document
            combined_doc_content = ' '.join(doc_vector_contents)
            
            # If no vector-specific content, fall back to full content
            if not combined_doc_content.strip() and mentions_for_doc:
                combined_doc_content = mentions_for_doc[0].get('content_text', '')
                print(f"   ⚠️ No vector-specific content, using full content as fallback")
            
            # Extract post_date for temporal context
            post_date = None
            for mention in mentions_for_doc:
                # Check for post_date in the mention (from Qdrant payload)
                if mention.get('post_date'):
                    post_date = mention.get('post_date')
                    break
            
            # Add document ID header with date for temporal awareness
            if post_date:
                attributed_content = f"[DOCUMENT {doc_id} - Posted: {post_date}]:\n{combined_doc_content}"
            else:
                attributed_content = f"[DOCUMENT {doc_id}]:\n{combined_doc_content}"
            all_content.append(attributed_content)
            doc_ids.append(doc_id)
            
            # Debug: show aggregation results
            original_length = len(mentions_for_doc[0].get('content_text', '')) if mentions_for_doc else 0
            final_length = len(combined_doc_content)
            # Content aggregation completed for doc: {doc_id}
            
            # Extract ALL structured metadata
            structured = mention.get('structured_data', {}) or mention.get('learning_metadata', {})
            
            # Also try getting it from enhanced_data if available
            if not structured and mention.get('enhanced_data', {}):
                enhanced = mention.get('enhanced_data', {})
                structured = enhanced.get('learning_metadata', {}) or enhanced.get('full_metadata', {})
            
            if structured:
                all_structured_data.append(structured)
        
        combined_content = "\n\n---\n\n".join(all_content)
        # Clean summary output
        total_docs = len(doc_mentions)
        vector_types_used = set()
        for mentions in doc_mentions.values():
            for mention in mentions:
                vector_types_used.add(mention.get('vector_type', 'unknown'))
        
        print(f"📊 SEARCH RESULTS: {total_docs} documents | {len(combined_content):,} chars | Types: {len(vector_types_used)}")
        
        # Citation flow debugging - Document selection
        print(f"🔍 CITATION FLOW DEBUG:")
        print(f"   Selected documents count: {len(doc_mentions)}")
        for i, (doc_id, mentions_for_doc) in enumerate(list(doc_mentions.items())[:10]):  # Limit to first 10
            title = "NO_TITLE"
            if mentions_for_doc:
                title = mentions_for_doc[0].get('main_lesson', mentions_for_doc[0].get('title', 'NO_TITLE'))[:50]
            print(f"   Doc {i+1}: {doc_id} - {title}")
        
        # Show document relevance summary instead of individual details
        if total_docs > 0:
            print(f"   📋 Top documents: {list(doc_mentions.keys())[:3]}")
        
        # Aggregate structured insights
        print(f"   🧠 Processing {len(all_structured_data)} enhanced documents...")
        all_key_takeaways = []
        all_actionable_insights = []
        all_specific_examples = []
        all_practical_applications = []
        all_technologies = []
        all_tools = []
        
        for structured in all_structured_data:
            all_key_takeaways.extend(structured.get('key_takeaways', []))
            all_actionable_insights.extend(structured.get('actionable_insights', []))
            all_specific_examples.extend(structured.get('specific_examples', []))
            all_practical_applications.extend(structured.get('practical_applications', []))
            
            # Extract technology and tool mentions
            for tech in structured.get('technologies_mentioned', []):
                if isinstance(tech, dict):
                    all_technologies.append(f"{tech.get('name', '')}: {tech.get('context', '')}")
            for tool in structured.get('tools_and_platforms', []):
                if isinstance(tool, dict):
                    all_tools.append(f"{tool.get('name', '')}: {tool.get('use_case', '')}")
        
        # Create context-aware intelligent prompt
        prompt = self._build_context_aware_prompt(
            query, entity, combined_content, conversation_context, is_followup,
            all_key_takeaways, all_actionable_insights, all_specific_examples,
            all_practical_applications, all_technologies, all_tools
        )
        
        print(f"🤖 CONTEXT-AWARE PROMPT DEBUG:")
        print(f"   Has conversation context: {conversation_context.get('has_context', False)}")
        print(f"   Is follow-up query: {is_followup}")
        if conversation_context.get('has_context'):
            print(f"   Previous queries: {conversation_context.get('previous_queries', [])}")
            print(f"   Conversation topics: {conversation_context.get('mentioned_entities', [])}")
        
        # Citation flow debugging - Before Claude synthesis
        import re
        doc_refs_in_prompt = re.findall(r'\[DOCUMENT ([^\]]+)\]', prompt)
        edhonour_refs_in_prompt = re.findall(r'\[edhonour_[^\]]+\]', prompt)
        print(f"🔍 DOCUMENTS SENT TO CLAUDE:")
        print(f"   Document IDs in synthesis prompt: {doc_refs_in_prompt}")
        print(f"   Prompt contains {len(edhonour_refs_in_prompt)} [edhonour_] references: {edhonour_refs_in_prompt[:5]}")  # Show first 5
        print(f"   Prompt length: {len(prompt):,} characters")
        
        # SYNTHESIS: Use local Qwen model if enabled, otherwise use API
        try:
            if self.use_local_synthesis and self.local_synthesizer:
                # Use local Qwen for final synthesis
                print("🤖 Using LOCAL QWEN for document synthesis")
                synthesis_result = self.local_synthesizer.extract_with_context_aware_prompt(
                    prompt=prompt,
                    temperature=0.1,
                    max_tokens=2500,
                    timeout=60.0
                )
                
                print(f"🤖 LOCAL SYNTHESIS: Generated response with confidence {synthesis_result['confidence']}")
                
                # Citation flow debugging - After local synthesis
                local_answer = synthesis_result.get('answer', '')
                edhonour_citations_in_local = re.findall(r'\[edhonour_[^\]]+\]', local_answer)
                print(f"🔍 LOCAL SYNTHESIS CITATION OUTPUT:")
                print(f"   Response contains {len(edhonour_citations_in_local)} [edhonour_] citations")
                print(f"   Citations found: {edhonour_citations_in_local[:10]}")  # Show first 10
                
                # Create RAGExtractionResult directly from local synthesis
                return RAGExtractionResult(
                    primary_answer=synthesis_result['answer'],
                    confidence=synthesis_result['confidence'],
                    source_type=synthesis_result['response_type'],
                    supporting_details=[synthesis_result['reasoning'], f"Documents processed: {final_doc_count}"],
                    follow_up_suggestions=synthesis_result['follow_up_questions'],
                    documents_used=doc_ids,
                    extraction_reasoning=f"Local Qwen synthesis: {synthesis_result['reasoning']}"
                )
                
            elif self.use_anthropic:
                # Add timeout to prevent hanging
                response = self.client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=2500,
                    timeout=30.0  # 30 second timeout
                )
                raw_content = response.content[0].text
            else:
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=2500
                )
                raw_content = response.choices[0].message.content
            
            print(f"🤖 LLM Response length: {len(raw_content)} characters")
            
            # Citation flow debugging - After Claude response
            edhonour_citations_in_response = re.findall(r'\[edhonour_[^\]]+\]', raw_content)
            print(f"🔍 CLAUDE CITATION OUTPUT:")
            print(f"   Response contains {len(edhonour_citations_in_response)} [edhonour_] citations")
            print(f"   Citations found: {edhonour_citations_in_response[:10]}")  # Show first 10
            if len(edhonour_citations_in_response) > 10:
                print(f"   ... and {len(edhonour_citations_in_response) - 10} more citations")
            
            # Try to parse JSON response with robust cleaning
            try:
                # Claude returns valid JSON - just parse it directly
                result = json.loads(raw_content)
                
                # Extract structured response
                answer = result.get('answer', '')
                confidence = result.get('confidence', 0.8)
                response_type = result.get('response_type', 'context_aware')
                reasoning = result.get('reasoning', 'Context-aware extraction')
                follow_up_questions = result.get('follow_up_questions', [])
                
                # Citation generation debug
                print(f"🔍 CITATION GENERATION DEBUG:")
                print(f"   Raw Claude response: {raw_content[:500]}...")
                print(f"   Contains [edhonour_ references: {raw_content.count('[edhonour_')}")
                print(f"   Extracted answer: {answer[:500]}...")
                print(f"   Answer contains [edhonour_ references: {answer.count('[edhonour_')}")
                
                print(f"✅ Successfully parsed context-aware JSON response")
                print(f"   Response type: {response_type}")
                print(f"   Confidence: {confidence}")
                print(f"   Follow-ups: {len(follow_up_questions)}")
                
            except json.JSONDecodeError as je:
                print(f"⚠️ JSON parsing failed: {je}")
                print(f"Raw response preview: {raw_content[:200]}...")
                
                # STEP 2: Advanced fallback - try to extract answer field using regex
                answer = self._extract_answer_from_broken_json(raw_content)
                confidence = 0.7
                response_type = "context_aware_fallback" 
                reasoning = "Advanced fallback extraction (JSON parsing failed)"
                follow_up_questions = [f"More about {entity}", f"Examples of {entity} usage"]
                
            except Exception as parse_error:
                print(f"⚠️ Response parsing error: {parse_error}")
                # Clean response of control characters
                clean_text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', raw_content)
                result = {
                    "answer": clean_text,
                    "response_type": "context_aware_fallback",
                    "confidence": 0.7,
                    "reasoning": "Context-aware extraction (parsing failed)",
                    "follow_up_questions": [f"More about {entity}", f"Examples of {entity}"]
                }
            # Add context metadata to result for debugging
            context_metadata = {
                'conversation_depth': conversation_context.get('conversation_depth', 0),
                'is_followup': is_followup,
                'context_entities': conversation_context.get('mentioned_entities', []),
                'previous_queries_count': len(conversation_context.get('previous_queries', []))
            }
            
            return RAGExtractionResult(
                primary_answer=answer,
                confidence=confidence,
                source_type=response_type,
                supporting_details=[reasoning, f"Context: {context_metadata}"],
                follow_up_suggestions=follow_up_questions,
                documents_used=doc_ids,
                extraction_reasoning=f"Context-aware extraction: {reasoning}"
            )
            
        except Exception as e:
            print(f"⚠️ Intelligent extraction error: {e}")
            # Fallback to basic extraction
            return RAGExtractionResult(
                primary_answer=f"Based on Ed's content about {entity}: {combined_content[:500]}...",
                confidence=0.6,
                source_type="fallback",
                supporting_details=["Fallback extraction due to API error"],
                follow_up_suggestions=[f"More details about {entity}"],
                documents_used=doc_ids,
                extraction_reasoning="Fallback extraction after API error"
            )
    
    def _build_context_aware_prompt(
        self,
        query: str,
        entity: str, 
        combined_content: str,
        conversation_context: Dict,
        is_followup: bool,
        all_key_takeaways: List,
        all_actionable_insights: List,
        all_specific_examples: List,
        all_practical_applications: List,
        all_technologies: List,
        all_tools: List
    ) -> str:
        """Build context-aware prompt that incorporates conversation history"""
        
        # Enhanced prompt with improved structure and quality guidelines
        
        # Build conversation context if available
        context_part = ""
        if conversation_context.get('has_context', False) and is_followup:
            previous_queries = conversation_context.get('previous_queries', [])
            mentioned_entities = conversation_context.get('mentioned_entities', [])
            context_part = f"""
CONVERSATION CONTEXT:
- This is a follow-up question building on previous discussion
- Previous topics: {', '.join(mentioned_entities[:3]) if mentioned_entities else 'None'}
- If user uses pronouns ("it", "that", "them"), they likely refer to: {', '.join(mentioned_entities[:3]) if mentioned_entities else entity}
"""

        # Create comprehensive documents text including structured knowledge
        structured_additions = []
        if all_key_takeaways:
            structured_additions.append(f"Key Takeaways: {'; '.join(all_key_takeaways[:5])}")
        if all_actionable_insights:
            structured_additions.append(f"Actionable Insights: {'; '.join(all_actionable_insights[:5])}")
        if all_specific_examples:
            structured_additions.append(f"Examples: {'; '.join(all_specific_examples[:3])}")
        if all_practical_applications:
            structured_additions.append(f"Applications: {'; '.join(all_practical_applications[:3])}")
        if all_technologies:
            structured_additions.append(f"Technologies: {'; '.join(all_technologies[:5])}")
        if all_tools:
            structured_additions.append(f"Tools: {'; '.join(all_tools[:5])}")
        
        documents_text = combined_content
        if structured_additions:
            documents_text += f"\n\nSTRUCTURED INSIGHTS:\n{chr(10).join(structured_additions)}"

        # Enhanced prompt for Claude synthesis
        full_prompt = f"""Based on the provided content about Ed's insights, generate a comprehensive response following these guidelines:

CONTENT STRUCTURE:
- Open with a clear, direct answer to the user's question
- Organize information in logical sections with clear flow
- Use bullet points for multiple related concepts
- Provide specific examples and concrete details from Ed's content
- Group related information together rather than scattering it

WRITING QUALITY:
- Write conversationally but professionally 
- Avoid repetitive phrasing between sections
- Make each point distinct and valuable
- Connect related concepts to show relationships
- Explain WHY Ed recommends something, not just WHAT

DEPTH & INSIGHT:
- Include both conceptual understanding and practical implementation
- Highlight Ed's unique perspectives and reasoning
- Address potential follow-up questions proactively
- Provide context for recommendations
- If you notice Ed's opinion has evolved over time (based on post dates), mention it naturally like "Initially Ed believed X (early dates), but has since evolved to Y (recent dates)"
- When there's a clear temporal pattern in opinions, acknowledge the evolution without overemphasizing it

CITATION INTEGRATION:
- Place citations [edhonour_ID] at the END of each sentence that uses that source
- Do NOT put citations on section headers or titles
- Each sentence should have its own citation if it comes from a specific source
- Don't over-cite - one citation per sentence is sufficient

CITATION PLACEMENT RULES:
- Citations go at the END of sentences, before the period [edhonour_ID].
- For bullet points, put the citation at the end of each bullet point [edhonour_ID]
- NEVER put citations on headers or section titles
- If multiple sources support one sentence, list them together [edhonour_ID1, edhonour_ID2]

Example:
❌ WRONG (citation on header):
1. Master Essential AI Skills [edhonour_ABC123]
- Learn prompt engineering fundamentals
- Practice with real examples

✅ CORRECT (citation on sentences):
1. Master Essential AI Skills
- Learn prompt engineering fundamentals through structured exercises [edhonour_ABC123].
- Practice with real examples from production systems [edhonour_XYZ789].

- Every factual statement needs its source citation at the sentence end
- Headers should NOT have citations - only the content sentences
- This enables proper source tracking for each piece of information
{context_part}
User's question: {query}

Relevant content: {documents_text}

Return your response as valid JSON:
{{
    "answer": "Your comprehensive response here (escape double quotes with \\" if needed)",
    "response_type": "definition|tips|comprehensive|opinion|comparison|implementation|followup",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of how you interpreted the query and chose your approach",
    "follow_up_questions": ["suggested follow-up 1", "suggested follow-up 2"]
}}

Generate a focused, comprehensive response that maximizes value for the user."""
        
        return full_prompt
    
    def _assess_query_complexity(self, query: str) -> str:
        """Assess query complexity for dynamic document selection"""
        query_lower = query.lower()
        
        comprehensive_indicators = [
            'everything', 'comprehensive', 'detailed', 'complete', 'full guide',
            'tell me all', 'explain everything', 'in depth', 'thorough'
        ]
        
        moderate_indicators = [
            'tips', 'advice', 'how to', 'best practices', 'recommendations',
            'steps', 'process', 'examples', 'implementation'
        ]
        
        if any(indicator in query_lower for indicator in comprehensive_indicators):
            return "comprehensive"
        elif any(indicator in query_lower for indicator in moderate_indicators):
            return "moderate"
        else:
            return "simple"
    
    
    # ==========================================
    # HELPER METHODS - INITIALIZATION AND SETUP  
    # ==========================================
    
    def _load_all_documents(self):
        """Load and index all enhanced documents"""
        print("📁 Loading enhanced documents...")
        print("📁 Loading enhanced documents...")
        
        self.documents = []
        self.enhanced_content_index = {}
        self.entity_content_map = {}
        
        
        self.documents = []
        self.enhanced_content_index = {}
        self.entity_content_map = {}
        
        enhanced_path = Path(self.enhanced_data_path)
        if not enhanced_path.exists():
            print(f"⚠️ Enhanced data path not found: {enhanced_path}")
            return
            
        # Load ALL JSON files from enhanced_processed path ONLY
        json_files = list(enhanced_path.glob("**/*.json"))
        print(f"📄 Found {len(json_files)} JSON files in enhanced_processed")
        print(f"📁 Loading from path: {enhanced_path.absolute()}")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    enhanced_content = json.load(f)
                
                # Use document_id from enhanced file, fallback to filename without "enhanced_" prefix
                doc_id = enhanced_content.get('document_id') or json_file.stem.replace('enhanced_', '')
                
                # Store in comprehensive index
                self.enhanced_content_index[doc_id] = enhanced_content
                self.documents.append(enhanced_content)
                
            except Exception as e:
                print(f"⚠️ Error loading {json_file}: {e}")
        
        print(f"✅ Loaded {len(self.documents)} documents")
    
    # ==========================================
    # CONVERSATION CONTEXT METHODS
    # ==========================================
    
    def get_conversation_debug_info(self) -> Dict[str, Any]:
        """
        Get conversation tracking debug information for web interface and testing.
        
        Returns:
            Dictionary with conversation state and debug info
        """
        if not hasattr(self, 'conversation_tracker'):
            return {"error": "ConversationTracker not initialized"}
        
        return {
            "conversation_summary": self.conversation_tracker.get_conversation_summary(),
            "recent_context": [
                {
                    "query": ctx.query,
                    "timestamp": ctx.timestamp,
                    "entities": ctx.entities_mentioned,
                    "doc_count": len(ctx.document_ids)
                }
                for ctx in self.conversation_tracker.get_recent_context(limit=5)
            ],
            "used_documents_count": len(self.conversation_tracker.get_used_document_ids()),
            "debug_state": self.conversation_tracker.debug_conversation_state()
        }
    
    def clear_conversation_context(self):
        """Clear conversation context (for new sessions or testing)"""
        if hasattr(self, 'conversation_tracker'):
            self.conversation_tracker.clear_session()
            print("🧹 Conversation context cleared")
    
