"""
MetadataScorer - Handles metadata-based relevance scoring and ranking
Extracted from GPT4ExtractionEngine as part of monolithic refactoring
"""

from typing import List, Dict, Any, Set


class MetadataScorer:
    """Handles metadata-based scoring, ranking, and deduplication of search results"""
    
    def __init__(self, enhanced_content_index: Dict[str, Dict]):
        """Initialize with enhanced content index"""
        self.enhanced_content_index = enhanced_content_index
    
    def search_metadata_fields(self, query: str) -> List[Dict[str, Any]]:
        """Search in structured metadata fields for enhanced relevance"""
        results = []
        query_lower = query.lower().strip()
        
        print(f"   📊 Searching metadata fields for: '{query}'")
        
        # Extract potential tools from query
        potential_tools = self._extract_tools_from_query(query_lower)
        
        for doc_id, enhanced_content in self.enhanced_content_index.items():
            metadata_score = 0
            match_reasons = []
            
            # Get learning_metadata section
            learning_metadata = enhanced_content.get('learning_metadata', {})
            
            # 1. Check natural_questions for exact or partial matches
            natural_questions = learning_metadata.get('natural_questions', [])
            for question in natural_questions:
                if question and isinstance(question, str):
                    if query_lower in question.lower() or question.lower() in query_lower:
                        metadata_score += 100  # High boost for question match
                        match_reasons.append(f"question_match: '{question}'")
                        break
            
            # 2. Check tools_and_platforms for tool mentions
            tools_and_platforms = learning_metadata.get('tools_and_platforms', [])
            if isinstance(tools_and_platforms, list):
                doc_tools = []
                for tool_item in tools_and_platforms:
                    if isinstance(tool_item, dict) and 'name' in tool_item:
                        tool_name = tool_item['name'].lower()
                        doc_tools.append(tool_name)
                        # Check if query mentions this tool (avoid single letters)
                        if len(tool_name) > 1 and (tool_name in query_lower or any(pt in tool_name for pt in potential_tools)):
                            tool_score = 80 if tool_item.get('recommendation') == 'strongly_recommended' else 60
                            metadata_score += tool_score
                            match_reasons.append(f"tool_match: '{tool_item['name']}'")
            
            # 3. Check related_topics for topic relevance
            related_topics = learning_metadata.get('related_topics', [])
            if isinstance(related_topics, list):
                for topic in related_topics:
                    if isinstance(topic, str) and topic.lower() in query_lower:
                        metadata_score += 40  # Medium boost for topic match
                        match_reasons.append(f"topic_match: '{topic}'")
            
            # 4. Check technologies_mentioned for tech stack relevance
            technologies_mentioned = learning_metadata.get('technologies_mentioned', [])
            if isinstance(technologies_mentioned, list):
                for tech_item in technologies_mentioned:
                    if isinstance(tech_item, dict) and 'name' in tech_item:
                        tech_name = tech_item['name'].lower()
                        if len(tech_name) > 1 and tech_name in query_lower:
                            metadata_score += 50  # Medium-high boost for tech match
                            match_reasons.append(f"tech_match: '{tech_item['name']}'")
            
            # 5. Check content_category for category matching
            content_category = learning_metadata.get('content_category', '')
            if content_category:
                category_terms = {
                    'tool_review': ['what is', 'review', 'compare'],
                    'tutorial': ['how to', 'guide', 'tutorial'],
                    'best_practices': ['best practices', 'tips', 'advice'],
                    'comparison': ['vs', 'versus', 'compare', 'difference']
                }
                
                if content_category in category_terms:
                    if any(term in query_lower for term in category_terms[content_category]):
                        metadata_score += 30  # Category relevance boost
                        match_reasons.append(f"category_match: '{content_category}'")
            
            # Only include results with meaningful metadata matches
            if metadata_score > 0:
                results.append({
                    'doc_id': doc_id,
                    'content_text': enhanced_content.get('content_text', ''),
                    'enhanced_data': enhanced_content,
                    'search_type': 'metadata_enhanced',
                    'metadata_score': metadata_score,
                    'similarity_score': 0.0,  # Will be updated if vector result exists
                    'match_reasons': match_reasons,
                    'main_lesson': enhanced_content.get('main_lesson', ''),
                    'key_takeaways': enhanced_content.get('key_takeaways', []),
                    'actionable_insights': enhanced_content.get('actionable_insights', [])
                })
        
        # Sort by metadata score (highest relevance first)
        results.sort(key=lambda x: x.get('metadata_score', 0), reverse=True)
        
        print(f"   📊 Found {len(results)} metadata-enhanced results")
        if results:
            top_3 = results[:3]
            for i, result in enumerate(top_3, 1):
                print(f"      {i}. {result['doc_id']} (score: {result['metadata_score']}) - {', '.join(result['match_reasons'][:2])}")
        
        return results
    
    def _extract_tools_from_query(self, query: str) -> List[str]:
        """Extract potential tool names from query"""
        # Common tools mentioned in the dataset
        known_tools = [
            'claude code', 'cursor', 'vs code', 'windsurf', 'gemini cli',
            'github copilot', 'chatgpt', 'claude', 'gemini', 'anthropic',
            'openai', 'tensorflow', 'pytorch', 'react', 'vue', 'angular',
            'bootstrap', 'tailwind', 'docker', 'kubernetes'
        ]
        
        found_tools = []
        for tool in known_tools:
            if tool in query:
                found_tools.append(tool)
        
        return found_tools
    
    def deduplicate_and_rank(self, results: List[Dict[str, Any]], entity: str) -> List[Dict[str, Any]]:
        """Remove duplicates and rank by relevance with metadata-enhanced scoring"""
        seen_doc_ids = set()
        unique_results = []
        
        # Enhanced priority with metadata search
        search_type_priority = {
            'metadata_enhanced': 4,  # Highest priority for metadata matches
            'vector': 3, 
            'entity_index': 2, 
            'enhanced_content': 1
        }
        
        # Multi-factor sorting with metadata score
        sorted_results = sorted(results, key=lambda x: (
            search_type_priority.get(x.get('search_type', ''), 0),
            x.get('metadata_score', 0),  # Include metadata score
            x.get('similarity_score', 0),
            x.get('mention_count', 0)
        ), reverse=True)
        
        for result in sorted_results:
            doc_id = result.get('doc_id', '')
            if doc_id and doc_id not in seen_doc_ids:
                seen_doc_ids.add(doc_id)
                unique_results.append(result)
        
        return unique_results
    
    def deduplicate_and_rank_with_context(
        self, 
        results: List[Dict[str, Any]], 
        entity: str, 
        conversation_context: Dict, 
        used_document_ids: Set[str], 
        is_followup: bool
    ) -> List[Dict[str, Any]]:
        """Context-aware deduplication and ranking with metadata scoring"""
        seen_doc_ids = set()
        unique_results = []
        
        # Enhanced search type priority for context-aware results
        search_type_priority = {
            'vector_with_metadata': 6,     # HIGHEST: Vector similarity + metadata relevance
            'metadata_enhanced': 5,        # Metadata-only matches
            'context_aware_vector': 4,
            'vector_semantic': 3, 
            'entity_index': 2, 
            'context_aware_content': 2,
            'enhanced_content': 1
        }
        
        # Sort with context-aware ranking including metadata scores
        sorted_results = sorted(results, key=lambda x: (
            search_type_priority.get(x.get('search_type', ''), 0),
            x.get('metadata_score', 0),  # CRITICAL: Include metadata score in ranking
            x.get('similarity_score', 0) + x.get('context_boost', 0),
            x.get('context_terms_matched', 0),
            x.get('mention_count', 0)
        ), reverse=True)
        
        for result in sorted_results:
            doc_id = result.get('doc_id', '')
            
            # Flag previously shown documents but allow them to compete on relevance
            was_recently_shown = doc_id in used_document_ids
                
            if doc_id and doc_id not in seen_doc_ids:
                seen_doc_ids.add(doc_id)
                
                # Add context metadata for debugging
                result['context_aware'] = True
                result['previously_shown'] = was_recently_shown
                
                unique_results.append(result)
        
        previously_shown_count = len([r for r in unique_results if r.get('previously_shown', False)])
        print(f"   🎯 Context filtering: {len(results)} → {len(unique_results)} results ({previously_shown_count} previously shown, flagged but competing)")
        
        # CLEAN RANKING SUMMARY - Show scoring overview
        if sorted_results:
            top_result = sorted_results[0]
            sim_score = top_result.get('similarity_score', 0)
            meta_score = top_result.get('metadata_score', 0) 
            ctx_score = top_result.get('context_boost', 0)
            total_score = sim_score + meta_score + ctx_score
            top_doc = top_result.get('doc_id', 'unknown')[:15]
            
            print(f"   🏆 TOP RESULT: {top_doc} | sim:{sim_score:.3f} meta:{meta_score:.3f} ctx:{ctx_score:.3f} total:{total_score:.3f}")
            
            # Show score distribution summary
            meta_scores = [r.get('metadata_score', 0) for r in sorted_results[:5] if r.get('metadata_score', 0) > 0]
            ctx_scores = [r.get('context_boost', 0) for r in sorted_results[:5] if r.get('context_boost', 0) > 0]
            
            if meta_scores:
                print(f"   🎯 METADATA: {len(meta_scores)} docs with scores (avg: {sum(meta_scores)/len(meta_scores):.1f})")
            if ctx_scores:
                print(f"   🎯 CONTEXT: {len(ctx_scores)} docs boosted (avg: {sum(ctx_scores)/len(ctx_scores):.1f})")
        
        return unique_results
    
    def apply_context_scoring(self, results: List[Dict[str, Any]], conversation_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply context-based scoring boost to results"""
        if not conversation_context.get('has_context', False):
            return results
        
        context_keywords = conversation_context.get('keywords', [])
        if not context_keywords:
            return results
        
        print(f"   🎯 Applying context scoring with keywords: {context_keywords[:5]}")
        
        for result in results:
            context_boost = 0
            context_terms_matched = 0
            content_text = result.get('content_text', '').lower()
            
            # Check for context keyword matches
            for keyword in context_keywords[:10]:  # Check top 10 context keywords
                if keyword.lower() in content_text:
                    context_boost += 5  # 5 points per context keyword match
                    context_terms_matched += 1
            
            # Apply diminishing returns for too many matches (avoid keyword stuffing)
            if context_terms_matched > 5:
                context_boost = context_boost * 0.8
            
            result['context_boost'] = context_boost
            result['context_terms_matched'] = context_terms_matched
        
        return results
    
    def calculate_metadata_relevance_score(self, doc: Dict[str, Any], query: str) -> float:
        """Calculate a relevance score based on metadata fields"""
        score = 0.0
        query_lower = query.lower()
        
        # Check various metadata fields for relevance
        enhanced_data = doc.get('enhanced_data', {})
        learning_metadata = enhanced_data.get('learning_metadata', {})
        
        # Natural questions boost
        questions = learning_metadata.get('natural_questions', [])
        for question in questions:
            if isinstance(question, str) and query_lower in question.lower():
                score += 20.0
        
        # Tools and platforms boost
        tools = learning_metadata.get('tools_and_platforms', [])
        for tool in tools:
            if isinstance(tool, dict) and 'name' in tool:
                if tool['name'].lower() in query_lower:
                    score += 15.0
        
        # Related topics boost
        topics = learning_metadata.get('related_topics', [])
        for topic in topics:
            if isinstance(topic, str) and topic.lower() in query_lower:
                score += 10.0
        
        # Technologies mentioned boost
        techs = learning_metadata.get('technologies_mentioned', [])
        for tech in techs:
            if isinstance(tech, dict) and 'name' in tech:
                if tech['name'].lower() in query_lower:
                    score += 12.0
        
        return score