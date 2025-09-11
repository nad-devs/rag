"""
DocumentSearcher - Handles all vector and text-based document search functionality
Extracted from GPT4ExtractionEngine as part of monolithic refactoring
"""

from typing import List, Dict, Any, Set
import json
from qdrant_client import QdrantClient


class DocumentSearcher:
    """Handles multi-strategy document search across vector collections and enhanced indices"""
    
    def __init__(self, qdrant_client: QdrantClient, collections: Dict[str, str], enhanced_data: List[Dict], embedding_model=None):
        """Initialize with Qdrant client and collections"""
        self.qdrant_client = qdrant_client
        self.collections = collections
        self.enhanced_data = enhanced_data
        self.embedding_model = embedding_model
        
        # Create enhanced data lookups for fast text search
        self._build_enhanced_indices()
    
    def _build_enhanced_indices(self):
        """Build indices for fast text-based searching"""
        self.entity_index = {}
        self.content_index = {}
        
        for doc in self.enhanced_data:
            doc_id = doc.get('document_id', doc.get('post_id', ''))  # Try document_id first, fallback to post_id
            
            # Entity index - map entities to documents
            entities = doc.get('entities', [])
            for entity in entities:
                entity_lower = entity.lower()
                if entity_lower not in self.entity_index:
                    self.entity_index[entity_lower] = []
                self.entity_index[entity_lower].append(doc)
            
            # Content index - map content words to documents  
            content_text = doc.get('content_text', '').lower()
            for word in content_text.split():
                if len(word) > 3:  # Skip short words
                    if word not in self.content_index:
                        self.content_index[word] = []
                    self.content_index[word].append(doc)
    
    def determine_vector_strategy(self, user_query: str, depth: int) -> Dict[str, int]:
        """Determine optimal vector search strategy based on query and depth"""
        base_strategy = {
            'content': 8,
            'qa': 8, 
            'tech': 8,
            'context': 6
        }
        
        # Adjust based on depth
        if depth == 0:  # First query - broader search
            multiplier = 1.0
        elif depth == 1:  # Follow-up - focus on specific content
            multiplier = 0.8
            base_strategy['content'] = 10  # More content focus
        else:  # Deep follow-up - very focused
            multiplier = 0.6
            base_strategy['content'] = 12
        
        # Query-based adjustments
        query_lower = user_query.lower()
        if any(word in query_lower for word in ['how', 'what', 'why', 'when', 'where']):
            base_strategy['qa'] += 2
        
        if any(word in query_lower for word in ['tool', 'software', 'app', 'platform']):
            base_strategy['tech'] += 2
            
        if any(word in query_lower for word in ['example', 'case', 'story', 'experience']):
            base_strategy['context'] += 3
        
        # Apply multiplier and ensure minimum values
        for key in base_strategy:
            base_strategy[key] = max(2, int(base_strategy[key] * multiplier))
            
        return base_strategy
    
    def search_vector_type(self, vector_type: str, entity: str, limit: int) -> List[Dict[str, Any]]:
        """Search a specific vector collection"""
        try:
            collection_name = self.collections.get(vector_type)
            if not collection_name:
                print(f"   ⚠️  Collection not found for vector_type: {vector_type}")
                return []
            
            # Convert text to vector using embedding model
            if self.embedding_model:
                query_vector = self.embedding_model.encode(entity, convert_to_tensor=False).tolist()
            else:
                print(f"   ⚠️  No embedding model available for vector search")
                return []
            
            # Use query_points instead of deprecated search method
            search_response = self.qdrant_client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=limit
            )
            
            # Debug output
            print(f"   🔍 Vector search in {collection_name}:")
            print(f"      Response type: {type(search_response)}")
            
            # Access points from the response
            search_results = search_response.points if hasattr(search_response, 'points') else []
            print(f"      Found {len(search_results)} results")
            if search_results and len(search_results) > 0:
                print(f"      First result score: {search_results[0].score}")
            
            results = []
            for result in search_results:
                payload = result.payload
                score_value = float(result.score)
                payload['score'] = score_value
                payload['similarity_score'] = score_value  # ADD: Map to expected field name
                payload['search_type'] = 'vector_semantic'  # ADD: Search type for ranking
                payload['vector_type'] = collection_name  # Use actual collection name instead of mapped key
                # Map document_id to doc_id for compatibility with other components
                if 'document_id' in payload:
                    payload['doc_id'] = payload['document_id']
                
                # DEBUG: Verify score is set
                if score_value > 0:
                    print(f"      ✅ Score preserved: {payload.get('doc_id', 'unknown')[:20]} = {score_value:.3f}")
                
                results.append(payload)
            
            return results
            
        except Exception as e:
            print(f"   ❌ Vector search error for {vector_type}: {str(e)}")
            return []
    
    def cross_document_search(self, entity: str, user_query: str) -> List[Dict[str, Any]]:
        """Perform cross-document search for comprehensive results"""
        try:
            # Search across all collections with the full query context
            all_results = []
            
            for vector_type in self.collections.keys():
                results = self.search_vector_type(vector_type, f"{entity} {user_query}", 5)
                all_results.extend(results)
            
            # Sort by score and return top results
            all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
            return all_results[:15]
            
        except Exception as e:
            print(f"   ❌ Cross-document search error: {str(e)}")
            return []
    
    def vector_search_entity(self, entity: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Main vector search with dynamic quality-based allocation"""
        try:
            # STEP 1: Quality Assessment - small sample from each collection
            print(f"   🔍 Quality assessment for '{entity}'...")
            quality_scores = {}
            sample_results = {}
            
            for vector_type in self.collections.keys():
                # Get top 3 results for quality assessment
                sample = self.search_vector_type(vector_type, entity, 3)
                if sample:
                    # Calculate quality metrics
                    top_score = sample[0].get('score', 0.0)
                    raw_scores = [r.get('score', 0.0) for r in sample]
                    try:
                        scores = [float(s) for s in raw_scores]
                        avg_score = sum(scores) / len(scores) if scores else 0.0
                    except Exception:
                        scores = [0.0] * len(raw_scores)
                        avg_score = 0.0
                    quality_scores[vector_type] = {
                        'top_score': top_score,
                        'avg_score': avg_score,
                        'quality_rating': self._calculate_quality_rating(top_score, avg_score)
                    }
                    sample_results[vector_type] = sample
                    print(f"     {vector_type}: top={top_score:.3f}, avg={avg_score:.3f}")
                else:
                    quality_scores[vector_type] = {
                        'top_score': 0.0,
                        'avg_score': 0.0,
                        'quality_rating': 'poor'
                    }
                    sample_results[vector_type] = []
            
            # STEP 2: Dynamic Allocation Strategy
            allocation = self._calculate_dynamic_allocation(quality_scores, limit)
            print(f"   📊 Dynamic allocation: {allocation}")
            
            # STEP 3: Fetch results based on dynamic allocation
            all_results = []
            total_fetched = 0
            
            for vector_type, allocated_limit in allocation.items():
                if allocated_limit > 3:  # We already have 3 from quality assessment
                    additional_needed = allocated_limit - 3
                    additional_results = self.search_vector_type(vector_type, entity, allocated_limit)[3:]
                    final_results = sample_results[vector_type] + additional_results
                else:
                    final_results = sample_results[vector_type][:allocated_limit]
                
                all_results.extend(final_results)
                total_fetched += len(final_results)
                print(f"     📦 {vector_type}: fetched {len(final_results)}")
            
            print(f"   ✅ Total vector results: {total_fetched}")
            return all_results
            
        except Exception as e:
            print(f"   ❌ Vector search error: {str(e)}")
            return self.vector_search_entity_fallback(entity, limit)
    
    def _calculate_quality_rating(self, top_score: float, avg_score: float) -> str:
        """Calculate quality rating based on scores"""
        if top_score >= 0.85 and avg_score >= 0.75:
            return 'excellent'
        elif top_score >= 0.75 and avg_score >= 0.65:
            return 'good'
        elif top_score >= 0.65 and avg_score >= 0.55:
            return 'fair'
        else:
            return 'poor'
    
    def _calculate_dynamic_allocation(self, quality_scores: Dict, total_limit: int) -> Dict[str, int]:
        """Calculate dynamic allocation based on quality scores"""
        # Base allocation (minimum for each collection)
        base_allocation = {vt: 2 for vt in self.collections.keys()}
        remaining_limit = total_limit - sum(base_allocation.values())
        
        if remaining_limit <= 0:
            return base_allocation
        
        # Calculate quality-based bonus allocation
        quality_weights = {}
        total_weight = 0
        
        for vector_type, scores in quality_scores.items():
            top_score = scores['top_score']
            avg_score = scores['avg_score']
            
            # Weight calculation favoring higher scores
            if top_score >= 0.8:
                weight = 4.0
            elif top_score >= 0.7:
                weight = 3.0
            elif top_score >= 0.6:
                weight = 2.0
            elif top_score >= 0.5:
                weight = 1.0
            else:
                weight = 0.5
            
            # Boost for consistent quality (high average)
            if avg_score >= 0.7:
                weight *= 1.3
            elif avg_score >= 0.6:
                weight *= 1.1
            
            quality_weights[vector_type] = weight
            total_weight += weight
        
        # Distribute remaining limit based on weights
        final_allocation = base_allocation.copy()
        
        if total_weight > 0:
            for vector_type, weight in quality_weights.items():
                bonus = int((weight / total_weight) * remaining_limit)
                final_allocation[vector_type] += bonus
        
        # Ensure we don't exceed total limit
        current_total = sum(final_allocation.values())
        if current_total > total_limit:
            # Proportionally reduce allocations
            scale_factor = total_limit / current_total
            for vector_type in final_allocation:
                final_allocation[vector_type] = max(1, int(final_allocation[vector_type] * scale_factor))
        
        return final_allocation
    
    def vector_search_entity_fallback(self, entity: str, limit: int) -> List[Dict[str, Any]]:
        """Fallback vector search with simple equal allocation"""
        try:
            print(f"   🔄 Using fallback vector search for '{entity}'")
            per_collection_limit = max(1, limit // len(self.collections))
            
            all_results = []
            for vector_type in self.collections.keys():
                results = self.search_vector_type(vector_type, entity, per_collection_limit)
                all_results.extend(results)
            
            # Sort by score and return top results
            all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
            return all_results[:limit]
            
        except Exception as e:
            print(f"   ❌ Fallback search failed: {str(e)}")
            return []
    
    def search_enhanced_entity_index(self, entity: str) -> List[Dict[str, Any]]:
        """Search enhanced data using entity index"""
        try:
            entity_lower = entity.lower()
            direct_matches = self.entity_index.get(entity_lower, [])
            
            # Also search for partial matches
            partial_matches = []
            for indexed_entity, docs in self.entity_index.items():
                if entity_lower in indexed_entity or indexed_entity in entity_lower:
                    partial_matches.extend(docs)
            
            # Combine and deduplicate
            all_matches = direct_matches + partial_matches
            seen_ids = set()
            unique_matches = []
            
            for doc in all_matches:
                doc_id = doc.get('document_id', doc.get('post_id', ''))  # Try document_id first, fallback to post_id
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    unique_matches.append(doc)
            
            return unique_matches[:15]
            
        except Exception as e:
            print(f"   ❌ Entity index search error: {str(e)}")
            return []
    
    def search_enhanced_content_text(self, entity: str) -> List[Dict[str, Any]]:
        """Search enhanced data using content text index"""
        try:
            entity_words = entity.lower().split()
            matching_docs = []
            
            for word in entity_words:
                if len(word) > 3:  # Skip short words
                    word_matches = self.content_index.get(word, [])
                    matching_docs.extend(word_matches)
            
            # Count occurrences and sort by relevance
            doc_counts = {}
            for doc in matching_docs:
                doc_id = doc.get('document_id', doc.get('post_id', ''))  # Try document_id first, fallback to post_id
                if doc_id not in doc_counts:
                    doc_counts[doc_id] = {'doc': doc, 'count': 0}
                doc_counts[doc_id]['count'] += 1
            
            # Sort by count (relevance) and return top matches
            sorted_docs = sorted(doc_counts.values(), key=lambda x: x['count'], reverse=True)
            return [item['doc'] for item in sorted_docs[:10]]
            
        except Exception as e:
            print(f"   ❌ Content text search error: {str(e)}")
            return []
    
    def vector_search_entity_with_context(
        self, 
        entity: str, 
        conversation_context: Dict[str, Any],
        used_document_ids: Set[str],
        is_followup: bool,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Context-aware vector search with conversation history"""
        try:
            print(f"   🔍 Context-aware vector search for '{entity}'")
            print(f"   📝 Context: {conversation_context.get('has_context', False)}")
            print(f"   🔄 Follow-up: {is_followup}")
            print(f"   📋 Used docs: {len(used_document_ids)}")
            
            # Determine search strategy based on context
            if conversation_context.get('has_context', False):
                # Use previous context to enhance search
                context_keywords = conversation_context.get('keywords', [])
                enhanced_query = f"{entity} {' '.join(context_keywords[:3])}"
            else:
                enhanced_query = entity
            
            # Adjust search strategy for follow-ups
            if is_followup:
                # For follow-ups, diversify search to avoid repetition
                strategy = self.determine_vector_strategy(enhanced_query, depth=1)
            else:
                strategy = self.determine_vector_strategy(enhanced_query, depth=0)
            
            all_results = []
            for vector_type, type_limit in strategy.items():
                results = self.search_vector_type(vector_type, enhanced_query, type_limit)
                all_results.extend(results)
            
            # Filter out previously used documents if this is a follow-up
            if used_document_ids:
                filtered_results = []
                for result in all_results:
                    doc_id = result.get('doc_id', result.get('post_id', ''))  # Try doc_id first, fallback to post_id
                    if doc_id not in used_document_ids:
                        filtered_results.append(result)
                all_results = filtered_results
                print(f"   🔄 Filtered out {len(all_results)} previously used docs")
            
            # Sort by score and return top results  
            all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
            return all_results[:limit]
            
        except Exception as e:
            print(f"   ❌ Context-aware search error: {str(e)}")
            return self.vector_search_entity(entity, limit)
    
    def search_enhanced_content_text_with_context(
        self,
        entity: str,
        conversation_context: Dict[str, Any], 
        used_document_ids: Set[str],
        is_followup: bool
    ) -> List[Dict[str, Any]]:
        """Context-aware enhanced content text search"""
        try:
            print(f"   🔍 Context-aware content search for '{entity}'")
            
            # Get base content search results
            base_results = self.search_enhanced_content_text(entity)
            
            # Apply context filtering if available
            if conversation_context.get('has_context', False):
                context_keywords = conversation_context.get('keywords', [])
                if context_keywords:
                    # Boost results that contain context keywords
                    for result in base_results:
                        content_text = result.get('content_text', '').lower()
                        boost_score = 0
                        for keyword in context_keywords[:5]:  # Top 5 context keywords
                            if keyword.lower() in content_text:
                                boost_score += 1
                        result['context_boost'] = boost_score
                    
                    # Sort by context boost then by original relevance
                    base_results.sort(key=lambda x: (x.get('context_boost', 0), x.get('relevance_score', 0)), reverse=True)
            
            # Filter out used documents for follow-ups
            if used_document_ids:
                filtered_results = []
                for result in base_results:
                    doc_id = result.get('doc_id', result.get('post_id', ''))  # Try doc_id first, fallback to post_id
                    if doc_id not in used_document_ids:
                        filtered_results.append(result)
                base_results = filtered_results
            
            return base_results[:10]
            
        except Exception as e:
            print(f"   ❌ Context-aware content search error: {str(e)}")
            return self.search_enhanced_content_text(entity)