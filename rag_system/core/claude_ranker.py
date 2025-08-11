"""
ClaudeRanker - Handles Claude-based document ranking and series detection
Extracted from GPT4ExtractionEngine as part of monolithic refactoring
"""

from typing import List, Dict, Any
import json
import re


class ClaudeRanker:
    """Handles Claude-based document ranking, series detection, and relevance scoring"""
    
    def __init__(self, client, enhanced_content_index: Dict[str, Dict]):
        """Initialize with Claude client and enhanced content index"""
        self.client = client
        self.enhanced_content_index = enhanced_content_index
        self._last_claude_scores = []
    
    def should_use_claude_ranking(self, query: str, document_count: int, query_complexity: str) -> bool:
        """
        Determine if we should use Claude-based document ranking
        
        Use ranking when:
        - Many documents available (>15)
        - Complex queries that benefit from contextual understanding
        - Queries where precision matters more than speed
        """
        # Always use ranking if we have many documents
        if document_count > 25:
            return True
            
        # Use for complex queries that need precise document selection
        complex_indicators = [
            'best', 'top', 'most important', 'compare', 'vs', 'difference',
            'how to start', 'which', 'should', 'what is the'
        ]
        
        query_lower = query.lower()
        if any(indicator in query_lower for indicator in complex_indicators):
            return True
            
        # Use for comprehensive queries
        if query_complexity == "comprehensive":
            return True
            
        # Use for longer, detailed queries (likely need precise answers)
        if len(query.split()) > 8:
            return True
            
        return False
    
    def claude_rank_documents(self, query: str, mentions: List[Dict]) -> List[Dict]:
        """
        Use Claude 3.5 to rank documents by relevance to the query
        
        Args:
            query: User's original question
            mentions: List of document mentions to rank
            
        Returns:
            List of mentions sorted by Claude's relevance scoring
        """
        if not mentions:
            return mentions
            
        # Limit documents for faster Claude ranking
        max_ranking_docs = 20  # Increased from 15 to improve recall while maintaining speed
        if len(mentions) > max_ranking_docs:
            mentions = mentions[:max_ranking_docs]
            
        print(f"🎯 CLAUDE RANKING: Scoring {len(mentions)} documents for query relevance")
        
        # Store original order for comparison
        original_order = [(i+1, mention.get('doc_id', 'unknown')[:20]) for i, mention in enumerate(mentions)]
        
        # Prepare document summaries for Claude
        doc_summaries = []
        for i, mention in enumerate(mentions):
            content_text = mention.get('content_text', '')
            doc_id = mention.get('doc_id', '')
            
            # Get enhanced metadata
            metadata = mention.get('learning_metadata', {}) or mention.get('structured_data', {})
            
            # Create rich document summary
            summary = {
                'index': i,
                'doc_id': doc_id,
                'content_preview': content_text[:400] if content_text else "No content preview available",
                'main_lesson': metadata.get('main_lesson', 'Not available'),
                'content_category': metadata.get('content_category', 'unknown'),
                'key_takeaways': metadata.get('key_takeaways', [])[:3],  # First 3 takeaways
                'natural_questions': metadata.get('natural_questions', [])[:2],  # First 2 questions
                'difficulty_level': metadata.get('difficulty_level', 'unknown'),
                'content_length': len(content_text)
            }
            doc_summaries.append(summary)
        
        # Create Claude ranking prompt
        ranking_prompt = self._create_claude_ranking_prompt(query, doc_summaries)
        
        try:
            # Call Claude API for ranking with timeout
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                temperature=0.1,  # Low temperature for consistent ranking
                timeout=15.0,  # Fast timeout for ranking
                messages=[
                    {
                        "role": "user",
                        "content": ranking_prompt
                    }
                ]
            )
            
            # Parse Claude's response
            ranking_text = response.content[0].text.strip()
            scores = self._parse_claude_ranking_response(ranking_text, len(mentions))
            
            # Sort documents by Claude's scores
            scored_mentions = list(zip(mentions, scores))
            scored_mentions.sort(key=lambda x: x[1], reverse=True)  # Sort by score descending
            
            ranked_mentions = [mention for mention, score in scored_mentions]
            
            # Store scores for threshold filtering
            self._last_claude_scores = [score for mention, score in scored_mentions]
            
            # Claude ranking effectiveness - Simple order comparison
            original_doc_order = [mention.get('doc_id', '') for mention in mentions]
            claude_doc_order = [mention.get('doc_id', '') for mention in ranked_mentions]
            
            if original_doc_order == claude_doc_order:
                print(f"   ❌ NOT HELPFUL - Claude kept exact same order")
            else:
                print(f"   ✅ HELPFUL - Claude changed document order")
            
            # Debug output
            print(f"🎯 CLAUDE RANKING: Completed scoring")
            
            return ranked_mentions
            
        except Exception as e:
            print(f"⚠️ Claude ranking failed: {e}")
            print(f"📊 Falling back to original document order")
            return mentions
    
    def _create_claude_ranking_prompt(self, query: str, doc_summaries: List[Dict]) -> str:
        """Create the prompt for Claude to rank documents"""
        
        prompt = f"""You are helping rank documents for relevance to a user's query. Please score each document from 1-10 for how well it answers the user's question.

SCORING CRITERIA:
- 10: Perfect match - directly and comprehensively answers the query
- 8-9: Highly relevant - contains specific, actionable information about the topic
- 6-7: Somewhat relevant - mentions the topic with useful context
- 4-5: Loosely related - tangentially connected or partially relevant
- 1-3: Not relevant - different topic or very generic discussion

USER QUERY: "{query}"

DOCUMENTS TO SCORE:
"""
        
        for i, doc in enumerate(doc_summaries):
            prompt += f"""
Document {i+1}:
Doc ID: {doc['doc_id']}
Main Lesson: {doc['main_lesson']}
Category: {doc['content_category']} | Difficulty: {doc['difficulty_level']}
Content Preview: {doc['content_preview']}
Key Takeaways: {', '.join(doc['key_takeaways']) if doc['key_takeaways'] else 'None listed'}
Natural Questions: {', '.join(doc['natural_questions']) if doc['natural_questions'] else 'None listed'}
---"""
        
        prompt += f"""

Please respond with ONLY a JSON array of {len(doc_summaries)} scores (integers 1-10), one for each document in order:
[score1, score2, score3, ...]

Example response: [9, 3, 7, 10, 5]"""
        
        return prompt
    
    def _parse_claude_ranking_response(self, response_text: str, expected_count: int) -> List[int]:
        """Parse Claude's scoring response into a list of integers"""
        try:
            # Look for JSON array in the response
            json_match = re.search(r'\[[\d,\s]+\]', response_text)
            if json_match:
                json_str = json_match.group()
                scores = json.loads(json_str)
                
                # Validate scores
                if len(scores) == expected_count and all(isinstance(s, int) and 1 <= s <= 10 for s in scores):
                    return scores
            
            print(f"⚠️ Invalid Claude ranking response: {response_text[:200]}")
            
        except Exception as e:
            print(f"⚠️ Error parsing Claude ranking: {e}")
        
        # Fallback: return neutral scores maintaining original order with slight preference for earlier docs
        return [7 - min(i * 0.1, 2) for i in range(expected_count)]
    
    def detect_and_boost_series(self, query: str, mentions: List[Dict]) -> List[Dict]:
        """
        Detect if any documents are part of a series and boost related series parts
        
        Args:
            query: User's original question
            mentions: List of document mentions from vector search
            
        Returns:
            Enhanced mentions list with series parts boosted
        """
        # Find series in top results
        series_found = {}
        original_doc_ids = {mention.get('doc_id', mention.get('post_id', '')) for mention in mentions if mention.get('doc_id') or mention.get('post_id')}
        
        # Check top 10 results for series indicators
        for i, mention in enumerate(mentions[:10]):
            doc_id = mention.get('doc_id', mention.get('post_id', ''))
            learning_metadata = {}
            
            # The vector search mentions have empty metadata, so always lookup from enhanced_content_index
            if doc_id and doc_id in self.enhanced_content_index:
                content = self.enhanced_content_index[doc_id]
                learning_metadata = content.get('full_metadata', {})
            
            # Series check for document: {doc_id}
            
            if learning_metadata.get('is_part_of_series', False):
                series_title = learning_metadata.get('series_title', '')
                if series_title and series_title not in series_found:
                    series_found[series_title] = {
                        'found_part': mention,
                        'found_position': i,
                        'series_metadata': learning_metadata
                    }
                    print(f"   ✅ Found series: '{series_title}' (Part {learning_metadata.get('part_number', '?')})")
        
        if not series_found:
            print(f"📊 SERIES DETECTION: No series detected in top results")
            return mentions
        
        print(f"📊 SERIES DETECTION: Found {len(series_found)} series in top results")
        
        # Promote and reorder series parts
        all_series_parts = []
        non_series_mentions = []
        
        for series_title, series_info in series_found.items():
            # Check if the query is specifically about this series topic
            query_lower = query.lower()
            series_title_lower = series_title.lower()
            
            # Calculate series relevance to query
            series_relevance = 0
            
            # Strong indicators: series title words in query
            series_words = series_title_lower.split()
            for word in series_words:
                if len(word) > 3 and word in query_lower:  # Ignore short words like "the", "of"
                    series_relevance += 2
            
            # Weak indicators: general topic overlap
            if any(word in series_title_lower for word in ['coding', 'vibe', 'ai', 'programming']):
                if any(word in query_lower for word in ['coding', 'vibe', 'ai', 'programming']):
                    series_relevance += 1
            
            # Only promote series if there's sufficient relevance
            SERIES_RELEVANCE_THRESHOLD = 3  # Require strong topic overlap
            
            if series_relevance >= SERIES_RELEVANCE_THRESHOLD:
                print(f"   🎯 Promoting relevant series: '{series_title}' (relevance: {series_relevance})")
                
                # Find ALL parts of this series in the enhanced index
                series_parts_from_index = []
                for doc_id, content in self.enhanced_content_index.items():
                    content_metadata = content.get('full_metadata', {})
                    if (content_metadata.get('is_part_of_series', False) and 
                        content_metadata.get('series_title', '') == series_title):
                        
                        # Create mention structure for this series part
                        series_part = {
                            'doc_id': doc_id,
                            'content_text': content['content_text'],
                            'enhanced_data': content,
                            'learning_metadata': content_metadata,
                            'source_type': 'series_promoted',
                            'part_number': content_metadata.get('part_number', 999)
                        }
                        series_parts_from_index.append(series_part)
                
                # Sort series parts by part number
                series_parts_from_index.sort(key=lambda x: x['part_number'])
                
                # Add all series parts to the promoted list
                for part in series_parts_from_index:
                    all_series_parts.append(part)
                    was_in_original = part['doc_id'] in original_doc_ids
                    status = "promoted" if was_in_original else "added"
                    print(f"   📈 {status.title()}: {part['doc_id']} (Part {part['part_number']})")
            else:
                print(f"   ⏭️ Skipping irrelevant series: '{series_title}' (relevance: {series_relevance} < {SERIES_RELEVANCE_THRESHOLD})")
        
        # Separate non-series mentions from original list
        for mention in mentions:
            doc_id = mention['doc_id']
            # Check if this doc is part of any detected series
            is_series_part = False
            if doc_id in self.enhanced_content_index:
                content = self.enhanced_content_index[doc_id]
                metadata = content.get('full_metadata', {})
                if metadata.get('is_part_of_series', False):
                    series_title = metadata.get('series_title', '')
                    if series_title in series_found:
                        is_series_part = True
            
            if not is_series_part:
                non_series_mentions.append(mention)
        
        # Reorder: Series parts first (in order), then non-series mentions
        reordered_mentions = all_series_parts + non_series_mentions
        
        if all_series_parts:
            print(f"📊 SERIES PROMOTION: Promoted {len(all_series_parts)} series parts to top positions")
            return reordered_mentions
        else:
            print(f"📊 SERIES PROMOTION: No series parts to promote")
            return mentions
    
    def apply_series_threshold_filter(self, query: str, ranked_mentions: List[Dict], original_mentions: List[Dict]) -> List[Dict]:
        """
        Apply threshold filtering to series parts based on Claude scores
        
        Args:
            query: User's original question
            ranked_mentions: Documents ranked by Claude with scores
            original_mentions: Original vector search results (always kept)
            
        Returns:
            Filtered list with low-scoring series parts removed
        """
        if not hasattr(self, '_last_claude_scores') or not self._last_claude_scores:
            # No scores available, return as-is
            return ranked_mentions
        
        # Create mapping of doc_id to Claude score
        doc_scores = {}
        for i, mention in enumerate(ranked_mentions):
            if i < len(self._last_claude_scores):
                doc_scores[mention['doc_id']] = self._last_claude_scores[i]
        
        # Identify original documents (always keep these)
        original_doc_ids = {mention['doc_id'] for mention in original_mentions}
        
        # Apply threshold filtering
        SERIES_SCORE_THRESHOLD = 6.5  # Keep series parts scoring 6.5+ out of 10
        filtered_mentions = []
        series_parts_filtered = 0
        
        for mention in ranked_mentions:
            doc_id = mention['doc_id']
            doc_score = doc_scores.get(doc_id, 7.0)  # Default score if missing
            
            # Always keep original documents that aren't series parts
            if doc_id in original_doc_ids and mention.get('source_type') != 'series_promoted':
                filtered_mentions.append(mention)
            # For promoted series parts, apply threshold
            elif mention.get('source_type') == 'series_promoted':
                if doc_score >= SERIES_SCORE_THRESHOLD:
                    filtered_mentions.append(mention)
                    print(f"   ✅ Kept series part: {doc_id} (score: {doc_score})")
                else:
                    series_parts_filtered += 1
                    print(f"   ❌ Filtered series part: {doc_id} (score: {doc_score} < {SERIES_SCORE_THRESHOLD})")
            # Keep other documents
            else:
                filtered_mentions.append(mention)
        
        if series_parts_filtered > 0:
            print(f"📊 SERIES FILTERING: Removed {series_parts_filtered} low-scoring series parts")
        
        return filtered_mentions
    
    def rank_documents_by_query_type(self, query: str, mentions: List[Dict], entity_type: str) -> List[Dict]:
        """Rank documents based on query type and entity categorization"""
        if not mentions:
            return mentions
        
        # Different ranking strategies based on entity type
        if entity_type == 'tool_definition':
            # For tool definitions, prioritize comprehensive explanations
            return self._rank_for_tool_definition(mentions)
        elif entity_type == 'comparison':
            # For comparisons, prioritize documents that mention multiple tools/options
            return self._rank_for_comparison(mentions, query)
        elif entity_type == 'best_practices':
            # For best practices, prioritize actionable advice
            return self._rank_for_best_practices(mentions)
        elif entity_type == 'numbered_list':
            # For numbered lists, prioritize structured content
            return self._rank_for_numbered_list(mentions)
        else:
            # Default ranking by relevance score
            return sorted(mentions, key=lambda x: x.get('similarity_score', 0), reverse=True)
    
    def _rank_for_tool_definition(self, mentions: List[Dict]) -> List[Dict]:
        """Rank documents for tool definition queries"""
        def tool_definition_score(mention):
            score = mention.get('similarity_score', 0)
            content = mention.get('content_text', '').lower()
            
            # Boost for definition keywords
            if any(word in content for word in ['what is', 'definition', 'explanation', 'overview']):
                score += 0.1
            
            # Boost for comprehensive content
            if len(content) > 1000:
                score += 0.05
            
            return score
        
        return sorted(mentions, key=tool_definition_score, reverse=True)
    
    def _rank_for_comparison(self, mentions: List[Dict], query: str) -> List[Dict]:
        """Rank documents for comparison queries"""
        def comparison_score(mention):
            score = mention.get('similarity_score', 0)
            content = mention.get('content_text', '').lower()
            
            # Boost for comparison keywords
            comparison_words = ['vs', 'versus', 'compared', 'difference', 'better', 'worse']
            comparison_count = sum(1 for word in comparison_words if word in content)
            score += comparison_count * 0.05
            
            return score
        
        return sorted(mentions, key=comparison_score, reverse=True)
    
    def _rank_for_best_practices(self, mentions: List[Dict]) -> List[Dict]:
        """Rank documents for best practices queries"""
        def best_practices_score(mention):
            score = mention.get('similarity_score', 0)
            content = mention.get('content_text', '').lower()
            
            # Boost for actionable keywords
            actionable_words = ['should', 'must', 'avoid', 'recommend', 'tip', 'practice']
            actionable_count = sum(1 for word in actionable_words if word in content)
            score += actionable_count * 0.03
            
            return score
        
        return sorted(mentions, key=best_practices_score, reverse=True)
    
    def _rank_for_numbered_list(self, mentions: List[Dict]) -> List[Dict]:
        """Rank documents for numbered list queries"""
        def numbered_list_score(mention):
            score = mention.get('similarity_score', 0)
            content = mention.get('content_text', '').lower()
            
            # Boost for numbered/structured content
            import re
            numbered_pattern = r'\d+\.|•|\-'
            numbered_count = len(re.findall(numbered_pattern, content))
            score += min(numbered_count * 0.01, 0.1)  # Cap the boost
            
            return score
        
        return sorted(mentions, key=numbered_list_score, reverse=True)