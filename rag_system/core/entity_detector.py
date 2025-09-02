"""
EntityDetector - Handles query analysis and entity extraction
Extracted from GPT4ExtractionEngine as part of monolithic refactoring
"""

from typing import List, Dict, Any
import re


class EntityDetector:
    """Handles entity extraction from user queries and query analysis"""
    
    def __init__(self, client, use_mistral: bool = True, local_entity_extractor=None):
        """Initialize with LLM client (Mistral preferred) and optional local extractor"""
        self.client = client
        self.use_mistral = use_mistral
        self.local_entity_extractor = local_entity_extractor
    
    def extract_entity_from_query(self, query: str, conversation_context: Dict[str, Any] = None) -> str:
        """
        Extract entity using both Claude and Local models for comparison
        
        Args:
            query: User's natural language question
            
        Returns:
            Main entity/topic as string (Claude result for now)
        """
        import time
        step_start = time.time()
        
        # Define Mistral extraction function 
        def mistral_extract(query_text: str) -> str:
            # Time prompt construction
            prompt_start = time.time()
            if conversation_context and conversation_context.get('entities'):
                previous_entities = ", ".join(conversation_context['entities'][-3:])
                prompt = f"""Previous conversation was about: {previous_entities}
Current question: "{query_text}"

If the current question contains pronouns like "it", "this", "that", resolve them using the previous conversation context.
Extract the main topic/entity:"""
            else:
                prompt = f"""Extract the main topic/entity from this user question: "{query_text}" """
            print(f"🔍 Entity prompt construction: {time.time() - prompt_start:.2f}s")

            # Time LLM API call
            print(f"🔍 MISTRAL DEBUG:")
            print(f"   Query length: {len(query_text)} chars")
            print(f"   Prompt length: {len(prompt)} chars") 
            print(f"   Prompt preview: {prompt[:200]}...")
            
            api_start = time.time()
            if self.use_mistral:
                print(f"   Model: mistral-large-latest")
                print(f"   Max tokens: 20")
                print(f"   Temperature: 0.1")
                
                # Use Mistral for entity extraction
                response = self.client.chat.completions.create(
                    model="mistral-large-latest",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=20
                )
            else:
                print(f"   Model: gpt-4")
                print(f"   Max tokens: 20")
                print(f"   Temperature: 0.1")
                
                # Fallback to other models
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=20
                )
            
            api_duration = time.time() - api_start
            print(f"🔍 MISTRAL RESPONSE:")
            print(f"   API call duration: {api_duration:.2f}s")
            print(f"   Response length: {len(str(response))} chars")
            print(f"   Response preview: {str(response)[:200]}...")
            print(f"   Response object type: {type(response)}")
            print(f"🔍 Entity LLM API call: {api_duration:.2f}s")
            
            # Time response processing
            process_start = time.time()
            result = response.choices[0].message.content.strip().lower()
            print(f"🔍 Entity response processing: {time.time() - process_start:.2f}s")
            
            return result
        
        # Use hybrid extractor if available
        entity = None
        if self.local_entity_extractor and self.local_entity_extractor.available:
            try:
                print(f"🔍 LOCAL ENTITY EXTRACTOR DEBUG:")
                print(f"   Query length: {len(query)} chars")
                print(f"   Query preview: {query[:200]}...")
                print(f"   Local extractor type: {type(self.local_entity_extractor)}")
                print(f"   Local extractor available: {self.local_entity_extractor.available}")
                
                local_start = time.time()
                result = self.local_entity_extractor.extract_entity(query)
                local_duration = time.time() - local_start
                
                print(f"🔍 LOCAL ENTITY EXTRACTOR RESPONSE:")
                print(f"   Extraction duration: {local_duration:.2f}s")
                print(f"   Result type: {type(result)}")
                print(f"   Result preview: {str(result)[:200]}...")
                print(f"🔍 Local entity extraction: {local_duration:.2f}s")
                
                # Return Claude result for now (but we see the comparison)
                entity = result["primary_entity"]
            except Exception as e:
                print(f"⚠️ Hybrid entity extraction failed: {e}")
        
        # Fallback to Mistral only if local extractor didn't work
        if entity is None:
            try:
                entity = mistral_extract(query)
            except Exception as e:
                print(f"⚠️ Claude entity extraction error: {e}")
                # Simple fallback
                query_lower = query.lower()
                common_entities = ["claude code", "gemini cli", "ai", "debugging", "rag", "python", "javascript"]
                
                for common_entity in common_entities:
                    if common_entity in query_lower:
                        entity = common_entity
                        break
                
                if entity is None:
                    words = [word for word in query.split() if len(word) > 3]
                    entity = words[0].lower() if words else "general"
        
        print(f"🔍 Total entity extraction: {time.time() - step_start:.2f}s")
        return entity
    
    def extract_query_concepts(self, query: str) -> List[str]:
        """Extract key concepts from user query for cross-document search"""
        # Simple keyword extraction - could be enhanced with NLP
        concepts = []
        important_words = query.lower().split()
        
        # Filter out common words and extract meaningful concepts
        stop_words = {'what', 'is', 'are', 'how', 'do', 'does', 'can', 'could', 'should', 'would', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        for word in important_words:
            if len(word) > 3 and word not in stop_words:
                concepts.append(word)
        
        return concepts[:5]  # Limit to top 5 concepts
    
    def categorize_entity_type(self, entity: str, query: str) -> str:
        """Categorize entity type for optimized document selection
        Based on statistical analysis of 8 query types across 446 documents"""
        
        entity_lower = entity.lower()
        query_lower = query.lower()
        
        # Tool definitions (need 6 docs for comparisons)
        tools = ['claude code', 'cursor', 'gemini cli', 'github copilot', 'copilot', 'anthropic', 'openai']
        if any(tool in entity_lower for tool in tools) and ('what is' in query_lower or 'tell me about' in query_lower):
            return 'tool_definition'
        
        # Comparisons (need 8 docs for multiple perspectives)
        if ' vs ' in query_lower or ' versus ' in query_lower or 'comparison' in query_lower or 'compare' in query_lower:
            return 'comparison'
        
        # Best practices (need 7 docs for diverse approaches)
        if 'best practices' in query_lower or 'practices for' in query_lower:
            return 'best_practices'
        
        # Numbered lists (5 docs work well)
        if any(num in query_lower for num in ['5 ', '3 ', '10 ', '7 ', '4 ', 'top ', ' reasons', ' topics']):
            return 'numbered_list'
        
        # How-to guides (need 6 docs for step-by-step)
        if 'how to' in query_lower or 'how do' in query_lower:
            return 'howto'
        
        # Technical deep-dives (need 6 docs for depth)
        technical_terms = ['architecture', 'system', 'database', 'api', 'framework', 'protocol']
        if any(term in entity_lower for term in technical_terms):
            return 'technical'
        
        # Concept definitions (need 5 docs for clear explanation)
        if 'what is' in query_lower or 'explain' in query_lower:
            return 'concept_definition'
        
        # Default to concept_definition
        return 'concept_definition'
    
    def expand_query_with_context(self, entity: str, conversation_context: Dict[str, Any]) -> str:
        """Expand entity query with conversation context"""
        if not conversation_context.get('has_context', False):
            return entity
        
        context_keywords = conversation_context.get('keywords', [])
        if not context_keywords:
            return entity
        
        # Add top 2-3 context keywords to entity for more precise search
        expanded_entity = entity
        for keyword in context_keywords[:3]:
            if keyword.lower() not in entity.lower():
                expanded_entity += f" {keyword}"
        
        print(f"   🔍 Expanded entity: '{entity}' → '{expanded_entity}'")
        return expanded_entity
    
    def detect_query_intent(self, query: str) -> Dict[str, Any]:
        """Detect the intent and complexity of the query"""
        query_lower = query.lower()
        
        # Detect question types
        question_patterns = {
            'definition': ['what is', 'what are', 'define', 'explain'],
            'how_to': ['how to', 'how do', 'how can'],
            'comparison': ['vs', 'versus', 'compare', 'difference between'],
            'listing': ['list', 'show me', 'give me', 'top', 'best'],
            'recommendation': ['recommend', 'suggest', 'advice', 'should']
        }
        
        detected_types = []
        for intent_type, patterns in question_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                detected_types.append(intent_type)
        
        # Detect complexity indicators
        complexity_indicators = {
            'simple': ['what is'],
            'medium': ['how to', 'explain', 'compare'],
            'complex': ['best practices', 'architecture', 'implementation', 'comprehensive']
        }
        
        complexity = 'simple'
        for level, indicators in complexity_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                complexity = level
        
        # Detect if query asks for specific count
        count_match = re.search(r'(\d+)', query)
        requested_count = int(count_match.group(1)) if count_match else None
        
        return {
            'types': detected_types or ['general'],
            'complexity': complexity,
            'requested_count': requested_count,
            'is_follow_up': any(word in query_lower for word in ['also', 'more', 'another', 'additionally', 'furthermore'])
        }
    
    def extract_domain_context(self, entity: str) -> Dict[str, Any]:
        """Extract domain-specific context from entity"""
        entity_lower = entity.lower()
        
        # Technical domains
        domains = {
            'ai_tools': ['claude', 'chatgpt', 'gemini', 'copilot', 'ai'],
            'development': ['coding', 'programming', 'development', 'software'],
            'tools': ['vs code', 'cursor', 'cli', 'terminal', 'editor'],
            'web_dev': ['react', 'vue', 'angular', 'javascript', 'frontend', 'backend'],
            'data': ['database', 'sql', 'data', 'analytics'],
            'devops': ['docker', 'kubernetes', 'deployment', 'cloud']
        }
        
        detected_domains = []
        for domain, keywords in domains.items():
            if any(keyword in entity_lower for keyword in keywords):
                detected_domains.append(domain)
        
        # Extract role context
        role_keywords = {
            'developer': ['developer', 'programmer', 'engineer'],
            'designer': ['designer', 'ui', 'ux'],
            'manager': ['manager', 'lead', 'pm'],
            'student': ['student', 'beginner', 'learning']
        }
        
        detected_roles = []
        for role, keywords in role_keywords.items():
            if any(keyword in entity_lower for keyword in keywords):
                detected_roles.append(role)
        
        return {
            'domains': detected_domains or ['general'],
            'roles': detected_roles or ['general'],
            'is_technical': len(detected_domains) > 0,
            'is_role_specific': len(detected_roles) > 0
        }