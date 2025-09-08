"""
Qdrant Vector Database Manager for Instagram Learning Content
Implements 4-vector strategy: Content, Q&A, Tech, Context
"""

import os
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import qdrant_client
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, Range, MatchValue
from sentence_transformers import SentenceTransformer
import uuid
from datetime import datetime

@dataclass
class VectorDocument:
    """Structured document for vector storage"""
    document_id: str
    content_vector: List[float]
    qa_vector: List[float] 
    tech_vector: List[float]
    context_vector: List[float]
    metadata: Dict[str, Any]
    source_url: str
    shortcode: str

class QdrantVectorManager:
    """Manages Qdrant vector database with 4-vector strategy"""
    
    def __init__(self, 
                 qdrant_url: str = None,
                 qdrant_api_key: str = None,
                 embedding_model: str = "intfloat/e5-large-v2"):
        """Initialize Qdrant manager with E5-Large-V2 embeddings"""
        
        # Qdrant connection
        self.qdrant_url = qdrant_url or os.getenv('QDRANT_URL', ':memory:')  # Local for testing
        self.qdrant_api_key = qdrant_api_key or os.getenv('QDRANT_API_KEY')
        
        # Initialize client
        if self.qdrant_url == ':memory:':
            print("🔧 Using local in-memory Qdrant for testing")
            self.client = QdrantClient(":memory:")
        else:
            print(f"🔗 Connecting to Qdrant Cloud: {self.qdrant_url}")
            self.client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key
            )
        
        # Initialize embedding model (force CPU to avoid CUDA memory issues)
        print(f"🤖 Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model, device='cpu')
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        print("💡 Using CPU for embeddings to avoid CUDA memory issues")
        
        # Collection names for 4-vector strategy
        self.collections = {
            'content': 'instagram_content_vectors',
            'qa': 'instagram_qa_vectors', 
            'tech': 'instagram_tech_vectors',
            'context': 'instagram_context_vectors'
        }
        
        print(f"✅ Qdrant Vector Manager initialized")
        print(f"📊 Embedding dimension: {self.embedding_dim}")
    
    def create_collections(self):
        """Alias for setup_collections for backward compatibility"""
        return self.setup_collections()
    
    def setup_collections(self):
        """Create Qdrant collections for 4-vector strategy"""
        
        print("🔧 Setting up Qdrant collections...")
        
        vector_config = VectorParams(
            size=self.embedding_dim,
            distance=Distance.COSINE
        )
        
        for vector_type, collection_name in self.collections.items():
            try:
                # Delete if exists (for testing)
                try:
                    self.client.delete_collection(collection_name)
                    print(f"🗑️ Deleted existing {vector_type} collection")
                except:
                    pass
                
                # Create collection
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=vector_config
                )
                print(f"✅ Created {vector_type} collection: {collection_name}")
                
            except Exception as e:
                print(f"❌ Error creating {vector_type} collection: {e}")
        
        print("🎉 All collections created successfully!")
    
    def _create_vector_content(self, metadata: Dict[str, Any], content_text: str) -> Dict[str, str]:
        """Create optimized text for each vector type - FIXED STRATEGY"""
        
        # PRIMARY VECTOR: Core learning content (main lesson + content + key takeaways)
        # This should contain the bulk of searchable content to avoid fragmentation
        primary_parts = [
            metadata.get('main_lesson', ''),
            content_text,
            ' '.join(metadata.get('key_takeaways', [])),
            ' '.join(metadata.get('actionable_insights', []))  # Move insights to primary
        ]
        if metadata.get('is_part_of_series'):
            primary_parts.append(f"Part {metadata.get('part_number')} of {metadata.get('series_title')}")
        primary_vector_text = ' '.join(filter(None, primary_parts))
        
        # ENTITY VECTOR: All structured entities (technologies + tools + companies)
        # Focus on entity names and their contexts for entity-specific queries
        entity_parts = []
        for tech in metadata.get('technologies_mentioned', []):
            tech_name = tech.get('name', '')
            tech_context = tech.get('context', '')
            if tech_name:
                entity_parts.append(f"{tech_name}: {tech_context}")
        
        for tool in metadata.get('tools_and_platforms', []):
            tool_name = tool.get('name', '')
            tool_use = tool.get('use_case', '')
            if tool_name:
                entity_parts.append(f"{tool_name}: {tool_use}")
        
        for company in metadata.get('companies_discussed', []):
            # Handle both old format (dict) and new format (string)
            if isinstance(company, dict):
                company_name = company.get('name', '')
                company_context = company.get('context', '')
                if company_name:
                    entity_parts.append(f"{company_name}: {company_context}")
            elif isinstance(company, str):
                # New Claude format - companies as strings
                if company:
                    entity_parts.append(company)
        
        entity_vector_text = ' '.join(filter(None, entity_parts))
        
        # Q&A VECTOR: Question-answer patterns for FAQ-style queries
        qa_parts = [
            ' '.join(metadata.get('natural_questions', [])),
            ' '.join(metadata.get('search_scenarios', [])),
            f"Topic: {metadata.get('main_lesson', '')}"  # Add topic context
        ]
        qa_vector_text = ' '.join(filter(None, qa_parts))
        
        # CONTEXT VECTOR: Practical application information
        context_parts = [
            ' '.join(metadata.get('practical_applications', [])),
            ' '.join(metadata.get('specific_examples', [])),
            ' '.join(metadata.get('prerequisites', [])),
            ' '.join(metadata.get('next_steps', [])),
            ' '.join(metadata.get('related_topics', []))  # Add related topics
        ]
        context_vector_text = ' '.join(filter(None, context_parts))
        
        return {
            'content': primary_vector_text,      # Renamed from content to primary
            'tech': entity_vector_text,          # Renamed from tech to entity  
            'qa': qa_vector_text,               # Keep as qa
            'context': context_vector_text       # Keep as context
        }
    
    def _prepare_metadata_for_storage(self, enhanced_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare metadata for Qdrant storage with proper filtering"""
        
        metadata = enhanced_data['learning_metadata']
        source = enhanced_data['source_metadata']
        
        return {
            # Core identifiers
            'document_id': enhanced_data['document_id'],
            'shortcode': source['shortcode'],
            'profile': source['profile'],
            'url': source['url'],
            
            # Learning metadata
            'main_lesson': metadata.get('main_lesson', ''),
            'content_category': metadata.get('content_category', 'unknown'),
            
            # Series information
            'is_part_of_series': metadata.get('is_part_of_series', False),
            'series_title': metadata.get('series_title', ''),
            'part_number': metadata.get('part_number', 0),
            
            # Source metadata
            'duration': source.get('duration', 0),
            'position_in_profile': source['position_in_profile'],
            'extraction_timestamp': source['extraction_timestamp']
        }
    
    def index_document(self, enhanced_data: Dict[str, Any]) -> bool:
        """Index a single enhanced document across all 4 vectors"""
        
        try:
            document_id = enhanced_data['document_id']
            content_text = enhanced_data['content_text']
            learning_metadata = enhanced_data['learning_metadata']  # Get nested learning_metadata
            
            print(f"📄 Indexing document: {document_id}")
            
            # Create vector content for each type using the correct nested structure
            vector_texts = self._create_vector_content(learning_metadata, content_text)
            
            # Generate embeddings for each vector type
            embeddings = {}
            for vector_type, text in vector_texts.items():
                if text.strip():  # Only embed if there's content
                    embeddings[vector_type] = self.embedding_model.encode(text).tolist()
                else:
                    # Use a zero vector if no content
                    embeddings[vector_type] = [0.0] * self.embedding_dim
            
            # Prepare metadata
            storage_metadata = self._prepare_metadata_for_storage(enhanced_data)
            
            # Add content text and full learning metadata for retrieval using correct nested structure
            storage_metadata.update({
                'content_text': content_text,
                'key_takeaways': learning_metadata.get('key_takeaways', []),
                'actionable_insights': learning_metadata.get('actionable_insights', []),
                'technologies_mentioned': learning_metadata.get('technologies_mentioned', []),
                'practical_applications': learning_metadata.get('practical_applications', []),
                'specific_examples': learning_metadata.get('specific_examples', [])
            })
            
            # Create point ID
            point_id = str(uuid.uuid4())
            
            # Insert into each collection
            for vector_type, embedding in embeddings.items():
                collection_name = self.collections[vector_type]
                
                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=storage_metadata
                )
                
                self.client.upsert(
                    collection_name=collection_name,
                    points=[point]
                )
            
            print(f"✅ Indexed {document_id} across all 4 vectors")
            return True
            
        except Exception as e:
            print(f"❌ Error indexing document {enhanced_data.get('document_id', 'unknown')}: {e}")
            return False
    
    def search_by_vector_type(self, 
                             query: str, 
                             vector_type: str = 'content',
                             limit: int = 5,
                             filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search using specific vector type with optional filters"""
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Build filters
            filter_conditions = []
            if filters:
                for key, value in filters.items():
                    if isinstance(value, str):
                        filter_conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
                    elif isinstance(value, list):
                        for v in value:
                            filter_conditions.append(FieldCondition(key=key, match=MatchValue(value=v)))
            
            query_filter = Filter(must=filter_conditions) if filter_conditions else None
            
            # Search
            collection_name = self.collections[vector_type]
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                query_filter=query_filter,
                limit=limit,
                with_payload=True
            )
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'score': result.score,
                    'document_id': result.payload['document_id'],
                    'shortcode': result.payload['shortcode'],
                    'url': result.payload['url'],
                    'main_lesson': result.payload['main_lesson'],
                    'content_category': result.payload['content_category'],
                    'series_info': {
                        'is_series': result.payload.get('is_part_of_series', False),
                        'series_title': result.payload.get('series_title', ''),
                        'part_number': result.payload.get('part_number', 0)
                    },
                    'metadata': result.payload
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []
    
    def search(self, collection_name: str, query_text: str, limit: int = 5, filters: Dict = None) -> List[Dict[str, Any]]:
        """Main search method - required by debug tools"""
        return self.search_vector(collection_name, query_text, limit, filters)
    
    def search_vector(self, collection_name: str, query_text: str, limit: int = 5, filters: Dict = None) -> List[Dict[str, Any]]:
        """Search vector by collection name (for hybrid retriever compatibility)"""
        
        try:
            # Extract vector type from collection name
            vector_type = None
            for vtype, cname in self.collections.items():
                if cname == collection_name:
                    vector_type = vtype
                    break
            
            if not vector_type:
                print(f"❌ Unknown collection: {collection_name}")
                return []
            
            # Use existing search method
            results = self.search_by_vector_type(query_text, vector_type, limit, filters)
            
            # Format for hybrid retriever compatibility
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'score': result['score'],
                    'payload': {
                        'document_id': result['document_id'],
                        'learning_metadata': {
                            'main_lesson': result['main_lesson'],
                            'content_category': result['content_category'],
                            'key_takeaways': result['metadata'].get('key_takeaways', []),
                            'actionable_insights': result['metadata'].get('actionable_insights', []),
                            'technologies_mentioned': result['metadata'].get('technologies_mentioned', []),
                            'practical_applications': result['metadata'].get('practical_applications', []),
                            'specific_examples': result['metadata'].get('specific_examples', []),
                            'is_part_of_series': result['metadata'].get('is_part_of_series', False),
                            'series_title': result['metadata'].get('series_title', ''),
                            'part_number': result['metadata'].get('part_number', 0)
                        },
                        'content_text': result['metadata'].get('content_text', ''),
                        'source_metadata': {
                            'url': result['url'],
                            'shortcode': result['shortcode'],
                            'profile': result['metadata'].get('profile', ''),
                            'duration': result['metadata'].get('duration', 0)
                        }
                    }
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ Search vector error: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics for all collections"""
        
        stats = {}
        for vector_type, collection_name in self.collections.items():
            try:
                info = self.client.get_collection(collection_name)
                stats[vector_type] = {
                    'name': collection_name,
                    'points_count': info.points_count,
                    'vectors_count': info.vectors_count
                }
            except Exception as e:
                stats[vector_type] = {'error': str(e)}
        
        return stats
    
    def document_exists_in_qdrant(self, document_id: str) -> bool:
        """Check if document already exists in any collection (lightweight duplicate check)"""
        
        try:
            # Check primary content collection first (most reliable)
            collection_name = self.collections['content']
            
            # Search for exact document_id match
            results = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
                ),
                limit=1,
                with_payload=False  # Just checking existence
            )
            
            return len(results[0]) > 0  # results is tuple (points, next_page_offset)
            
        except Exception as e:
            print(f"⚠️ Error checking document existence for {document_id}: {e}")
            return False  # Assume doesn't exist if error, let index_document handle any issues
    
    def process_enhanced_file(self, enhanced_file_path: str) -> Dict[str, Any]:
        """Process a single enhanced file for vectorization"""
        
        try:
            # Load enhanced file
            with open(enhanced_file_path, 'r', encoding='utf-8') as f:
                enhanced_data = json.load(f)
            
            document_id = enhanced_data.get('document_id', 'unknown')
            
            # Check for duplicates
            if self.document_exists_in_qdrant(document_id):
                return {
                    'success': True,
                    'skipped': True,
                    'reason': 'Document already exists in Qdrant',
                    'document_id': document_id,
                    'file_path': enhanced_file_path
                }
            
            # Index the document using existing method
            success = self.index_document(enhanced_data)
            
            if success:
                return {
                    'success': True,
                    'skipped': False,
                    'document_id': document_id,
                    'file_path': enhanced_file_path
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to index document',
                    'document_id': document_id,
                    'file_path': enhanced_file_path
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'file_path': enhanced_file_path
            }
    
    def batch_process_enhanced_files(self, enhanced_file_paths: List[str]) -> Dict[str, Any]:
        """Process multiple enhanced files for vectorization"""
        
        results = {
            'successful': [],
            'skipped': [],
            'failed': [],
            'total_processed': 0,
            'total_added': 0,
            'total_skipped': 0,
            'start_time': datetime.now()
        }
        
        print(f"🚀 Starting vectorization of {len(enhanced_file_paths)} enhanced files...")
        
        for i, file_path in enumerate(enhanced_file_paths, 1):
            print(f"📄 Processing {i}/{len(enhanced_file_paths)}: {os.path.basename(file_path)}")
            
            result = self.process_enhanced_file(file_path)
            
            if result['success']:
                if result.get('skipped'):
                    results['skipped'].append(result)  
                    results['total_skipped'] += 1
                    print(f"⏭️ Skipped: {result['document_id']} (already exists)")
                else:
                    results['successful'].append(result)
                    results['total_added'] += 1
                    print(f"✅ Added: {result['document_id']}")
            else:
                results['failed'].append(result)
                print(f"❌ Failed: {result.get('document_id', 'unknown')} - {result.get('error', 'unknown error')}")
            
            results['total_processed'] += 1
            
            # Memory management: cleanup every 10 documents to prevent CUDA OOM
            if i % 10 == 0:
                print(f"💾 Processed {i} documents, clearing GPU cache...")
                import gc
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                import time
                time.sleep(0.5)  # Brief pause for memory cleanup
        
        results['end_time'] = datetime.now()
        results['duration'] = (results['end_time'] - results['start_time']).total_seconds()
        
        # Print summary
        self._print_vectorization_summary(results)
        
        return results
    
    def process_new_enhanced_files(self, enhanced_processed_dir: str = None) -> Dict[str, Any]:
        """Main method: Find and process all enhanced files for vectorization"""
        
        if enhanced_processed_dir is None:
            # Default to the standard location
            enhanced_processed_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 
                'enhanced_processed'
            )
        
        print(f"🔍 Scanning enhanced files in: {enhanced_processed_dir}")
        
        # Find all enhanced JSON files
        enhanced_files = []
        if os.path.exists(enhanced_processed_dir):
            for root, dirs, files in os.walk(enhanced_processed_dir):
                for file in files:
                    if file.startswith('enhanced_') and file.endswith('.json'):
                        enhanced_files.append(os.path.join(root, file))
        
        if not enhanced_files:
            print("ℹ️ No enhanced files found to process.")
            return {
                'successful': [],
                'skipped': [],
                'failed': [],
                'total_processed': 0,
                'total_added': 0,
                'total_skipped': 0
            }
        
        print(f"📋 Found {len(enhanced_files)} enhanced files")
        
        # Process all files
        return self.batch_process_enhanced_files(enhanced_files)
    
    def process_specific_files(self, file_paths: List) -> Dict[str, Any]:
        """Process only the specified list of enhanced files"""
        
        # Convert Path objects to strings if needed
        file_paths_str = [str(path) for path in file_paths]
        
        print(f"📋 Processing {len(file_paths_str)} specific files")
        
        # Process the specific files
        return self.batch_process_enhanced_files(file_paths_str)
    
    def _print_vectorization_summary(self, results: Dict):
        """Print vectorization batch processing summary"""
        print("\n" + "="*60)
        print("📊 VECTORIZATION SUMMARY")
        print("="*60)
        print(f"📁 Total processed: {results['total_processed']}")
        print(f"✅ Successfully added: {results['total_added']}")
        print(f"⏭️ Skipped (duplicates): {results['total_skipped']}")
        print(f"❌ Failed: {len(results['failed'])}")
        print(f"⏱️ Processing time: {results.get('duration', 0):.1f}s")
        
        if results['failed']:
            print(f"\n❌ Failed files:")
            for failure in results['failed']:
                print(f"  • {os.path.basename(failure['file_path'])}: {failure.get('error', 'unknown error')}")
        
        if results['total_added'] > 0:
            print(f"\n🎯 {results['total_added']} new documents added to vector database!")
        
        print("="*60)