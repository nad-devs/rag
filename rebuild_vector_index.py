#!/usr/bin/env python3
"""
Rebuild Vector Index - Index all enhanced documents into Qdrant
Uses the vector_manager's batch processing capability
"""

import sys
from pathlib import Path

# Add paths for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir / 'rag_system'))

try:
    from core.vector_manager import QdrantVectorManager
    
    print("🚀 REBUILDING VECTOR INDEX")
    print("=" * 60)
    
    # Create vector manager
    print("🔧 Creating vector manager...")
    vector_manager = QdrantVectorManager(qdrant_url="http://localhost:6333")
    
    # Ensure collections exist first
    print("\n🔧 Ensuring collections exist...")
    vector_manager.setup_collections()
    
    # Check collections status
    print("\n📊 Checking collections status...")
    stats = vector_manager.get_collection_stats()
    for vector_type, stat in stats.items():
        if 'error' in stat:
            print(f"❌ {vector_type}: {stat['error']}")
        else:
            print(f"✅ {vector_type}: {stat['points_count']} points")
    
    # Process all enhanced files
    print("\n🔍 Processing enhanced files...")
    enhanced_processed_dir = str(current_dir / 'rag_system' / 'enhanced_processed')
    
    results = vector_manager.process_new_enhanced_files(enhanced_processed_dir)
    
    # Check final status
    print("\n📊 Final collection status:")
    stats = vector_manager.get_collection_stats()
    for vector_type, stat in stats.items():
        if 'error' in stat:
            print(f"❌ {vector_type}: {stat['error']}")
        else:
            print(f"✅ {vector_type}: {stat['points_count']} points")
    
    print("\n🎯 Vector index rebuild complete!")
    
except Exception as e:
    print(f"💥 Critical error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)