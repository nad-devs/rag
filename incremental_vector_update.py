#!/usr/bin/env python3
"""
Incremental Vector Update - Only process enhanced documents not yet in Qdrant
Uses Qdrant's built-in duplicate detection for efficiency
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# Add paths for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir / 'rag_system'))

try:
    from core.vector_manager import QdrantVectorManager
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

class IncrementalVectorUpdater:
    """Manages incremental updates to vector database using Qdrant's duplicate detection"""
    
    def __init__(self, qdrant_url: str = "http://localhost:6333"):
        self.qdrant_url = qdrant_url
        self.vector_manager = QdrantVectorManager(qdrant_url=qdrant_url)
        
    def find_all_enhanced_files(self, enhanced_dir: Path) -> List[Path]:
        """Find all enhanced JSON files"""
        enhanced_files = []
        
        # Find all enhanced JSON files
        for file_path in enhanced_dir.rglob("enhanced_*.json"):
            enhanced_files.append(file_path)
        
        print(f"📂 Found {len(enhanced_files)} enhanced files total")
        return enhanced_files
    
    def process_files_incrementally(self, files_to_process: List[Path]) -> Dict:
        """Process files using Qdrant's built-in duplicate detection"""
        if not files_to_process:
            return {
                'total_processed': 0,
                'total_added': 0,
                'total_skipped': 0,
                'failed': [],
                'processing_time': 0
            }
        
        print(f"🔄 Processing {len(files_to_process)} files (Qdrant will skip duplicates)...")
        
        start_time = datetime.now()
        
        # Process files using existing vector manager method
        # Qdrant's duplicate detection will handle skipping existing documents
        results = self.vector_manager.process_specific_files(files_to_process)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        results['processing_time'] = processing_time
        return results
    
    def run_incremental_update(self, enhanced_dir: str = None) -> Dict:
        """Main method to run incremental vector update"""
        if enhanced_dir is None:
            enhanced_dir = current_dir / 'rag_system' / 'enhanced_processed'
        else:
            enhanced_dir = Path(enhanced_dir)
        
        print("🚀 INCREMENTAL VECTOR UPDATE")
        print("=" * 60)
        
        # Ensure collections exist
        print("🔧 Ensuring collections exist...")
        self.vector_manager.setup_collections()
        
        # Find all enhanced files (Qdrant will handle duplicate detection)
        print("\n🔍 Scanning for enhanced files...")
        files_to_process = self.find_all_enhanced_files(enhanced_dir)
        
        if not files_to_process:
            print("✨ No enhanced files found!")
            return {
                'total_processed': 0,
                'total_added': 0,
                'total_skipped': 0,
                'failed': [],
                'processing_time': 0,
                'message': 'No files to process'
            }
        
        print(f"📋 Found {len(files_to_process)} files to process")
        
        # Process the files
        results = self.process_files_incrementally(files_to_process)
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 INCREMENTAL UPDATE SUMMARY")
        print("=" * 60)
        print(f"📁 Files processed: {len(files_to_process)}")
        print(f"✅ Successfully added: {results.get('total_added', 0)}")
        print(f"⏭️ Skipped: {results.get('total_skipped', 0)}")
        
        failed_list = results.get('failed', [])
        failed_count = len(failed_list) if isinstance(failed_list, list) else failed_list
        print(f"❌ Failed: {failed_count}")
        print(f"⏱️ Processing time: {results.get('processing_time', 0):.1f}s")
        print("=" * 60)
        
        # Check final collection status
        print("\n📊 Final collection status:")
        stats = self.vector_manager.get_collection_stats()
        for vector_type, stat in stats.items():
            if 'error' in stat:
                print(f"❌ {vector_type}: {stat['error']}")
            else:
                print(f"✅ {vector_type}: {stat['points_count']} points")
        
        print(f"\n🎯 Incremental update complete!")
        
        return results

def main():
    """Main function for CLI usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Incremental vector database update')
    parser.add_argument('--enhanced-dir', help='Path to enhanced files directory')
    parser.add_argument('--qdrant-url', default='http://localhost:6333', help='Qdrant URL')
    parser.add_argument('--force-full', action='store_true', help='Force full rebuild instead of incremental')
    
    args = parser.parse_args()
    
    if args.force_full:
        print("🔄 Force full rebuild requested - running rebuild_vector_index.py instead")
        import subprocess
        result = subprocess.run([sys.executable, 'rebuild_vector_index.py'], 
                              cwd=current_dir)
        return result.returncode
    
    try:
        updater = IncrementalVectorUpdater(qdrant_url=args.qdrant_url)
        results = updater.run_incremental_update(args.enhanced_dir)
        
        # Return appropriate exit code
        failed_count = results.get('failed', [])
        if isinstance(failed_count, list):
            failed_count = len(failed_count)
        
        if failed_count > 0:
            return 1
        else:
            return 0
            
    except Exception as e:
        print(f"💥 Critical error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())