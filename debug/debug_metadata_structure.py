#!/usr/bin/env python3
"""
Debug metadata structure to fix series detection
"""

import sys
sys.path.append('..')
sys.path.append('../rag_system')  
from claude_rag_engine import ClaudeRAGEngine

def debug_metadata_structure():
    print("🔍 DEBUGGING METADATA STRUCTURE")
    print("=" * 50)
    
    engine = ClaudeRAGEngine('../rag_system/enhanced_processed/')
    
    # Check a known series document
    doc_id = 'edhonour_DMFoZKKyRDm'
    print(f'\n=== ENHANCED CONTENT INDEX for {doc_id} ===')
    
    if doc_id in engine.enhanced_content_index:
        content = engine.enhanced_content_index[doc_id]
        print(f'Keys in enhanced_content_index[{doc_id}]:')
        for key in content.keys():
            print(f'  - {key}: {type(content[key])}')
        
        # Check full_metadata specifically
        if 'full_metadata' in content:
            metadata = content['full_metadata']
            print(f'\nKeys in full_metadata:')
            for key in sorted(metadata.keys()):
                value = metadata[key]
                if isinstance(value, (list, dict)):
                    print(f'  - {key}: {type(value)} (len={len(value)})')
                else:
                    print(f'  - {key}: {value}')
                
            print(f'\nSeries fields:')
            print(f'  is_part_of_series: {metadata.get("is_part_of_series")}')
            print(f'  series_title: {metadata.get("series_title")}')
            print(f'  part_number: {metadata.get("part_number")}')
        else:
            print('  ❌ No full_metadata found')
    else:
        print(f'❌ Document {doc_id} not found in enhanced_content_index')
    
    print(f'\n=== SAMPLE MENTION STRUCTURE FROM VECTOR SEARCH ===')
    # Get a sample mention to see structure
    try:
        result = engine._find_entity_content_with_context('coding tips', {'has_context': False}, set(), False)
        if result:
            sample_mention = result[0]
            print('Keys in vector search mention:')
            for key in sorted(sample_mention.keys()):
                value = sample_mention[key]
                if isinstance(value, (list, dict)):
                    print(f'  - {key}: {type(value)} (len={len(value)})')
                else:
                    print(f'  - {key}: {type(value)}')
                    
            # Check if it has enhanced_data
            if 'enhanced_data' in sample_mention:
                enhanced = sample_mention['enhanced_data']
                print(f'\nKeys in mention.enhanced_data:')
                for key in sorted(enhanced.keys()):
                    value = enhanced[key]
                    if isinstance(value, (list, dict)):
                        print(f'  - {key}: {type(value)} (len={len(value)})')
                    else:
                        print(f'  - {key}: {type(value)}')
        else:
            print('❌ No mentions returned from vector search')
    except Exception as e:
        print(f'❌ Error getting vector search results: {e}')

if __name__ == "__main__":
    debug_metadata_structure()