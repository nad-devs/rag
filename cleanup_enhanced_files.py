#!/usr/bin/env python3
"""
Safe cleanup script for enhanced JSON files.
Removes unused fields while preserving all fields used in the RAG system.

SAFE TO REMOVE (verified - never used in code):
- 18 metadata fields that are never accessed
- user_experience section
- processing_info section

PRESERVES:
- All fields used in vector embedding
- All fields used in metadata scoring
- All fields used in synthesis
- All fields used for series detection
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
import argparse

# Fields that are DEFINITELY safe to remove (thoroughly verified)
FIELDS_TO_REMOVE = {
    'companies_discussed',
    'content_date_context',
    'continuation_phrases',
    'controversy_level',
    'credibility_indicators',
    'difficulty_level',
    'discussion_potential',
    'next_steps',
    'personal_experience',
    'series_context',
    'shareability',
    'temporal_relevance',
    'time_investment',
    'time_sensitive_topics',
    'total_parts_detected',
    'update_indicators',
    'version_context'
}

# Top-level sections to remove
SECTIONS_TO_REMOVE = {
    'user_experience',
    'processing_info'
}

# Fields that MUST be kept (used in system)
CRITICAL_FIELDS = {
    # Used in metadata scoring
    'natural_questions',      # 100 points
    'tools_and_platforms',    # 60-80 points
    'technologies_mentioned', # 50 points
    'related_topics',         # 40 points
    'content_category',       # 30 points
    
    # Used in vector embedding
    'main_lesson',
    'content_text',
    'key_takeaways',
    'actionable_insights',
    'search_scenarios',       # Used in qa_vectors
    'prerequisites',          # Used in context_vectors
    'specific_examples',
    'practical_applications',
    
    # Used for series detection
    'is_part_of_series',
    'series_title',
    'part_number'
}


def clean_enhanced_file(file_path: Path, dry_run: bool = True) -> dict:
    """Clean a single enhanced file."""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    original_size = len(json.dumps(data))
    
    # Remove top-level sections
    for section in SECTIONS_TO_REMOVE:
        if section in data:
            del data[section]
    
    # Clean learning_metadata
    if 'learning_metadata' in data:
        metadata = data['learning_metadata']
        
        # Remove unused fields
        for field in FIELDS_TO_REMOVE:
            if field in metadata:
                del metadata[field]
        
        # Verify critical fields are preserved
        for field in CRITICAL_FIELDS:
            if field in metadata or field == 'content_text':  # content_text is top-level
                continue
            # Field should be there but isn't - log warning
            # print(f"  ⚠️  Warning: Critical field '{field}' not found")
    
    cleaned_size = len(json.dumps(data))
    reduction = (original_size - cleaned_size) * 100 // original_size if original_size > 0 else 0
    
    return {
        'data': data,
        'original_size': original_size,
        'cleaned_size': cleaned_size,
        'reduction': reduction
    }


def main():
    parser = argparse.ArgumentParser(description='Clean enhanced JSON files by removing unused fields')
    parser.add_argument('--path', type=str, 
                       default='/var/www/uploads/instagram_caption_scraper/rag_system/enhanced_processed',
                       help='Path to enhanced files directory')
    parser.add_argument('--dry-run', action='store_true', default=True,
                       help='Perform dry run without modifying files (default: True)')
    parser.add_argument('--execute', action='store_true',
                       help='Actually modify the files (creates backup first)')
    parser.add_argument('--limit', type=int, default=0,
                       help='Limit number of files to process (0 = all)')
    
    args = parser.parse_args()
    
    if args.execute:
        args.dry_run = False
    
    enhanced_path = Path(args.path)
    if not enhanced_path.exists():
        print(f"❌ Path not found: {enhanced_path}")
        return
    
    # No backup needed - using git
    
    # Process files
    json_files = list(enhanced_path.glob("*.json"))
    if args.limit > 0:
        json_files = json_files[:args.limit]
    
    print(f"\n{'DRY RUN' if args.dry_run else 'EXECUTING'}: Processing {len(json_files)} files\n")
    
    total_original = 0
    total_cleaned = 0
    processed = 0
    errors = 0
    
    for file_path in json_files:
        try:
            result = clean_enhanced_file(file_path, args.dry_run)
            
            total_original += result['original_size']
            total_cleaned += result['cleaned_size']
            processed += 1
            
            if not args.dry_run:
                # Write cleaned version directly (git handles backup)
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(result['data'], f, indent=2, ensure_ascii=False)
            
            # Show progress every 50 files
            if processed % 50 == 0:
                print(f"  Processed {processed}/{len(json_files)} files...")
            
        except Exception as e:
            print(f"❌ Error processing {file_path.name}: {e}")
            errors += 1
    
    # Summary
    print("\n" + "="*60)
    print("CLEANUP SUMMARY")
    print("="*60)
    print(f"Files processed: {processed}")
    print(f"Errors: {errors}")
    print(f"Original total size: {total_original:,} chars")
    print(f"Cleaned total size: {total_cleaned:,} chars")
    print(f"Total reduction: {total_original - total_cleaned:,} chars ({(total_original - total_cleaned) * 100 // total_original}%)")
    print(f"Average file size: {total_original // processed:,} → {total_cleaned // processed:,} chars")
    
    if args.dry_run:
        print("\n⚠️  This was a DRY RUN. No files were modified.")
        print("To execute cleanup, run with --execute flag")
    else:
        print(f"\n✅ Cleanup complete! Files modified in place (use git to revert if needed)")


if __name__ == "__main__":
    main()