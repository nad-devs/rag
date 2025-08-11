#!/usr/bin/env python3
"""
Debug Claude ranking to understand why expected documents score low
"""

import sys
sys.path.append('..')
sys.path.append('../rag_system')  
from claude_rag_engine import ClaudeRAGEngine

def debug_claude_ranking_for_query(query, expected_doc, description):
    """Debug Claude ranking for a specific query"""
    print(f"\n{'='*80}")
    print(f"🔍 DEBUGGING CLAUDE RANKING")
    print(f"Query: '{query}'")
    print(f"Expected: {expected_doc}")
    print(f"Why: {description}")
    print(f"{'='*80}")
    
    try:
        engine = ClaudeRAGEngine('../rag_system/enhanced_processed/')
        engine.conversation_tracker.clear_session()
        
        # Step 1: Get vector search results
        print("\n📊 STEP 1: Vector Search Results")
        print("-" * 40)
        
        vector_results = engine._find_entity_content_with_context(
            query, 
            {'has_context': False}, 
            set(), 
            False
        )
        
        expected_in_vector = False
        expected_vector_position = None
        
        print(f"Found {len(vector_results)} vector results:")
        for i, result in enumerate(vector_results[:20]):
            doc_id = result.get('doc_id', 'unknown')
            score = result.get('score', 0)
            if doc_id == expected_doc:
                expected_in_vector = True
                expected_vector_position = i + 1
                print(f"   {i+1:2d}. {doc_id} (score: {score:.3f}) ⭐ EXPECTED!")
            else:
                print(f"   {i+1:2d}. {doc_id} (score: {score:.3f})")
        
        if not expected_in_vector:
            print(f"❌ Expected doc {expected_doc} NOT found in vector results!")
            return
        
        print(f"✅ Expected doc found at vector position {expected_vector_position}")
        
        # Step 2: Test Claude ranking directly
        print(f"\n🤖 STEP 2: Claude Ranking Test")
        print("-" * 40)
        
        # Get top 20 for Claude ranking
        top_20_for_ranking = vector_results[:20]
        
        # Ensure our expected document is included
        if not any(r.get('doc_id') == expected_doc for r in top_20_for_ranking):
            # Find and add it
            for result in vector_results:
                if result.get('doc_id') == expected_doc:
                    top_20_for_ranking.append(result)
                    break
        
        print(f"Sending {len(top_20_for_ranking)} documents to Claude for ranking...")
        
        # Use the internal Claude ranking method
        ranked_results = engine._claude_rank_documents(query, top_20_for_ranking)
        
        # Check Claude scores
        if hasattr(engine, '_last_claude_scores') and engine._last_claude_scores:
            print(f"\n📊 Claude Scores:")
            expected_claude_position = None
            expected_claude_score = None
            
            for i, (result, score) in enumerate(zip(ranked_results, engine._last_claude_scores)):
                doc_id = result.get('doc_id', 'unknown')
                if doc_id == expected_doc:
                    expected_claude_position = i + 1
                    expected_claude_score = score
                    print(f"   {i+1:2d}. {doc_id} (score: {score}) ⭐ EXPECTED!")
                else:
                    print(f"   {i+1:2d}. {doc_id} (score: {score})")
            
            if expected_claude_score is not None:
                print(f"\n📈 Expected document Claude analysis:")
                print(f"   Vector position: {expected_vector_position}")
                print(f"   Claude position: {expected_claude_position}")
                print(f"   Claude score: {expected_claude_score}")
                print(f"   Threshold (6.5): {'✅ PASS' if expected_claude_score >= 6.5 else '❌ FAIL'}")
                
                if expected_claude_score < 6.5:
                    print(f"\n⚠️ ISSUE: Claude scored {expected_doc} only {expected_claude_score}/10")
                    print(f"   This is below the 6.5 threshold, so it gets filtered out")
                    
                    # Show content preview
                    if expected_doc in engine.enhanced_content_index:
                        content = engine.enhanced_content_index[expected_doc]
                        content_text = content.get('content_text', '')[:200]
                        metadata = content.get('full_metadata', {})
                        main_lesson = metadata.get('main_lesson', 'N/A')
                        
                        print(f"\n📄 Expected document content preview:")
                        print(f"   Main lesson: {main_lesson}")
                        print(f"   Content: {content_text}...")
                        
                        # Check for direct keyword matches
                        query_words = query.lower().split()
                        content_lower = content_text.lower()
                        matches = [word for word in query_words if word in content_lower]
                        print(f"   Direct keyword matches: {matches}")
            else:
                print(f"❌ Expected document not found in Claude ranking results!")
        else:
            print(f"❌ No Claude scores available!")
        
        # Step 3: Show what actually gets selected
        print(f"\n🎯 STEP 3: Final Selection")
        print("-" * 40)
        
        # Apply threshold filtering
        final_selection = []
        if hasattr(engine, '_last_claude_scores') and engine._last_claude_scores:
            for result, score in zip(ranked_results, engine._last_claude_scores):
                if score >= 6.5:
                    final_selection.append((result.get('doc_id'), score))
        
        print(f"Documents passing 6.5 threshold:")
        for i, (doc_id, score) in enumerate(final_selection):
            indicator = " ⭐ EXPECTED!" if doc_id == expected_doc else ""
            print(f"   {i+1}. {doc_id} (score: {score}){indicator}")
        
        if not any(doc_id == expected_doc for doc_id, _ in final_selection):
            print(f"❌ Expected document {expected_doc} did not make final selection!")
        
    except Exception as e:
        print(f"❌ Error during debugging: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("🔍 CLAUDE RANKING DEBUG TOOL")
    print("="*80)
    print("This tool analyzes why Claude ranks documents lower than expected")
    print()
    
    test_cases = [
        {
            'query': 'AI coding tips for backend developers',
            'expected': 'edhonour_DLaBB2Ruatz',
            'description': 'Should match backend AI tips document'
        },
        {
            'query': 'What is the most important vibe coding tip?',
            'expected': 'edhonour_DMD1IAqSFh3',
            'description': 'Should match activity log vibe coding tip'
        }
    ]
    
    for test in test_cases:
        debug_claude_ranking_for_query(
            test['query'],
            test['expected'],
            test['description']
        )
    
    print(f"\n{'='*80}")
    print("🔍 DEBUG COMPLETE")
    print("="*80)
    print("Check the Claude scores above to understand why documents are filtered out.")
    print("If Claude scores are too low, we may need to:")
    print("1. Lower the threshold further (currently 6.5)")
    print("2. Improve the Claude ranking prompt")
    print("3. Ensure document content includes the right keywords")

if __name__ == "__main__":
    main()