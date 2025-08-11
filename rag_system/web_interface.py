#!/usr/bin/env python3
"""
Simple Web Interface for Instagram CEO Content Search
Shows answers + Instagram reel links for easy browsing
"""

from flask import Flask, render_template, request, jsonify, session
import os
import sys
import re
from dotenv import load_dotenv
import uuid
from datetime import datetime

# Setup
load_dotenv()
# Ensure Qdrant URL is set for Docker instance
os.environ['QDRANT_URL'] = 'http://localhost:6333'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your-secret-key-for-sessions-change-in-production')

# Global engine instance
engine = None

def format_answer_for_web(answer_text, instagram_links=None):
    """Format plain text answer with natural, clickable content (no ugly citations)"""
    if not answer_text:
        return ""
    
    # Create a mapping of document IDs to Instagram URLs for invisible clickability
    doc_to_info = {}
    if instagram_links:
        for link in instagram_links:
            if 'doc_id' in link and 'url' in link:
                doc_to_info[link['doc_id']] = {
                    'url': link['url'],
                    'title': link.get('title', 'Ed\'s Instagram Content'),
                    'shortcode': link.get('doc_id', '').replace('edhonour_', '')
                }
    
    # Clean up any remaining JSON artifacts
    if answer_text.strip().startswith('{') and '"answer"' in answer_text:
        json_match = re.search(r'"answer"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', answer_text, re.DOTALL)
        if json_match:
            answer_text = json_match.group(1)
            answer_text = answer_text.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
        else:
            answer_text = re.sub(r'[{}",]', '', answer_text)
            answer_text = re.sub(r'"[^"]*":', '', answer_text)
    
    # Basic HTML formatting
    lines = answer_text.split('\n')
    result_lines = []
    in_ordered_list = False
    in_unordered_list = False
    
    for line in lines:
        line = line.strip()
        if not line:
            if in_ordered_list:
                result_lines.append('</ol>')
                in_ordered_list = False
            if in_unordered_list:
                result_lines.append('</ul>')
                in_unordered_list = False
            result_lines.append('<br>')
            continue
        
        # Handle numbered lists
        if re.match(r'^\d+\.', line):
            if not in_ordered_list:
                if in_unordered_list:
                    result_lines.append('</ul>')
                    in_unordered_list = False
                result_lines.append('<ol>')
                in_ordered_list = True
            result_lines.append(f'<li>{line[line.find(".")+1:].strip()}</li>')
        # Handle bullet points
        elif line.startswith('- ') or line.startswith('• '):
            if not in_unordered_list:
                if in_ordered_list:
                    result_lines.append('</ol>')
                    in_ordered_list = False
                result_lines.append('<ul>')
                in_unordered_list = True
            result_lines.append(f'<li>{line[2:].strip()}</li>')
        # Handle headers
        elif line.endswith(':') and len(line) > 3 and not line.startswith('http'):
            if in_ordered_list:
                result_lines.append('</ol>')
                in_ordered_list = False
            if in_unordered_list:
                result_lines.append('</ul>')
                in_unordered_list = False
            result_lines.append(f'<h4 style="color: #4a5568; margin: 15px 0 8px 0; font-weight: 600;">{line}</h4>')
        else:
            if in_ordered_list:
                result_lines.append('</ol>')
                in_ordered_list = False
            if in_unordered_list:
                result_lines.append('</ul>')
                in_unordered_list = False
            result_lines.append(f'<p style="margin: 8px 0;">{line}</p>')
    
    if in_unordered_list:
        result_lines.append('</ul>')
    if in_ordered_list:
        result_lines.append('</ol>')
    
    final_html = '\n'.join(result_lines)
    
    # NATURAL CLICKABLE CONTENT: Remove ugly citations and make content itself clickable
    if doc_to_info:
        def extract_and_clean_citations(text):
            """Extract content with citations and create clean, clickable content"""
            import re
            
            # Pattern to find content followed by citation: "text [doc_id]"
            pattern = r'([^.!?\[\n]*?)(\[([^\]]+)\])'
            
            segments = []
            last_end = 0
            
            for match in re.finditer(pattern, text):
                # Add any text before this match
                if match.start() > last_end:
                    uncited_text = text[last_end:match.start()].strip()
                    if uncited_text:
                        segments.append({'text': uncited_text, 'clickable': False})
                
                # Process the cited content
                content = match.group(1).strip()
                doc_id = match.group(3)
                
                if content and doc_id in doc_to_info:
                    # Make this content clickable
                    segments.append({
                        'text': content,
                        'clickable': True,
                        'doc_id': doc_id,
                        'url': doc_to_info[doc_id]['url'],
                        'title': f"From Ed's Instagram: {doc_to_info[doc_id]['shortcode']}"
                    })
                elif content:
                    # Content without valid citation
                    segments.append({'text': content, 'clickable': False})
                
                last_end = match.end()
            
            # Add any remaining text
            if last_end < len(text):
                remaining_text = text[last_end:].strip()
                if remaining_text:
                    segments.append({'text': remaining_text, 'clickable': False})
            
            return segments
        
        def render_segments(segments):
            """Render segments as clean, naturally clickable HTML"""
            html_parts = []
            
            for segment in segments:
                text = segment['text']
                if not text:
                    continue
                    
                if segment['clickable']:
                    # Create clearly visible clickable content with good hover effect
                    html_parts.append(
                        f'<span class="natural-clickable" '
                        f'data-url="{segment["url"]}" '
                        f'title="Click to see Ed\'s original post: {segment["title"]}" '
                        f'style="cursor: pointer; transition: all 0.2s ease; border-radius: 4px; padding: 3px 6px; '
                        f'background-color: rgba(102, 126, 234, 0.15); '
                        f'border: 1px solid rgba(102, 126, 234, 0.3); '
                        f'text-decoration: underline; text-decoration-color: rgba(102, 126, 234, 0.6);">'
                        f'{text}'
                        f'</span>'
                    )
                else:
                    # Regular non-clickable text
                    html_parts.append(text)
            
            return ''.join(html_parts)
        
        # Apply the natural clickable transformation
        segments = extract_and_clean_citations(final_html)
        final_html = render_segments(segments)
    
    # Add some overall structure styling
    return f'<div style="font-family: -apple-system, BlinkMacSystemFont, sans-serif; color: #2d3748;">{final_html}</div>'

def initialize_engine():
    """Initialize the GPT-4 extraction engine"""
    global engine
    if engine is None:
        print("🚀 Initializing LOCAL Multi-Vector System...")
        from claude_rag_engine import ClaudeRAGEngine
        enhanced_data_path = 'enhanced_processed'
        engine = ClaudeRAGEngine(enhanced_data_path)
        print("✅ System ready with LOCAL Qwen synthesis!")

@app.route('/')
def index():
    """Serve the main search interface"""
    return render_template('search.html')

@app.route('/search', methods=['POST'])
def search():
    """Handle search requests"""
    initialize_engine()
    
    data = request.get_json()
    query = data.get('query', '').strip()
    
    if not query:
        return jsonify({'error': 'Please enter a search query'})
    
    try:
        # Generate a unique session ID for this query
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
        
        # Perform the search
        result = engine.extract_smart_answer(query)
        
        # Get Instagram links from documents used in answer
        instagram_links = []
        if hasattr(result, 'documents_used') and result.documents_used:
            print(f"DEBUG: Getting Instagram links from documents used in answer")
            print(f"DEBUG: result.documents_used: {result.documents_used}")
            
            # Get Instagram links for the documents that contributed to the answer
            for doc_id in result.documents_used:
                # Find the document in the engine's loaded documents
                for doc in engine.documents:
                    if doc.get('document_id') == doc_id:
                        source_metadata = doc.get('source_metadata', {})
                        url = source_metadata.get('url', '')
                        if url:
                            instagram_links.append({
                                'doc_id': doc_id,
                                'url': url,
                                'title': doc.get('learning_metadata', {}).get('main_lesson', 'Ed\'s Instagram Content')[:100]
                            })
                        break
            
            print(f"DEBUG: Using {len(instagram_links)} documents that contributed to the answer")
        
        # Format the answer for web display with clickable content
        formatted_answer = format_answer_for_web(result.primary_answer, instagram_links)
        
        response_data = {
            'answer': formatted_answer,
            'confidence': result.confidence,
            'answer_docs': len(result.documents_used) if hasattr(result, 'documents_used') else 0,
            'instagram_links': instagram_links,
            'session_id': session['session_id'],
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Search error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Search failed: {str(e)}'})

@app.route('/debug/conversation')
def debug_conversation():
    """Debug endpoint to view conversation state"""
    initialize_engine()
    
    if hasattr(engine, 'conversation_tracker'):
        context = engine.conversation_tracker.get_context_for_response_generation()
        used_docs = list(engine.conversation_tracker.get_used_document_ids())
        
        return jsonify({
            'conversation_context': context,
            'used_document_ids': used_docs,
            'total_queries': len(engine.conversation_tracker.query_history) if hasattr(engine.conversation_tracker, 'query_history') else 0
        })
    else:
        return jsonify({'error': 'Conversation tracking not available'})

@app.route('/clear-conversation', methods=['POST'])
def clear_conversation():
    """Clear conversation history"""
    initialize_engine()
    
    if hasattr(engine, 'conversation_tracker') and hasattr(engine.conversation_tracker, 'clear_history'):
        engine.conversation_tracker.clear_history()
        return jsonify({'message': 'Conversation history cleared successfully'})
    else:
        return jsonify({'error': 'Conversation clearing not available'})

if __name__ == '__main__':
    print("🌐 Starting Instagram CEO Content Search Interface...")
    print("📱 This will show Instagram reel links with your answers!")
    print(f"🔍 Debug conversation state at: http://localhost:5000/debug/conversation")
    print(f"🧹 Clear conversation at: POST http://localhost:5000/clear-conversation")
    app.run(debug=True, host='0.0.0.0', port=5000)