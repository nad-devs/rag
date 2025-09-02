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

def process_text_with_citations_clean(text, doc_to_info):
    """
    Clean, single-pass citation processor that:
    1. Finds all citations in the text
    2. Wraps sentences with data-reel-id attributes
    3. Replaces citations with Instagram reel buttons
    """
    import re
    import html
    
    # DEBUG: Show exactly what we're receiving
    print("\n" + "="*80)
    print("🔍 DEBUG: process_text_with_citations_clean")
    print("="*80)
    print(f"📝 Raw text length: {len(text) if text else 0}")
    print(f"📝 Raw text preview (first 500 chars):")
    print(text[:500] if text else "EMPTY")
    print(f"\n📊 doc_to_info keys ({len(doc_to_info)} total):")
    for key in list(doc_to_info.keys())[:5]:
        print(f"  - {key}")
    print("\n🔍 Looking for citations in text...")
    
    # Find ALL citations in the text first
    citation_pattern = r'\[([^\]]+)\]'
    all_citations = re.findall(citation_pattern, text)
    print(f"📌 Found {len(all_citations)} citations in text:")
    for cit in all_citations[:10]:  # Show first 10
        print(f"  - [{cit}]")
    
    if not text or not doc_to_info:
        return html.escape(text) if text else ""
    
    # Split text into paragraphs first
    paragraphs = text.split('\n\n')
    processed_paragraphs = []
    
    for paragraph in paragraphs:
        if not paragraph.strip():
            continue
            
        # Process each paragraph line by line
        lines = paragraph.split('\n')
        processed_lines = []
        
        for line in lines:
            if not line.strip():
                continue
                
            # Check if this line has citations
            citation_pattern = r'\[([^\]]+)\]'
            citations_in_line = re.findall(citation_pattern, line)
            
            if citations_in_line:
                # Process line with citations
                processed_line = ""
                last_end = 0
                
                for match in re.finditer(citation_pattern, line):
                    citation_text = match.group(1)  # Could be "doc_id1, doc_id2, doc_id3"
                    start = match.start()
                    end = match.end()
                    
                    # Get the text before the citation
                    text_before = line[last_end:start].strip()
                    
                    if text_before:
                        # Split multiple citations by comma
                        doc_ids = [id.strip() for id in citation_text.split(',')]
                        
                        # Find which doc_ids we have URLs for
                        valid_doc_ids = [doc_id for doc_id in doc_ids if doc_id in doc_to_info]
                        
                        if valid_doc_ids:
                            # Use the first valid doc_id for the main sentence wrapping
                            first_doc_id = valid_doc_ids[0]
                            
                            # Create data-reel-ids attribute with ALL doc_ids
                            all_ids = ' '.join(valid_doc_ids)
                            
                            # Wrap the text with data attribute
                            processed_line += (
                                f'<span class="reel-sentence" data-reel-id="{html.escape(first_doc_id)}" data-reel-ids="{html.escape(all_ids)}">'
                                f'{html.escape(text_before)}'
                                f'</span>'
                            )
                            
                            # Add a button for each valid doc_id
                            for doc_id in valid_doc_ids:
                                info = doc_to_info[doc_id]
                                processed_line += (
                                    f' <a href="{html.escape(info["url"])}" '
                                    f'target="_blank" '
                                    f'class="inline-reel-link" '
                                    f'data-reel-id="{html.escape(doc_id)}" '
                                    f'onmouseover="highlightReelSentences(\'{html.escape(doc_id)}\')" '
                                    f'onmouseout="clearReelSentenceHighlight()">'
                                    f'📱 View Reel'
                                    f'</a>'
                                )
                        else:
                            # No valid URLs for any citations, just show the text
                            processed_line += html.escape(text_before)
                            print(f"⚠️ No URLs found for citations: {doc_ids}")
                    
                    last_end = end
                
                # Add any remaining text after the last citation
                if last_end < len(line):
                    remaining = line[last_end:].strip()
                    if remaining:
                        processed_line += f' {html.escape(remaining)}'
                
                processed_lines.append(processed_line)
            else:
                # No citations in this line, just escape and add
                processed_lines.append(html.escape(line))
        
        # Join lines and wrap in paragraph
        if processed_lines:
            paragraph_html = '<br>'.join(processed_lines)
            processed_paragraphs.append(f'<p class="simple-paragraph">{paragraph_html}</p>')
    
    return '\n'.join(processed_paragraphs)


def show_line_to_reel_mapping(response_text, instagram_links):
    """
    Show in terminal which Instagram reel each line/paragraph references
    Uses the existing citation system that's already built into responses
    """
    print("\n" + "="*80)
    print("📋 PARAGRAPH-TO-REEL MAPPING")
    print("="*80)
    
    # Create lookup from doc_id to URL
    link_lookup = {}
    for link in instagram_links:
        link_lookup[link['doc_id']] = {
            'url': link['url'], 
            'title': link['title']
        }
    
    import re
    paragraphs = response_text.split('\n\n')  # Split by double newlines
    citation_pattern = r'\[([^\]]+)\]'
    
    for i, paragraph in enumerate(paragraphs):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        print(f"\n📝 PARAGRAPH {i+1}:")
        
        # Find all citations in this paragraph
        citations = re.findall(citation_pattern, paragraph)
        
        # Show paragraph text without citations
        clean_text = re.sub(citation_pattern, '', paragraph).strip()
        print(f"   {clean_text[:100]}{'...' if len(clean_text) > 100 else ''}")
        
        # Show which reels this paragraph references
        if citations:
            print(f"   📎 Instagram Sources:")
            for citation in set(citations):  # Remove duplicates
                if citation in link_lookup:
                    url = link_lookup[citation]['url']
                    title = link_lookup[citation]['title'][:80]
                    print(f"      • {url}")
                    print(f"        Topic: {title}")
                else:
                    print(f"      • {citation}: (URL not found)")
        else:
            print(f"   📎 General knowledge (no specific reel)")
    
    print("\n" + "="*80 + "\n")

def extract_existing_citations_by_line(response_text, document_details):
    """
    Extract existing citations [edhonour_xyz] from each line to show which reels each line references
    """
    import re
    
    lines = response_text.split('\n')
    line_debug_info = []
    
    # Pattern to find citations like [edhonour_abc123] or [doc_id]
    citation_pattern = r'\[([^\]]+)\]'
    
    for i, line in enumerate(lines):
        if not line.strip():
            continue
            
        # Find all citations in this line
        citations = re.findall(citation_pattern, line)
        
        # Get details for each citation
        citation_details = []
        for citation in citations:
            if citation in document_details:
                doc_data = document_details[citation]
                citation_details.append({
                    'doc_id': citation,
                    'instagram_url': doc_data.get('instagram_url', ''),
                    'main_lesson': doc_data.get('main_lesson', '')[:100],
                    'post_id': doc_data.get('post_id', citation)
                })
        
        # Clean line content (remove citations for display)
        clean_line = re.sub(citation_pattern, '', line).strip()
        
        if clean_line or citation_details:  # Include if has content or citations
            line_debug_info.append({
                'line_number': i + 1,
                'line_content': clean_line,
                'original_line': line,
                'citations': citation_details,
                'citation_count': len(citations)
            })
    
    return line_debug_info

def format_single_paragraph(paragraph_text):
    """Format a single paragraph with proper HTML structure"""
    if not paragraph_text.strip():
        return ""
    
    # First, check if this is structured content (has numbered lists or multiple sections)
    if has_structured_content(paragraph_text):
        return format_structured_content(paragraph_text)
    
    # Otherwise, handle as simple paragraph or line-based content
    lines = [line.strip() for line in paragraph_text.split('\n') if line.strip()]
    if not lines:
        return ""
    
    # Check if this is a main header (ends with colon, not a URL)
    first_line = lines[0]
    if first_line.endswith(':') and len(first_line) > 3 and not first_line.startswith('http'):
        # This is a section header
        header_html = f'<h3 class="section-header">{first_line}</h3>'
        
        # Process remaining lines as content
        if len(lines) > 1:
            content_lines = lines[1:]
            content_html = format_content_lines(content_lines)
            return f'<div class="section-block">{header_html}{content_html}</div>'
        else:
            return f'<div class="section-block">{header_html}</div>'
    else:
        # Regular paragraph or list content
        content_html = format_content_lines(lines)
        return f'<div class="content-block">{content_html}</div>'

def has_structured_content(text):
    """Check if text contains structured content like numbered lists"""
    import re
    # Look for numbered items (1. 2. 3. etc.) within the text
    numbered_pattern = r'\d+\.\s+[A-Z]'
    return bool(re.search(numbered_pattern, text))

def format_structured_content(text):
    """Format content that has numbered lists and headers mixed together"""
    import re
    
    # Split on numbered items while preserving the numbers
    parts = re.split(r'(\d+\.\s+)', text)
    
    result_html = []
    current_section_header = ""
    
    i = 0
    while i < len(parts):
        part = parts[i].strip()
        if not part:
            i += 1
            continue
        
        # Check if this is a number (like "1. ")
        if re.match(r'^\d+\.\s*$', part):
            if i + 1 < len(parts):
                # Get the content after the number
                number = part.strip()
                content = parts[i + 1].strip()
                
                # Check if content starts with a header (ends with colon)
                header_match = re.match(r'^([^:]+:)\s*(.*)', content)
                if header_match:
                    item_header = header_match.group(1)
                    item_content = header_match.group(2).strip()
                    
                    # Format as numbered section with sub-content
                    result_html.append(f'<div class="numbered-section">')
                    result_html.append(f'<h4 class="numbered-header">{number} {item_header}</h4>')
                    
                    if item_content:
                        # Process any bullet points in the content
                        formatted_content = format_bullet_content(item_content)
                        result_html.append(formatted_content)
                    
                    result_html.append(f'</div>')
                else:
                    # Regular numbered item without sub-header
                    result_html.append(f'<div class="numbered-item">')
                    result_html.append(f'<strong>{number}</strong> {content}')
                    result_html.append(f'</div>')
                
                i += 2  # Skip both number and content parts
            else:
                i += 1
        else:
            # Check if this is a main header (before any numbered items)
            if part.endswith(':') and not re.search(r'\d+\.', part):
                result_html.append(f'<h3 class="section-header">{part}</h3>')
            else:
                # Regular content
                formatted_content = format_bullet_content(part)
                result_html.append(formatted_content)
            i += 1
    
    return f'<div class="structured-content">{"".join(result_html)}</div>'

def format_bullet_content(content):
    """Format content that may contain bullet points"""
    if not content.strip():
        return ""
    
    # Split on bullet points while preserving them
    import re
    bullet_parts = re.split(r'(-\s+)', content)
    
    result = []
    in_list = False
    
    for i, part in enumerate(bullet_parts):
        part = part.strip()
        if not part:
            continue
        
        if part == '-' or part == '- ':
            # This is a bullet marker
            if not in_list:
                result.append('<ul class="bullet-list">')
                in_list = True
            
            # Get the next part as bullet content
            if i + 1 < len(bullet_parts):
                bullet_content = bullet_parts[i + 1].strip()
                result.append(f'<li class="bullet-item">{bullet_content}</li>')
        elif not re.match(r'^-\s*$', part) and i == 0:
            # First part without bullets - regular text
            if in_list:
                result.append('</ul>')
                in_list = False
            result.append(f'<p class="formatted-paragraph">{part}</p>')
    
    if in_list:
        result.append('</ul>')
    
    return ''.join(result)

def format_content_lines(lines):
    """Format lines of content with proper list detection and styling"""
    if not lines:
        return ""
    
    result_html = []
    in_ordered_list = False
    in_unordered_list = False
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check for numbered list items (1. 2. 3. etc.)
        numbered_match = re.match(r'^(\d+)\.(.+)', line)
        if numbered_match:
            if not in_ordered_list:
                if in_unordered_list:
                    result_html.append('</ul>')
                    in_unordered_list = False
                result_html.append('<ol class="formatted-list">')
                in_ordered_list = True
            
            content = numbered_match.group(2).strip()
            result_html.append(f'<li class="list-item"><strong>{numbered_match.group(1)}.</strong> {content}</li>')
            
        # Check for bullet points (- or •)
        elif line.startswith('- ') or line.startswith('• '):
            if not in_unordered_list:
                if in_ordered_list:
                    result_html.append('</ol>')
                    in_ordered_list = False
                result_html.append('<ul class="formatted-list">')
                in_unordered_list = True
            
            content = line[2:].strip()
            result_html.append(f'<li class="list-item">{content}</li>')
            
        # Check for sub-headers (lines ending with colon)
        elif line.endswith(':') and len(line) > 3 and not line.startswith('http'):
            if in_ordered_list:
                result_html.append('</ol>')
                in_ordered_list = False
            if in_unordered_list:
                result_html.append('</ul>')
                in_unordered_list = False
            result_html.append(f'<h4 class="sub-header">{line}</h4>')
            
        # Regular paragraph text
        else:
            if in_ordered_list:
                result_html.append('</ol>')
                in_ordered_list = False
            if in_unordered_list:
                result_html.append('</ul>')
                in_unordered_list = False
            result_html.append(f'<p class="formatted-paragraph">{line}</p>')
    
    # Close any remaining lists
    if in_ordered_list:
        result_html.append('</ol>')
    if in_unordered_list:
        result_html.append('</ul>')
    
    return ''.join(result_html)

def process_citations_in_formatted_html(formatted_paragraphs, doc_to_info):
    """Process citations in the already formatted HTML to make them clickable"""
    import re
    
    def make_text_clickable(match):
        """Replace citation patterns with clickable spans"""
        text_before = match.group(1).strip()
        doc_id = match.group(2)
        
        if doc_id in doc_to_info and text_before:
            info = doc_to_info[doc_id]
            return (
                f'<span class="natural-clickable" '
                f'data-url="{info["url"]}" '
                f'data-reel-id="{doc_id}" '
                f'title="Click to see Ed\'s original post: {info["shortcode"]}" '
                f'onclick="window.open(\'{info["url"]}\', \'_blank\')" '
                f'style="cursor: pointer; transition: all 0.2s ease; border-radius: 4px; padding: 2px 4px; '
                f'background-color: rgba(102, 126, 234, 0.15); '
                f'border: 1px solid rgba(102, 126, 234, 0.3); '
                f'text-decoration: underline; text-decoration-color: rgba(102, 126, 234, 0.6);">'
                f'{text_before}'
                f'</span>'
            )
        elif text_before:
            # Return text without citation if no valid link
            return text_before
        return match.group(0)
    
    # Process each formatted paragraph
    processed_paragraphs = []
    for paragraph_html in formatted_paragraphs:
        # Find patterns like "text [doc_id]" and make the text clickable
        citation_pattern = r'([^<>\[\n]*?)\s*\[([^\]]+)\]'
        processed_html = re.sub(citation_pattern, make_text_clickable, paragraph_html)
        processed_paragraphs.append(processed_html)
    
    return '<div class="formatted-content">' + ''.join(processed_paragraphs) + '</div>'








def format_answer_for_web(answer_text, instagram_links=None):
    """Format answer with clean paragraphs and inline Instagram links"""
    if not answer_text:
        return ""
    
    # Create a mapping of document IDs to Instagram URLs
    doc_to_info = {}
    if instagram_links:
        for link in instagram_links:
            if 'doc_id' in link and 'url' in link:
                # Fix URL protocol issues upfront
                url = link['url']
                if url.startswith('//'):
                    url = 'https:' + url
                elif not url.startswith(('http://', 'https://')):
                    url = 'https://' + url
                    
                doc_to_info[link['doc_id']] = {
                    'url': url,
                    'title': link.get('title', 'Ed\'s Instagram Content'),
                    'shortcode': link.get('doc_id', '').replace('edhonour_', '')
                }
        print(f"📊 doc_to_info has {len(doc_to_info)} entries: {list(doc_to_info.keys())}")
    
    # Clean up any JSON artifacts
    if answer_text.strip().startswith('{') and '"answer"' in answer_text:
        json_match = re.search(r'"answer"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', answer_text, re.DOTALL)
        if json_match:
            answer_text = json_match.group(1)
            answer_text = answer_text.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
    
    # Use the new clean processor
    formatted_html = process_text_with_citations_clean(answer_text, doc_to_info)
    
    return f'<div class="clean-formatted-content">{formatted_html}</div>'

def format_answer_with_paragraph_highlighting(answer_text, instagram_links, paragraph_mapping):
    """Format answer with visual highlighting for paragraphs from different reels"""
    if not answer_text:
        return ""
    
    # Define distinct colors for different reels
    reel_colors = [
        '#E3F2FD',  # Light blue
        '#F3E5F5',  # Light purple  
        '#E8F5E8',  # Light green
        '#FFF3E0',  # Light orange
        '#FCE4EC',  # Light pink
        '#F1F8E9',  # Light lime
        '#E0F2F1',  # Light teal
        '#FFF8E1'   # Light yellow
    ]
    
    # Create color mapping for Instagram links
    link_to_color = {}
    color_index = 0
    for link in instagram_links:
        if link['doc_id'] not in link_to_color:
            link_to_color[link['doc_id']] = reel_colors[color_index % len(reel_colors)]
            color_index += 1
    
    # Process the answer text paragraph by paragraph
    paragraphs = answer_text.split('\n\n')
    enhanced_paragraphs = []
    
    for i, paragraph in enumerate(paragraphs):
        paragraph = paragraph.strip()
        if not paragraph:
            enhanced_paragraphs.append('')
            continue
            
        # Check if this paragraph has sources in our mapping
        paragraph_sources = []
        if i < len(paragraph_mapping) and paragraph_mapping[i]['has_sources']:
            paragraph_sources = paragraph_mapping[i]['sources']
        
        if paragraph_sources:
            # Get the primary color (use first reel's color)
            primary_reel = paragraph_sources[0]['doc_id']
            bg_color = link_to_color.get(primary_reel, '#F5F5F5')
            
            # Build tooltip content
            tooltip_sources = []
            for source in paragraph_sources[:3]:
                tooltip_sources.append(f"📱 {source['title'][:50]}...")
            tooltip_text = "Sources:\\n" + "\\n".join(tooltip_sources)
            
            # Create enhanced paragraph with styling and citations converted to clickable links
            enhanced_paragraph = format_paragraph_with_citations(
                paragraph, 
                instagram_links, 
                bg_color, 
                tooltip_text,
                f"paragraph-{i}"
            )
        else:
            # No sources - regular formatting
            enhanced_paragraph = format_paragraph_with_citations(paragraph, instagram_links)
        
        enhanced_paragraphs.append(enhanced_paragraph)
    
    # Add CSS and JavaScript
    css_js = """
    <style>
    .paragraph-highlighted {
        padding: 15px;
        margin: 10px 0;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        position: relative;
        cursor: help;
    }
    .paragraph-normal {
        padding: 10px 0;
        margin: 8px 0;
    }
    .reel-tooltip {
        position: absolute;
        background: #333;
        color: white;
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 11px;
        max-width: 250px;
        z-index: 1000;
        display: none;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        white-space: pre-line;
    }
    .natural-clickable {
        cursor: pointer;
        transition: all 0.2s ease;
        border-radius: 4px;
        padding: 2px 4px;
        background-color: rgba(102, 126, 234, 0.15);
        border: 1px solid rgba(102, 126, 234, 0.3);
        text-decoration: underline;
        text-decoration-color: rgba(102, 126, 234, 0.6);
    }
    .natural-clickable:hover {
        background-color: rgba(102, 126, 234, 0.25);
    }
    </style>
    
    <script>
    function showReelTooltip(tooltipId, event) {
        const tooltip = document.getElementById(tooltipId);
        if (tooltip) {
            tooltip.style.display = 'block';
            tooltip.style.left = event.pageX + 10 + 'px';
            tooltip.style.top = event.pageY - 10 + 'px';
        }
    }
    
    function hideReelTooltip(tooltipId) {
        const tooltip = document.getElementById(tooltipId);
        if (tooltip) {
            tooltip.style.display = 'none';
        }
    }
    
    function openInstagramReel(url) {
        window.open(url, '_blank');
    }
    
    function highlightSameReel(reelId) {
        // Remove any existing highlights
        clearReelHighlight();
        
        // Find all sources from the same reel and highlight them
        const sources = document.querySelectorAll(`[data-reel-id="${reelId}"]`);
        sources.forEach(source => {
            source.style.backgroundColor = '#FFE082';  // Bright yellow highlight
            source.style.border = '2px solid #FF9800';
            source.style.fontWeight = 'bold';
            source.style.transform = 'scale(1.05)';
        });
        
        // Also highlight paragraphs that contain this reel
        const allParagraphs = document.querySelectorAll('.paragraph-highlighted');
        allParagraphs.forEach(paragraph => {
            const sourcesInParagraph = paragraph.querySelectorAll(`[data-reel-id="${reelId}"]`);
            if (sourcesInParagraph.length > 0) {
                paragraph.style.boxShadow = '0 0 15px rgba(255, 152, 0, 0.5)';
                paragraph.style.borderLeft = '6px solid #FF9800';
                paragraph.style.transform = 'scale(1.02)';
            }
        });
    }
    
    function clearReelHighlight() {
        // Clear source highlights
        const allSources = document.querySelectorAll('.reel-source');
        allSources.forEach(source => {
            source.style.backgroundColor = '';
            source.style.border = '';
            source.style.fontWeight = '';
            source.style.transform = '';
        });
        
        // Clear paragraph highlights
        const allParagraphs = document.querySelectorAll('.paragraph-highlighted');
        allParagraphs.forEach(paragraph => {
            paragraph.style.boxShadow = '';
            paragraph.style.borderLeft = '4px solid #667eea';  // Reset to original
            paragraph.style.transform = '';
        });
    }
    </script>
    """
    
    return f'{css_js}<div style="font-family: -apple-system, BlinkMacSystemFont, sans-serif; color: #2d3748;">{"<br>".join(enhanced_paragraphs)}</div>'

def format_paragraph_with_citations(paragraph_text, instagram_links, bg_color=None, tooltip_text=None, tooltip_id=None):
    """Format a single paragraph, converting citations to clickable links"""
    import re
    
    # Create lookup for Instagram links
    link_lookup = {}
    for link in instagram_links:
        link_lookup[link['doc_id']] = link
    
    # Find and replace citations with clickable links
    citation_pattern = r'\[([^\]]+)\]'
    
    def replace_citation(match):
        doc_id = match.group(1)
        if doc_id in link_lookup:
            link_info = link_lookup[doc_id]
            title = f"📱 {link_info['title'][:60]}... (Click to open Instagram)"
            # Add data attribute for reel highlighting and fix onclick with proper escaping
            safe_url = link_info["url"].replace("'", "\\'")
            return f'<span class="natural-clickable reel-source" data-reel-id="{doc_id}" onclick="openInstagramReel(\'{safe_url}\')" onmouseover="highlightSameReel(\'{doc_id}\')" onmouseout="clearReelHighlight()" title="{title}">🎥 Source</span>'
        return match.group(0)  # Keep original if no link found
    
    # Replace citations with clickable elements
    enhanced_text = re.sub(citation_pattern, replace_citation, paragraph_text)
    
    # Format as HTML paragraph
    lines = enhanced_text.split('\n')
    html_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Handle headers (lines ending with :)
        if line.endswith(':') and not line.startswith('http'):
            html_lines.append(f'<h4 style="color: #4a5568; margin: 15px 0 8px 0; font-weight: 600;">{line}</h4>')
        # Handle numbered lists
        elif re.match(r'^\d+\.', line):
            html_lines.append(f'<div style="margin: 5px 0;"><strong>{line}</strong></div>')
        # Handle bullet points
        elif line.startswith('- ') or line.startswith('• '):
            html_lines.append(f'<div style="margin: 5px 0 5px 20px;">{line}</div>')
        else:
            html_lines.append(f'<div style="margin: 8px 0;">{line}</div>')
    
    content_html = ''.join(html_lines)
    
    # Apply styling based on whether it has sources
    if bg_color and tooltip_text and tooltip_id:
        # Highlighted paragraph with sources
        tooltip_div = f'<div id="tooltip-{tooltip_id}" class="reel-tooltip">{tooltip_text}</div>'
        return f'<div class="paragraph-highlighted" style="background-color: {bg_color};" onmouseover="showReelTooltip(\'tooltip-{tooltip_id}\', event)" onmouseout="hideReelTooltip(\'tooltip-{tooltip_id}\')">{content_html}{tooltip_div}</div>'
    else:
        # Normal paragraph
        return f'<div class="paragraph-normal">{content_html}</div>'

def add_citation_hover_tooltips(html_content, instagram_links):
    """Simply enhance the existing clickable citations with better tooltips"""
    
    # Create lookup for Instagram links
    link_lookup = {}
    for link in instagram_links:
        link_lookup[link['doc_id']] = {
            'url': link['url'],
            'title': link['title']
        }
    
    # Enhance existing clickable spans that already have citations
    import re
    
    def enhance_clickable_span(match):
        original_span = match.group(0)
        doc_id = match.group(1)  # Extract doc_id from data-url
        
        if doc_id in link_lookup:
            # Extract the URL from the existing span
            url = link_lookup[doc_id]['url']
            title = link_lookup[doc_id]['title'][:80]
            
            # Enhance the title attribute with better tooltip info
            enhanced_title = f"📱 Instagram Source: {title}\\nClick to open: {url}"
            
            # Replace the existing title with enhanced one
            enhanced_span = re.sub(r'title="[^"]*"', f'title="{enhanced_title}"', original_span)
            return enhanced_span
        
        return original_span
    
    # Find and enhance existing clickable spans
    pattern = r'<span class="natural-clickable"[^>]*data-url="([^"]*)"[^>]*>(.*?)</span>'
    enhanced_html = re.sub(pattern, enhance_clickable_span, html_content)
    
    return enhanced_html

def initialize_engine():
    """Initialize the GPT-4 extraction engine"""
    global engine
    if engine is None:
        print("🚀 Initializing LOCAL Multi-Vector System...")
        from claude_rag_engine import ClaudeRAGEngine
        # Use absolute path to enhanced_processed directory
        enhanced_data_path = '/var/www/uploads/instagram_caption_scraper/rag_system/enhanced_processed'
        print(f"📁 Looking for enhanced data in: {enhanced_data_path}")
        engine = ClaudeRAGEngine(enhanced_data_path)
        print("✅ System ready with LOCAL Qwen synthesis!")

@app.route('/')
def index():
    """Serve the main search interface"""
    return render_template('search.html')

@app.route('/search', methods=['POST'])
def search():
    """Handle search requests"""
    print(f"🔍 SEARCH REQUEST RECEIVED")
    initialize_engine()
    
    data = request.get_json()
    query = data.get('query', '').strip()
    debug_mode = data.get('debug', False)  # Allow debug mode from frontend
    print(f"📝 Query: {query}")
    
    if not query:
        return jsonify({'error': 'Please enter a search query'})
    
    try:
        # Generate a unique session ID for this query
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
        
        # Perform the search
        result = engine.extract_smart_answer(query)
        
        # DEBUG: Show what the RAG engine returned
        print("\n" + "="*80)
        print("🎯 DEBUG: RAG ENGINE RESPONSE")
        print("="*80)
        print(f"📝 Primary answer length: {len(result.primary_answer) if hasattr(result, 'primary_answer') else 0}")
        print(f"📝 Primary answer preview (first 1000 chars):")
        if hasattr(result, 'primary_answer'):
            print(result.primary_answer[:1000])
        print(f"\n📚 Documents used: {result.documents_used if hasattr(result, 'documents_used') else 'NONE'}")
        print("="*80 + "\n")
        
        # Get detailed document information for debugging
        document_details = {}
        instagram_links = []
        if hasattr(result, 'documents_used') and result.documents_used:
            print(f"DEBUG: Getting Instagram links from documents used in answer")
            print(f"DEBUG: result.documents_used: {result.documents_used}")
            
            # Import Path and json for direct file access
            from pathlib import Path
            import json
            
            # Get Instagram links and detailed info for the documents that contributed to the answer
            for doc_id in result.documents_used:
                doc_found = False
                url = ''
                learning_metadata = {}
                
                # First try to find in engine.documents
                for doc in engine.documents:
                    if doc.get('document_id') == doc_id:
                        doc_found = True
                        source_metadata = doc.get('source_metadata', {})
                        url = source_metadata.get('url', '')
                        learning_metadata = doc.get('learning_metadata', {})
                        
                        # Store detailed document info for debugging
                        document_details[doc_id] = {
                            'main_lesson': learning_metadata.get('main_lesson', ''),
                            'content_text': doc.get('content_text', ''),
                            'key_takeaways': learning_metadata.get('key_takeaways', []),
                            'actionable_insights': learning_metadata.get('actionable_insights', []),
                            'technologies_mentioned': learning_metadata.get('technologies_mentioned', []),
                            'instagram_url': url,
                            'post_id': doc.get('post_id', doc_id)
                        }
                        break
                
                # If not found in memory, search for the file directly
                if not doc_found or not url:
                    enhanced_path = Path('/var/www/uploads/instagram_caption_scraper/rag_system/enhanced_processed')
                    # Simple direct file path - no subdirectories needed now
                    file_path = enhanced_path / f"enhanced_{doc_id.replace('edhonour_', '')}.json"
                    
                    if file_path.exists():
                        # Load the file directly
                        with open(file_path, 'r', encoding='utf-8') as f:
                            doc_data = json.load(f)
                            source_metadata = doc_data.get('source_metadata', {})
                            url = source_metadata.get('url', '')
                            learning_metadata = doc_data.get('learning_metadata', {})
                            
                            if not doc_found:
                                # Store document details if not found before
                                document_details[doc_id] = {
                                    'main_lesson': learning_metadata.get('main_lesson', ''),
                                    'content_text': doc_data.get('content_text', ''),
                                    'key_takeaways': learning_metadata.get('key_takeaways', []),
                                    'actionable_insights': learning_metadata.get('actionable_insights', []),
                                    'technologies_mentioned': learning_metadata.get('technologies_mentioned', []),
                                    'instagram_url': url,
                                    'post_id': doc_data.get('post_id', doc_id)
                                }
                            print(f"✅ Found document {doc_id} via direct file access: {url}")
                    else:
                        print(f"❌ File not found: {file_path}")
                
                # Add to instagram_links if we have a URL
                if url:
                    # Ensure URL has proper protocol
                    if url.startswith('//'):
                        url = 'https:' + url
                    elif not url.startswith(('http://', 'https://')):
                        url = 'https://' + url
                    instagram_links.append({
                        'doc_id': doc_id,
                        'url': url,
                        'title': learning_metadata.get('main_lesson', 'Ed\'s Instagram Content')[:100]
                    })
                else:
                    # Fallback: construct URL from doc_id
                    if doc_id.startswith('edhonour_'):
                        shortcode = doc_id.replace('edhonour_', '')
                        url = f'https://www.instagram.com/reel/{shortcode}/'
                        instagram_links.append({
                            'doc_id': doc_id,
                            'url': url,
                            'title': 'Ed\'s Instagram Content'
                        })
                        print(f"⚠️ Using constructed URL for {doc_id}: {url}")
        
        # Generate line-by-line debugging if debug mode is enabled
        line_debugging = None
        if debug_mode:
            line_debugging = extract_existing_citations_by_line(result.primary_answer, document_details)
            print("🔍 DEBUG MODE: Extracted existing citations by line")
        
        # ALWAYS show line-to-reel mapping in terminal for debugging
        show_line_to_reel_mapping(result.primary_answer, instagram_links)
        
        # Format the answer for web display with improved structure and clickable citations
        formatted_answer = format_answer_for_web(result.primary_answer, instagram_links)
        
        response_data = {
            'answer': formatted_answer,
            'confidence': result.confidence,
            'answer_docs': len(result.documents_used) if hasattr(result, 'documents_used') else 0,
            'instagram_links': instagram_links,
            'session_id': session['session_id'],
            'timestamp': datetime.now().isoformat(),
            'debug_info': line_debugging if debug_mode else None,
            'document_details': document_details if debug_mode else None
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

@app.route('/debug/line-sources', methods=['POST'])
def debug_line_sources():
    """Debug endpoint to show line-by-line source mapping for a query"""
    initialize_engine()
    
    data = request.get_json()
    query = data.get('query', '').strip()
    
    if not query:
        return jsonify({'error': 'Please provide a query'})
    
    try:
        # Get the response
        result = engine.extract_smart_answer(query)
        
        # Get document details
        document_details = {}
        for doc_id in result.documents_used:
            for doc in engine.documents:
                if doc.get('document_id') == doc_id:
                    source_metadata = doc.get('source_metadata', {})
                    learning_metadata = doc.get('learning_metadata', {})
                    document_details[doc_id] = {
                        'main_lesson': learning_metadata.get('main_lesson', ''),
                        'instagram_url': source_metadata.get('url', ''),
                        'post_id': doc.get('post_id', doc_id)
                    }
                    break
        
        # Extract line-by-line citations
        line_debug = extract_existing_citations_by_line(result.primary_answer, document_details)
        
        return jsonify({
            'query': query,
            'response': result.primary_answer,
            'line_debug': line_debug,
            'total_lines': len(line_debug),
            'documents_used': result.documents_used
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Debug failed: {str(e)}'})

def consolidate_duplicate_reel_buttons(html_content):
    """
    Remove duplicate reel buttons within logical sections, add data-reel-id attributes.
    Keeps the LAST occurrence of each reel button per section for natural placement.
    """
    import re
    
    # First, add data-reel-id attributes by extracting reel IDs from URLs
    def add_reel_id_attribute(match):
        full_link = match.group(0)
        url = match.group(1)
        
        # Extract reel ID from Instagram URL
        reel_id_match = re.search(r'/reel/([^/]+)/', url)
        if reel_id_match:
            reel_shortcode = reel_id_match.group(1)
            reel_id = f"edhonour_{reel_shortcode}"
            
            # Add data-reel-id attribute to the link
            enhanced_link = full_link.replace(
                'class="inline-reel-link"',
                f'class="inline-reel-link" data-reel-id="{reel_id}"'
            )
            return enhanced_link
        
        return full_link
    
    # Add data-reel-id attributes to all buttons
    button_pattern = r'<a href="([^"]+)" target="_blank" class="inline-reel-link">📱 View Reel</a>'
    html_content = re.sub(button_pattern, add_reel_id_attribute, html_content)
    
    # Split content into logical sections (paragraphs separated by </p><p>)
    sections = html_content.split('</p><p class="simple-paragraph">')
    
    consolidated_sections = []
    
    for section in sections:
        # Find all reel buttons in this section with their reel IDs
        enhanced_button_pattern = r'<a href="[^"]+" target="_blank" class="inline-reel-link" data-reel-id="([^"]+)">📱 View Reel</a>'
        buttons = list(re.finditer(enhanced_button_pattern, section))
        
        if len(buttons) <= 1:
            # No duplicates in this section
            consolidated_sections.append(section)
            continue
        
        # Group buttons by reel ID and find last occurrence of each
        reel_last_positions = {}
        for button in buttons:
            reel_id = button.group(1)
            reel_last_positions[reel_id] = button
        
        # Remove all buttons first
        section_without_buttons = re.sub(enhanced_button_pattern, '', section)
        
        # Add back only the last occurrence of each unique reel
        final_section = section_without_buttons
        for reel_id, last_button in reel_last_positions.items():
            final_section += ' ' + last_button.group(0)
        
        consolidated_sections.append(final_section)
    
    # Rejoin sections
    return '</p><p class="simple-paragraph">'.join(consolidated_sections)


if __name__ == '__main__':
    import argparse
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0', help='Host to run on')
    parser.add_argument('--port', type=int, default=80, help='Port to run on')
    parser.add_argument('--dev', action='store_true', help='Run in development mode with Flask server')
    args = parser.parse_args()
    
    # Only run Flask dev server if --dev flag is passed
    if args.dev or '--dev' in sys.argv:
        print("🌐 Starting Instagram CEO Content Search Interface (Development Mode)...")
        print("📱 This will show Instagram reel links with your answers!")
        print(f"🔍 Debug conversation state at: http://localhost:{args.port}/debug/conversation")
        print(f"🧹 Clear conversation at: POST http://localhost:{args.port}/clear-conversation")
        app.run(debug=True, host=args.host, port=args.port)
    else:
        print("🚀 For production, use Gunicorn:")
        print(f"   sudo gunicorn --bind 0.0.0.0:80 --workers 3 --timeout 300 rag_system.web_interface:app")