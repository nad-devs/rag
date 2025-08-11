#!/usr/bin/env python3
"""
Instagram Content Enhancer: Production Claude 3.5 Sonnet Enhancement System
Converts raw Instagram data to enhanced_processed format for daily automation
"""

import json
import os
import sys
import time
from typing import Dict, Any, List
from datetime import datetime
import openai
from anthropic import Anthropic
from dotenv import load_dotenv

# Setup
load_dotenv()
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class ProductionEnhancer:
    def __init__(self):
        self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.anthropic_client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        
        # Cost tracking (per 1M tokens)
        self.costs = {
            'gpt-4': {'input': 30.00, 'output': 60.00},  # $30/$60 per 1M tokens
            'claude-3-5-sonnet': {'input': 3.00, 'output': 15.00}  # $3/$15 per 1M tokens
        }
        
        # Content preprocessing patterns
        self.content_fixes = {
            # Claude Code vs Cloud Code disambiguation 
            'claude_code_patterns': [
                (r'\bcloud code\b', 'Claude Code'),  # Fix common transcription error
                (r'\bClaude code\b', 'Claude Code'),  # Standardize capitalization
                (r'\bclaude Code\b', 'Claude Code'),  # Fix partial capitalization
            ],
            # Other common transcription fixes
            'general_fixes': [
                (r'\bgemini cli\b', 'Gemini CLI'),
                (r'\bgemini CLI\b', 'Gemini CLI'), 
                (r'\bgpt 4\b', 'GPT-4'),
                (r'\bgpt4\b', 'GPT-4'),
                (r'\bRAG\b', 'RAG'),
                (r'\bapi\b', 'API'),
                (r'\bui\b', 'UI'),
                (r'\bux\b', 'UX'),
            ]
        }
        
    def preprocess_content(self, content: str) -> str:
        """Apply content fixes and preprocessing before enhancement"""
        import re
        
        if not content:
            return content
            
        processed_content = content
        
        # Apply Claude Code fixes (most important)
        for pattern, replacement in self.content_fixes['claude_code_patterns']:
            processed_content = re.sub(pattern, replacement, processed_content, flags=re.IGNORECASE)
        
        # Apply general transcription fixes
        for pattern, replacement in self.content_fixes['general_fixes']:
            processed_content = re.sub(pattern, replacement, processed_content, flags=re.IGNORECASE)
        
        # Clean up extra whitespace
        processed_content = ' '.join(processed_content.split())
        
        return processed_content
        
    def create_enhancement_prompt(self, raw_content: str) -> str:
        """Create the enhancement prompt based on existing enhanced file structure"""
        
        prompt = f"""You are an expert content analyzer specializing in extracting structured learning metadata from Instagram educational content. Your task is to analyze video transcription content and create comprehensive learning-focused metadata.

IMPORTANT: This content comes from video transcription and may contain common transcription errors. Please fix these mistakes before analysis:
- "Quad" should be "Claude" (common AI transcription error)
- "quad" should be "Claude" 
- "cloud code" should be "Claude Code"
- Other obvious transcription mistakes for technical terms

CONTENT TO ANALYZE:
{raw_content}

ANALYSIS REQUIREMENTS:
Transform this Instagram educational content into a comprehensive learning resource with rich metadata. Focus on making the content discoverable and useful for learners.

OUTPUT FORMAT - Return ONLY valid JSON with this exact structure:

{{
  "learning_metadata": {{
    "main_lesson": "Single sentence describing the core learning objective",
    "key_takeaways": ["3-5 specific bullet points of key information learned"],
    "actionable_insights": ["2-3 specific actions viewers can take based on this content"],
    "content_category": "Choose from: technical_tutorial|industry_opinion|career_advice|tool_review|comparison|implementation_guide|quick_tip",
    "difficulty_level": "beginner|intermediate|advanced",
    "time_investment": "quick_tip|short_lesson|comprehensive_guide",
    "is_part_of_series": true/false,
    "series_title": "Title if part of series, empty string if not",
    "part_number": 0,
    "total_parts_detected": 0,
    "series_context": "Context about where this fits in the series, empty if not series",
    "continuation_phrases": ["phrases that indicate this continues from/to other content"],
    "content_date_context": "current|timeless|dated",
    "temporal_relevance": "current|timeless|historical",
    "time_sensitive_topics": ["topics that may become outdated"],
    "update_indicators": ["phrases suggesting this needs updates"],
    "version_context": "Information about versions/updates mentioned",
    "technologies_mentioned": [
      {{
        "name": "Technology name",
        "context": "How it's discussed in the content",
        "sentiment": "positive|negative|neutral",
        "recommendation": "recommended|not_recommended|neutral"
      }}
    ],
    "tools_and_platforms": [
      {{
        "name": "Tool/platform name", 
        "use_case": "How it's used or recommended"
      }}
    ],
    "companies_discussed": ["List of companies mentioned"],
    "natural_questions": ["3-5 questions users would naturally ask about this topic"],
    "search_scenarios": ["3-5 scenarios when someone would search for this content"],
    "related_topics": ["3-5 related topics for exploration"],
    "prerequisites": ["Knowledge needed to understand this content"],
    "next_steps": ["What to learn/do after this content"],
    "practical_applications": ["Real-world applications of this knowledge"],
    "credibility_indicators": ["What makes this content credible"],
    "personal_experience": true/false,
    "specific_examples": ["Concrete examples provided in the content"],
    "controversy_level": "low|medium|high",
    "shareability": "highly_shareable|moderately_shareable|niche_audience",
    "discussion_potential": "high|medium|low"
  }},
  "content_text": "The full transcribed text from the video",
  "user_experience": {{
    "learning_path": {{
      "current_level": "beginner|intermediate|advanced",
      "time_to_consume": "quick_tip|short_lesson|comprehensive_guide", 
      "prerequisites_met": true/false,
      "next_steps_available": true/false
    }},
    "discoverability": {{
      "search_friendly": true/false,
      "category_clear": true/false,
      "actionable_content": true/false
    }},
    "engagement_metrics": {{
      "discussion_potential": "high|medium|low",
      "shareability": "highly_shareable|moderately_shareable|niche_audience",
      "controversy_level": "low|medium|high"
    }},
    "knowledge_value": {{
      "personal_experience": true/false,
      "specific_examples": number_of_examples,
      "credibility_score": 1-5
    }}
  }}
}}

ANALYSIS GUIDELINES:
1. Extract learning value from conversational, informal content
2. Focus on practical, actionable insights
3. Consider the educational progression and skill level
4. Identify technical concepts and tools accurately
5. Generate search-friendly questions and scenarios
6. Assess the temporal relevance and update needs
7. Determine appropriate difficulty and prerequisites
8. Extract technology sentiments and recommendations accurately
9. SERIES DETECTION: Look for multi-part content indicators:
   - Numbered patterns: "Number one", "Number two", "Part 1", "Part 2", etc.
   - Series titles in content like "Critical [topic] tips", "Top [number] reasons", etc.
   - Continuation phrases: "next episode", "in the next video", "coming up"
   - If series detected: set is_part_of_series=true, extract series_title, determine part_number
   - Count total parts when multiple numbers mentioned or obvious from context
   - Use continuation_phrases to capture linking words between episodes

IMPORTANT: Return ONLY the JSON structure. No additional text or explanation."""

        return prompt

    def enhance_with_gpt4(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance content using GPT-4"""
        content_text = raw_data.get('caption', '') or raw_data.get('transcription_data', {}).get('full_text', '')
        prompt = self.create_enhancement_prompt(content_text)
        
        start_time = time.time()
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert content analyzer. Return only valid JSON."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2500
            )
            
            processing_time = time.time() - start_time
            raw_response = response.choices[0].message.content
            
            # Parse JSON response
            enhancement_data = json.loads(raw_response)
            
            # Add metadata
            result = {
                "document_id": f"edhonour_{raw_data.get('shortcode', 'unknown')}",
                "source_metadata": {
                    "profile": raw_data.get('profile_username', 'edhonour'),
                    "shortcode": raw_data.get('shortcode', ''),
                    "url": raw_data.get('url', ''),
                    "video_url": raw_data.get('video_url_extracted', ''),
                    "confidence_score": raw_data.get('processing_info', {}).get('confidence_score', 1),
                    "language": raw_data.get('transcription_data', {}).get('language', 'en'),
                    "position_in_profile": raw_data.get('position_in_profile', 0),
                    "extraction_timestamp": raw_data.get('extraction_timestamp', time.time())
                },
                **enhancement_data,
                "processing_info": {
                    "model_used": "gpt-4",
                    "extraction_method": "user_focused_learning",
                    "content_length": len(content_text),
                    "word_count": len(content_text.split()),
                    "processing_timestamp": time.time(),
                    "processing_time": processing_time
                }
            }
            
            # Calculate cost
            input_tokens = len(prompt.split()) * 1.3  # Rough token estimate
            output_tokens = len(raw_response.split()) * 1.3
            cost = (input_tokens * self.costs['gpt-4']['input'] + output_tokens * self.costs['gpt-4']['output']) / 1000000
            
            return {
                'result': result,
                'cost': cost,
                'processing_time': processing_time,
                'tokens': {'input': input_tokens, 'output': output_tokens}
            }
            
        except Exception as e:
            return {'error': str(e), 'cost': 0, 'processing_time': processing_time}

    def enhance_with_claude(self, raw_data: Dict[str, Any], max_retries: int = 3) -> Dict[str, Any]:
        """Enhance content using Claude 3.5 Sonnet with preprocessing and retry logic"""
        # Extract and preprocess content
        raw_content = raw_data.get('caption', '') or raw_data.get('transcription_data', {}).get('full_text', '')
        content_text = self.preprocess_content(raw_content)
        
        if not content_text or len(content_text.strip()) < 10:
            return {'error': 'Insufficient content for enhancement', 'cost': 0, 'processing_time': 0}
        
        prompt = self.create_enhancement_prompt(content_text)
        
        start_time = time.time()
        last_error = None
        
        # Retry logic for robustness
        for attempt in range(max_retries):
            try:
                response = self.anthropic_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=2500,
                    temperature=0.1,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                processing_time = time.time() - start_time
                raw_response = response.content[0].text
                
                # Clean response before parsing
                cleaned_response = self._clean_json_response(raw_response)
                
                # Parse JSON response
                enhancement_data = json.loads(cleaned_response)
                
                # Validate response structure
                if not self._validate_enhancement_structure(enhancement_data):
                    raise ValueError("Invalid enhancement structure returned")
                
                break  # Success - exit retry loop
                
            except (json.JSONDecodeError, ValueError) as e:
                last_error = f"Attempt {attempt + 1}: JSON/Validation error: {str(e)}"
                if attempt < max_retries - 1:
                    print(f"⚠️  {last_error}, retrying...")
                    time.sleep(2)  # Wait before retry
                    continue
                else:
                    processing_time = time.time() - start_time
                    return {'error': f"Failed after {max_retries} attempts: {last_error}", 'cost': 0, 'processing_time': processing_time}
                    
            except Exception as e:
                last_error = f"Attempt {attempt + 1}: API error: {str(e)}"
                if attempt < max_retries - 1:
                    print(f"⚠️  {last_error}, retrying...")
                    time.sleep(5)  # Longer wait for API errors
                    continue
                else:
                    processing_time = time.time() - start_time
                    return {'error': f"Failed after {max_retries} attempts: {last_error}", 'cost': 0, 'processing_time': processing_time}
        
        # Add metadata to successful result
        try:
            result = {
                "document_id": f"edhonour_{raw_data.get('shortcode', 'unknown')}",
                "source_metadata": {
                    "profile": raw_data.get('profile_username', 'edhonour'),
                    "shortcode": raw_data.get('shortcode', ''),
                    "url": raw_data.get('url', ''),
                    "video_url": raw_data.get('video_url_extracted', ''),
                    "confidence_score": raw_data.get('processing_info', {}).get('confidence_score', 1),
                    "language": raw_data.get('transcription_data', {}).get('language', 'en'),
                    "position_in_profile": raw_data.get('position_in_profile', 0),
                    "extraction_timestamp": raw_data.get('extraction_timestamp', time.time())
                },
                **enhancement_data,
                "processing_info": {
                    "model_used": "claude-3-5-sonnet",
                    "extraction_method": "user_focused_learning", 
                    "content_length": len(content_text),
                    "word_count": len(content_text.split()),
                    "processing_timestamp": time.time(),
                    "processing_time": processing_time
                }
            }
            
            # Calculate cost
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cost = (input_tokens * self.costs['claude-3-5-sonnet']['input'] + output_tokens * self.costs['claude-3-5-sonnet']['output']) / 1000000
            
            return {
                'result': result,
                'cost': cost,
                'processing_time': processing_time,
                'tokens': {'input': input_tokens, 'output': output_tokens}
            }
            
        except Exception as e:
            return {'error': str(e), 'cost': 0, 'processing_time': processing_time}

    def _clean_json_response(self, raw_response: str) -> str:
        """Clean JSON response from Claude to handle common formatting issues"""
        import re
        
        # Remove any text before the first {
        if '{' in raw_response:
            start_idx = raw_response.find('{')
            raw_response = raw_response[start_idx:]
        
        # Remove any text after the last }
        if '}' in raw_response:
            end_idx = raw_response.rfind('}') + 1
            raw_response = raw_response[:end_idx]
        
        # Remove control characters
        cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', raw_response)
        
        return cleaned
    
    def _validate_enhancement_structure(self, data: Dict) -> bool:
        """Validate that the enhancement data has required structure"""
        try:
            # Check for required top-level keys
            required_keys = ['learning_metadata', 'content_text', 'user_experience']
            if not all(key in data for key in required_keys):
                return False
            
            # Check learning_metadata has essential fields
            learning_meta = data.get('learning_metadata', {})
            essential_fields = ['main_lesson', 'key_takeaways', 'actionable_insights', 'content_category']
            if not all(field in learning_meta for field in essential_fields):
                return False
            
            # Basic content validation
            if not data.get('content_text') or len(data.get('content_text', '')) < 10:
                return False
                
            return True
            
        except Exception:
            return False
    
    def enhance_single_file(self, raw_file_path: str, output_dir: str = None) -> Dict:
        """Production method to enhance a single file"""
        try:
            # Load raw file
            with open(raw_file_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            print(f"🔄 Enhancing: {raw_data.get('shortcode', 'unknown')}")
            
            # Enhance with Claude
            result = self.enhance_with_claude(raw_data)
            
            if 'error' in result:
                print(f"❌ Enhancement failed: {result['error']}")
                return result
            
            # Determine output path
            if output_dir is None:
                # Use the enhanced_processed directory structure
                # Get project root (where this script's parent directory is)
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                enhanced_dir = os.path.join(project_root, 'rag_system', 'enhanced_processed')
                
                # Maintain directory structure
                raw_dir_name = os.path.basename(os.path.dirname(raw_file_path))
                output_subdir = os.path.join(enhanced_dir, raw_dir_name)
            else:
                output_subdir = output_dir
            
            # Create output directory if needed
            os.makedirs(output_subdir, exist_ok=True)
            
            # Generate output filename
            shortcode = raw_data.get('shortcode', 'unknown')
            output_filename = f"enhanced_{shortcode}.json"
            output_path = os.path.join(output_subdir, output_filename)
            
            # Save enhanced file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result['result'], f, indent=2, ensure_ascii=False)
            
            print(f"✅ Enhanced file saved: {output_path}")
            print(f"💰 Cost: ${result['cost']:.4f} | ⏱️ Time: {result['processing_time']:.1f}s")
            
            return {
                'success': True,
                'input_file': raw_file_path,
                'output_file': output_path,
                'cost': result['cost'],
                'processing_time': result['processing_time'],
                'shortcode': shortcode
            }
            
        except Exception as e:
            error_msg = f"Failed to enhance {raw_file_path}: {str(e)}"
            print(f"❌ {error_msg}")
            return {'error': error_msg, 'success': False}
    
    def enhance_batch(self, file_list: List[str], max_concurrent: int = 3) -> Dict:
        """Production method to enhance multiple files with progress tracking"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results = {
            'successful': [],
            'failed': [],
            'total_cost': 0.0,
            'total_time': 0.0,
            'start_time': datetime.now()
        }
        
        print(f"🚀 Starting batch enhancement of {len(file_list)} files...")
        
        # Process files with limited concurrency
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self.enhance_single_file, file_path): file_path 
                for file_path in file_list
            }
            
            # Process results as they complete
            for i, future in enumerate(as_completed(future_to_file), 1):
                file_path = future_to_file[future]
                
                try:
                    result = future.result()
                    
                    if result.get('success'):
                        results['successful'].append(result)
                        results['total_cost'] += result.get('cost', 0)
                        results['total_time'] += result.get('processing_time', 0)
                    else:
                        results['failed'].append({'file': file_path, 'error': result.get('error', 'Unknown error')})
                    
                except Exception as e:
                    results['failed'].append({'file': file_path, 'error': str(e)})
                
                # Progress update
                print(f"📊 Progress: {i}/{len(file_list)} files processed")
        
        results['end_time'] = datetime.now()
        results['duration'] = (results['end_time'] - results['start_time']).total_seconds()
        
        # Print summary
        self._print_batch_summary(results)
        
        return results
    
    def _print_batch_summary(self, results: Dict):
        """Print batch processing summary"""
        print("\n" + "="*60)
        print("📋 BATCH ENHANCEMENT SUMMARY")
        print("="*60)
        print(f"✅ Successful: {len(results['successful'])}")
        print(f"❌ Failed: {len(results['failed'])}")
        print(f"💰 Total cost: ${results['total_cost']:.4f}")
        print(f"⏱️ Total processing time: {results['total_time']:.1f}s")
        print(f"🕐 Wall clock time: {results['duration']:.1f}s")
        
        if results['failed']:
            print(f"\n❌ Failed files:")
            for failure in results['failed']:
                print(f"  • {os.path.basename(failure['file'])}: {failure['error']}")
        
        print("="*60)

    def run_test(self, test_files: list) -> list:
        """Run quality comparison test (for backward compatibility)"""
        results = []
        
        for raw_file, enhanced_file in test_files:
            print(f"\\n🧪 Testing: {os.path.basename(raw_file)}")
            
            # Load files
            with open(raw_file, 'r') as f:
                raw_data = json.load(f)
            with open(enhanced_file, 'r') as f:
                original_enhanced = json.load(f)
            
            # Test GPT-4
            print("  ⏳ Testing GPT-4...")
            gpt4_result = self.enhance_with_gpt4(raw_data)
            
            # Test Claude 3.5 Sonnet
            print("  ⏳ Testing Claude 3.5 Sonnet...")
            claude_result = self.enhance_with_claude(raw_data)
            
            # Compare quality
            comparison = self.compare_quality(gpt4_result, claude_result, original_enhanced)
            
            # Print results
            self.print_comparison(raw_file, comparison)
            
            results.append({
                'file': os.path.basename(raw_file),
                'comparison': comparison,
                'gpt4_result': gpt4_result,
                'claude_result': claude_result
            })
            
        return results

    def compare_quality(self, gpt4_result: Dict, claude_result: Dict, original_enhanced: Dict) -> Dict:
        """Compare the quality of both models against the original enhanced file"""
        
        def count_fields(data: Dict) -> int:
            """Count non-empty fields in learning_metadata"""
            learning_meta = data.get('result', {}).get('learning_metadata', {})
            count = 0
            for key, value in learning_meta.items():
                if value and value != [] and value != "" and value != 0:
                    count += 1
            return count
            
        def check_key_fields(data: Dict) -> Dict:
            """Check presence of key fields"""
            learning_meta = data.get('result', {}).get('learning_metadata', {})
            return {
                'main_lesson': bool(learning_meta.get('main_lesson')),
                'key_takeaways': len(learning_meta.get('key_takeaways', [])),
                'actionable_insights': len(learning_meta.get('actionable_insights', [])),
                'technologies_mentioned': len(learning_meta.get('technologies_mentioned', [])),
                'natural_questions': len(learning_meta.get('natural_questions', [])),
                'search_scenarios': len(learning_meta.get('search_scenarios', [])),
                'related_topics': len(learning_meta.get('related_topics', []))
            }
        
        # Analyze original (GPT-4 enhanced file)
        orig_learning = original_enhanced.get('learning_metadata', {})
        orig_fields = {
            'main_lesson': bool(orig_learning.get('main_lesson')),
            'key_takeaways': len(orig_learning.get('key_takeaways', [])),
            'actionable_insights': len(orig_learning.get('actionable_insights', [])),
            'technologies_mentioned': len(orig_learning.get('technologies_mentioned', [])),
            'natural_questions': len(orig_learning.get('natural_questions', [])),
            'search_scenarios': len(orig_learning.get('search_scenarios', [])),
            'related_topics': len(orig_learning.get('related_topics', []))
        }
        
        return {
            'original_enhanced': {
                'field_count': len([v for v in orig_learning.values() if v and v != [] and v != "" and v != 0]),
                'key_fields': orig_fields
            },
            'gpt4_test': {
                'field_count': count_fields(gpt4_result),
                'key_fields': check_key_fields(gpt4_result),
                'cost': gpt4_result.get('cost', 0),
                'processing_time': gpt4_result.get('processing_time', 0)
            },
            'claude_test': {
                'field_count': count_fields(claude_result),
                'key_fields': check_key_fields(claude_result),
                'cost': claude_result.get('cost', 0),
                'processing_time': claude_result.get('processing_time', 0)
            }
        }

    def run_test(self, test_files: list):
        """Run the comparison test on specified files"""
        results = []
        
        for raw_file, enhanced_file in test_files:
            print(f"\\n🧪 Testing: {os.path.basename(raw_file)}")
            
            # Load files
            with open(raw_file, 'r') as f:
                raw_data = json.load(f)
            with open(enhanced_file, 'r') as f:
                original_enhanced = json.load(f)
            
            # Test GPT-4
            print("  ⏳ Testing GPT-4...")
            gpt4_result = self.enhance_with_gpt4(raw_data)
            
            # Test Claude 3.5 Sonnet
            print("  ⏳ Testing Claude 3.5 Sonnet...")
            claude_result = self.enhance_with_claude(raw_data)
            
            # Compare quality
            comparison = self.compare_quality(gpt4_result, claude_result, original_enhanced)
            
            # Print results
            self.print_comparison(raw_file, comparison)
            
            results.append({
                'file': os.path.basename(raw_file),
                'comparison': comparison,
                'gpt4_result': gpt4_result,
                'claude_result': claude_result
            })
            
        return results

    def print_comparison(self, filename: str, comparison: Dict):
        """Print comparison results"""
        print(f"\\n📊 RESULTS for {os.path.basename(filename)}:")
        print("=" * 60)
        
        orig = comparison['original_enhanced']
        gpt4 = comparison['gpt4_test'] 
        claude = comparison['claude_test']
        
        print(f"FIELD COMPLETENESS:")
        print(f"  Original Enhanced: {orig['field_count']} fields")
        print(f"  GPT-4 Test:       {gpt4['field_count']} fields")
        print(f"  Claude Test:      {claude['field_count']} fields")
        
        print(f"\\nKEY FIELD COMPARISON:")
        print(f"  Main Lesson:       Orig: {orig['key_fields']['main_lesson']} | GPT4: {gpt4['key_fields']['main_lesson']} | Claude: {claude['key_fields']['main_lesson']}")
        print(f"  Key Takeaways:     Orig: {orig['key_fields']['key_takeaways']} | GPT4: {gpt4['key_fields']['key_takeaways']} | Claude: {claude['key_fields']['key_takeaways']}")
        print(f"  Tech Mentioned:    Orig: {orig['key_fields']['technologies_mentioned']} | GPT4: {gpt4['key_fields']['technologies_mentioned']} | Claude: {claude['key_fields']['technologies_mentioned']}")
        print(f"  Natural Questions: Orig: {orig['key_fields']['natural_questions']} | GPT4: {gpt4['key_fields']['natural_questions']} | Claude: {claude['key_fields']['natural_questions']}")
        
        print(f"\\nCOST & PERFORMANCE:")
        print(f"  GPT-4:   ${gpt4['cost']:.4f} ({gpt4['processing_time']:.1f}s)")
        print(f"  Claude:  ${claude['cost']:.4f} ({claude['processing_time']:.1f}s)")
        print(f"  Savings: ${gpt4['cost'] - claude['cost']:.4f} ({((gpt4['cost'] - claude['cost']) / gpt4['cost']) * 100:.1f}%)")

def main():
    """Main production function with CLI interface"""
    import argparse
    from daily_file_detector import DailyFileDetector
    
    parser = argparse.ArgumentParser(description='Instagram Content Enhancement System')
    parser.add_argument('--test-quality', action='store_true', help='Run quality comparison test')
    parser.add_argument('--detect-files', action='store_true', help='Detect new files needing enhancement')
    parser.add_argument('--enhance-new', action='store_true', help='Enhance all newly detected files')
    parser.add_argument('--enhance-file', help='Enhance a specific file')
    parser.add_argument('--enhance-batch', nargs='+', help='Enhance specific files')
    parser.add_argument('--base-path', help='Base path to project directory')
    parser.add_argument('--max-concurrent', type=int, default=3, help='Max concurrent enhancements')
    
    args = parser.parse_args()
    
    enhancer = ProductionEnhancer()
    
    if args.test_quality:
        # Run quality test mode
        print("🧪 Running Quality Comparison Test...")
        test_files = [
            (
                "/mnt/c/Users/arjun/Desktop/Scraper/instagram_caption_scraper/output/edhonour_20250714_202318/DL6K4ZZNxeN.json",
                "/mnt/c/Users/arjun/Desktop/Scraper/instagram_caption_scraper/rag_system/enhanced_processed/edhonour_20250714_202318/enhanced_DL6K4ZZNxeN.json"
            )
        ]
        results = enhancer.run_test(test_files)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"enhancement_test_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to: {results_file}")
        
    elif args.detect_files:
        # Detect new files
        detector = DailyFileDetector(args.base_path)
        report = detector.run_detection()
        
        if report['files_to_process']:
            print(f"\\n🎯 Found {len(report['files_to_process'])} files ready for enhancement!")
            return 0
        else:
            print("\\n✨ No new files found.")
            return 1
            
    elif args.enhance_new:
        # Enhance all newly detected files
        detector = DailyFileDetector(args.base_path)
        report = detector.run_detection(verbose=False)
        
        if not report['files_to_process']:
            print("✨ No new files to enhance!")
            return 1
        
        file_paths = [file_info['file_path'] for file_info in report['files_to_process']]
        print(f"🚀 Enhancing {len(file_paths)} new files...")
        
        results = enhancer.enhance_batch(file_paths, args.max_concurrent)
        
        return 0 if len(results['failed']) == 0 else 1
        
    elif args.enhance_file:
        # Enhance single file
        result = enhancer.enhance_single_file(args.enhance_file)
        return 0 if result.get('success') else 1
        
    elif args.enhance_batch:
        # Enhance specific files
        results = enhancer.enhance_batch(args.enhance_batch, args.max_concurrent)
        return 0 if len(results['failed']) == 0 else 1
        
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    exit(main())