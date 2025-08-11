"""
Debug Helper for Instagram Caption Scraper

This module provides comprehensive debugging functionality to identify
why caption extraction is failing, including HTML inspection, screenshots,
and detailed logging.
"""

import logging
import os
import time
import re
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException


class DebugHelper:
    """Comprehensive debugging helper for Instagram scraper."""
    
    # Updated Instagram selectors for 2025
    INSTAGRAM_SELECTORS = {
        'caption_containers': [
            # Post description selectors
            'article div[data-testid="post-content"]',
            'article div[role="button"] + div',
            'div[data-testid="post-content"] span',
            'article span[dir="auto"]',
            'article div h1 + div span',
            'div[data-testid="caption"] span',
            'article header + div span',
            'article div[style*="word-wrap"] span',
            
            # Legacy selectors
            'article div.-webkit-box span',
            'article div.C4VMK span',
            'article div.C7I1f span',
            'div.ZyFrc span',
            'div.C4VMK span',
        ],
        
        'video_elements': [
            'video',
            'div[role="button"] video',
            'article video',
            'div[data-testid="video-player"] video',
            'div[data-testid="media-container"] video',
        ],
        
        'text_elements': [
            'span', 'div', 'p', 'h1', 'h2', 'h3',
            'article span', 'article div', 'article p',
            '[data-testid*="caption"]', '[data-testid*="text"]',
            '[aria-label*="caption"]', '[aria-label*="text"]',
        ],
        
        'challenge_indicators': [
            'div[data-testid="challenge"]',
            'div[data-testid="suspicious-login"]',
            'form[data-testid="challenge-form"]',
            'div[data-testid="checkpoint-challenge"]',
            'div[role="dialog"][aria-label*="challenge"]',
            'div[role="dialog"][aria-label*="verify"]',
        ],
        
        'login_indicators': [
            'form[data-testid="royal_login_form"]',
            'input[name="username"]',
            'input[aria-label="Phone number, username, or email"]',
            'button[type="submit"]',
            'div[data-testid="login-page"]',
        ]
    }
    
    def __init__(self, driver: webdriver, logger: logging.Logger, debug_dir: str = "debug_output"):
        """Initialize debug helper."""
        self.driver = driver
        self.logger = logger
        self.debug_dir = Path(debug_dir)
        self.debug_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.debug_dir / "html").mkdir(exist_ok=True)
        (self.debug_dir / "screenshots").mkdir(exist_ok=True)
        (self.debug_dir / "logs").mkdir(exist_ok=True)
        
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def comprehensive_page_debug(self, url: str, save_html: bool = True, 
                                take_screenshot: bool = True) -> Dict[str, Any]:
        """Perform comprehensive debugging of the current page state."""
        debug_info = {
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id,
            'url': url,
            'debug_actions': []
        }
        
        try:
            # 1. Basic page information
            page_info = self._analyze_page_state()
            debug_info['page_info'] = page_info
            
            # 2. Save HTML source
            if save_html:
                html_file = self._save_page_source(url)
                debug_info['html_file'] = html_file
                debug_info['debug_actions'].append('html_saved')
            
            # 3. Take screenshot
            if take_screenshot:
                screenshot_file = self._take_debug_screenshot(url)
                debug_info['screenshot_file'] = screenshot_file
                debug_info['debug_actions'].append('screenshot_taken')
            
            # 4. Analyze page content
            content_analysis = self._analyze_page_content()
            debug_info['content_analysis'] = content_analysis
            
            # 5. Test selectors
            selector_results = self._test_all_selectors()
            debug_info['selector_results'] = selector_results
            
            # 6. Check for challenges/blocks
            challenge_check = self._check_for_challenges()
            debug_info['challenge_check'] = challenge_check
            
            # 7. Video analysis if present
            video_analysis = self._analyze_video_elements()
            debug_info['video_analysis'] = video_analysis
            
            # 8. Text extraction attempts
            text_extraction = self._attempt_text_extraction()
            debug_info['text_extraction'] = text_extraction
            
            # Save debug summary
            self._save_debug_summary(debug_info)
            
            return debug_info
            
        except Exception as e:
            self.logger.error(f"Debug analysis error: {str(e)}")
            debug_info['error'] = str(e)
            return debug_info
    
    def _analyze_page_state(self) -> Dict[str, Any]:
        """Analyze basic page state information."""
        try:
            current_url = self.driver.current_url
            title = self.driver.title
            
            # Check if we're on Instagram
            is_instagram = 'instagram.com' in current_url.lower()
            
            # Check for redirects
            is_redirect = '/accounts/login' in current_url or '/challenge' in current_url
            
            # Get page dimensions
            window_size = self.driver.get_window_size()
            
            # Check JavaScript errors
            js_errors = []
            try:
                logs = self.driver.get_log('browser')
                js_errors = [log for log in logs if log['level'] == 'SEVERE']
            except Exception:
                pass
            
            page_info = {
                'current_url': current_url,
                'title': title,
                'is_instagram': is_instagram,
                'is_redirect': is_redirect,
                'window_size': window_size,
                'js_errors_count': len(js_errors),
                'js_errors': js_errors[:5] if js_errors else []  # First 5 errors
            }
            
            self.logger.info(f"Page Analysis: {current_url} | Title: '{title}' | Instagram: {is_instagram}")
            
            return page_info
            
        except Exception as e:
            self.logger.error(f"Page state analysis error: {str(e)}")
            return {'error': str(e)}
    
    def _save_page_source(self, url: str) -> str:
        """Save the complete HTML source of the page."""
        try:
            html_content = self.driver.page_source
            
            # Create filename
            url_part = re.sub(r'[^a-zA-Z0-9]', '_', url.split('/')[-2] if '/' in url else 'page')
            filename = f"{self.session_id}_{url_part}_source.html"
            filepath = self.debug_dir / "html" / filename
            
            # Save HTML
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            # Save a snippet for quick inspection
            snippet_file = filepath.with_suffix('.snippet.html')
            snippet = html_content[:5000] + "\n\n... [Content truncated] ...\n\n" + html_content[-2000:]
            with open(snippet_file, 'w', encoding='utf-8') as f:
                f.write(snippet)
            
            self.logger.info(f"HTML source saved: {filepath}")
            self.logger.info(f"HTML snippet saved: {snippet_file}")
            
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"HTML save error: {str(e)}")
            return ""
    
    def _take_debug_screenshot(self, url: str) -> str:
        """Take a screenshot of the current page."""
        try:
            # Create filename
            url_part = re.sub(r'[^a-zA-Z0-9]', '_', url.split('/')[-2] if '/' in url else 'page')
            filename = f"{self.session_id}_{url_part}_screenshot.png"
            filepath = self.debug_dir / "screenshots" / filename
            
            # Take screenshot
            self.driver.save_screenshot(str(filepath))
            
            self.logger.info(f"Screenshot saved: {filepath}")
            
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Screenshot error: {str(e)}")
            return ""
    
    def _analyze_page_content(self) -> Dict[str, Any]:
        """Analyze the content structure of the page."""
        try:
            # Count different element types
            element_counts = {}
            for tag in ['div', 'span', 'article', 'section', 'video', 'img', 'button', 'a']:
                try:
                    count = len(self.driver.find_elements(By.TAG_NAME, tag))
                    element_counts[tag] = count
                except Exception:
                    element_counts[tag] = 0
            
            # Find all text-containing elements
            text_elements = []
            try:
                all_elements = self.driver.find_elements(By.XPATH, "//*[text()]")
                for elem in all_elements[:5]:  # Reduced from 20 to 5 for speed
                    try:
                        text = elem.text.strip()
                        if text and len(text) > 3:
                            text_elements.append({
                                'tag': elem.tag_name,
                                'text': text[:100],  # First 100 chars
                                'class': elem.get_attribute('class') or '',
                                'id': elem.get_attribute('id') or ''
                            })
                    except Exception:
                        continue
            except Exception as e:
                self.logger.debug(f"Text element analysis error: {str(e)}")
            
            # Check for common Instagram indicators
            instagram_indicators = {
                'has_article_tag': len(self.driver.find_elements(By.TAG_NAME, 'article')) > 0,
                'has_data_testid': len(self.driver.find_elements(By.CSS_SELECTOR, '[data-testid]')) > 0,
                'has_instagram_class': len(self.driver.find_elements(By.CSS_SELECTOR, '[class*="instagram"]')) > 0,
                'has_react_root': len(self.driver.find_elements(By.ID, 'react-root')) > 0,
            }
            
            content_analysis = {
                'element_counts': element_counts,
                'text_elements_found': len(text_elements),
                'text_elements_sample': text_elements[:5],  # First 5 for inspection
                'instagram_indicators': instagram_indicators,
                'total_elements': len(self.driver.find_elements(By.XPATH, "//*"))
            }
            
            self.logger.info(f"Content Analysis: {element_counts}")
            self.logger.info(f"Text elements found: {len(text_elements)}")
            
            return content_analysis
            
        except Exception as e:
            self.logger.error(f"Content analysis error: {str(e)}")
            return {'error': str(e)}
    
    def _test_all_selectors(self) -> Dict[str, Any]:
        """Test all known Instagram selectors."""
        selector_results = {}
        
        for category, selectors in self.INSTAGRAM_SELECTORS.items():
            category_results = {}
            
            for i, selector in enumerate(selectors):
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    
                    result = {
                        'found_count': len(elements),
                        'elements': []
                    }
                    
                    # Get details for first element only (reduced from 3)
                    for elem in elements[:1]:
                        try:
                            elem_info = {
                                'tag': elem.tag_name,
                                'text': elem.text.strip()[:100] if elem.text else '',
                                'visible': elem.is_displayed(),
                                'class': elem.get_attribute('class') or '',
                                'id': elem.get_attribute('id') or ''
                            }
                            result['elements'].append(elem_info)
                        except Exception:
                            continue
                    
                    category_results[f"selector_{i}"] = {
                        'selector': selector,
                        'result': result
                    }
                    
                    if len(elements) > 0:
                        self.logger.info(f"✅ Selector found elements: {selector} ({len(elements)} elements)")
                    else:
                        self.logger.debug(f"❌ Selector found nothing: {selector}")
                        
                except Exception as e:
                    category_results[f"selector_{i}"] = {
                        'selector': selector,
                        'error': str(e)
                    }
                    self.logger.debug(f"Selector error: {selector} - {str(e)}")
            
            selector_results[category] = category_results
        
        return selector_results
    
    def _check_for_challenges(self) -> Dict[str, Any]:
        """Check if we're being challenged or blocked by Instagram."""
        challenge_info = {
            'has_challenge': False,
            'has_login_prompt': False,
            'challenge_type': None,
            'indicators_found': []
        }
        
        try:
            # Check for challenge indicators
            for selector in self.INSTAGRAM_SELECTORS['challenge_indicators']:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        challenge_info['has_challenge'] = True
                        challenge_info['indicators_found'].append(selector)
                        self.logger.warning(f"🚨 Challenge indicator found: {selector}")
                except Exception:
                    continue
            
            # Check for login prompts
            for selector in self.INSTAGRAM_SELECTORS['login_indicators']:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        challenge_info['has_login_prompt'] = True
                        challenge_info['indicators_found'].append(selector)
                        self.logger.warning(f"🔐 Login prompt found: {selector}")
                except Exception:
                    continue
            
            # Check URL for challenges
            current_url = self.driver.current_url
            if '/challenge' in current_url:
                challenge_info['has_challenge'] = True
                challenge_info['challenge_type'] = 'url_challenge'
            elif '/accounts/login' in current_url:
                challenge_info['has_login_prompt'] = True
                challenge_info['challenge_type'] = 'login_required'
            
            # Check page title for blocks
            title = self.driver.title.lower()
            if any(word in title for word in ['login', 'challenge', 'verify', 'suspicious']):
                challenge_info['has_challenge'] = True
                challenge_info['challenge_type'] = 'title_indicator'
            
            return challenge_info
            
        except Exception as e:
            self.logger.error(f"Challenge check error: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_video_elements(self) -> Dict[str, Any]:
        """Analyze video elements on the page."""
        video_info = {
            'video_count': 0,
            'videos': [],
            'has_playable_video': False
        }
        
        try:
            for selector in self.INSTAGRAM_SELECTORS['video_elements']:
                try:
                    videos = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    for video in videos:
                        try:
                            video_data = {
                                'selector': selector,
                                'visible': video.is_displayed(),
                                'paused': self.driver.execute_script("return arguments[0].paused;", video),
                                'duration': self.driver.execute_script("return arguments[0].duration;", video),
                                'current_time': self.driver.execute_script("return arguments[0].currentTime;", video),
                                'has_controls': video.get_attribute('controls') is not None,
                                'src': video.get_attribute('src') or 'No src'
                            }
                            video_info['videos'].append(video_data)
                            
                            if video.is_displayed():
                                video_info['has_playable_video'] = True
                                
                        except Exception as e:
                            video_info['videos'].append({
                                'selector': selector,
                                'error': str(e)
                            })
                except Exception:
                    continue
            
            video_info['video_count'] = len(video_info['videos'])
            
            if video_info['video_count'] > 0:
                self.logger.info(f"📹 Found {video_info['video_count']} video elements")
            
            return video_info
            
        except Exception as e:
            self.logger.error(f"Video analysis error: {str(e)}")
            return {'error': str(e)}
    
    def _attempt_text_extraction(self) -> Dict[str, Any]:
        """Attempt various text extraction methods."""
        extraction_results = {
            'methods_tried': [],
            'text_found': [],
            'best_candidates': []
        }
        
        try:
            # Method 1: Try caption container selectors
            for selector in self.INSTAGRAM_SELECTORS['caption_containers']:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    for elem in elements:
                        text = elem.text.strip()
                        if text and len(text) > 10:  # Reasonable caption length
                            extraction_results['text_found'].append({
                                'method': f'caption_selector: {selector}',
                                'text': text[:200],  # First 200 chars
                                'length': len(text),
                                'element_tag': elem.tag_name
                            })
                            
                            # Score as best candidate if long enough
                            if len(text) > 50:
                                extraction_results['best_candidates'].append({
                                    'text': text,
                                    'method': selector,
                                    'confidence': 'high' if len(text) > 100 else 'medium'
                                })
                                
                    extraction_results['methods_tried'].append(f'caption_selector: {selector}')
                except Exception:
                    continue
            
            # Method 2: General text element search
            try:
                all_text_elements = self.driver.find_elements(By.XPATH, "//span[string-length(text()) > 20] | //div[string-length(text()) > 20]")
                for elem in all_text_elements[:10]:  # First 10 elements
                    text = elem.text.strip()
                    if text and not self._is_ui_text(text):
                        extraction_results['text_found'].append({
                            'method': 'general_text_search',
                            'text': text[:200],
                            'length': len(text),
                            'element_tag': elem.tag_name
                        })
                        
                extraction_results['methods_tried'].append('general_text_search')
            except Exception:
                pass
            
            # Method 3: Article-specific search
            try:
                articles = self.driver.find_elements(By.TAG_NAME, 'article')
                for article in articles:
                    text = article.text.strip()
                    if text:
                        # Split into lines and look for caption-like content
                        lines = [line.strip() for line in text.split('\\n') if line.strip()]
                        for line in lines:
                            if len(line) > 20 and not self._is_ui_text(line):
                                extraction_results['text_found'].append({
                                    'method': 'article_text_search',
                                    'text': line[:200],
                                    'length': len(line),
                                    'element_tag': 'article'
                                })
                                
                extraction_results['methods_tried'].append('article_text_search')
            except Exception:
                pass
            
            return extraction_results
            
        except Exception as e:
            self.logger.error(f"Text extraction error: {str(e)}")
            return {'error': str(e)}
    
    def _is_ui_text(self, text: str) -> bool:
        """Check if text is likely UI element rather than content."""
        ui_keywords = [
            'like', 'comment', 'share', 'follow', 'view', 'play', 'pause',
            'instagram', 'stories', 'reels', 'igtv', 'more', 'show more',
            'tap to', 'swipe', 'click', 'press', 'settings', 'options'
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in ui_keywords) or len(text) < 5
    
    def _save_debug_summary(self, debug_info: Dict[str, Any]) -> str:
        """Save a comprehensive debug summary."""
        try:
            filename = f"{self.session_id}_debug_summary.json"
            filepath = self.debug_dir / "logs" / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(debug_info, f, indent=2, default=str)
            
            # Also create a human-readable summary
            summary_file = filepath.with_suffix('.txt')
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(f"Instagram Scraper Debug Summary\\n")
                f.write(f"{'='*50}\\n")
                f.write(f"Session ID: {self.session_id}\\n")
                f.write(f"Timestamp: {debug_info.get('timestamp', 'unknown')}\\n")
                f.write(f"URL: {debug_info.get('url', 'unknown')}\\n\\n")
                
                # Page info
                page_info = debug_info.get('page_info', {})
                f.write(f"Page Information:\\n")
                f.write(f"  Current URL: {page_info.get('current_url', 'unknown')}\\n")
                f.write(f"  Title: {page_info.get('title', 'unknown')}\\n")
                f.write(f"  Is Instagram: {page_info.get('is_instagram', False)}\\n")
                f.write(f"  Is Redirect: {page_info.get('is_redirect', False)}\\n\\n")
                
                # Challenge check
                challenge_info = debug_info.get('challenge_check', {})
                f.write(f"Challenge/Block Check:\\n")
                f.write(f"  Has Challenge: {challenge_info.get('has_challenge', False)}\\n")
                f.write(f"  Has Login Prompt: {challenge_info.get('has_login_prompt', False)}\\n")
                f.write(f"  Challenge Type: {challenge_info.get('challenge_type', 'none')}\\n\\n")
                
                # Text extraction results
                text_results = debug_info.get('text_extraction', {})
                f.write(f"Text Extraction Results:\\n")
                f.write(f"  Methods Tried: {len(text_results.get('methods_tried', []))}\\n")
                f.write(f"  Text Found: {len(text_results.get('text_found', []))}\\n")
                f.write(f"  Best Candidates: {len(text_results.get('best_candidates', []))}\\n")
                
                best_candidates = text_results.get('best_candidates', [])
                if best_candidates:
                    f.write(f"\\n  Best Caption Candidates:\\n")
                    for i, candidate in enumerate(best_candidates[:3], 1):
                        f.write(f"    {i}. {candidate.get('text', '')[:100]}...\\n")
            
            self.logger.info(f"Debug summary saved: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Debug summary save error: {str(e)}")
            return ""
    
    def log_all_found_text(self, min_length: int = 10) -> List[str]:
        """Log all text found on the page for manual inspection."""
        found_texts = []
        
        try:
            self.logger.info("🔍 MANUAL INSPECTION - All Text Elements Found:")
            self.logger.info("=" * 60)
            
            # Get all elements with text
            elements = self.driver.find_elements(By.XPATH, f"//text()[string-length(normalize-space(.)) >= {min_length}]/..")
            
            for i, elem in enumerate(elements[:20], 1):  # First 20 elements
                try:
                    text = elem.text.strip()
                    if text and len(text) >= min_length:
                        tag = elem.tag_name
                        classes = elem.get_attribute('class') or 'no-class'
                        
                        self.logger.info(f"{i:2d}. [{tag}] {classes[:30]:<30} | {text[:100]}")
                        found_texts.append(text)
                        
                except Exception:
                    continue
            
            self.logger.info("=" * 60)
            self.logger.info(f"Total text elements logged: {len(found_texts)}")
            
            return found_texts
            
        except Exception as e:
            self.logger.error(f"Text logging error: {str(e)}")
            return []