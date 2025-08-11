"""
Video Transcription Module for Instagram Videos

This module handles downloading Instagram videos and extracting captions
using speech-to-text transcription as the primary method.
"""

import os
import re
import tempfile
import time
import logging
import requests
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from urllib.parse import urlparse, parse_qs
import json

try:
    import whisper
    import torch
    WHISPER_AVAILABLE = True
    WHISPER_IMPORT_ERROR = None
    
    # TorchAudio is optional - Whisper can work without it
    try:
        import torchaudio
        TORCHAUDIO_AVAILABLE = True
    except Exception as e:
        TORCHAUDIO_AVAILABLE = False
        TORCHAUDIO_ERROR = str(e)
        # Don't fail Whisper import if only torchaudio has issues
        
except ImportError as e:
    WHISPER_AVAILABLE = False
    WHISPER_IMPORT_ERROR = str(e)
    TORCHAUDIO_AVAILABLE = False
    TORCHAUDIO_ERROR = "Whisper not available"

try:
    from moviepy.editor import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError as e:
    MOVIEPY_AVAILABLE = False
    MOVIEPY_IMPORT_ERROR = str(e)

try:
    import ffmpeg
    FFMPEG_PYTHON_AVAILABLE = True
except ImportError as e:
    FFMPEG_PYTHON_AVAILABLE = False
    FFMPEG_PYTHON_IMPORT_ERROR = str(e)


class VideoTranscriber:
    """Main class for downloading videos and extracting captions via transcription."""
    
    def __init__(self, driver, logger: logging.Logger, temp_dir: str = "temp_videos", force_cpu: bool = False):
        """Initialize the video transcriber."""
        self.driver = driver
        self.logger = logger
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        self.force_cpu = force_cpu
        
        # Initialize Whisper model (large-v3 for best accuracy with GPU acceleration)
        self.whisper_model = None
        self.whisper_model_loaded = False
        
        if WHISPER_AVAILABLE:
            try:
                self.logger.info("Loading Whisper model (this may take a moment on first run)...")
                
                # Load model with error handling for torchaudio issues
                if not TORCHAUDIO_AVAILABLE:
                    self.logger.warning("⚠️  TorchAudio not available - using basic audio processing")
                
                # Use large-v3 for best accuracy with multiple GPUs
                # Check GPU memory and select best device
                if self.force_cpu:
                    device = "cpu"
                    self.logger.info("🖥️ Forcing CPU usage as requested")
                else:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    if torch.cuda.is_available():
                        gpu_count = torch.cuda.device_count()
                        self.logger.info(f"🚀 Detected {gpu_count} GPU(s) - using GPU acceleration")
                
                self.whisper_model = whisper.load_model("large-v3", device=device)
                self.whisper_model_loaded = True
                self.logger.info("✅ Whisper model loaded successfully")
                
                if not TORCHAUDIO_AVAILABLE:
                    self.logger.info("💡 Note: Using Whisper without TorchAudio (basic functionality)")
                    
            except Exception as e:
                self.logger.error(f"❌ Failed to load Whisper model: {str(e)}")
                if "torchaudio" in str(e).lower():
                    self.logger.info("💡 TorchAudio compatibility issue detected")
                    self.logger.info("💡 Try: pip uninstall torch torchaudio && pip install torch==2.1.0 torchaudio==2.1.0")
                else:
                    self.logger.info("💡 Try: pip install openai-whisper torch")
        else:
            self.logger.warning("❌ Whisper not available. Install with: pip install openai-whisper torch")
            if WHISPER_IMPORT_ERROR:
                self.logger.debug(f"Whisper import error: {WHISPER_IMPORT_ERROR}")
        
        self.session = requests.Session()
        self._setup_session()
    
    def _setup_session(self):
        """Setup requests session with proper headers."""
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
    
    def extract_video_url(self, instagram_url: str) -> Optional[str]:
        """Extract the direct video URL using yt-dlp (primary method)."""
        try:
            self.logger.info("🎥 Extracting video URL using yt-dlp...")
            
            # Primary method: yt-dlp (most reliable)
            video_url = self._extract_with_ytdlp(instagram_url)
            if video_url:
                self.logger.info("✅ Successfully extracted video URL with yt-dlp")
                return video_url
            
            # Fallback method: Simple DOM-based extraction (lightweight)
            self.logger.info("🔄 Trying fallback DOM extraction...")
            video_url = self._extract_video_from_elements_simple()
            if video_url:
                self.logger.info("✅ Successfully extracted video URL with DOM fallback")
                return video_url

            self.logger.warning("❌ No video URL found with any extraction method")
            return None
                
        except Exception as e:
            self.logger.error(f"Error extracting video URL: {str(e)}")
            return None
    
    def _extract_video_from_elements(self) -> Optional[str]:
        """Extract video URL from HTML video elements."""
        try:
            # Enhanced video selectors for logged-in Instagram
            video_selectors = [
                'video[src]',
                'video',
                'div[data-testid="video-player"] video',
                'article video',
                'div[role="button"] video',
                'div[data-testid="video"] video',
                'video source[src]',
                'source[type*="video"]'
            ]
            
            self.logger.debug(f"🔍 Looking for video elements with {len(video_selectors)} selectors...")
            
            for selector in video_selectors:
                try:
                    elements = self.driver.find_elements("css selector", selector)
                    self.logger.debug(f"🔍 Selector '{selector}' found {len(elements)} elements")
                    
                    for i, element in enumerate(elements):
                        # Check if element is visible
                        is_displayed = element.is_displayed()
                        self.logger.debug(f"  Element {i+1}: displayed={is_displayed}")
                        
                        if not is_displayed:
                            continue
                        
                        # Try direct src attribute
                        src = element.get_attribute('src')
                        self.logger.debug(f"  Element {i+1} src: {src[:100] if src else 'None'}...")
                        
                        if src and self._is_valid_video_url(src):
                            self.logger.info(f"✅ Found video URL via {selector}: {src[:100]}...")
                            return src
                        
                        # Try source child elements
                        sources = element.find_elements("css selector", "source")
                        self.logger.debug(f"  Element {i+1}: found {len(sources)} source children")
                        
                        for j, source in enumerate(sources):
                            src = source.get_attribute('src')
                            self.logger.debug(f"    Source {j+1} src: {src[:100] if src else 'None'}...")
                            
                            if src and self._is_valid_video_url(src):
                                self.logger.info(f"✅ Found video URL via source in {selector}: {src[:100]}...")
                                return src
                                
                except Exception as e:
                    self.logger.debug(f"Error with selector {selector}: {str(e)}")
                    continue
            
            self.logger.warning("❌ No video elements found with valid src attributes")
            return None
            
        except Exception as e:
            self.logger.debug(f"Element extraction error: {str(e)}")
            return None
    
    def _is_valid_video_url(self, url: str) -> bool:
        """Check if URL is a valid video URL."""
        if not url or not url.startswith('http'):
            return False
        
        # Instagram video URL patterns
        valid_patterns = [
            'scontent',  # Instagram CDN
            'fbcdn',     # Facebook CDN
            '.mp4',      # MP4 files
            '.m4v',      # M4V files
            'video'      # Contains video keyword
        ]
        
        url_lower = url.lower()
        return any(pattern in url_lower for pattern in valid_patterns)
    
    def _extract_video_from_network(self) -> Optional[str]:
        """Extract video URL from browser network logs."""
        try:
            # Get browser logs to find network requests
            logs = self.driver.get_log('performance')
            
            for log in logs:
                try:
                    log_message = json.loads(log['message'])
                    if log_message.get('message', {}).get('method') == 'Network.responseReceived':
                        response = log_message['message']['params']['response']
                        url = response.get('url', '')
                        
                        if self._is_valid_video_url(url):
                            self.logger.info(f"✅ Found video URL from network logs: {url[:100]}...")
                            return url
                            
                except Exception:
                    continue
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Network extraction error: {str(e)}")
            return None
    
    def _download_with_api_fallback(self, instagram_url: str) -> Optional[str]:
        """Extract video URL using API methods (pyigdl, yt-dlp, etc.)"""
        try:
            # Try multiple API methods in order of speed/reliability
            api_methods = [
                ("pyigdl", self._extract_with_pyigdl),
                ("yt-dlp", self._extract_with_ytdlp),
                ("instaloader", self._extract_with_instaloader)
            ]
            
            for method_name, method_func in api_methods:
                try:
                    self.logger.debug(f"🔄 Trying {method_name} API...")
                    video_url = method_func(instagram_url)
                    if video_url:
                        self.logger.info(f"✅ Got video URL from {method_name}")
                        return video_url
                except Exception as e:
                    self.logger.debug(f"❌ {method_name} failed: {e}")
                    continue
            
            return None
            
        except Exception as e:
            self.logger.debug(f"API extraction error: {str(e)}")
            return None
    
    def _extract_with_pyigdl(self, instagram_url: str) -> Optional[str]:
        """Extract video URL using pyigdl API"""
        try:
            from pyigdl import IGDownloader
            
            data = IGDownloader(instagram_url)
            if data and len(data) > 0:
                download_url = data[0].get("download_link")
                if download_url:
                    self.logger.debug(f"pyigdl returned URL: {download_url[:100]}...")
                    return download_url
            
            return None
            
        except ImportError:
            self.logger.debug("pyigdl not installed, skipping...")
            return None
        except Exception as e:
            self.logger.debug(f"pyigdl error: {e}")
            return None
    
    def _extract_with_ytdlp(self, instagram_url: str) -> Optional[str]:
        """Extract video URL using yt-dlp, prioritizing combined video+audio formats"""
        try:
            import subprocess
            import json
            
            # First, try to get download URL with yt-dlp's smart format selection
            download_url = self._get_ytdlp_download_url(instagram_url)
            if download_url:
                return download_url
            
            # Fallback: manual format analysis
            cmd = [
                "yt-dlp",
                "--dump-json",
                "--quiet",
                instagram_url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                info = json.loads(result.stdout)
                
                # Strategy 1: Look for combined video+audio formats first
                if 'formats' in info and info['formats']:
                    combined_formats = []
                    video_only_formats = []
                    
                    for format_info in info['formats']:
                        if not format_info.get('url'):
                            continue
                        
                        vcodec = format_info.get('vcodec', 'none')
                        acodec = format_info.get('acodec', 'none')
                        
                        # Combined format: has both video and audio
                        if vcodec != 'none' and acodec != 'none':
                            combined_formats.append(format_info)
                        # Video-only format: backup option
                        elif vcodec != 'none' and acodec == 'none':
                            video_only_formats.append(format_info)
                    
                    # Prioritize combined formats
                    if combined_formats:
                        # Sort by quality (height, then bitrate)
                        combined_formats.sort(key=lambda x: (
                            x.get('height', 0),
                            x.get('tbr', 0)
                        ), reverse=True)
                        
                        best_format = combined_formats[0]
                        video_url = best_format['url']
                        resolution = f"{best_format.get('width', '?')}x{best_format.get('height', '?')}"
                        format_id = best_format.get('format_id', 'unknown')
                        
                        self.logger.info(f"✅ yt-dlp found combined video+audio format ({format_id}, {resolution}): {video_url[:100]}...")
                        return video_url
                    
                    # Fallback: use video-only format (will need separate audio handling)
                    elif video_only_formats:
                        video_only_formats.sort(key=lambda x: (
                            x.get('height', 0),
                            x.get('tbr', 0)
                        ), reverse=True)
                        
                        best_format = video_only_formats[0]
                        video_url = best_format['url']
                        resolution = f"{best_format.get('width', '?')}x{best_format.get('height', '?')}"
                        format_id = best_format.get('format_id', 'unknown')
                        
                        self.logger.warning(f"⚠️ yt-dlp found video-only format ({format_id}, {resolution}): {video_url[:100]}...")
                        self.logger.warning("⚠️ This format has no audio track - audio extraction may fail")
                        return video_url
                
                # Strategy 2: Check requested_formats as fallback
                if 'requested_formats' in info and info['requested_formats']:
                    self.logger.debug("Checking requested_formats for combined streams...")
                    for format_info in info['requested_formats']:
                        vcodec = format_info.get('vcodec', 'none')
                        acodec = format_info.get('acodec', 'none')
                        
                        # Look for combined format
                        if vcodec != 'none' and acodec != 'none' and format_info.get('url'):
                            video_url = format_info['url']
                            format_id = format_info.get('format_id', 'unknown')
                            self.logger.info(f"✅ yt-dlp found combined format from requested_formats ({format_id}): {video_url[:100]}...")
                            return video_url
                
                # Strategy 3: Check if there's a direct URL (single format)
                if 'url' in info:
                    video_url = info['url']
                    self.logger.info(f"✅ yt-dlp found direct video URL: {video_url[:100]}...")
                    return video_url
                
                self.logger.warning("yt-dlp: No suitable video formats found in response")
            else:
                self.logger.error(f"yt-dlp failed with return code {result.returncode}")
                if result.stderr:
                    self.logger.error(f"yt-dlp stderr: {result.stderr}")
            
            return None
            
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.debug(f"yt-dlp error: {e}")
            return None
    
    def _get_ytdlp_download_url(self, instagram_url: str) -> Optional[str]:
        """Get download URL using yt-dlp's smart format selection."""
        try:
            import subprocess
            
            # Use yt-dlp's format selection to get best video+audio format
            cmd = [
                "yt-dlp",
                "--get-url",
                "--format", "best[height<=1080][ext=mp4]/best[ext=mp4]/best",
                "--quiet",
                instagram_url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and result.stdout.strip():
                video_url = result.stdout.strip()
                self.logger.info(f"✅ yt-dlp --get-url found video URL: {video_url[:100]}...")
                return video_url
            else:
                self.logger.debug("yt-dlp --get-url failed or returned empty result")
                return None
                
        except Exception as e:
            self.logger.debug(f"yt-dlp --get-url error: {e}")
            return None
    
    def _extract_video_from_elements_simple(self) -> Optional[str]:
        """Simple DOM-based video extraction without complex analysis."""
        try:
            self.logger.debug("🔍 Attempting simple DOM video extraction...")
            
            # Basic video selectors (no complex analysis)
            video_selectors = [
                'video[src]',
                'video source[src]',
                'video'
            ]
            
            for selector in video_selectors:
                try:
                    elements = self.driver.find_elements("css selector", selector)
                    self.logger.debug(f"Selector '{selector}' found {len(elements)} elements")
                    
                    for element in elements:
                        # Get src attribute
                        src = element.get_attribute('src')
                        if src and src.startswith('http') and any(ext in src.lower() for ext in ['.mp4', '.m4v', 'video']):
                            self.logger.info(f"✅ Found video URL via simple DOM: {src[:100]}...")
                            return src
                        
                        # Check source child elements
                        sources = element.find_elements("css selector", "source")
                        for source in sources:
                            src = source.get_attribute('src')
                            if src and src.startswith('http') and any(ext in src.lower() for ext in ['.mp4', '.m4v', 'video']):
                                self.logger.info(f"✅ Found video URL via source: {src[:100]}...")
                                return src
                                
                except Exception as e:
                    self.logger.debug(f"Error with selector {selector}: {str(e)}")
                    continue
            
            self.logger.warning("❌ No video URLs found via simple DOM extraction")
            return None
            
        except Exception as e:
            self.logger.debug(f"Simple DOM extraction error: {str(e)}")
            return None
    
    def _extract_with_instaloader(self, instagram_url: str) -> Optional[str]:
        """Extract video URL using instaloader"""
        try:
            import instaloader
            
            # Extract shortcode from URL
            if '/reel/' in instagram_url:
                shortcode = instagram_url.split('/reel/')[-1].rstrip('/')
            elif '/p/' in instagram_url:
                shortcode = instagram_url.split('/p/')[-1].rstrip('/')
            else:
                return None
            
            L = instaloader.Instaloader()
            post = instaloader.Post.from_shortcode(L.context, shortcode)
            
            if post.is_video and post.video_url:
                self.logger.debug(f"instaloader returned URL: {post.video_url[:100]}...")
                return post.video_url
            
            return None
            
        except ImportError:
            self.logger.debug("instaloader not installed, skipping...")
            return None
        except Exception as e:
            self.logger.debug(f"instaloader error: {e}")
            return None
    
    def _extract_video_url_javascript(self) -> Optional[str]:
        """Try to extract video URL using JavaScript."""
        try:
            self.logger.debug("🔍 Attempting JavaScript video URL extraction...")
            
            # JavaScript to find video elements and their sources
            js_script = """
            var videos = document.querySelectorAll('video');
            var result = {
                video_count: videos.length,
                urls: [],
                video_info: []
            };
            
            for (var i = 0; i < videos.length; i++) {
                var video = videos[i];
                var info = {
                    index: i,
                    src: video.src || null,
                    currentSrc: video.currentSrc || null,
                    style_display: window.getComputedStyle(video).display,
                    offsetWidth: video.offsetWidth,
                    offsetHeight: video.offsetHeight,
                    source_count: 0,
                    sources: []
                };
                
                if (video.src) {
                    result.urls.push(video.src);
                }
                if (video.currentSrc) {
                    result.urls.push(video.currentSrc);
                }
                
                var sources = video.querySelectorAll('source');
                info.source_count = sources.length;
                
                for (var j = 0; j < sources.length; j++) {
                    var source = sources[j];
                    if (source.src) {
                        result.urls.push(source.src);
                        info.sources.push({
                            src: source.src,
                            type: source.type || null
                        });
                    }
                }
                
                result.video_info.push(info);
            }
            
            return result;
            """
            
            result = self.driver.execute_script(js_script)
            self.logger.debug(f"🔍 JavaScript found {result.get('video_count', 0)} video elements")
            
            # Log detailed video information
            for info in result.get('video_info', []):
                self.logger.debug(f"  Video {info['index']}: size={info['offsetWidth']}x{info['offsetHeight']}, display={info['style_display']}")
                self.logger.debug(f"    src: {info['src'][:100] if info['src'] else 'None'}...")
                self.logger.debug(f"    currentSrc: {info['currentSrc'][:100] if info['currentSrc'] else 'None'}...")
                self.logger.debug(f"    sources: {info['source_count']}")
                for src_info in info['sources']:
                    self.logger.debug(f"      - {src_info['src'][:100]}... (type: {src_info['type']})")
            
            video_urls = result.get('urls', [])
            if video_urls and len(video_urls) > 0:
                self.logger.debug(f"🔍 Found {len(video_urls)} potential video URLs via JavaScript")
                # Return the first valid URL
                for url in video_urls:
                    if url and url.startswith('http'):
                        self.logger.info(f"✅ Found video URL via JavaScript: {url[:100]}...")
                        return url
            else:
                self.logger.warning("❌ No video URLs found via JavaScript")
            
            return None
            
        except Exception as e:
            self.logger.debug(f"JavaScript extraction failed: {str(e)}")
            return None
    
    def _extract_video_url_page_source(self) -> Optional[str]:
        """Try to extract video URL from page source."""
        try:
            self.logger.debug("🔍 Attempting page source video URL extraction...")
            
            page_source = self.driver.page_source
            self.logger.debug(f"🔍 Page source length: {len(page_source)} characters")
            
            # Look for video URLs in various formats
            video_patterns = [
                r'"video_url":"([^"]+)"',
                r'"video_url_original":"([^"]+)"',
                r'"video_dash_manifest":"([^"]+)"',
                r'<video[^>]+src="([^"]+)"',
                r'<source[^>]+src="([^"]+)"',
                r'https://[^"\s]+\.mp4[^"\s]*',
                r'https://[^"\s]+\.m4v[^"\s]*',
                r'https://scontent[^"\s]+\.mp4[^"\s]*'
            ]
            
            total_matches = 0
            for i, pattern in enumerate(video_patterns):
                matches = re.findall(pattern, page_source)
                self.logger.debug(f"🔍 Pattern {i+1} '{pattern}' found {len(matches)} matches")
                total_matches += len(matches)
                
                for j, match in enumerate(matches):
                    if match and match.startswith('http'):
                        # Clean up the URL
                        video_url = match.replace('\\/', '/').replace('\\u0026', '&')
                        self.logger.debug(f"  Match {j+1}: {video_url[:100]}...")
                        
                        if any(ext in video_url.lower() for ext in ['.mp4', '.m4v', 'video']):
                            self.logger.info(f"✅ Found video URL in page source: {video_url[:100]}...")
                            return video_url
            
            self.logger.warning(f"❌ No valid video URLs found in page source ({total_matches} total matches)")
            return None
            
        except Exception as e:
            self.logger.debug(f"Page source extraction failed: {str(e)}")
            return None
    
    def download_video(self, video_url: str, shortcode: str) -> Optional[str]:
        """Download the video from the given URL with improved headers and error handling."""
        try:
            self.logger.info(f"🚀 Downloading video with improved method: {video_url[:100]}...")
            
            # Create filename
            timestamp = int(time.time())
            filename = f"{shortcode}_{timestamp}.mp4"
            filepath = self.temp_dir / filename
            
            # Try improved download method first
            if self._download_video_fast(video_url, filepath):
                return str(filepath)
            
            # Fallback to original method if fast method fails
            self.logger.warning("Fast download failed, trying fallback method...")
            return self._download_video_fallback(video_url, filepath)
            
        except Exception as e:
            self.logger.error(f"Video download error: {str(e)}")
            return None
    
    def _download_video_fast(self, video_url: str, output_path) -> bool:
        """Fast video download with proper Instagram headers."""
        try:
            # Instagram-optimized headers
            headers = {
                'User-Agent': 'Instagram 219.0.0.12.117 Android (24/7.0; 640dpi; 1440x2560; samsung; SM-G930F; herolte; samsungexynos8890; en_US; 138226743)',
                'Accept': '*/*',
                'Accept-Encoding': 'gzip, deflate',
                'Accept-Language': 'en-US,en;q=0.9',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'DNT': '1',
                'Pragma': 'no-cache',
                'Sec-Fetch-Dest': 'video',
                'Sec-Fetch-Mode': 'cors',
                'Sec-Fetch-Site': 'cross-site',
                'Referer': 'https://www.instagram.com/',
                'Origin': 'https://www.instagram.com',
                'X-Instagram-AJAX': '1',
                'X-Requested-With': 'XMLHttpRequest'
            }
            
            # Copy cookies from selenium session if possible
            try:
                cookies = self.driver.get_cookies()
                cookie_dict = {}
                for cookie in cookies:
                    cookie_dict[cookie['name']] = cookie['value']
                
                if cookie_dict:
                    self.logger.debug(f"Using {len(cookie_dict)} cookies from browser session")
            except Exception:
                cookie_dict = {}
            
            self.logger.debug("🔄 Starting fast download with Instagram headers...")
            start_time = time.time()
            
            # Create session with improved headers
            session = requests.Session()
            session.headers.update(headers)
            
            # Add cookies to session
            for name, value in cookie_dict.items():
                session.cookies.set(name, value)
            
            # Download with streaming and progress tracking
            response = session.get(video_url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Check if response is actually video content
            content_type = response.headers.get('content-type', '').lower()
            if 'video' not in content_type and 'octet-stream' not in content_type:
                self.logger.warning(f"⚠️  Unexpected content type: {content_type}")
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=16384):  # Larger chunk size for speed
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        
                        # Log progress for large files (every 2MB)
                        if total_size > 0 and downloaded_size % (2 * 1024 * 1024) == 0:
                            progress = (downloaded_size / total_size) * 100
                            speed = downloaded_size / (time.time() - start_time) / 1024 / 1024  # MB/s
                            self.logger.debug(f"📊 Progress: {progress:.1f}% ({speed:.1f} MB/s)")
            
            download_time = time.time() - start_time
            file_size = output_path.stat().st_size
            speed = (file_size / 1024 / 1024) / download_time if download_time > 0 else 0
            
            self.logger.info(f"✅ Fast download completed: {file_size / 1024 / 1024:.1f} MB in {download_time:.1f}s ({speed:.1f} MB/s)")
            
            # Verify the file is a valid video
            if file_size < 1000:  # Less than 1KB is probably not a video
                self.logger.error("❌ Downloaded file is too small to be a valid video")
                self._cleanup_file(output_path)
                return False
            
            # Basic video file validation
            if not self._validate_video_file(output_path):
                self.logger.error("❌ Downloaded file failed video validation")
                self._cleanup_file(output_path)
                return False
            
            return True
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"❌ Fast download request failed: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"❌ Fast download error: {str(e)}")
            return False
    
    def _download_video_fallback(self, video_url: str, output_path) -> Optional[str]:
        """Fallback download method using original approach."""
        try:
            self.logger.info("🔄 Using fallback download method...")
            
            # Copy cookies from selenium session if possible
            try:
                cookies = self.driver.get_cookies()
                for cookie in cookies:
                    self.session.cookies.set(cookie['name'], cookie['value'], domain=cookie.get('domain'))
            except Exception:
                pass
            
            # Download the video with original method
            response = self.session.get(video_url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        
                        # Log progress for large files
                        if total_size > 0 and downloaded_size % (1024 * 1024) == 0:  # Every MB
                            progress = (downloaded_size / total_size) * 100
                            self.logger.debug(f"Fallback progress: {progress:.1f}%")
            
            file_size = output_path.stat().st_size
            self.logger.info(f"✅ Fallback download successful: {file_size / 1024 / 1024:.1f} MB")
            
            # Verify the file is a valid video
            if file_size < 1000:  # Less than 1KB is probably not a video
                self.logger.error("Downloaded file is too small to be a valid video")
                self._cleanup_file(output_path)
                return None
            
            return str(output_path)
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"❌ Fallback download failed: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"❌ Fallback download error: {str(e)}")
            return None
    
    def _validate_video_file(self, filepath) -> bool:
        """Basic validation to check if downloaded file is a valid video."""
        try:
            # Check file size
            if filepath.stat().st_size < 1000:
                return False
            
            # Check file signature (basic check for common video formats)
            with open(filepath, 'rb') as f:
                header = f.read(12)
            
            # MP4 file signatures
            mp4_signatures = [
                b'\x00\x00\x00\x18ftypmp4',  # MP4
                b'\x00\x00\x00\x1cftypisom',  # MP4 ISO
                b'\x00\x00\x00\x20ftypmp41',  # MP4 v1
                b'\x00\x00\x00\x20ftypmp42',  # MP4 v2
            ]
            
            # Check for MP4 signatures
            for signature in mp4_signatures:
                if header.startswith(signature[:8]):
                    return True
            
            # If no specific signature found, assume it's valid (some videos have different headers)
            return True
            
        except Exception as e:
            self.logger.debug(f"Video validation error: {str(e)}")
            return True  # If we can't validate, assume it's valid
    
    def extract_audio(self, video_path: str) -> Optional[str]:
        """Extract audio from the downloaded video."""
        try:
            self.logger.info("Extracting audio from video...")
            
            video_file = Path(video_path)
            audio_file = video_file.with_suffix('.wav')
            
            if MOVIEPY_AVAILABLE:
                # Use moviepy for audio extraction
                try:
                    with VideoFileClip(str(video_file)) as video:
                        if video.audio is None:
                            self.logger.warning("Video has no audio track")
                            return None
                        
                        # Extract audio
                        video.audio.write_audiofile(
                            str(audio_file),
                            verbose=False,
                            logger=None  # Suppress moviepy logging
                        )
                        
                        self.logger.info(f"Audio extracted successfully: {audio_file}")
                        return str(audio_file)
                        
                except Exception as e:
                    self.logger.warning(f"Moviepy extraction failed: {str(e)}, trying ffmpeg...")
            
            # Fallback to ffmpeg if moviepy fails or not available
            return self._extract_audio_ffmpeg(video_path, str(audio_file))
            
        except Exception as e:
            self.logger.error(f"Audio extraction error: {str(e)}")
            return None
    
    def _extract_audio_ffmpeg(self, video_path: str, audio_path: str) -> Optional[str]:
        """Extract audio using ffmpeg command line."""
        try:
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # PCM 16-bit little-endian
                '-ar', '16000',  # 16kHz sample rate (good for speech)
                '-ac', '1',  # Mono
                '-y',  # Overwrite output file
                audio_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout
            )
            
            if result.returncode == 0:
                self.logger.info(f"Audio extracted with ffmpeg: {audio_path}")
                return audio_path
            else:
                self.logger.error(f"ffmpeg failed: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            self.logger.error("ffmpeg extraction timed out")
            return None
        except FileNotFoundError:
            self.logger.error("ffmpeg not found. Please install ffmpeg.")
            return None
        except Exception as e:
            self.logger.error(f"ffmpeg extraction error: {str(e)}")
            return None
    
    def transcribe_audio(self, audio_path: str) -> Optional[Dict[str, Any]]:
        """Transcribe audio to text using Whisper."""
        try:
            if not WHISPER_AVAILABLE or not self.whisper_model_loaded:
                error_msg = "Whisper not available for transcription"
                if not WHISPER_AVAILABLE:
                    error_msg += f". Import error: {WHISPER_IMPORT_ERROR}"
                self.logger.error(error_msg)
                return None
            
            self.logger.info("Transcribing audio with Whisper...")
            start_time = time.time()
            
            # Transcribe with Whisper
            result = self.whisper_model.transcribe(
                audio_path,
                language="en",  # You can make this configurable
                fp16=False,  # Disable fp16 for better compatibility
            )
            
            processing_time = time.time() - start_time
            
            # Extract segments with timestamps
            segments = []
            for segment in result.get('segments', []):
                segments.append({
                    'start': round(segment['start'], 2),
                    'end': round(segment['end'], 2),
                    'text': segment['text'].strip(),
                    'confidence': segment.get('avg_logprob', 0)  # Whisper confidence
                })
            
            transcription_result = {
                'full_text': result['text'].strip(),
                'language': result.get('language', 'en'),
                'segments': segments,
                'processing_time': round(processing_time, 2),
                'transcription_method': 'openai_whisper',
                'model_used': 'large-v3',
                'confidence_score': self._calculate_overall_confidence(segments)
            }
            
            self.logger.info(f"Transcription completed in {processing_time:.1f}s: {len(result['text'])} characters")
            
            return transcription_result
            
        except Exception as e:
            self.logger.error(f"Transcription error: {str(e)}")
            return None
    
    def _calculate_overall_confidence(self, segments: List[Dict]) -> float:
        """Calculate overall confidence score from segment confidences."""
        if not segments:
            return 0.0
        
        # Average confidence weighted by segment length
        total_duration = 0
        weighted_confidence = 0
        
        for segment in segments:
            duration = segment['end'] - segment['start']
            confidence = max(0, segment.get('confidence', 0))  # Ensure non-negative
            
            weighted_confidence += confidence * duration
            total_duration += duration
        
        if total_duration == 0:
            return 0.0
        
        # Convert log probability to a 0-1 scale (approximate)
        avg_confidence = weighted_confidence / total_duration
        normalized_confidence = max(0, min(1, (avg_confidence + 3) / 3))  # Rough normalization
        
        return round(normalized_confidence, 3)
    
    def process_video(self, instagram_url: str, shortcode: str) -> Optional[Dict[str, Any]]:
        """Complete video processing pipeline: extract URL -> download -> transcribe."""
        try:
            self.logger.info(f"Starting video processing for {shortcode}")
            
            # Step 1: Extract video URL
            video_url = self.extract_video_url(instagram_url)
            if not video_url:
                return {
                    'success': False,
                    'error': 'Could not extract video URL from Instagram page',
                    'method': 'video_transcription'
                }
            
            # Step 2: Download video
            video_path = self.download_video(video_url, shortcode)
            if not video_path:
                return {
                    'success': False,
                    'error': 'Failed to download video',
                    'video_url': video_url,
                    'method': 'video_transcription'
                }
            
            try:
                # Step 3: Extract audio
                audio_path = self.extract_audio(video_path)
                if not audio_path:
                    return {
                        'success': False,
                        'error': 'Failed to extract audio from video (video may have no audio)',
                        'video_path': video_path,
                        'method': 'video_transcription'
                    }
                
                try:
                    # Step 4: Transcribe audio
                    transcription = self.transcribe_audio(audio_path)
                    if not transcription:
                        return {
                            'success': False,
                            'error': 'Failed to transcribe audio',
                            'audio_path': audio_path,
                            'method': 'video_transcription'
                        }
                    
                    # Success! Return complete result
                    result = {
                        'success': True,
                        'caption': transcription['full_text'],
                        'method': 'video_transcription',
                        'transcription_data': transcription,
                        'video_url': video_url,
                        'video_path': video_path,
                        'audio_path': audio_path
                    }
                    
                    self.logger.info(f"Video processing completed successfully: {len(transcription['full_text'])} characters transcribed")
                    return result
                    
                finally:
                    # Clean up audio file
                    if audio_path:
                        self._cleanup_file(audio_path)
                        
            finally:
                # Clean up video file
                self._cleanup_file(video_path)
                
        except Exception as e:
            self.logger.error(f"Video processing error: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'method': 'video_transcription'
            }
    
    def _cleanup_file(self, filepath: str) -> None:
        """Clean up temporary file."""
        try:
            if filepath and os.path.exists(filepath):
                os.remove(filepath)
                self.logger.debug(f"Cleaned up temporary file: {filepath}")
        except Exception as e:
            self.logger.warning(f"Failed to clean up file {filepath}: {str(e)}")
    
    def cleanup_temp_files(self) -> None:
        """Clean up all temporary files in the temp directory."""
        try:
            if self.temp_dir.exists():
                for file in self.temp_dir.iterdir():
                    if file.is_file():
                        file.unlink()
                        self.logger.debug(f"Cleaned up: {file}")
                
                # Remove directory if empty
                try:
                    self.temp_dir.rmdir()
                except OSError:
                    pass  # Directory not empty, that's ok
                    
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {str(e)}")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about available transcription capabilities."""
        info = {
            'whisper_available': WHISPER_AVAILABLE,
            'moviepy_available': MOVIEPY_AVAILABLE,
            'ffmpeg_python_available': FFMPEG_PYTHON_AVAILABLE,
            'ffmpeg_available': self._check_ffmpeg(),
            'torchaudio_available': TORCHAUDIO_AVAILABLE,
            'recommended_method': 'video_transcription'
        }
        
        if WHISPER_AVAILABLE and self.whisper_model_loaded:
            info['whisper_model_loaded'] = True
            info['whisper_model'] = 'large-v3'
            info['whisper_mode'] = 'with_torchaudio' if TORCHAUDIO_AVAILABLE else 'basic'
        else:
            info['whisper_model_loaded'] = False
            if not WHISPER_AVAILABLE and WHISPER_IMPORT_ERROR:
                info['whisper_error'] = WHISPER_IMPORT_ERROR
        
        if not MOVIEPY_AVAILABLE:
            info['moviepy_error'] = MOVIEPY_IMPORT_ERROR
            
        if not FFMPEG_PYTHON_AVAILABLE:
            info['ffmpeg_python_error'] = FFMPEG_PYTHON_IMPORT_ERROR
            
        if not TORCHAUDIO_AVAILABLE and 'TORCHAUDIO_ERROR' in globals():
            info['torchaudio_error'] = TORCHAUDIO_ERROR
        
        return info
    
    def _check_ffmpeg(self) -> bool:
        """Check if ffmpeg is available."""
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5)
            return True
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            return False