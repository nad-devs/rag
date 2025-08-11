import json
import csv
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from urllib.parse import urlparse, parse_qs
from logging.handlers import RotatingFileHandler

from .config import Config


def setup_logging(config: Config) -> logging.Logger:
    """Set up logging configuration based on config settings."""
    logger = logging.getLogger("instagram_scraper")
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Set logging level
    log_level = getattr(logging, config.logging.level.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # File handler with rotation (only if file logging is enabled)
    file_handler = None
    if config.logging.file:
        log_file_path = Path(config.logging.file)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            config.logging.file,
            maxBytes=config.logging.max_file_size,
            backupCount=config.logging.backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # Formatter
    formatter = logging.Formatter(config.logging.format)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    if file_handler:
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def save_data(data: List[Dict[str, Any]], config: Config, filename: Optional[str] = None) -> str:
    """Save scraped data to file in specified format."""
    if not config.output.save_to_file:
        return ""
    
    # Create output directory if it doesn't exist
    output_dir = Path(config.output.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename if not provided
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"instagram_captions_{timestamp}"
    
    # Determine file extension based on format
    if config.output.format.lower() == "json":
        file_path = output_dir / f"{filename}.json"
        save_as_json(data, file_path)
    elif config.output.format.lower() == "csv":
        file_path = output_dir / f"{filename}.csv"
        save_as_csv(data, file_path)
    else:
        raise ValueError(f"Unsupported output format: {config.output.format}")
    
    return str(file_path)


def save_as_json(data: List[Dict[str, Any]], file_path: Path) -> None:
    """Save data as JSON file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        raise RuntimeError(f"Error saving JSON file: {str(e)}")


def save_as_csv(data: List[Dict[str, Any]], file_path: Path) -> None:
    """Save data as CSV file."""
    if not data:
        return
    
    try:
        fieldnames = data[0].keys()
        with open(file_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
    except Exception as e:
        raise RuntimeError(f"Error saving CSV file: {str(e)}")



def is_valid_instagram_url(url: str) -> bool:
    """Validate if URL is a valid Instagram URL."""
    instagram_patterns = [
        "instagram.com/p/",
        "instagram.com/reel/",
        "instagram.com/tv/",
        "/reel/",  # Also match profile-based reel URLs like /username/reel/ABC123/
        "/p/",     # Also match profile-based post URLs like /username/p/ABC123/
        "/tv/"     # Also match profile-based TV URLs like /username/tv/ABC123/
    ]
    return any(pattern in url.lower() for pattern in instagram_patterns)


def is_valid_instagram_profile_url(url: str) -> bool:
    """Validate if URL is a valid Instagram profile URL."""
    if not url or not isinstance(url, str):
        return False
    
    # Import the validation function from scraper
    from .scraper import InstagramScraper
    return InstagramScraper.is_valid_instagram_profile_url(url)


def extract_post_id_from_url(url: str) -> Optional[str]:
    """Extract post ID from Instagram URL."""
    try:
        # Handle different Instagram URL formats
        if "/p/" in url:
            return url.split("/p/")[1].split("/")[0]
        elif "/reel/" in url:
            return url.split("/reel/")[1].split("/")[0]
        elif "/tv/" in url:
            return url.split("/tv/")[1].split("/")[0]
        return None
    except Exception:
        return None


def format_caption_text(caption: str) -> str:
    """Clean and format caption text."""
    if not caption:
        return ""
    
    # Remove excessive whitespace
    caption = " ".join(caption.split())
    
    # Remove common Instagram artifacts
    caption = caption.replace("... more", "")
    caption = caption.replace("Show more", "")
    
    return caption.strip()


def get_timestamp() -> str:
    """Get current timestamp in ISO format."""
    return datetime.now().isoformat()


def ensure_directory_exists(directory_path: str) -> None:
    """Ensure directory exists, create if it doesn't."""
    Path(directory_path).mkdir(parents=True, exist_ok=True)


def sanitize_filename(filename: str) -> str:
    """Sanitize filename by removing invalid characters."""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename


def get_file_size(file_path: str) -> int:
    """Get file size in bytes."""
    try:
        return os.path.getsize(file_path)
    except OSError:
        return 0


def create_backup_file(file_path: str) -> str:
    """Create a backup of an existing file."""
    if not os.path.exists(file_path):
        return ""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{file_path}.backup_{timestamp}"
    
    try:
        import shutil
        shutil.copy2(file_path, backup_path)
        return backup_path
    except Exception as e:
        raise RuntimeError(f"Error creating backup: {str(e)}")


def validate_credentials(username: str, password: str) -> bool:
    """Basic validation for Instagram credentials."""
    if not username or not password:
        return False
    
    if len(username) < 1 or len(password) < 6:
        return False
    
    return True


def get_instagram_url_info(url: str) -> Dict[str, Any]:
    """Extract comprehensive information from Instagram URL."""
    info = {
        'url': url,
        'is_valid': False,
        'type': None,
        'shortcode': None,
        'username': None,
        'story_id': None
    }
    
    if not is_valid_instagram_url(url):
        return info
    
    info['is_valid'] = True
    
    try:
        parsed = urlparse(url)
        path = parsed.path
        
        # Determine URL type and extract information
        if '/p/' in path:
            info['type'] = 'post'
            info['shortcode'] = extract_post_id_from_url(url)
        elif '/reel/' in path:
            info['type'] = 'reel'
            info['shortcode'] = extract_post_id_from_url(url)
        elif '/tv/' in path:
            info['type'] = 'igtv'
            info['shortcode'] = extract_post_id_from_url(url)
        elif '/stories/' in path:
            info['type'] = 'story'
            # Extract username and story ID from stories URL
            story_match = re.search(r'/stories/([A-Za-z0-9_.]+)/([0-9]+)', path)
            if story_match:
                info['username'] = story_match.group(1)
                info['story_id'] = story_match.group(2)
        
    except Exception as e:
        info['error'] = str(e)
    
    return info


def normalize_instagram_url(url: str) -> str:
    """Normalize Instagram URL to a standard format."""
    if not url:
        return url
    
    # Add protocol if missing
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    # Parse URL
    try:
        parsed = urlparse(url)
        
        # Ensure www. prefix for consistency
        netloc = parsed.netloc.lower()
        if netloc == 'instagram.com':
            netloc = 'www.instagram.com'
        
        # Remove trailing slash if present
        path = parsed.path.rstrip('/')
        
        # Reconstruct URL
        return f"https://{netloc}{path}/"
        
    except Exception:
        return url

def analyze_url_file(file_path: str) -> Dict[str, Any]:
    """Analyze a file containing Instagram URLs and return statistics."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    analysis = {
        'file_path': file_path,
        'file_size': get_file_size(file_path),
        'total_lines': 0,
        'empty_lines': 0,
        'comment_lines': 0,
        'total_urls': 0,
        'valid_urls': 0,
        'invalid_urls': 0,
        'url_types': {},
        'duplicate_urls': 0,
        'issues': []
    }
    
    try:
        urls_seen = set()
        
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                analysis['total_lines'] += 1
                line = line.strip()
                
                if not line:
                    analysis['empty_lines'] += 1
                    continue
                
                if line.startswith('#'):
                    analysis['comment_lines'] += 1
                    continue
                
                analysis['total_urls'] += 1
                
                # Check for duplicates
                if line in urls_seen:
                    analysis['duplicate_urls'] += 1
                    analysis['issues'].append(f"Line {line_num}: Duplicate URL: {line}")
                else:
                    urls_seen.add(line)
                
                # Validate URL
                info = get_instagram_url_info(line)
                if info['is_valid']:
                    analysis['valid_urls'] += 1
                    url_type = info.get('type', 'unknown')
                    analysis['url_types'][url_type] = analysis['url_types'].get(url_type, 0) + 1
                else:
                    analysis['invalid_urls'] += 1
                    analysis['issues'].append(f"Line {line_num}: Invalid URL: {line}")
        
    except Exception as e:
        analysis['error'] = str(e)
    
    return analysis

def print_url_validation_report(valid_urls: List[str], invalid_urls: List[str], url_info: List[Dict[str, Any]]) -> None:
    """Print a detailed URL validation report."""
    total_urls = len(valid_urls) + len(invalid_urls)
    
    print(f"\n📊 URL Validation Report")
    print(f"{'='*50}")
    print(f"Total URLs: {total_urls}")
    print(f"Valid URLs: {len(valid_urls)} ({len(valid_urls)/total_urls*100:.1f}%)")
    print(f"Invalid URLs: {len(invalid_urls)} ({len(invalid_urls)/total_urls*100:.1f}%)")
    
    if url_info:
        # Count by type
        type_counts = {}
        for info in url_info:
            if info['is_valid']:
                url_type = info.get('type', 'unknown')
                type_counts[url_type] = type_counts.get(url_type, 0) + 1
        
        if type_counts:
            print(f"\nValid URLs by type:")
            for url_type, count in type_counts.items():
                print(f"  {url_type.capitalize()}: {count}")
    
    if invalid_urls:
        print(f"\n❌ Invalid URLs:")
        for i, url in enumerate(invalid_urls[:5], 1):  # Show first 5
            print(f"  {i}. {url}")
        if len(invalid_urls) > 5:
            print(f"  ... and {len(invalid_urls) - 5} more")

def test_url_examples() -> None:
    """Test URL validation with various examples."""
    test_urls = [
        # Valid URLs
        "https://www.instagram.com/p/ABC123/",
        "https://instagram.com/reel/XYZ789",
        "http://www.instagram.com/tv/DEF456/",
        "https://www.instagram.com/stories/username/123456789/",
        
        # Invalid URLs
        "https://facebook.com/post/123",
        "https://twitter.com/status/456",
        "not_a_url",
        "instagram.com/invalid",
        "https://www.instagram.com/user/",
    ]
    
    print("\n🧪 URL Validation Test Results:")
    print("=" * 50)
    
    for url in test_urls:
        info = get_instagram_url_info(url)
        status = "✅ VALID" if info['is_valid'] else "❌ INVALID"
        type_info = f" ({info['type']})" if info['type'] else ""
        shortcode_info = f" [ID: {info['shortcode']}]" if info['shortcode'] else ""
        
        print(f"{status:<10} {url}{type_info}{shortcode_info}")

def get_user_agent_list() -> List[str]:
    """Get a list of updated common user agents for rotation."""
    return [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ]


def save_individual_reel_files(profile_data: Dict[str, Any], config: Config) -> List[str]:
    """Save each reel as a separate file."""
    if not config.output.save_to_file:
        return []
    
    # Create output directory if it doesn't exist
    output_dir = Path(config.output.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    profile_username = profile_data.get('profile_username', 'unknown')
    scan_timestamp = profile_data.get('scan_timestamp', datetime.now().timestamp())
    
    # Create timestamp string for folder organization
    timestamp_str = datetime.fromtimestamp(scan_timestamp).strftime("%Y%m%d_%H%M%S")
    
    # Create profile-specific directory
    profile_dir = output_dir / f"{profile_username}_{timestamp_str}"
    profile_dir.mkdir(parents=True, exist_ok=True)
    
    # Save each reel as individual file
    for i, reel_data in enumerate(profile_data.get('reels', []), 1):
        try:
            # Extract shortcode for filename
            shortcode = reel_data.get('shortcode', f'reel_{i}')
            
            # Create filename
            if config.output.format.lower() == "json":
                filename = f"{shortcode}.json"
                file_path = profile_dir / filename
                
                # Save as JSON
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(reel_data, f, indent=2, ensure_ascii=False, default=str)
                    
            elif config.output.format.lower() == "csv":
                filename = f"{shortcode}.csv"
                file_path = profile_dir / filename
                
                # Flatten the data for CSV
                flattened_data = flatten_reel_data(reel_data)
                
                # Save as CSV
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=flattened_data.keys())
                    writer.writeheader()
                    writer.writerow(flattened_data)
            
            saved_files.append(str(file_path))
            
        except Exception as e:
            print(f"Error saving reel {i}: {str(e)}")
            continue
    
    return saved_files


def flatten_reel_data(reel_data: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten reel data structure for CSV export."""
    flattened = {}
    
    # Basic fields
    basic_fields = ['url', 'caption', 'timestamp', 'success', 'shortcode', 'url_type', 'extraction_method']
    for field in basic_fields:
        flattened[field] = reel_data.get(field, '')
    
    # Transcription data
    transcription_data = reel_data.get('transcription_data', {})
    if transcription_data:
        flattened['transcription_full_text'] = transcription_data.get('full_text', '')
        flattened['transcription_language'] = transcription_data.get('language', '')
        flattened['transcription_segments_count'] = len(transcription_data.get('segments', []))
    
    # Processing info
    processing_info = reel_data.get('processing_info', {})
    if processing_info:
        flattened['processing_method'] = processing_info.get('method', '')
        flattened['whisper_model'] = processing_info.get('whisper_model', '')
        flattened['confidence_score'] = processing_info.get('confidence_score', 0)
        flattened['processing_time'] = processing_info.get('processing_time', 0)
        flattened['language_detected'] = processing_info.get('language_detected', '')
    
    # Profile metadata
    flattened['profile_username'] = reel_data.get('profile_username', '')
    flattened['position_in_profile'] = reel_data.get('position_in_profile', '')
    flattened['extraction_timestamp'] = reel_data.get('extraction_timestamp', '')
    
    return flattened



def get_processed_reels(output_dir: Path, profile_username: str) -> set:
    """Get set of already processed reel shortcodes to avoid duplicates."""
    processed = set()
    
    # Look for existing profile directories
    pattern = f"{profile_username}_*"
    for profile_dir in output_dir.glob(pattern):
        if profile_dir.is_dir():
            # Get all JSON/CSV files in the directory
            for file_path in profile_dir.glob("*.json"):
                shortcode = file_path.stem  # filename without extension
                processed.add(shortcode)
            for file_path in profile_dir.glob("*.csv"):
                shortcode = file_path.stem
                processed.add(shortcode)
    
    return processed


def create_batch_resume_file(output_dir: Path, profile_username: str, batch_info: Dict[str, Any]) -> str:
    """Create a resume file to track batch processing progress."""
    resume_file = output_dir / f"{profile_username}_batch_progress.json"
    
    with open(resume_file, 'w', encoding='utf-8') as f:
        json.dump(batch_info, f, indent=2, ensure_ascii=False, default=str)
    
    return str(resume_file)


def load_batch_resume_file(output_dir: Path, profile_username: str) -> Optional[Dict[str, Any]]:
    """Load batch processing progress if it exists."""
    resume_file = output_dir / f"{profile_username}_batch_progress.json"
    
    if resume_file.exists():
        try:
            with open(resume_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None
    
    return None


def filter_unprocessed_reels(all_reels: List[Dict[str, Any]], processed_shortcodes: set) -> List[Dict[str, Any]]:
    """Filter out already processed reels."""
    unprocessed = []
    
    for reel in all_reels:
        shortcode = reel.get('shortcode')
        if not shortcode:
            # Extract shortcode from URL if not present
            url = reel.get('url', '')
            if '/reel/' in url:
                shortcode = url.split('/reel/')[1].split('/')[0].split('?')[0]
        
        if shortcode and shortcode not in processed_shortcodes:
            unprocessed.append(reel)
    
    return unprocessed


def create_batches(reels: List[Dict[str, Any]], batch_size: int = 50) -> List[List[Dict[str, Any]]]:
    """Split reels into batches."""
    batches = []
    for i in range(0, len(reels), batch_size):
        batches.append(reels[i:i + batch_size])
    return batches


def convert_existing_json_to_individual_files(json_file_path: str, config: Config) -> Dict[str, Any]:
    """Convert existing combined JSON file to individual reel files."""
    try:
        # Load existing JSON
        with open(json_file_path, 'r', encoding='utf-8') as f:
            reels_data = json.load(f)
        
        if not reels_data:
            return {'success': False, 'error': 'No data in JSON file'}
        
        # Extract profile info from first reel or filename
        first_reel = reels_data[0] if reels_data else {}
        
        # Try to get profile username from various sources
        profile_username = (
            first_reel.get('source_profile') or 
            first_reel.get('profile_username') or
            'unknown_profile'
        )
        
        # Create synthetic profile result structure
        profile_result = {
            'profile_username': profile_username,
            'scan_timestamp': first_reel.get('profile_scan_timestamp') or first_reel.get('timestamp') or time.time(),
            'reels': reels_data,
            'total_reels_processed': len(reels_data),
            'successful_extractions': sum(1 for r in reels_data if r.get('success', False)),
            'failed_extractions': sum(1 for r in reels_data if not r.get('success', False))
        }
        
        # Get already processed reels to avoid duplicates
        output_dir = Path(config.output.output_directory)
        processed_shortcodes = get_processed_reels(output_dir, profile_username)
        
        print(f"Found {len(processed_shortcodes)} already processed reels")
        
        # Filter out already processed reels
        unprocessed_reels = filter_unprocessed_reels(reels_data, processed_shortcodes)
        profile_result['reels'] = unprocessed_reels
        
        print(f"Converting {len(unprocessed_reels)} unprocessed reels to individual files...")
        
        if unprocessed_reels:
            # Save individual files
            individual_files = save_individual_reel_files(profile_result, config)
            
            return {
                'success': True,
                'total_reels': len(reels_data),
                'already_processed': len(processed_shortcodes),
                'newly_converted': len(unprocessed_reels),
                'individual_files': individual_files,
                'profile_username': profile_username
            }
        else:
            return {
                'success': True,
                'total_reels': len(reels_data),
                'already_processed': len(processed_shortcodes),
                'newly_converted': 0,
                'message': 'All reels already converted to individual files'
            }
            
    except Exception as e:
        return {'success': False, 'error': str(e)}


def print_banner() -> None:
    """Print application banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║               Instagram Caption Scraper                      ║
    ║                    Version 1.0.0                            ║
    ║                Enhanced URL Processing                        ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)