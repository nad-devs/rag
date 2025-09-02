#!/usr/bin/env python3
"""
Instagram Caption Scraper - Main Entry Point

This script provides a command-line interface for scraping Instagram video captions
using Selenium WebDriver.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Optional
from datetime import datetime

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.scraper import InstagramScraper
from src.config import Config
from src.utils import (
    print_banner, 
    is_valid_instagram_profile_url,
    validate_credentials,
    ensure_directory_exists,
    save_individual_reel_files
)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Instagram Video Caption Scraper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Profile scraping with login
  python main.py --profile-url "https://www.instagram.com/edhonour/" --max-reels 25 --login
  
  # Interactive login with manual browser login
  python main.py --profile-url "https://www.instagram.com/edhonour/" --max-reels 25 --interactive-login
  
  # Browser handoff - login in your default browser
  python main.py --profile-url "https://www.instagram.com/edhonour/" --max-reels 25 --browser-handoff
  
  # Profile scraping with debug mode
  python main.py --profile-url "https://www.instagram.com/username/" --max-reels 10 --login --debug-mode
  
  # Incremental update - only new reels since last scrape
  python main.py --profile-url "https://www.instagram.com/edhonour/" --incremental --browser-handoff
  
  # Validate collection and fill missing reels up to position 500
  python main.py --profile-url "https://www.instagram.com/edhonour/" --validate-up-to 500 --browser-handoff
        """
    )
    
    # Profile URL (required)
    parser.add_argument(
        '--profile-url', '-p',
        type=str,
        required=True,
        help='Instagram profile URL to scrape reels from (e.g., https://www.instagram.com/username/)'
    )
    
    # Profile scraping options (must be right after profile-url for proper parsing)
    parser.add_argument(
        '--max-reels',
        type=int,
        default=25,
        help='Maximum number of reels to scrape from profile (default: 25)'
    )
    parser.add_argument(
        '--skip',
        type=int,
        default=0,
        help='Skip the first N reels when scraping profile (useful for batch processing, default: 0)'
    )
    
    # Gap detection for validation
    parser.add_argument(
        '--validate-up-to',
        type=int,
        help='Validate collection up to N reels and process only missing ones (e.g., --validate-up-to 500)'
    )
    
    # Incremental update mode
    parser.add_argument(
        '--incremental',
        action='store_true',
        help='Incremental update mode: only scrape reels posted after the latest existing reel date. Skips pinned reels.'
    )
    
    # Authentication options
    auth_group = parser.add_argument_group('authentication')
    auth_group.add_argument(
        '--login',
        action='store_true',
        help='Enable Instagram login to bypass login walls'
    )
    auth_group.add_argument(
        '--interactive-login',
        action='store_true',
        help='Open browser for manual interactive login (including 2FA support)'
    )
    auth_group.add_argument(
        '--browser-handoff',
        action='store_true',
        help='Generate link to login in your default browser, then continue with scraping'
    )
    auth_group.add_argument(
        '--username',
        type=str,
        help='Instagram username for login (optional)'
    )
    auth_group.add_argument(
        '--password',
        type=str,
        help='Instagram password for login (optional)'
    )
    
    # Configuration options
    config_group = parser.add_argument_group('configuration')
    config_group.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    config_group.add_argument(
        '--output', '-o',
        type=str,
        help='Output filename (without extension)'
    )
    config_group.add_argument(
        '--format',
        choices=['json', 'csv'],
        help='Output format (overrides config file)'
    )
    
    # Browser options
    browser_group = parser.add_argument_group('browser settings')
    browser_group.add_argument(
        '--headless',
        action='store_true',
        help='Run browser in headless mode'
    )
    browser_group.add_argument(
        '--browser',
        choices=['chrome', 'firefox'],
        help='Browser to use (overrides config file)'
    )
    browser_group.add_argument(
        '--use-firefox',
        action='store_true',
        help='Force use of Firefox browser (useful for WSL/Linux)'
    )
    
    # Video transcription options
    transcription_group = parser.add_argument_group('video transcription')
    transcription_group.add_argument(
        '--language',
        type=str,
        default='en',
        help='Language for transcription (default: en)'
    )
    
    # Profile scraping options (max-reels defined above)
    
    # Debug options
    debug_group = parser.add_argument_group('debugging')
    debug_group.add_argument(
        '--debug-html',
        action='store_true',
        help='Save page HTML source for debugging caption extraction failures'
    )
    debug_group.add_argument(
        '--screenshot',
        action='store_true',
        help='Take screenshots during caption extraction for debugging'
    )
    debug_group.add_argument(
        '--debug-mode',
        action='store_true',
        help='Enable comprehensive debugging (includes HTML saving and screenshots)'
    )
    debug_group.add_argument(
        '--force-cpu',
        action='store_true',
        help='Force CPU usage for video transcription (avoid GPU memory issues)'
    )
    
    # Other options
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--version',
        action='version',
        version='Instagram Caption Scraper 1.0.0'
    )
    
    return parser.parse_args()


def validate_inputs(args: argparse.Namespace) -> bool:
    """Validate command line inputs."""
    # Check config file exists
    if not os.path.exists(args.config):
        print(f"Error: Configuration file '{args.config}' not found.")
        return False
    
    # Validate profile URL (now required)
    if not args.profile_url or not is_valid_instagram_profile_url(args.profile_url):
        print(f"Error: Invalid Instagram profile URL: {args.profile_url}")
        return False
    
    # Validate credentials if provided
    if args.username and args.password:
        if not validate_credentials(args.username, args.password):
            print("Error: Invalid credentials provided.")
            return False
    elif args.username or args.password:
        print("Error: Both username and password must be provided together.")
        return False
    
    # Validate max-reels if profile URL is provided
    if hasattr(args, 'profile_url') and args.profile_url:
        if not hasattr(args, 'max_reels') or args.max_reels is None:
            print("Warning: --max-reels not properly set, using default of 25")
            args.max_reels = 25
        elif args.max_reels < 1:
            print("Error: --max-reels must be at least 1")
            return False
    
    return True


def analyze_existing_files(profile_username: str, output_dir: str) -> dict:
    """Analyze existing output files to find which shortcodes we have."""
    import json
    import glob
    import os
    
    existing_shortcodes = set()
    profile_dirs = glob.glob(os.path.join(output_dir, f"{profile_username}_*"))
    
    for profile_dir in profile_dirs:
        json_files = glob.glob(os.path.join(profile_dir, "*.json"))
        
        for json_file in json_files:
            try:
                # Extract shortcode from filename (e.g., "DImGEdCO1-W.json" -> "DImGEdCO1-W")
                filename = os.path.basename(json_file)
                shortcode = os.path.splitext(filename)[0]
                existing_shortcodes.add(shortcode)
            except Exception as e:
                print(f"Warning: Could not process {json_file}: {e}")
    
    return {
        'existing_shortcodes': existing_shortcodes,
        'total_files': len(existing_shortcodes)
    }


def analyze_existing_reels_for_incremental(profile_username: str, output_dir: str) -> dict:
    """Analyze existing reel files to find the latest post date and positions."""
    import json
    import glob
    import os
    from datetime import datetime
    
    existing_shortcodes = set()
    latest_post_date = None
    latest_position = None
    reel_data = []
    
    profile_dirs = glob.glob(os.path.join(output_dir, f"{profile_username}_*"))
    
    for profile_dir in profile_dirs:
        json_files = glob.glob(os.path.join(profile_dir, "*.json"))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                shortcode = data.get('shortcode')
                post_date = data.get('post_date')  # This will be None initially, but we'll add it
                position = data.get('position_in_profile')
                
                if shortcode:
                    existing_shortcodes.add(shortcode)
                    reel_data.append({
                        'shortcode': shortcode,
                        'post_date': post_date,
                        'position_in_profile': position,
                        'file_path': json_file
                    })
                    
                    # Track the latest position (smallest number = most recent)
                    if position is not None:
                        if latest_position is None or position < latest_position:
                            latest_position = position
                    
                    # Track the latest post date if available
                    if post_date:
                        if latest_post_date is None or post_date > latest_post_date:
                            latest_post_date = post_date
                            
            except Exception as e:
                print(f"Warning: Could not process {json_file}: {e}")
    
    return {
        'existing_shortcodes': existing_shortcodes,
        'total_files': len(existing_shortcodes),
        'latest_post_date': latest_post_date,
        'latest_position': latest_position,
        'reel_data': reel_data
    }


def extract_shortcode_from_url(url: str) -> str:
    """Extract shortcode from Instagram URL."""
    import re
    # Match Instagram reel/post URLs and extract shortcode
    match = re.search(r'/(?:p|reel)/([A-Za-z0-9_-]+)/', url)
    if match:
        return match.group(1)
    return ""


def find_missing_shortcodes(existing_shortcodes: set, discovered_urls: list) -> list:
    """Find URLs with shortcodes that we don't have yet."""
    missing_urls = []
    
    for url in discovered_urls:
        shortcode = extract_shortcode_from_url(url)
        if shortcode and shortcode not in existing_shortcodes:
            missing_urls.append(url)
    
    return missing_urls


def validate_profile_url(args: argparse.Namespace) -> bool:
    """Validate profile URL."""
    if not args.profile_url:
        print("Error: Profile URL is required.")
        return False
    
    if not is_valid_instagram_profile_url(args.profile_url):
        print(f"Error: Invalid Instagram profile URL: {args.profile_url}")
        return False
    
    return True


def check_wsl_environment():
    """Check WSL environment and provide helpful information."""
    try:
        with open('/proc/version', 'r') as f:
            proc_version = f.read().lower()
            if 'microsoft' in proc_version or 'wsl' in proc_version:
                return True
    except FileNotFoundError:
        pass
    return False

def update_config_from_args(config: Config, args: argparse.Namespace) -> None:
    """Update configuration with command line arguments."""
    if args.format:
        config.output.format = args.format
    
    if args.browser:
        config.selenium.browser = args.browser
    elif getattr(args, 'use_firefox', False):  # Handle --use-firefox flag
        config.selenium.browser = 'firefox'
    
    # Set headless mode FIRST if requested (can be overridden by interactive login)
    if args.headless:
        config.selenium.headless = True
    
    # Interactive login requires non-headless mode (overrides headless flag)
    if hasattr(args, 'interactive_login') and args.interactive_login:
        config.selenium.headless = False
        print("🖥️  Interactive login mode: Browser will open in non-headless mode")
        
        # Check WSL environment and warn user
        if check_wsl_environment():
            print("⚠️  WSL environment detected!")
            print("   Interactive login requires GUI support.")
            print("   If browser fails to open, see WSL display setup instructions.")
    
    # Browser handoff can use headless mode since user logs in separately
    elif hasattr(args, 'browser_handoff') and args.browser_handoff:
        config.selenium.headless = True  # Can use headless since no browser interaction needed
        print("🔗 Browser handoff mode: You'll login in your default browser")
    
    if args.verbose:
        config.logging.level = "DEBUG"


def main() -> int:
    """Main function."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Print banner
        print_banner()
        
        # Validate inputs
        if not validate_inputs(args):
            return 1
        
        # Load configuration
        try:
            config = Config(args.config)
            config.validate()
        except Exception as e:
            print(f"Error loading configuration: {str(e)}")
            return 1
        
        # Update config with command line arguments
        update_config_from_args(config, args)
        
        # Update transcription language if specified
        if hasattr(args, 'language') and args.language:
            # Add transcription language to config
            if not hasattr(config, 'transcription'):
                config.transcription = type('obj', (object,), {})()
            config.transcription.language = args.language
        
        # Validate profile URL
        if not validate_profile_url(args):
            return 1
        
        print(f"Profile to process: {args.profile_url}")
        
        # Show browser information if verbose
        if args.verbose:
            if getattr(args, 'use_firefox', False):
                print("🦊 Firefox browser forced by --use-firefox flag")
            elif args.browser:
                print(f"🌐 Browser set to: {args.browser}")
        
        # Ensure required directories exist
        ensure_directory_exists("logs")
        ensure_directory_exists(config.output.output_directory)
        
        # Create debug directory if debug mode is enabled
        debug_mode = args.debug_mode or args.debug_html or args.screenshot
        if debug_mode:
            ensure_directory_exists("debug_output")
            ensure_directory_exists("debug_output/html")
            ensure_directory_exists("debug_output/screenshots")
            ensure_directory_exists("debug_output/logs")
        
        # Initialize scraper
        results = []
        
        try:
            # Determine debug mode
            debug_mode = args.debug_mode or args.debug_html or args.screenshot
            
            # Pass debug flags to scraper
            force_firefox = getattr(args, 'use_firefox', False)
            force_cpu = getattr(args, 'force_cpu', False)
            scraper = InstagramScraper(
                config,  # Pass the modified config object, not args.config
                force_firefox=force_firefox,
                debug_mode=debug_mode,
                force_cpu=force_cpu
            )
            
            # Handle browser handoff BEFORE starting browser session
            # Apply browser handoff to single URLs too if it's specified
            if hasattr(args, 'browser_handoff') and args.browser_handoff:
                print("🔗 Starting browser handoff login mode...")
                login_success = scraper.browser_handoff_login()
                
                if not login_success:
                    print("❌ Browser handoff failed.")
                    return 1
            
            # NOW start the browser session (will connect to existing Chrome)
            scraper.start_session()
            print("Starting browser session...")
                
            # Show browser info in verbose mode
            if args.verbose:
                    browser_info = scraper.get_browser_info()
                    if 'browser_name' in browser_info:
                        print(f"📊 Browser: {browser_info['browser_name']} {browser_info['browser_version']}")
                        print(f"📱 Platform: {browser_info['platform']}")
                        print(f"📐 Window size: {browser_info['window_size']}")
                        print(f"👻 Headless mode: {browser_info['headless']}")
                        
                        # Show transcription capabilities
                        transcriber_status = browser_info.get('video_transcriber_status', {})
                        if transcriber_status:
                            print(f"🎤 Whisper available: {transcriber_status.get('whisper_available', False)}")
                            print(f"🎬 MoviePy available: {transcriber_status.get('moviepy_available', False)}")
                            print(f"🔧 FFmpeg available: {transcriber_status.get('ffmpeg_available', False)}")
                
            # Handle other login options (browser handoff already done)
            if hasattr(args, 'interactive_login') and args.interactive_login:
                print("🔐 Starting interactive login mode...")
                login_success = scraper.interactive_login()
                
                if not login_success:
                    print("❌ Interactive login failed. You may still be able to access public content.")
                    print("   Consider trying browser handoff or using regular login with credentials.")
            elif args.login:
                # Use command line credentials if provided, otherwise use config/env credentials
                if args.username and args.password:
                    print(f"🔐 Logging into Instagram using provided credentials...")
                    login_success = scraper.login(args.username, args.password)
                else:
                    # Use credentials from config/environment (.env file)
                    import os
                    from dotenv import load_dotenv
                    load_dotenv()
                    
                    env_username = os.getenv('INSTAGRAM_USERNAME') or config.instagram.credentials.username
                    env_password = os.getenv('INSTAGRAM_PASSWORD') or config.instagram.credentials.password
                    
                    if env_username and env_password:
                        print(f"🔐 Logging into Instagram using credentials from .env file...")
                        login_success = scraper.login(env_username, env_password)
                    else:
                        print("❌ No Instagram credentials found")
                        print("   Please set INSTAGRAM_USERNAME and INSTAGRAM_PASSWORD in .env file")
                        print("   or provide --username and --password arguments")
                        login_success = False
                
                if not login_success:
                    print("⚠️  Login failed or skipped. Continuing without authentication...")
                    print("   Note: Private profiles and some content may not be accessible")
                
            # Process profile
            print(f"Processing profile: {args.profile_url}")
            
            # Determine debug flags for individual extraction
            save_html = args.debug_mode or args.debug_html
            take_screenshot = args.debug_mode or args.screenshot
            
            if debug_mode:
                print(f"🔍 Debug mode enabled - HTML: {save_html}, Screenshots: {take_screenshot}")
            
            results = []
            
            # Extract profile username from URL
            profile_username = args.profile_url.rstrip('/').split('/')[-1]
                
            # Handle validation mode
            if hasattr(args, 'validate_up_to') and args.validate_up_to:
                print(f"🔍 Validation mode: discovering first {args.validate_up_to} reels and checking for gaps")
                print(f"Profile: {args.profile_url}")
                
                # Analyze existing files
                analysis = analyze_existing_files(profile_username, config.output.output_directory)
                existing_shortcodes = analysis['existing_shortcodes']
                
                print(f"📊 Analysis of existing files:")
                print(f"   Total files found: {analysis['total_files']}")
                print(f"   Existing shortcodes: {len(existing_shortcodes)}")
                
                # Set discovery parameters - we need to discover enough reels to check against
                args.max_reels = args.validate_up_to
                args.skip = 0  # Don't skip any during discovery
                
                # Set validation mode flag
                validation_mode = True
                target_shortcodes = existing_shortcodes
            # Handle incremental update mode
            elif hasattr(args, 'incremental') and args.incremental:
                print(f"🔄 Incremental update mode: discovering new reels since last scrape")
                print(f"Profile: {args.profile_url}")
                
                # Analyze existing files for incremental data
                analysis = analyze_existing_reels_for_incremental(profile_username, config.output.output_directory)
                existing_shortcodes = analysis['existing_shortcodes']
                latest_position = analysis['latest_position']
                latest_post_date = analysis['latest_post_date']
                
                print(f"📊 Analysis of existing files:")
                print(f"   Total files found: {analysis['total_files']}")
                print(f"   Latest position found: {latest_position}")
                if latest_post_date:
                    print(f"   Latest post date found: {latest_post_date}")
                else:
                    print(f"   No post dates found - will use position-based filtering")
                
                # For incremental mode, we start from the beginning but stop when we reach known content
                # The scraper will handle the filtering internally
                args.max_reels = latest_position if latest_position else 50  # Default to 50 if no position found
                args.skip = 0  # Always start from the beginning for incremental
                
                # Set incremental mode flag
                validation_mode = True  # Reuse validation logic for filtering
                target_shortcodes = existing_shortcodes
                
                # Add incremental-specific metadata
                incremental_mode = True
                incremental_data = {
                    'latest_position': latest_position,
                    'latest_post_date': latest_post_date,
                    'existing_shortcodes': existing_shortcodes
                }
            else:
                validation_mode = False
                target_shortcodes = None
                incremental_mode = False
                incremental_data = None
                
            print(f"🔍 Profile-based reel scraping mode")
            print(f"Profile: {args.profile_url}")
            print(f"Target reels: {args.max_reels}")
            if args.skip > 0:
                print(f"Skipping first: {args.skip} reels")
            
            # Use profile scraping method
            profile_result = scraper.scrape_profile_reels(
                args.profile_url,
                max_reels=args.max_reels,
                skip_first=args.skip,
                save_html=save_html,
                take_screenshot=take_screenshot,
                validation_mode=validation_mode,
                existing_shortcodes=target_shortcodes,
                incremental_mode=incremental_mode if 'incremental_mode' in locals() else False,
                incremental_data=incremental_data if 'incremental_data' in locals() else None
            )
            
            if profile_result and profile_result.get('success'):
                # For profile results, we treat the entire profile as one "result"
                # but the individual reels are in profile_result['reels']
                results = [profile_result]
                print(f"✅ Profile scraping completed:")
                print(f"   Discovered: {profile_result.get('total_reels_discovered', 0)} reels")
                print(f"   Processed: {profile_result.get('total_reels_processed', 0)} reels")
                print(f"   Successful: {profile_result.get('successful_extractions', 0)} extractions")
                print(f"   Failed: {profile_result.get('failed_extractions', 0)} extractions")
            else:
                error_msg = profile_result.get('error', 'Unknown error') if profile_result else 'No result returned'
                print(f"❌ Profile scraping failed: {error_msg}")
                results = [profile_result] if profile_result else []
            
            # Close scraper session
            scraper.close_session()
        
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            return 1
        except Exception as e:
            print(f"Error during scraping: {str(e)}")
            return 1
        
        # Save results for profile scraping
        if results and config.output.save_to_file:
            try:
                profile_result = results[0]
                if profile_result.get('success') and 'reels' in profile_result:
                    # Save each reel as individual file
                    individual_files = save_individual_reel_files(profile_result, config)
                    print(f"✅ Saved {len(individual_files)} individual reel files in profile directory")
                else:
                    print(f"Profile scraping failed: {profile_result.get('error', 'Unknown error')}")
            except Exception as e:
                print(f"Error saving results: {str(e)}")
                return 1
        
        # Print summary
        if results:
            profile_result = results[0]
            if profile_result.get('success'):
                total_reels = profile_result.get('total_reels_processed', 0)
                successful_count = profile_result.get('successful_extractions', 0)
                failed_count = profile_result.get('failed_extractions', 0)
                
                print(f"\nProfile Scraping Summary:")
                print(f"  Profile: {profile_result.get('profile_username', 'Unknown')}")
                print(f"  Reels discovered: {profile_result.get('total_reels_discovered', 0)}")
                print(f"  Reels processed: {total_reels}")
                print(f"  Successful extractions: {successful_count}")
                print(f"  Failed extractions: {failed_count}")
                if total_reels > 0:
                    success_rate = (successful_count / total_reels) * 100
                    print(f"  Success rate: {success_rate:.1f}%")
            else:
                print(f"\nProfile Scraping Summary:")
                print(f"  Status: Failed")
                print(f"  Error: {profile_result.get('error', 'Unknown error')}")
        else:
            # Regular URL processing summary
            successful_count = sum(1 for r in results if r.get('success', False))
            print(f"\nSummary:")
            print(f"  Total URLs processed: {len(results)}")
            print(f"  Successful extractions: {successful_count}")
            print(f"  Failed extractions: {len(results) - successful_count}")
        
        if args.verbose:
            print("\nDetailed results:")
            
            if hasattr(args, 'profile_url') and args.profile_url and results:
                # Profile scraping verbose output
                profile_result = results[0]
                if profile_result.get('success') and 'reels' in profile_result:
                    print(f"Profile: {profile_result.get('profile_username', 'Unknown')}")
                    print(f"Profile URL: {profile_result.get('profile_url', '')}")
                    print(f"Reels found: {len(profile_result['reels'])}")
                    print(f"\nIndividual reel results:")
                    
                    for i, reel in enumerate(profile_result['reels'], 1):
                        status = "✓" if reel.get('success', False) else "✗"
                        caption_preview = reel.get('caption', '')[:50] + '...' if reel.get('caption') else 'No caption'
                        method = reel.get('extraction_method', 'unknown')
                        position = reel.get('position_in_profile', i)
                        
                        print(f"  {position}. {status} {reel.get('url', '')} - {caption_preview}")
                        print(f"       Method: {method}")
                        
                        # Show transcription info if available
                        if reel.get('success') and 'processing_info' in reel:
                            proc_info = reel['processing_info']
                            print(f"       🎤 Whisper model: {proc_info.get('whisper_model', 'unknown')}")
                            print(f"       ⏱️  Processing time: {proc_info.get('processing_time', 0):.1f}s")
                            print(f"       📊 Confidence: {proc_info.get('confidence_score', 0):.2f}")
                            print(f"       🌐 Language: {proc_info.get('language_detected', 'unknown')}")
                        
                        # Show debug info if available
                        if debug_mode and 'debug_info' in reel:
                            debug_info = reel['debug_info']
                            print(f"       🔍 Debug files created:")
                            if debug_info.get('html_file'):
                                print(f"          📄 HTML: {debug_info['html_file']}")
                            if debug_info.get('screenshot_file'):
                                print(f"          📸 Screenshot: {debug_info['screenshot_file']}")
                            
                            # Show challenge detection
                            challenge_check = debug_info.get('challenge_check', {})
                            if challenge_check.get('has_challenge') or challenge_check.get('has_login_prompt'):
                                print(f"          ⚠️  Challenges detected: {challenge_check}")
                else:
                    print(f"Profile scraping failed: {profile_result.get('error', 'Unknown error')}")
            else:
                # Regular URL processing verbose output
                for i, result in enumerate(results, 1):
                    status = "✓" if result.get('success', False) else "✗"
                    caption_preview = result.get('caption', '')[:50] + '...' if result.get('caption') else 'No caption'
                    method = result.get('extraction_method', 'unknown')
                    print(f"  {i}. {status} {result.get('url', '')} - {caption_preview}")
                    print(f"       Method: {method}")
                    
                    # Show transcription info if available
                    if result.get('success') and 'processing_info' in result:
                        proc_info = result['processing_info']
                        print(f"       🎤 Whisper model: {proc_info.get('whisper_model', 'unknown')}")
                        print(f"       ⏱️  Processing time: {proc_info.get('processing_time', 0):.1f}s")
                        print(f"       📊 Confidence: {proc_info.get('confidence_score', 0):.2f}")
                        print(f"       🌐 Language: {proc_info.get('language_detected', 'unknown')}")
                    
                    # Show debug info if available
                    if debug_mode and 'debug_info' in result:
                        debug_info = result['debug_info']
                        print(f"       🔍 Debug files created:")
                        if debug_info.get('html_file'):
                            print(f"          📄 HTML: {debug_info['html_file']}")
                        if debug_info.get('screenshot_file'):
                            print(f"          📸 Screenshot: {debug_info['screenshot_file']}")
                        
                        # Show challenge detection
                        challenge_check = debug_info.get('challenge_check', {})
                        if challenge_check.get('has_challenge') or challenge_check.get('has_login_prompt'):
                            print(f"          ⚠️  Challenges detected: {challenge_check}")
        
        # Show debug summary if debug mode was used
        if debug_mode:
            print(f"\n🔍 Debug Information:")
            print(f"   Debug output directory: debug_output/")
            print(f"   HTML sources saved: {sum(1 for r in results if r.get('debug_info', {}).get('html_file'))}")
            print(f"   Screenshots taken: {sum(1 for r in results if r.get('debug_info', {}).get('screenshot_file'))}")
            print(f"   Use these files to analyze why caption extraction failed")
        
        return 0
        
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())