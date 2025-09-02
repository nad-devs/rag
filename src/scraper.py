import logging
import time
import random
import re
import os
import platform
import shutil
import subprocess
from urllib.parse import urlparse, parse_qs
from typing import Dict, List, Optional, Any, Tuple
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.common.exceptions import (
    TimeoutException, 
    NoSuchElementException, 
    WebDriverException,
    StaleElementReferenceException,
    ElementNotInteractableException,
    SessionNotCreatedException
)
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.firefox import GeckoDriverManager
from fake_useragent import UserAgent

from .config import Config
from .utils import setup_logging, save_data
from .video_transcriber import VideoTranscriber
from .debug_helper import DebugHelper


class InstagramScraper:
    """Main class for scraping Instagram video captions using Selenium WebDriver."""
    
    def __init__(self, config_path: str = "config.yaml", force_firefox: bool = False, debug_mode: bool = False, force_cpu: bool = False):
        """Initialize the scraper with configuration."""
        # Allow config_path to be either a string path or a Config object
        if isinstance(config_path, str):
            self.config = Config(config_path)
        else:
            self.config = config_path
        self.logger = setup_logging(self.config)
        self.driver = None
        self.wait = None
        self.force_cpu = force_cpu
        self.user_agent = UserAgent()
        self.force_firefox = force_firefox
        self.debug_mode = debug_mode
        self.video_transcriber = None
        self.debug_helper = None
        self._available_browsers = self._detect_available_browsers()
        self._shared_profile_dir = None  # For browser handoff shared profile
        self._use_existing_chrome = False  # For connecting to existing Chrome instance
        self._debug_port = 9222  # Remote debugging port
        self._chrome_process = None  # Chrome process for cleanup
        
    def _detect_available_browsers(self) -> Dict[str, bool]:
        """Detect which browsers are available on the system."""
        browsers = {'chrome': False, 'firefox': False}
        
        # Check for Chrome/Chromium
        chrome_paths = [
            '/usr/bin/google-chrome',
            '/usr/bin/google-chrome-stable',
            '/usr/bin/chromium-browser',
            '/usr/bin/chromium',
            '/opt/google/chrome/chrome',
            '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
            'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
            'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe'
        ]
        
        # WSL-specific Chrome paths
        if self._is_wsl():
            chrome_paths.extend([
                '/mnt/c/Program Files/Google/Chrome/Application/chrome.exe',
                '/mnt/c/Program Files (x86)/Google/Chrome/Application/chrome.exe'
            ])
        
        for path in chrome_paths:
            if os.path.exists(path) or shutil.which('google-chrome') or shutil.which('chromium'):
                browsers['chrome'] = True
                break
        
        # Check for Firefox
        firefox_paths = [
            '/usr/bin/firefox',
            '/Applications/Firefox.app/Contents/MacOS/firefox',
            'C:\\Program Files\\Mozilla Firefox\\firefox.exe',
            'C:\\Program Files (x86)\\Mozilla Firefox\\firefox.exe'
        ]
        
        if self._is_wsl():
            firefox_paths.extend([
                '/mnt/c/Program Files/Mozilla Firefox/firefox.exe',
                '/mnt/c/Program Files (x86)/Mozilla Firefox/firefox.exe'
            ])
        
        for path in firefox_paths:
            if os.path.exists(path) or shutil.which('firefox'):
                browsers['firefox'] = True
                break
        
        self.logger.info(f"Available browsers: {browsers}")
        return browsers
    
    def _is_wsl(self) -> bool:
        """Check if running in Windows Subsystem for Linux."""
        try:
            with open('/proc/version', 'r') as f:
                return 'microsoft' in f.read().lower() or 'wsl' in f.read().lower()
        except FileNotFoundError:
            return False
    
    def _is_wsl2(self) -> bool:
        """Check if running specifically in WSL2."""
        try:
            with open('/proc/version', 'r') as f:
                content = f.read().lower()
                return 'microsoft' in content and 'wsl2' in content
        except FileNotFoundError:
            return False
    
    def _setup_wsl_display_environment(self) -> bool:
        """Setup display environment for WSL2 GUI applications."""
        if not self._is_wsl():
            return True  # Not WSL, no setup needed
        
        try:
            # Check if WSLg is available (Windows 11 22000+ or Windows 10 with WSLg)
            if self._check_wslg_available():
                self.logger.info("🖥️  WSLg detected - configuring for native WSL GUI support")
                os.environ['DISPLAY'] = ':0'
                os.environ['WAYLAND_DISPLAY'] = 'wayland-0'
                return True
            
            # Fall back to X11 forwarding
            if self._setup_x11_forwarding():
                self.logger.info("🖥️  X11 forwarding configured")
                return True
            
            # No display solution available
            self.logger.error("❌ No WSL display solution available")
            return False
            
        except Exception as e:
            self.logger.error(f"Error setting up WSL display: {str(e)}")
            return False
    
    def _check_wslg_available(self) -> bool:
        """Check if WSLg (Windows Subsystem for Linux GUI) is available."""
        try:
            # WSLg typically sets up these environment variables
            wayland_display = os.environ.get('WAYLAND_DISPLAY')
            display = os.environ.get('DISPLAY')
            
            # Check for WSLg-specific paths
            wslg_paths = [
                '/tmp/.X11-unix/X0',
                '/mnt/wslg/.X11-unix/X0',
                '/run/user/1000/wayland-0'
            ]
            
            wslg_available = any(os.path.exists(path) for path in wslg_paths)
            
            if wslg_available:
                self.logger.debug("WSLg paths detected")
                return True
            
            # Check if DISPLAY or WAYLAND_DISPLAY are already set by WSLg
            if wayland_display or (display and display.startswith(':')):
                self.logger.debug("WSLg environment variables detected")
                return True
            
            return False
            
        except Exception as e:
            self.logger.debug(f"WSLg check failed: {str(e)}")
            return False
    
    def _setup_x11_forwarding(self) -> bool:
        """Setup X11 forwarding for WSL2."""
        try:
            # Get Windows host IP from resolv.conf
            windows_host_ip = self._get_windows_host_ip()
            if not windows_host_ip:
                return False
            
            # Set DISPLAY environment variable for X11 forwarding
            display_var = f"{windows_host_ip}:0"
            os.environ['DISPLAY'] = display_var
            
            self.logger.debug(f"Set DISPLAY={display_var}")
            
            # Test X11 connection
            if self._test_x11_connection():
                return True
            
            return False
            
        except Exception as e:
            self.logger.debug(f"X11 setup failed: {str(e)}")
            return False
    
    def _get_windows_host_ip(self) -> str:
        """Get Windows host IP address from WSL2."""
        try:
            # Method 1: Parse /etc/resolv.conf
            with open('/etc/resolv.conf', 'r') as f:
                for line in f:
                    if line.startswith('nameserver'):
                        ip = line.split()[1]
                        if ip and not ip.startswith('127.'):
                            return ip
            
            # Method 2: Use ip route command
            import subprocess
            result = subprocess.run(['ip', 'route', 'show', 'default'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'default via' in line:
                        parts = line.split()
                        if len(parts) >= 3:
                            return parts[2]
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Failed to get Windows host IP: {str(e)}")
            return None
    
    def _test_x11_connection(self) -> bool:
        """Test if X11 connection is working."""
        try:
            import subprocess
            
            # Simple X11 test with xset
            result = subprocess.run(['xset', 'q'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
            
        except Exception:
            return False
    
    def _show_wsl_display_help(self) -> None:
        """Show helpful instructions for WSL display setup."""
        print("\n" + "="*70)
        print("🖥️  WSL2 DISPLAY SETUP REQUIRED")
        print("="*70)
        print("Interactive login requires a browser window, but WSL2 has no display server.")
        print()
        print("SOLUTION OPTIONS:")
        print()
        print("1. 📱 USE WSLg (Recommended - Windows 11 or Windows 10 with WSLg):")
        print("   - Update to Windows 11 build 22000+ OR")
        print("   - Install WSLg on Windows 10: https://github.com/microsoft/wslg")
        print("   - Restart WSL: wsl --shutdown && wsl")
        print()
        print("2. 🔧 SETUP X11 FORWARDING:")
        print("   - Install X Server on Windows (e.g., VcXsrv, Xming)")
        print("   - Configure X Server to allow connections")
        print("   - Run: export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0")
        print()
        print("3. 🪟 RUN FROM WINDOWS POWERSHELL (Alternative):")
        print("   - Install Python on Windows")
        print("   - Clone this project to Windows filesystem")
        print("   - Run from Windows PowerShell instead of WSL2")
        print()
        print("4. ⚡ QUICK FIX - Use credential login instead:")
        print("   - Set credentials in config.yaml")
        print("   - Use --login instead of --interactive-login")
        print("="*70)
        print("For detailed setup instructions, see: https://docs.microsoft.com/en-us/windows/wsl/tutorials/gui-apps")
        print("="*70)
    
    def _find_chrome_binary(self) -> Optional[str]:
        """Find Chrome binary path for different environments."""
        chrome_paths = [
            '/usr/bin/google-chrome',
            '/usr/bin/google-chrome-stable',
            '/usr/bin/chromium-browser',
            '/usr/bin/chromium',
            '/opt/google/chrome/chrome',
            '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
        ]
        
        # WSL-specific paths
        if self._is_wsl():
            chrome_paths.extend([
                '/mnt/c/Program Files/Google/Chrome/Application/chrome.exe',
                '/mnt/c/Program Files (x86)/Google/Chrome/Application/chrome.exe'
            ])
        
        # Check which command
        chrome_which = shutil.which('google-chrome') or shutil.which('chromium')
        if chrome_which:
            return chrome_which
        
        # Check specific paths
        for path in chrome_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def _setup_driver(self) -> webdriver:
        """Set up and configure the WebDriver with fallback logic."""
        # Force Firefox if requested
        if self.force_firefox:
            self.logger.info("Firefox browser forced by user")
            return self._setup_firefox_driver()
        
        # Determine preferred browser
        preferred_browser = self.config.selenium.browser.lower()
        
        # Try preferred browser first
        if preferred_browser == "chrome" and self._available_browsers.get('chrome', False):
            try:
                self.logger.info("Attempting to setup Chrome driver")
                return self._setup_chrome_driver()
            except Exception as e:
                self.logger.warning(f"Chrome setup failed: {str(e)}")
                if self._available_browsers.get('firefox', False):
                    self.logger.info("Falling back to Firefox")
                    return self._setup_firefox_driver()
                else:
                    raise WebDriverException("Chrome failed and Firefox not available")
        
        elif preferred_browser == "firefox" and self._available_browsers.get('firefox', False):
            try:
                self.logger.info("Attempting to setup Firefox driver")
                return self._setup_firefox_driver()
            except Exception as e:
                self.logger.warning(f"Firefox setup failed: {str(e)}")
                if self._available_browsers.get('chrome', False):
                    self.logger.info("Falling back to Chrome")
                    return self._setup_chrome_driver()
                else:
                    raise WebDriverException("Firefox failed and Chrome not available")
        
        # If preferred browser not available, try any available browser
        if self._available_browsers.get('chrome', False):
            try:
                self.logger.info("Preferred browser not available, trying Chrome")
                return self._setup_chrome_driver()
            except Exception as e:
                self.logger.warning(f"Chrome fallback failed: {str(e)}")
        
        if self._available_browsers.get('firefox', False):
            try:
                self.logger.info("Trying Firefox as final fallback")
                return self._setup_firefox_driver()
            except Exception as e:
                self.logger.warning(f"Firefox fallback failed: {str(e)}")
        
        # No browsers available
        available = [k for k, v in self._available_browsers.items() if v]
        if not available:
            raise WebDriverException(
                "No supported browsers found. Please install Chrome or Firefox.\n"
                "For WSL: Install Chrome in Windows and ensure it's accessible from WSL."
            )
        else:
            raise WebDriverException(f"All browser setups failed. Available: {available}")
    
    def _setup_chrome_driver(self) -> webdriver.Chrome:
        """Set up Chrome WebDriver with advanced anti-detection options and WSL support."""
        options = ChromeOptions()
        
        # WSL display setup for non-headless mode
        if not self.config.selenium.headless and self._is_wsl():
            display_setup_success = self._setup_wsl_display_environment()
            if not display_setup_success:
                self.logger.error("❌ WSL display setup failed for interactive mode")
                raise WebDriverException(
                    "WSL2 display not configured for interactive browser mode. "
                    "See help instructions for WSLg or X11 setup."
                )
        
        # Find Chrome binary
        chrome_binary = self._find_chrome_binary()
        if chrome_binary:
            options.binary_location = chrome_binary
            self.logger.info(f"Using Chrome binary: {chrome_binary}")
        
        # Basic Chrome options
        if self.config.selenium.headless:
            options.add_argument("--headless=new")  # Use new headless mode
        
        # Essential arguments for WSL/Linux environments
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-software-rasterizer")
        options.add_argument("--disable-background-timer-throttling")
        options.add_argument("--disable-backgrounding-occluded-windows")
        options.add_argument("--disable-renderer-backgrounding")
        options.add_argument("--disable-features=TranslateUI")
        options.add_argument("--disable-ipc-flooding-protection")
        
        # WSL-specific arguments
        if self._is_wsl():
            options.add_argument("--disable-features=VizDisplayCompositor")
            options.add_argument("--disable-features=VizServiceDisplayCompositor")
            options.add_argument("--use-gl=swiftshader")
            options.add_argument("--single-process")  # Sometimes helps in WSL
        
        # Anti-detection arguments
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-web-security")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-plugins")
        options.add_argument("--disable-images")  # Faster loading
        options.add_argument("--disable-notifications")
        options.add_argument("--disable-popup-blocking")
        options.add_argument("--disable-default-apps")
        
        # Memory and performance optimizations
        options.add_argument("--memory-pressure-off")
        options.add_argument("--max_old_space_size=4096")
        
        # Experimental options to avoid detection
        options.add_experimental_option("excludeSwitches", ["enable-automation", "enable-logging"])
        options.add_experimental_option('useAutomationExtension', False)
        
        # Prefs to disable various features
        prefs = {
            "profile.default_content_setting_values": {
                "notifications": 2,
                "geolocation": 2,
                "media_stream": 2,
            },
            "profile.managed_default_content_settings": {
                "images": 2  # Block images for faster loading
            }
        }
        options.add_experimental_option("prefs", prefs)
        
        # FORCE DESKTOP USER AGENT to get the classic login page (not mobile version)
        # This prevents the language selection issue
        # Randomize the Chrome version slightly to avoid detection
        import random
        chrome_version = random.choice(["119", "120", "121", "122"])
        desktop_user_agent = f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{chrome_version}.0.0.0 Safari/537.36"
        options.add_argument(f"--user-agent={desktop_user_agent}")
        self.logger.info(f"Using desktop User-Agent (Chrome {chrome_version}) to force classic login page")
        
        # Window size - ALWAYS use desktop resolution to avoid mobile version
        # Mobile version triggers language selection issues
        if self.config.selenium.window_size:
            width = self.config.selenium.window_size.width
            height = self.config.selenium.window_size.height
            # Ensure minimum desktop size
            if width < 1024:
                width = 1920
                height = 1080
                self.logger.info("Overriding to desktop resolution to avoid mobile login page")
            options.add_argument(f"--window-size={width},{height}")
        else:
            # Default to desktop resolution
            options.add_argument("--window-size=1920,1080")
        
        # Connect to existing Chrome if browser handoff was used
        if hasattr(self, '_use_existing_chrome') and self._use_existing_chrome:
            print("🔗 Connecting to your Chrome window...")
            
            # Create minimal options for existing Chrome connection
            minimal_options = ChromeOptions()
            minimal_options.add_experimental_option("debuggerAddress", "localhost:9222")
            
            try:
                driver_path = ChromeDriverManager().install()
                service = ChromeService(driver_path)
                driver = webdriver.Chrome(service=service, options=minimal_options)
                print("✅ Connected! Automation will happen in your Chrome window.")
                return driver
            except Exception as e:
                print(f"❌ Connection failed: {str(e)}")
                return None
        
        # User data directory for session persistence (fallback mode)
        # Use a unique temp directory in headless mode to avoid conflicts
        if self.config.selenium.headless:
            import tempfile
            temp_dir = tempfile.mkdtemp(prefix="chrome_scraper_")
            options.add_argument(f"--user-data-dir={temp_dir}")
            self.logger.info(f"Using temporary Chrome profile: {temp_dir}")
        elif self.config.selenium.user_data_dir:
            options.add_argument(f"--user-data-dir={self.config.selenium.user_data_dir}")
        
        try:
            # Download and setup ChromeDriver (normal mode)
            self.logger.info("Downloading ChromeDriver...")
            driver_path = ChromeDriverManager().install()
            self.logger.info(f"ChromeDriver installed at: {driver_path}")
            
            service = ChromeService(driver_path)
            driver = webdriver.Chrome(service=service, options=options)
            
            # Execute anti-detection scripts
            self._apply_stealth_settings(driver)
            
            self.logger.info("Chrome driver initialized successfully")
            return driver
            
        except SessionNotCreatedException as e:
            error_msg = str(e)
            if "chrome not reachable" in error_msg.lower():
                raise WebDriverException(
                    "Chrome browser is not reachable. "
                    "Please ensure Chrome is installed and accessible.\n"
                    "For WSL: Install Chrome in Windows or use --use-firefox flag."
                )
            elif "unknown error: cannot find chrome binary" in error_msg.lower():
                raise WebDriverException(
                    "Chrome binary not found. "
                    "Please install Google Chrome or Chromium.\n"
                    "For WSL: Install Chrome in Windows or use --use-firefox flag."
                )
            else:
                raise WebDriverException(f"Chrome session creation failed: {error_msg}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Chrome driver: {str(e)}")
            raise WebDriverException(f"Chrome driver initialization failed: {str(e)}")
    
    def _setup_firefox_driver(self) -> webdriver.Firefox:
        """Set up Firefox WebDriver with anti-detection options and improved error handling."""
        options = FirefoxOptions()
        
        # WSL display setup for non-headless mode
        if not self.config.selenium.headless and self._is_wsl():
            display_setup_success = self._setup_wsl_display_environment()
            if not display_setup_success:
                self.logger.error("❌ WSL display setup failed for interactive mode")
                raise WebDriverException(
                    "WSL2 display not configured for interactive browser mode. "
                    "See help instructions for WSLg or X11 setup."
                )
        
        if self.config.selenium.headless:
            options.add_argument("--headless")
        
        # Firefox preferences for anti-detection
        options.set_preference("dom.webdriver.enabled", False)
        options.set_preference("useAutomationExtension", False)
        options.set_preference("general.platform.override", "Win32")
        options.set_preference("general.useragent.vendor", "")
        options.set_preference("general.useragent.vendorSub", "")
        
        # Performance and stability preferences
        options.set_preference("browser.tabs.remote.autostart", False)
        options.set_preference("browser.tabs.remote.autostart.2", False)
        options.set_preference("dom.ipc.plugins.enabled", False)
        options.set_preference("dom.ipc.processCount", 1)
        
        # Disable images for faster loading
        options.set_preference("permissions.default.image", 2)
        
        # Disable notifications and other distractions
        options.set_preference("dom.push.enabled", False)
        options.set_preference("dom.webnotifications.enabled", False)
        
        # WSL-specific preferences
        if self._is_wsl():
            options.set_preference("layers.acceleration.disabled", True)
            options.set_preference("gfx.direct2d.disabled", True)
            options.set_preference("webgl.disabled", True)
        
        # User agent rotation
        if self.config.anti_detection.user_agent_rotation:
            user_agent = self._get_random_user_agent()
            options.set_preference("general.useragent.override", user_agent)
            self.logger.debug(f"Using User-Agent: {user_agent}")
        
        try:
            # Download and setup GeckoDriver
            self.logger.info("Downloading GeckoDriver...")
            driver_path = GeckoDriverManager().install()
            self.logger.info(f"GeckoDriver installed at: {driver_path}")
            
            service = FirefoxService(driver_path)
            driver = webdriver.Firefox(service=service, options=options)
            
            # Set window size
            if self.config.selenium.window_size:
                width = self.config.selenium.window_size.width
                height = self.config.selenium.window_size.height
                driver.set_window_size(width, height)
            else:
                driver.set_window_size(1920, 1080)
            
            self.logger.info("Firefox driver initialized successfully")
            return driver
            
        except SessionNotCreatedException as e:
            error_msg = str(e)
            if "firefox" in error_msg.lower() and "binary" in error_msg.lower():
                raise WebDriverException(
                    "Firefox binary not found. "
                    "Please install Mozilla Firefox.\n"
                    "Ubuntu/Debian: sudo apt install firefox\n"
                    "CentOS/RHEL: sudo yum install firefox"
                )
            else:
                raise WebDriverException(f"Firefox session creation failed: {error_msg}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Firefox driver: {str(e)}")
            raise WebDriverException(f"Firefox driver initialization failed: {str(e)}")
    
    def start_session(self) -> None:
        """Start a new scraping session with enhanced error handling."""
        try:
            self.logger.info("Starting Instagram scraper session")
            self.logger.info(f"Platform: {platform.system()} {platform.release()}")
            wsl_detected = self._is_wsl()
            wsl2_detected = self._is_wsl2()
            self.logger.info(f"WSL detected: {wsl_detected} (WSL2: {wsl2_detected})")
            self.logger.info(f"Available browsers: {self._available_browsers}")
            
            # Check for WSL display environment if non-headless mode
            if wsl_detected and not self.config.selenium.headless:
                self.logger.info("🖥️  WSL non-headless mode - checking display environment")
                display_var = os.environ.get('DISPLAY')
                wayland_var = os.environ.get('WAYLAND_DISPLAY')
                self.logger.info(f"Environment: DISPLAY={display_var}, WAYLAND_DISPLAY={wayland_var}")
            
            self.driver = self._setup_driver()
            self.wait = WebDriverWait(self.driver, self.config.selenium.page_load_timeout)
            self.driver.implicitly_wait(self.config.selenium.implicit_wait)
            
            # Initialize video transcriber (primary method)
            self.video_transcriber = VideoTranscriber(self.driver, self.logger, force_cpu=self.force_cpu)
            transcriber_info = self.video_transcriber.get_system_info()
            self.logger.info(f"Video transcriber initialized: {transcriber_info}")
            
            # Initialize debug helper if needed
            if self.debug_mode:
                self.debug_helper = DebugHelper(self.driver, self.logger)
                self.logger.info("Debug helper initialized")
            
            # Test basic Instagram access
            self._test_instagram_access()
            
            browser_name = self.driver.capabilities.get('browserName', 'unknown')
            browser_version = self.driver.capabilities.get('browserVersion', 'unknown')
            self.logger.info(f"Driver setup completed successfully using {browser_name} {browser_version}")
            
        except WebDriverException as e:
            self.logger.error(f"WebDriver error: {str(e)}")
            if self.driver:
                self.close_session()
            
            # Provide helpful error messages
            error_msg = str(e).lower()
            
            # WSL display-related errors
            if self._is_wsl() and ("display" in error_msg or "cannot connect" in error_msg):
                self.logger.error("❌ WSL display configuration issue detected")
                self._show_wsl_display_help()
            elif "chrome not reachable" in error_msg or "cannot find chrome binary" in error_msg:
                self.logger.error(
                    "Chrome browser setup failed. Try one of these solutions:\n"
                    "1. Install Chrome: sudo apt update && sudo apt install google-chrome-stable\n"
                    "2. Use Firefox instead: python main.py --use-firefox [other args]\n"
                    "3. For WSL: Install Chrome in Windows and ensure PATH is accessible"
                )
            elif "firefox" in error_msg and "binary" in error_msg:
                self.logger.error(
                    "Firefox browser setup failed. Try installing Firefox:\n"
                    "Ubuntu/Debian: sudo apt install firefox\n"
                    "CentOS/RHEL: sudo yum install firefox"
                )
            elif "wsl2 display not configured" in error_msg:
                # This is our custom error, help is already shown
                pass
            
            raise
        except Exception as e:
            self.logger.error(f"Failed to start session: {str(e)}")
            if self.driver:
                self.close_session()
            raise
    
    def close_session(self) -> None:
        """Close the scraping session and cleanup."""
        try:
            # Clean up video transcriber temporary files
            if self.video_transcriber:
                self.video_transcriber.cleanup_temp_files()
                self.logger.info("Video transcriber cleanup completed")
            
            if self.driver:
                self.driver.quit()
                self.logger.info("Driver session closed successfully")
                
            # Clean up Chrome process if we started one for browser handoff
            if hasattr(self, '_chrome_process') and self._chrome_process:
                try:
                    self._chrome_process.terminate()
                    self.logger.info("Chrome process terminated")
                except:
                    pass
                    
        except Exception as e:
            self.logger.error(f"Error closing session: {str(e)}")
    
    def browser_handoff_login(self) -> bool:
        """Simple browser handoff - just open Chrome and wait."""
        try:
            # Find Chrome binary
            chrome_paths = ['/usr/bin/google-chrome', '/usr/bin/google-chrome-stable']
            chrome_binary = None
            for path in chrome_paths:
                if os.path.exists(path):
                    chrome_binary = path
                    break
            
            if not chrome_binary:
                print("❌ Chrome not found!")
                return False
            
            print("\n🔗 SIMPLE BROWSER HANDOFF")
            print("="*50)
            print("1. Chrome will open with Instagram login")
            print("2. Login manually")
            print("3. Press Enter here when done")
            print("4. Script will use the SAME Chrome window")
            print("="*50)
            
            # Open Chrome with remote debugging and user data directory
            import subprocess
            import tempfile
            data_dir = os.path.join(tempfile.gettempdir(), "chrome_handoff")
            os.makedirs(data_dir, exist_ok=True)
            
            subprocess.Popen([
                chrome_binary,
                "--remote-debugging-port=9222",
                f"--user-data-dir={data_dir}",
                "https://www.instagram.com/accounts/login/"
            ])
            
            print("✅ Chrome opened!")
            
            # Auto-detect if already logged in by checking the page
            import time
            time.sleep(3)  # Give Chrome time to load
            
            try:
                # Connect to existing Chrome to check login status
                from selenium import webdriver
                from selenium.webdriver.chrome.options import Options
                
                options = Options()
                options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
                
                temp_driver = webdriver.Chrome(options=options)
                current_url = temp_driver.current_url
                temp_driver.quit()
                
                # If already on Instagram (not login page), skip manual confirmation
                if "instagram.com" in current_url and "login" not in current_url:
                    print("✅ Already logged into Instagram - proceeding automatically")
                else:
                    input("\n🔗 Press Enter after login...")
                    
            except Exception:
                # Fallback to manual confirmation if auto-detection fails
                input("\n🔗 Press Enter after login...")
            
            # Set flag to use existing Chrome
            self._use_existing_chrome = True
            print("✅ Will use your Chrome window for automation")
            return True
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return False
    
    
    def interactive_login(self) -> bool:
        """Open browser for manual interactive login with user guidance."""
        try:
            self.logger.info("🔐 Starting interactive login mode")
            
            # Check WSL environment and display setup
            if self._is_wsl():
                self.logger.info("🖥️  WSL environment detected")
                
                # Verify display is properly configured
                display_var = os.environ.get('DISPLAY')
                wayland_var = os.environ.get('WAYLAND_DISPLAY')
                
                if not display_var and not wayland_var:
                    self.logger.error("❌ No display environment configured in WSL")
                    self._show_wsl_display_help()
                    return False
                
                self.logger.info(f"🖥️  Display configured: DISPLAY={display_var}, WAYLAND_DISPLAY={wayland_var}")
            
            # Navigate to Instagram login page
            try:
                self.driver.get(self.config.instagram.login_url)
                self._random_delay(1, 2)
            except Exception as e:
                if self._is_wsl() and "cannot connect to display" in str(e).lower():
                    self.logger.error("❌ Cannot connect to display server in WSL")
                    self._show_wsl_display_help()
                    return False
                raise
            
            # Handle any initial popups
            self._handle_popups()
            
            print("\n" + "="*70)
            print("🔐 INTERACTIVE INSTAGRAM LOGIN")
            print("="*70)
            print("A browser window has been opened for you to log in to Instagram.")
            print()
            print("Please complete the following steps:")
            print("1. Enter your Instagram username and password")
            print("2. Complete any 2FA verification if prompted")
            print("3. Handle any security challenges if they appear")
            print("4. Wait until you see your Instagram home feed")
            print()
            print("The script will automatically detect when login is successful.")
            print("Do NOT close the browser window.")
            print("="*70)
            
            # Wait for user to complete login manually
            max_wait_time = 600  # 10 minutes
            check_interval = 5   # Check every 5 seconds
            elapsed_time = 0
            
            while elapsed_time < max_wait_time:
                current_url = self.driver.current_url.lower()
                
                # Check if we're successfully logged in
                if self._is_logged_in():
                    print("✅ Login detected successfully!")
                    self.logger.info("✅ Interactive login completed successfully")
                    return True
                
                # Check if still on login page
                if "login" in current_url:
                    # Still on login page, keep waiting
                    time.sleep(check_interval)
                    elapsed_time += check_interval
                    
                    # Show progress every 30 seconds
                    if elapsed_time % 30 == 0:
                        remaining = (max_wait_time - elapsed_time) // 60
                        print(f"⏳ Waiting for login completion... ({remaining} minutes remaining)")
                        print("   Please complete login in the browser window")
                    
                    continue
                
                # Check for challenge pages
                if "challenge" in current_url:
                    print("🔐 Security challenge detected - please complete it in the browser")
                    time.sleep(check_interval)
                    elapsed_time += check_interval
                    continue
                
                # If we're not on login page and not logged in, something might be wrong
                if "instagram.com" in current_url:
                    # We're on Instagram but login status unclear, check again
                    time.sleep(check_interval)
                    elapsed_time += check_interval
                    continue
                else:
                    # Redirected away from Instagram
                    print("❌ Redirected away from Instagram. Please try again.")
                    return False
            
            # Timeout reached
            print("⏰ Login timeout reached. Please try again with more time.")
            self.logger.error("❌ Interactive login timed out")
            return False
            
        except Exception as e:
            self.logger.error(f"Error during interactive login: {str(e)}")
            print(f"❌ Interactive login error: {str(e)}")
            return False
    
    def login(self, username: str = None, password: str = None) -> bool:
        """Login to Instagram with provided or configured credentials."""
        try:
            # Use provided credentials or fall back to config
            if not username:
                username = self.config.instagram.credentials.username
            if not password:
                password = self.config.instagram.credentials.password
            
            if not username or not password:
                self.logger.warning("No Instagram credentials provided")
                return False
            
            self.logger.info(f"🔐 Attempting Instagram login for user: {username[:3]}***")
            # Use the main Instagram page and let it redirect to login
            self.driver.get("https://www.instagram.com/")
            self._random_delay(1, 2)  # Reduced from 2-3
            
            # Screenshot 1: Initial page (debug mode only)
            if self.debug_mode:
                self.driver.save_screenshot("login_step1_initial.png")
                self.logger.debug("Screenshot saved: login_step1_initial.png")
            
            # Try to click the "Open Instagram" button instead of "Log in" which goes to Facebook
            open_clicked = False
            try:
                # Look for the "Open Instagram" button first
                open_selectors = [
                    "//button[contains(text(), 'Open Instagram')]",
                    "//a[contains(text(), 'Open Instagram')]",
                    "//div[contains(text(), 'Open Instagram')]",
                    "//*[text()='Open Instagram']",
                    "button[contains(@class, 'primary')]"  # Sometimes it's the primary button
                ]
                
                for selector in open_selectors:
                    try:
                        if selector.startswith("//"):
                            open_button = self.driver.find_element(By.XPATH, selector)
                        elif selector.startswith("button["):
                            open_button = self.driver.find_element(By.CSS_SELECTOR, selector)
                        else:
                            open_button = self.driver.find_element(By.XPATH, selector)
                        
                        if open_button and open_button.is_displayed():
                            self.logger.info(f"Found 'Open Instagram' button with selector: {selector}")
                            open_button.click()
                            open_clicked = True
                            self.logger.info("✅ Clicked 'Open Instagram' button")
                            break
                    except:
                        continue
                
                if open_clicked:
                    self._random_delay(1, 1.5)  # Reduced from 2-3
                    self.logger.info(f"Navigated to: {self.driver.current_url}")
            except Exception as e:
                self.logger.warning(f"Could not find 'Open Instagram' button: {e}")
            
            # Fallback: If clicking didn't work, navigate directly
            current_url = self.driver.current_url
            if "login" not in current_url:
                self.logger.info("Fallback: Navigating directly to login page")
                self.driver.get("https://www.instagram.com/accounts/login/")
                self._random_delay(1, 2)  # Reduced from 2-3
                
            # Screenshot 2: After navigation to login (debug mode only)
            if self.debug_mode:
                self.driver.save_screenshot("login_step2_after_nav.png")
                self.logger.debug("Screenshot saved: login_step2_after_nav.png")
            
            # Handle any initial popups and cookie consent
            self._handle_popups()
            
            # Try to accept cookies if present
            try:
                cookie_button = self.driver.find_element(By.XPATH, "//button[contains(text(), 'Allow') or contains(text(), 'Accept')]")
                if cookie_button.is_displayed():
                    cookie_button.click()
                    self.logger.info("Accepted cookie consent")
                    self._random_delay(1, 2)
            except:
                pass  # No cookie banner
            
            # Language selector removed - not needed with desktop User-Agent
            
            # Log current page for debugging
            self.logger.info(f"Current URL: {self.driver.current_url}")
            self.logger.info(f"Page title: {self.driver.title}")
            
            # Wait for login form with multiple selector strategies
            username_input = self._find_username_input()
            if not username_input:
                self.logger.error("Username input field not found")
                # Save screenshot to see what page we're on
                try:
                    self.driver.save_screenshot("login_page_issue.png")
                    self.logger.info("Screenshot saved to login_page_issue.png")
                except:
                    pass
                return False
            
            # Clear and enter username
            username_input.clear()
            self._human_type(username_input, username)
            self.logger.info(f"Entered username: {username}")
            
            # Screenshot after username (debug mode only)
            if self.debug_mode:
                self.driver.save_screenshot("login_step4_after_username.png")
                self.logger.debug("Screenshot saved: login_step4_after_username.png")
            self._random_delay(0.5, 1)  # Reduced from 1-2
            
            # Find password input
            password_input = self._find_password_input()
            if not password_input:
                self.logger.error("Password input field not found")
                return False
            
            password_input.clear()
            self._human_type(password_input, password)
            self.logger.info("Entered password (hidden)")
            
            # Screenshot after password (debug mode only)
            if self.debug_mode:
                self.driver.save_screenshot("login_step5_before_submit.png")
                self.logger.debug("Screenshot saved: login_step5_before_submit.png")
            self._random_delay(0.5, 1)  # Reduced from 1-2
            
            # Find and click login button
            login_button = self._find_login_button()
            if not login_button:
                self.logger.error("Login button not found")
                return False
                
            login_button.click()
            self.logger.info("Clicked login button")
            
            # Wait briefly to see what happens
            self._random_delay(1, 2)  # Reduced from 2-3
            
            # Wait a bit longer to let Instagram process and redirect
            self.logger.info("⏳ Waiting for login to process...")
            self._random_delay(3, 4)  # Give Instagram time to redirect
            
            # Check where we are after waiting
            current_url = self.driver.current_url.lower()
            self.logger.info(f"URL after login processing: {current_url}")
            
            # Screenshot after clicking login (debug mode only)
            if self.debug_mode:
                self.driver.save_screenshot("login_step6_after_click.png")
                self.logger.debug("Screenshot saved: login_step6_after_click.png")
            
            # RETRY LOGIC: If we're still on login page, Instagram rejected the attempt
            login_attempts = 0
            max_attempts = 3  # Increase attempts since Instagram is picky
            
            while "accounts/login" in current_url and login_attempts < max_attempts:
                login_attempts += 1
                self.logger.warning(f"⚠️ Still on login page, restarting login process (attempt {login_attempts}/{max_attempts})...")
                
                # Check for error messages first
                error_msg = self._check_login_errors()
                if error_msg and ("incorrect" in error_msg.lower() or "wrong" in error_msg.lower()):
                    self.logger.error(f"❌ Login failed with error: {error_msg}")
                    return False
                elif error_msg and ("try again" in error_msg.lower() or "too many" in error_msg.lower()):
                    # Rate limited - wait longer
                    self.logger.warning(f"⚠️ Rate limited: {error_msg}. Waiting 30 seconds...")
                    self._random_delay(25, 35)
                    login_attempts -= 1  # Don't count this as an attempt
                    continue
                
                # Restart the EXACT same login process from the beginning
                self.logger.info("🔄 Restarting login from the beginning...")
                
                # Start from main Instagram page (same as initial login does)
                self.driver.get("https://www.instagram.com/")
                self._random_delay(1, 1.5)  # Reduced for retry
                
                # Try to click "Open Instagram" button (same as initial)
                open_clicked = False
                try:
                    open_selectors = [
                        "//button[contains(text(), 'Open Instagram')]",
                        "//a[contains(text(), 'Open Instagram')]",
                        "//div[contains(text(), 'Open Instagram')]",
                        "//*[text()='Open Instagram']",
                        "button[contains(@class, 'primary')]"
                    ]
                    
                    for selector in open_selectors:
                        try:
                            if selector.startswith("//"):
                                open_button = self.driver.find_element(By.XPATH, selector)
                            elif selector.startswith("button["):
                                open_button = self.driver.find_element(By.CSS_SELECTOR, selector)
                            else:
                                open_button = self.driver.find_element(By.XPATH, selector)
                            
                            if open_button and open_button.is_displayed():
                                open_button.click()
                                open_clicked = True
                                self.logger.info("✅ Clicked 'Open Instagram' button")
                                break
                        except:
                            continue
                    
                    if open_clicked:
                        self._random_delay(1, 1.5)  # Reduced for retry
                except:
                    pass
                
                # Fallback: Navigate directly to login page (same as initial)
                current_url = self.driver.current_url
                if "login" not in current_url:
                    self.logger.info("Fallback: Navigating directly to login page")
                    self.driver.get("https://www.instagram.com/accounts/login/")
                    self._random_delay(1, 1.5)  # Reduced for retry
                
                # Handle any popups and cookie consent (same as initial)
                self._handle_popups()
                
                # Try to accept cookies if present
                try:
                    cookie_button = self.driver.find_element(By.XPATH, "//button[contains(text(), 'Allow') or contains(text(), 'Accept')]")
                    if cookie_button.is_displayed():
                        cookie_button.click()
                        self.logger.info("Accepted cookie consent")
                        self._random_delay(1, 2)
                except:
                    pass  # No cookie banner
                
                # Language selector not needed with desktop User-Agent
                
                # Log current page (same as initial)
                self.logger.info(f"Current URL: {self.driver.current_url}")
                self.logger.info(f"Page title: {self.driver.title}")
                
                # Find username input (same as initial)
                username_input = self._find_username_input()
                if not username_input:
                    self.logger.error("Username input field not found on retry")
                    self.driver.save_screenshot(f"login_retry_failed_{login_attempts}.png")
                    continue
                
                # Clear and enter username (same as initial)
                username_input.clear()
                self._human_type(username_input, username)
                self.logger.info(f"Entered username: {username}")
                self._random_delay(0.5, 1)  # Reduced for retry
                
                # Find password input (same as initial)
                password_input = self._find_password_input()
                if not password_input:
                    self.logger.error("Password input field not found on retry")
                    continue
                
                # Clear and enter password (same as initial)
                password_input.clear()
                self._human_type(password_input, password)
                self.logger.info("Entered password (hidden)")
                self._random_delay(0.5, 1)  # Reduced for retry
                
                # Find and click login button (same as initial)
                login_button = self._find_login_button()
                if not login_button:
                    self.logger.error("Login button not found on retry")
                    continue
                
                login_button.click()
                self.logger.info(f"Clicked login button (retry {login_attempts})")
                
                # Wait and check result (same as initial)
                self._random_delay(1, 2)  # Reduced for retry
                
                # Update current URL
                current_url = self.driver.current_url.lower()
                self.logger.info(f"URL after retry: {current_url}")
            
            # After retry attempts, take a final screenshot (debug mode only)
            if self.debug_mode:
                self.driver.save_screenshot("login_final_state.png")
                self.logger.debug("Screenshot saved: login_final_state.png")
            
            # Check current URL and handle different scenarios
            current_url = self.driver.current_url.lower()
            self.logger.info(f"Post-login URL: {current_url}")
            
            # If still on login page after retries, login failed
            if "accounts/login" in current_url:
                error_msg = self._check_login_errors()
                self.logger.error(f"❌ Login failed after retries: {error_msg}")
                self.driver.save_screenshot("login_failed.png")
                return False
            
            # Check for specific post-login pages
            if "onetap" in current_url:
                # This is the "Save Login Info" page
                self.logger.info("📱 'Save Login Info' page detected")
                self.driver.save_screenshot("save_login_info_page.png")
                
                # First check if this is the "Save Login Info" page (happens after successful login)
                save_info_found = False
                try:
                    # Check for "Save Your Login Info?" text or similar
                    page_text = self.driver.page_source.lower()
                    save_info_indicators = [
                        "save your login",
                        "save login info",
                        "remember password",
                        "keep me logged in"
                    ]
                    
                    if any(indicator in page_text for indicator in save_info_indicators):
                        self.logger.info("📱 'Save Login Info' page detected - clicking Save Info")
                        save_info_found = True
                        
                        # Try to click "Save Info" or "Not Now" button
                        save_buttons = [
                            "//button[contains(text(), 'Save Info')]",
                            "//button[contains(text(), 'Save')]",
                            "//button[contains(@aria-label, 'Save')]",
                            "button:contains('Save Info')",
                            "button[type='button']"  # Sometimes it's just a generic button
                        ]
                        
                        for selector in save_buttons:
                            try:
                                if selector.startswith("//"):
                                    buttons = self.driver.find_elements(By.XPATH, selector)
                                else:
                                    buttons = self.driver.find_elements(By.CSS_SELECTOR, selector)
                                    
                                for btn in buttons:
                                    btn_text = btn.text.lower()
                                    if btn.is_displayed() and ("save" in btn_text or btn_text == ""):
                                        btn.click()
                                        self.logger.info(f"✅ Clicked 'Save Info' button")
                                        self._random_delay(2, 3)
                                        save_info_found = True
                                        break
                                if save_info_found:
                                    break
                            except:
                                continue
                except Exception as e:
                    self.logger.debug(f"Error checking for Save Login Info page: {e}")
                
            
            # Only ask for OTP if we're on a two-factor/challenge page
            elif "two_factor" in current_url or "challenge" in current_url:
                self.logger.warning("🔐 Two-factor authentication required")
                self.driver.save_screenshot("otp_required.png")
                
                print("\n" + "="*60)
                print("🔐 INSTAGRAM TWO-FACTOR AUTHENTICATION")
                print("="*60)
                print("Enter the verification code sent to your phone/email.")
                print("="*60)
                
                otp_code = input("📱 Enter OTP code (or press Enter to skip): ").strip()
                
                if otp_code:
                    # Try to find and enter OTP
                    otp_entered = False
                    otp_selectors = [
                        "input[name='verificationCode']",
                        "input[aria-label*='code']",
                        "input[placeholder*='code']",
                        "input[type='tel']",
                        "input[type='number']",
                        "input[maxlength='6']",
                        "input[maxlength='8']",
                        "input:not([type='hidden']):not([type='password']):not([name='username'])"  # Any visible input
                    ]
                    
                    for selector in otp_selectors:
                        try:
                            inputs = self.driver.find_elements(By.CSS_SELECTOR, selector)
                            for otp_input in inputs:
                                if otp_input.is_displayed():
                                    otp_input.clear()
                                    otp_input.send_keys(otp_code)
                                    self.logger.info(f"Entered OTP in field: {selector}")
                                    otp_entered = True
                                    
                                    # Screenshot after entering OTP
                                    self.driver.save_screenshot("otp_entered.png")
                                    self.logger.info("Screenshot saved: otp_entered.png")
                                    
                                    # Try to submit OTP
                                    try:
                                        # Press Enter or find submit button
                                        otp_input.send_keys(Keys.RETURN)
                                        self.logger.info("Pressed Enter to submit OTP")
                                    except:
                                        # Try to find submit button
                                        for btn_sel in ["button[type='submit']", "button"]:
                                            try:
                                                buttons = self.driver.find_elements(By.CSS_SELECTOR, btn_sel)
                                                for btn in buttons:
                                                    if btn.is_displayed() and btn.is_enabled():
                                                        btn.click()
                                                        self.logger.info("Clicked submit button for OTP")
                                                        break
                                            except:
                                                continue
                                    break
                            if otp_entered:
                                break
                        except Exception as e:
                            continue
                    
                    if otp_entered:
                        print("✅ OTP entered, verifying...")
                        self._random_delay(3, 5)
                        
                        # Check if login successful after OTP
                        if self._is_logged_in():
                            self.logger.info("✅ Login successful after OTP!")
                            return True
                    else:
                        print("❌ Could not find OTP input field")
                        self.driver.save_screenshot("otp_field_not_found.png")
            
            # Handle 2FA/OTP if required - check URL and page content
            if "two_factor" in current_url or "challenge" in current_url:
                self.logger.warning("🔐 Two-factor authentication/OTP required")
                self.driver.save_screenshot("otp_required.png")
                self.logger.info("Screenshot saved: otp_required.png")
                return self._handle_two_factor_auth()
            
            # Also check for OTP input field presence
            try:
                otp_input = self.driver.find_element(By.CSS_SELECTOR, "input[name='verificationCode'], input[aria-label*='code'], input[placeholder*='code']")
                if otp_input.is_displayed():
                    self.logger.warning("🔐 OTP verification detected")
                    self.driver.save_screenshot("otp_page.png")
                    return self._handle_two_factor_auth()
            except:
                pass
            
            # Handle login errors - but give it more time and double-check
            if "login" in current_url and "two_factor" not in current_url:
                # Wait a bit more as Instagram sometimes takes time to redirect
                self._random_delay(3, 5)
                
                # Re-check URL after waiting
                current_url = self.driver.current_url.lower()
                
                # Double-check if we're actually logged in (sometimes URL updates slowly)
                if self._is_logged_in():
                    self.logger.info("✅ Login successful despite being on login URL")
                    return True
                    
                # If still on login page, check for errors
                if "login" in current_url:
                    error_msg = self._check_login_errors()
                    self.logger.error(f"❌ Login failed: {error_msg}")
                    # Save screenshot for debugging
                    try:
                        self.driver.save_screenshot("login_failed.png")
                        self.logger.info("Screenshot saved to login_failed.png")
                    except:
                        pass
                    return False
            
            # Handle suspicious activity challenges
            if "challenge" in current_url or "suspicious" in current_url:
                self.logger.warning("⚠️  Account requires manual verification due to suspicious activity")
                return False
            
            # Handle post-login popups and challenges
            self._handle_post_login_flow()
            
            # Verify login success - but be more lenient
            # If we're not on login page and no errors, assume success
            final_url = self.driver.current_url.lower()
            
            # Definitely failed if still on login page
            if "accounts/login" in final_url:
                error_msg = self._check_login_errors()
                self.logger.error(f"❌ Still on login page: {error_msg}")
                return False
            
            # Success indicators - check URL patterns
            # Remove any trailing # or parameters for cleaner comparison
            clean_url = final_url.split('#')[0].split('?')[0]
            
            success_indicators = [
                clean_url == "https://www.instagram.com/",  # Homepage exact
                clean_url == "https://www.instagram.com",  # Homepage without slash
                clean_url == "http://www.instagram.com/",  # HTTP version
                clean_url == "http://www.instagram.com",  # HTTP without slash
                "/accounts/edit" in final_url,  # Edit profile
                "/accounts/onetap" in final_url,  # Save login
                "/direct" in final_url,  # Messages
                # Most important: if we're on instagram.com but NOT on login page
                ("instagram.com" in final_url and "accounts/login" not in final_url)
            ]
            
            if any(success_indicators):
                self.logger.info("✅ Login successful (URL indicates logged in)")
                return True
            
            # Quick check for basic navigation elements (don't wait too long)
            try:
                # Just check if we can find ANY Instagram navigation element quickly
                self.logger.info("🔍 Quick check for logged-in elements...")
                quick_check = WebDriverWait(self.driver, 3).until(
                    lambda driver: (
                        driver.find_elements(By.CSS_SELECTOR, "svg[aria-label='Home']") or
                        driver.find_elements(By.CSS_SELECTOR, "a[href='/']") or
                        driver.find_elements(By.CSS_SELECTOR, "nav") or
                        driver.find_elements(By.CSS_SELECTOR, "[role='navigation']")
                    )
                )
                if quick_check:
                    self.logger.info("✅ Login successful (found navigation elements)")
                    return True
            except TimeoutException:
                self.logger.warning("⚠️ Could not find navigation elements quickly")
            
            # If we're not on login page and no errors, assume success
            if "accounts/login" not in final_url:
                self.logger.info("✅ Login successful (not on login page, assuming success)")
                return True
                
            self.logger.error("❌ Login verification failed")
            return False
                
        except Exception as e:
            self.logger.error(f"Login error: {str(e)}")
            return False
    
    def _find_username_input(self):
        """Find username input field using multiple strategies."""
        username_selectors = [
            "input[name='username']",
            "input[aria-label*='username']", 
            "input[aria-label*='Phone number, username, or email']",
            "input[placeholder*='username']",
            "input[placeholder*='Phone number, username, or email']",
            "input[type='text']",
            "form input[type='text']",
            "input[autocomplete='username']"
        ]
        
        for selector in username_selectors:
            try:
                element = self.wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, selector)))
                self.logger.debug(f"Found username input with: {selector}")
                return element
            except TimeoutException:
                continue
        
        return None
    
    def _find_password_input(self):
        """Find password input field using multiple strategies."""
        password_selectors = [
            "input[name='password']",
            "input[aria-label*='password']",
            "input[aria-label*='Password']", 
            "input[type='password']",
            "form input[type='password']",
            "input[autocomplete='current-password']"
        ]
        
        for selector in password_selectors:
            try:
                element = self.driver.find_element(By.CSS_SELECTOR, selector)
                self.logger.debug(f"Found password input with: {selector}")
                return element
            except NoSuchElementException:
                continue
        
        return None
    
    def _find_login_button(self):
        """Find login button using multiple strategies."""
        # First try CSS selectors
        login_selectors = [
            "button[type='submit']",
            "form button",
            "div[role='button']"
        ]
        
        for selector in login_selectors:
            try:
                element = self.driver.find_element(By.CSS_SELECTOR, selector)
                if element.is_displayed() and element.is_enabled():
                    self.logger.debug(f"Found login button with: {selector}")
                    return element
            except NoSuchElementException:
                continue
        
        # Then try by text content
        try:
            buttons = self.driver.find_elements(By.TAG_NAME, "button")
            for btn in buttons:
                if btn.text and "log in" in btn.text.lower():
                    self.logger.debug(f"Found login button by text: '{btn.text}'")
                    return btn
        except Exception:
            pass
        
        return None
    
    def _check_login_errors(self):
        """Check for login error messages on the page."""
        error_selectors = [
            "[role='alert']",
            ".error",
            "[data-testid='login-error-message']", 
            "p[role='alert']",
            "div[role='alert']",
            "span[role='alert']"
        ]
        
        for selector in error_selectors:
            try:
                error_elem = self.driver.find_element(By.CSS_SELECTOR, selector)
                if error_elem.is_displayed() and error_elem.text:
                    error_text = error_elem.text.lower()
                    # Only return error if it's actually about login
                    if any(word in error_text for word in ['incorrect', 'wrong', 'invalid', 'password', 'username', 'login', 'sorry']):
                        return error_elem.text
            except NoSuchElementException:
                continue
        
        # Check for specific error messages in page text, but be more careful
        try:
            page_text = self.driver.page_source.lower()
            
            # Check for actual login error messages (not just the word appearing anywhere)
            if "sorry, your password was incorrect" in page_text:
                return "Incorrect username or password"
            elif "the username you entered doesn't belong to an account" in page_text:
                return "Username not found"
            elif "we couldn't connect to instagram" in page_text:
                return "Connection error"
            elif "too many failed attempts" in page_text or "too many login attempts" in page_text:
                return "Too many login attempts"
            elif "please wait a few minutes" in page_text:
                return "Rate limited - please wait"
                
            # Don't return error just because these words appear somewhere on the page
            # (they might be in language selection or other unrelated content)
        except:
            pass
        
        return "Unknown login error"
    
    def _handle_two_factor_auth(self):
        """Handle two-factor authentication with user interaction."""
        try:
            self.logger.info("🔐 Two-factor authentication detected")
            print("\n" + "="*60)
            print("🔐 INSTAGRAM OTP/2FA VERIFICATION REQUIRED")
            print("="*60)
            print("Instagram sent an OTP to your WhatsApp/SMS/Email")
            print()
            
            # Since we're in headless mode, ask user for the code
            otp_code = input("📱 Please enter the OTP code you received: ").strip()
            
            if not otp_code:
                print("❌ No code entered")
                return False
            
            # Find OTP input field
            otp_selectors = [
                "input[name='verificationCode']",
                "input[aria-label*='code']",
                "input[placeholder*='code']",
                "input[type='tel']",  # Often OTP fields are tel type
                "input[type='number']",
                "input[maxlength='6']",  # Common for 6-digit OTP
                "input[maxlength='8']"   # Some OTPs are 8 digits
            ]
            
            otp_input = None
            for selector in otp_selectors:
                try:
                    otp_input = self.driver.find_element(By.CSS_SELECTOR, selector)
                    if otp_input.is_displayed():
                        self.logger.info(f"Found OTP input with selector: {selector}")
                        break
                except:
                    continue
            
            if not otp_input:
                print("❌ Could not find OTP input field")
                self.driver.save_screenshot("otp_field_not_found.png")
                return False
            
            # Enter the OTP
            otp_input.clear()
            otp_input.send_keys(otp_code)
            self.logger.info(f"Entered OTP: {otp_code[:2]}****")
            
            # Find and click submit button
            submit_selectors = [
                "button[type='submit']",
                "button:contains('Confirm')",
                "button:contains('Submit')",
                "button:contains('Verify')",
                "button:contains('Next')"
            ]
            
            for selector in submit_selectors:
                try:
                    if "contains" in selector:
                        # Use XPath for text content
                        text = selector.split("'")[1]
                        button = self.driver.find_element(By.XPATH, f"//button[contains(text(), '{text}')]")
                    else:
                        button = self.driver.find_element(By.CSS_SELECTOR, selector)
                    
                    if button.is_displayed() and button.is_enabled():
                        button.click()
                        self.logger.info("Clicked OTP submit button")
                        break
                except:
                    continue
            
            print("⏳ Verifying OTP...")
            self._random_delay(3, 5)
            
            # Wait for user to complete 2FA manually
            max_wait_time = 300  # 5 minutes
            check_interval = 3   # Check every 3 seconds
            elapsed_time = 0
            
            while elapsed_time < max_wait_time:
                current_url = self.driver.current_url.lower()
                
                # Check if we've moved past 2FA
                if "two_factor" not in current_url:
                    # Check if we're successfully logged in
                    if self._is_logged_in():
                        print("✅ 2FA completed successfully! Login successful.")
                        self.logger.info("✅ 2FA authentication successful")
                        return True
                    elif "login" in current_url:
                        print("❌ 2FA failed or was cancelled.")
                        self.logger.error("❌ 2FA authentication failed")
                        return False
                    elif "challenge" in current_url:
                        print("⚠️  Additional verification required.")
                        self.logger.warning("⚠️  Account requires additional verification")
                        return False
                
                # Still on 2FA page, keep waiting
                time.sleep(check_interval)
                elapsed_time += check_interval
                
                # Show progress every 30 seconds
                if elapsed_time % 30 == 0:
                    remaining = (max_wait_time - elapsed_time) // 60
                    print(f"⏳ Still waiting for 2FA completion... ({remaining} minutes remaining)")
            
            # Timeout reached
            print("⏰ 2FA timeout reached. Please try again.")
            self.logger.error("❌ 2FA authentication timed out")
            return False
            
        except Exception as e:
            self.logger.error(f"Error handling 2FA: {str(e)}")
            return False
    
    def _human_type(self, element, text: str):
        """Type text in a human-like manner."""
        for char in text:
            element.send_keys(char)
            time.sleep(random.uniform(0.05, 0.15))
    
    def _is_logged_in(self) -> bool:
        """Check if successfully logged into Instagram."""
        try:
            current_url = self.driver.current_url.lower()
            
            # Check URL indicators
            if "login" in current_url or "accounts/login" in current_url:
                return False
            
            if "challenge" in current_url:
                self.logger.warning("⚠️  Account challenged - may need manual verification")
                return False
            
            # Look for logged-in indicators
            logged_in_indicators = [
                "nav[role='navigation']",
                "a[href='/explore/']",
                "button[aria-label='New post']",
                "[data-testid='user-avatar']"
            ]
            
            for selector in logged_in_indicators:
                try:
                    if self.driver.find_elements(By.CSS_SELECTOR, selector):
                        return True
                except:
                    continue
            
            # Check if we can access a protected endpoint
            try:
                self.driver.get("https://www.instagram.com/accounts/edit/")
                self._random_delay(2, 3)
                if "accounts/edit" in self.driver.current_url:
                    return True
            except:
                pass
            
            return False
            
        except Exception as e:
            self.logger.debug(f"Login check error: {str(e)}")
            return False
    
    def _handle_post_login_flow(self):
        """Handle post-login prompts and popups."""
        try:
            # Common post-login popups to handle
            popup_selectors = [
                # "Turn on notifications" popup
                "button[aria-label='Not Now']",
                "button:contains('Not Now')",
                
                # "Save login info" popup  
                "button[role='button']:contains('Not Now')",
                
                # "Add to home screen" popup
                "button:contains('Cancel')",
                
                # Generic close buttons
                "button[aria-label='Close']",
                "svg[aria-label='Close']",
                
                # "Get the app" prompts
                "button:contains('Not now')",
                "a:contains('Not now')"
            ]
            
            for _ in range(3):  # Try multiple times as popups can appear sequentially
                self._random_delay(1, 2)
                popup_handled = False
                
                for selector in popup_selectors:
                    try:
                        elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                        for element in elements:
                            if element.is_displayed() and element.is_enabled():
                                element.click()
                                self.logger.debug(f"Handled post-login popup: {selector}")
                                popup_handled = True
                                self._random_delay(1, 2)
                                break
                    except Exception:
                        continue
                    
                    if popup_handled:
                        break
                
                if not popup_handled:
                    break
                    
        except Exception as e:
            self.logger.debug(f"Error handling post-login flow: {str(e)}")
    
    def bypass_login_wall(self) -> bool:
        """Attempt to bypass Instagram's login wall for public content."""
        try:
            current_url = self.driver.current_url
            
            # First check if we're already logged in - if so, no need to bypass anything
            if self._is_logged_in():
                self.logger.debug("Already logged in, no login wall to bypass")
                return True
            
            # Check if we're actually on the login page (URL check only)
            if "accounts/login" in current_url.lower():
                self.logger.warning("🚧 On login page - login wall detected")
                login_wall_detected = True
            else:
                # Only check page source for specific login wall text, not URLs
                login_wall_indicators = [
                    "Sign up to see photos and videos",
                    "Keep watching in the app",
                    "Sign up for Instagram",
                    self.config.instagram.selectors.signup_prompt,
                    self.config.instagram.selectors.app_download
                ]
                
                page_source = self.driver.page_source.lower()
                
                login_wall_detected = any(
                    indicator.lower() in page_source
                    for indicator in login_wall_indicators if indicator
                )
            
            if login_wall_detected:
                self.logger.warning("🚧 Login wall detected - attempting bypass")
                
                # Try to close login prompts
                close_selectors = [
                    "button[aria-label='Close']",
                    "svg[aria-label='Close']",
                    "button:contains('Not now')",
                    "a:contains('Not now')"
                ]
                
                for selector in close_selectors:
                    try:
                        elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                        for element in elements:
                            if element.is_displayed():
                                element.click()
                                self.logger.info(f"Closed login prompt: {selector}")
                                self._random_delay(1, 2)
                                return True
                    except Exception:
                        continue
                
                # If auto-login is enabled, try logging in
                if self.config.instagram.credentials.auto_login:
                    self.logger.info("🔄 Auto-login enabled, attempting login")
                    return self.login()
                
                return False
            
            return True  # No login wall detected
            
        except Exception as e:
            self.logger.error(f"Error bypassing login wall: {str(e)}")
            return False
    
    def extract_video_caption(self, video_url: str, save_html: bool = False, take_screenshot: bool = False) -> Optional[Dict[str, Any]]:
        """Extract caption from Instagram video using video download and transcription."""
        # Validate URL first
        if not self.is_valid_instagram_url(video_url):
            self.logger.error(f"Invalid Instagram URL: {video_url}")
            return {
                "url": video_url,
                "caption": None,
                "timestamp": time.time(),
                "success": False,
                "error": "Invalid URL format"
            }
        
        try:
            self.logger.info(f"Starting video transcription for: {video_url}")
            
            # Navigate to URL with error handling
            success = self._navigate_to_url(video_url)
            if not success:
                return {
                    "url": video_url,
                    "caption": None,
                    "timestamp": time.time(),
                    "success": False,
                    "error": "Failed to navigate to URL"
                }
            
            self._random_delay()
            
            # Handle login wall if encountered
            if not self.bypass_login_wall():
                self.logger.error("🚧 Could not bypass login wall")
                return {
                    "url": video_url,
                    "caption": None,
                    "timestamp": time.time(),
                    "success": False,
                    "error": "Login wall encountered - requires Instagram authentication"
                }
            
            # Run debugging if enabled (fast mode for production)
            debug_info = None
            if self.debug_mode and self.debug_helper and (save_html or take_screenshot):
                self.logger.info("🔍 Running page analysis for video extraction...")
                debug_info = self.debug_helper.comprehensive_page_debug(
                    video_url, 
                    save_html=save_html, 
                    take_screenshot=take_screenshot
                )
                
                # Log all found text for manual inspection
                if self.logger.level <= 10:  # DEBUG level is 10
                    self.debug_helper.log_all_found_text()
            
            # Extract shortcode for processing
            shortcode = self.extract_shortcode_from_url(video_url)
            if not shortcode:
                shortcode = f"unknown_{int(time.time())}"
            
            # Main extraction: Video download and transcription
            self.logger.info("🎥 Starting video download and transcription...")
            transcription_result = self.video_transcriber.process_video(video_url, shortcode)
            
            # Prepare base response
            caption_data = {
                "url": video_url,
                "caption": None,
                "timestamp": time.time(),
                "success": False,
                "shortcode": shortcode,
                "url_type": self.get_url_type(video_url),
                "extraction_method": "video_transcription"
            }
            
            # Add debug info if available
            if debug_info:
                caption_data["debug_info"] = debug_info
            
            # Process transcription result
            if transcription_result and transcription_result.get('success'):
                caption_data.update({
                    "caption": transcription_result['caption'],
                    "success": True,
                    "transcription_data": transcription_result.get('transcription_data', {}),
                    "video_url_extracted": transcription_result.get('video_url'),
                    "processing_info": {
                        "method": "video_download_and_transcription",
                        "whisper_model": transcription_result.get('transcription_data', {}).get('model_used', 'unknown'),
                        "confidence_score": transcription_result.get('transcription_data', {}).get('confidence_score', 0),
                        "processing_time": transcription_result.get('transcription_data', {}).get('processing_time', 0),
                        "language_detected": transcription_result.get('transcription_data', {}).get('language', 'en')
                    }
                })
                
                caption_text = transcription_result['caption']
                segments_count = len(transcription_result.get('transcription_data', {}).get('segments', []))
                self.logger.info(f"✅ Video transcription successful: {len(caption_text)} characters, {segments_count} segments")
                
            else:
                # Transcription failed
                error_msg = transcription_result.get('error', 'Unknown transcription error') if transcription_result else 'Video transcription returned no result'
                caption_data.update({
                    "success": False,
                    "error": error_msg
                })
                self.logger.error(f"❌ Video transcription failed: {error_msg}")
            
            return caption_data
            
        except Exception as e:
            self.logger.error(f"Error in video caption extraction: {str(e)}")
            return {
                "url": video_url,
                "caption": None,
                "timestamp": time.time(),
                "success": False,
                "error": str(e),
                "extraction_method": "video_transcription"
            }
    
    def extract_multiple_captions(self, video_urls: List[str]) -> List[Dict[str, Any]]:
        """Extract captions from multiple Instagram videos."""
        results = []
        
        for i, url in enumerate(video_urls, 1):
            self.logger.info(f"Processing video {i}/{len(video_urls)}")
            caption_data = self.extract_video_caption(url)
            
            if caption_data:
                results.append(caption_data)
            
            if i < len(video_urls):
                self._random_delay()
        
        self.logger.info(f"Completed processing {len(results)} videos")
        return results
    
    def scrape_profile_reels(self, profile_url: str, max_reels: int = 5, skip_first: int = 0, save_html: bool = False, take_screenshot: bool = False, validation_mode: bool = False, existing_shortcodes: Optional[set] = None, incremental_mode: bool = False, incremental_data: Optional[Dict] = None) -> Dict[str, Any]:
        """Scrape reels from an Instagram profile.
        
        Args:
            profile_url: Instagram profile URL (e.g., https://www.instagram.com/username/)
            max_reels: Maximum number of reels to discover (default: 5)
            skip_first: Skip the first N reels when processing (default: 0)
            save_html: Whether to save HTML for debugging
            take_screenshot: Whether to take screenshots for debugging
            validation_mode: If True, only process reels not in existing_shortcodes
            existing_shortcodes: Set of shortcodes already processed (validation mode only)
            incremental_mode: If True, only process new reels posted after the latest existing reel
            incremental_data: Dict containing latest position, post date, and existing shortcodes for incremental updates
            
        Returns:
            Dict containing profile metadata and reel processing results
        """
        # Validate profile URL
        if not self.is_valid_instagram_profile_url(profile_url):
            self.logger.error(f"Invalid Instagram profile URL: {profile_url}")
            return {
                "profile_url": profile_url,
                "success": False,
                "error": "Invalid profile URL format",
                "total_reels_discovered": 0,
                "reels": []
            }
        
        try:
            self.logger.info(f"🔍 Starting profile reel discovery for: {profile_url}")
            print(f"🔍 Starting profile reel discovery for: {profile_url}")
            
            # Extract username from profile URL
            username = self._extract_username_from_profile_url(profile_url)
            if not username:
                username = "unknown_profile"
            
            print(f"📝 Profile username: {username}")
            
            # Check current URL before navigation
            current_url_before = self.driver.current_url if self.driver else "No driver"
            print(f"📍 Current URL before navigation: {current_url_before}")
            
            # Navigate to profile URL
            print(f"🚀 Navigating to profile URL: {profile_url}")
            success = self._navigate_to_url(profile_url)
            
            if success:
                # Verify we're on the correct page
                current_url_after = self.driver.current_url
                print(f"✅ Navigation successful! Current URL: {current_url_after}")
                
                # Check if we're actually on the profile page
                if username.lower() in current_url_after.lower():
                    print(f"✅ Confirmed on {username}'s profile page")
                else:
                    print(f"⚠️  Warning: Expected to be on {username}'s profile, but current URL is: {current_url_after}")
                    
            if not success:
                print(f"❌ Failed to navigate to profile URL: {profile_url}")
                return {
                    "profile_url": profile_url,
                    "username": username,
                    "success": False,
                    "error": "Failed to navigate to profile URL",
                    "total_reels_discovered": 0,
                    "reels": []
                }
            
            print("⏱️  Waiting for page load...")
            self._random_delay(0.5, 1)
            
            # Handle login wall if encountered
            if not self.bypass_login_wall():
                self.logger.error("🚧 Could not bypass login wall for profile")
                return {
                    "profile_url": profile_url,
                    "username": username,
                    "success": False,
                    "error": "Login wall encountered - requires Instagram authentication",
                    "total_reels_discovered": 0,
                    "reels": []
                }
            
            # Navigate to reels tab
            reels_tab_url = f"{profile_url.rstrip('/')}/reels/"
            self.logger.info(f"📱 Navigating to reels tab: {reels_tab_url}")
            print(f"📱 Navigating to reels tab: {reels_tab_url}")
            
            success = self._navigate_to_url(reels_tab_url)
            
            if success:
                current_url_reels = self.driver.current_url
                print(f"✅ Reels tab navigation successful! Current URL: {current_url_reels}")
                
                # Verify we're on the reels tab
                if "/reels/" in current_url_reels:
                    print(f"✅ Confirmed on {username}'s reels tab")
                else:
                    print(f"⚠️  Warning: Expected to be on reels tab, but current URL is: {current_url_reels}")
            else:
                print(f"❌ Failed to navigate to reels tab: {reels_tab_url}")
                
            if not success:
                return {
                    "profile_url": profile_url,
                    "username": username,
                    "success": False,
                    "error": "Failed to navigate to reels tab",
                    "total_reels_discovered": 0,
                    "reels": []
                }
            
            print("⏱️  Waiting for reels tab load...")
            self._random_delay(0.5, 1)
            
            # Get first reel by clicking it and then navigate via arrows for additional reels
            self.logger.info(f"🔄 Finding first reel to click...")
            print(f"🔄 Looking for first reel on the page...")
            first_reel_url = self._click_first_reel_and_get_url()
            
            if not first_reel_url:
                print("❌ No reel found to click")
                self.logger.warning(f"⚠️  No reel found to click on profile: {username}")
                return {
                    "profile_url": profile_url,
                    "username": username,
                    "success": False,
                    "error": "No reel found to click",
                    "total_reels_discovered": 0,
                    "reels": []
                }
            
            print(f"✅ Got first reel URL: {first_reel_url}")
            
            # Build list of reel URLs using arrow navigation (restore working approach)
            reel_urls = [first_reel_url]
            
            # Discover reels (including those to skip)
            total_reels_needed = max_reels + skip_first
            
            # If we need more than 1 reel (counting skipped ones), navigate to additional reels
            if total_reels_needed > 1:
                self.logger.info(f"🔄 Discovering additional reels via arrow navigation (total needed: {total_reels_needed}, will skip first {skip_first})")
                print(f"🔄 Looking for additional reels via arrow navigation...")
                
                for reel_number in range(2, total_reels_needed + 1):
                    # Try to navigate to next reel using arrow
                    navigation_success = self._navigate_to_next_reel_with_arrow()
                    
                    if navigation_success:
                        current_reel_url = self.driver.current_url
                        # Clean URL (remove query parameters)
                        clean_url = current_reel_url.split('?')[0]
                        reel_urls.append(clean_url)
                        print(f"✅ Found reel {reel_number}: {clean_url}")
                        self.logger.info(f"🔄 Discovered reel {reel_number}/{total_reels_needed}: {clean_url}")
                    else:
                        self.logger.info(f"🔄 Arrow navigation failed - stopping at {len(reel_urls)} reels (reached end or no more arrows)")
                        print(f"🔄 No more reels available - stopping at {len(reel_urls)} reels")
                        break
                
                self.logger.info(f"✅ Reel discovery completed: {len(reel_urls)} reels found")
            else:
                print(f"✅ Single reel mode - processing only the first reel")
            
            if not reel_urls:
                self.logger.warning(f"⚠️  No reels found for profile: {username}")
                return {
                    "profile_url": profile_url,
                    "username": username,
                    "success": True,
                    "total_reels_discovered": 0,
                    "reels": [],
                    "message": "No reels found for this profile"
                }
            
            self.logger.info(f"✅ Discovered {len(reel_urls)} reel URLs")
            
            # Store total discovered count before applying skip
            total_discovered_before_skip = len(reel_urls)
            
            # Apply skip logic first
            if skip_first > 0:
                if skip_first >= len(reel_urls):
                    print(f"⚠️  Skip count ({skip_first}) is greater than or equal to discovered reels ({len(reel_urls)})")
                    return {
                        "profile_url": profile_url,
                        "profile_username": username,
                        "success": True,
                        "total_reels_discovered": total_discovered_before_skip,
                        "total_reels_skipped": len(reel_urls),
                        "total_reels_processed": 0,
                        "reels": [],
                        "message": f"All {len(reel_urls)} reels were skipped (skip_first={skip_first})"
                    }
                
                print(f"⏭️  Skipping first {skip_first} reels as requested")
                reel_urls = reel_urls[skip_first:]
                self.logger.info(f"Applied skip_first={skip_first}, processing {len(reel_urls)} remaining reels")
            
            # Check for already processed reels (fast duplicate detection)
            from .utils import get_processed_reels, filter_unprocessed_reels
            from pathlib import Path
            
            output_dir = Path(self.config.output.output_directory)
            processed_shortcodes = get_processed_reels(output_dir, username)
            
            if processed_shortcodes:
                print(f"🔍 Found {len(processed_shortcodes)} already processed reels - skipping duplicates")
                self.logger.info(f"Found {len(processed_shortcodes)} already processed reels")
            
            # Create reel objects with shortcodes for filtering
            reel_objects = []
            for i, url in enumerate(reel_urls, 1):
                shortcode = self.extract_shortcode_from_url(url)
                reel_objects.append({
                    'url': url,
                    'shortcode': shortcode,
                    'position': i  # 1-based position in profile
                })
            
            # Filter out already processed reels
            unprocessed_reels = filter_unprocessed_reels(reel_objects, processed_shortcodes)
            
            # Apply validation mode filtering if enabled
            if validation_mode and existing_shortcodes is not None:
                print(f"🎯 Validation mode: filtering out already processed shortcodes")
                filtered_reels = []
                skipped_existing = 0
                
                for reel in unprocessed_reels:
                    if reel['shortcode'] not in existing_shortcodes:
                        filtered_reels.append(reel)
                    else:
                        skipped_existing += 1
                
                pre_filter_count = len(unprocessed_reels)
                unprocessed_reels = filtered_reels
                print(f"📊 Validation filtering: {pre_filter_count} → {len(unprocessed_reels)} reels (skipped {skipped_existing} existing)")
                
                if not unprocessed_reels:
                    print("✅ All discovered reels already processed!")
                    return {
                        "profile_url": profile_url,
                        "profile_username": username,
                        "scan_timestamp": time.time(),
                        "total_reels_discovered": total_discovered_before_skip,
                        "total_reels_processed": 0,
                        "successful_extractions": 0,
                        "failed_extractions": 0,
                        "success": True,
                        "reels": [],
                        "message": "All discovered reels already processed"
                    }
            
            # Apply incremental mode filtering if enabled
            if incremental_mode and incremental_data is not None:
                print(f"🔄 Incremental mode: filtering for new reels only")
                latest_position = incremental_data.get('latest_position')
                latest_post_date = incremental_data.get('latest_post_date')
                
                filtered_reels = []
                skipped_pinned = 0
                skipped_existing = 0
                reached_existing_content = False
                
                for reel in unprocessed_reels:
                    position = reel['position']
                    shortcode = reel['shortcode']
                    
                    # Skip pinned reels (first 3 positions)
                    if self._is_pinned_reel(position):
                        skipped_pinned += 1
                        print(f"   Skipping pinned reel at position {position}: {shortcode}")
                        continue
                    
                    # Check if we've reached existing content by position
                    if latest_position and position >= latest_position:
                        print(f"   Reached existing content at position {position} (latest was {latest_position})")
                        reached_existing_content = True
                        break
                    
                    # Check if shortcode already exists (double-check)
                    if shortcode in incremental_data.get('existing_shortcodes', set()):
                        skipped_existing += 1
                        print(f"   Reached existing reel: {shortcode}")
                        reached_existing_content = True
                        break
                    
                    filtered_reels.append(reel)
                
                pre_filter_count = len(unprocessed_reels)
                unprocessed_reels = filtered_reels
                
                print(f"📊 Incremental filtering: {pre_filter_count} → {len(unprocessed_reels)} reels")
                print(f"   Skipped pinned: {skipped_pinned}")
                print(f"   Skipped existing: {skipped_existing}")
                print(f"   Reached existing content: {reached_existing_content}")
                
                if not unprocessed_reels:
                    message = "No new reels found" if reached_existing_content else "All reels are pinned or already processed"
                    print(f"✅ {message}")
                    return {
                        "profile_url": profile_url,
                        "profile_username": username,
                        "scan_timestamp": time.time(),
                        "total_reels_discovered": len(reel_urls),
                        "total_reels_processed": 0,
                        "successful_extractions": 0,
                        "failed_extractions": 0,
                        "success": True,
                        "reels": [],
                        "message": message,
                        "incremental_stats": {
                            "skipped_pinned": skipped_pinned,
                            "skipped_existing": skipped_existing,
                            "reached_existing_content": reached_existing_content
                        }
                    }
            
            total_discovered = len(reel_urls)
            total_unprocessed = len(unprocessed_reels)
            total_skipped = total_discovered - total_unprocessed
            
            print(f"📊 Duplicate Detection Results:")
            print(f"   Total discovered: {total_discovered}")
            print(f"   Already processed: {total_skipped}")
            print(f"   New to process: {total_unprocessed}")
            
            if total_unprocessed == 0:
                print("✅ All reels already processed - nothing new to extract!")
                return {
                    "profile_url": profile_url,
                    "profile_username": username,
                    "scan_timestamp": time.time(),
                    "total_reels_discovered": total_discovered_before_skip,
                    "total_reels_processed": 0,
                    "successful_extractions": 0,
                    "failed_extractions": 0,
                    "success": True,
                    "reels": [],
                    "message": "All reels already processed"
                }
            
            # Process only unprocessed reels
            scan_timestamp = time.time()
            results = []
            
            self.logger.info(f"🎬 Processing {total_unprocessed} new reels...")
            print(f"🎬 Processing {total_unprocessed} new reels...")
            
            for i, reel_obj in enumerate(unprocessed_reels, 1):
                reel_url = reel_obj['url']
                shortcode = reel_obj['shortcode']
                position = reel_obj['position']
                self.logger.info(f"Processing reel {i}/{len(unprocessed_reels)} (position {position}): {reel_url}")
                print(f"🎬 Processing reel {i}/{len(unprocessed_reels)} (position {position}): {shortcode} - {reel_url}")
                
                # Navigate to the specific reel URL to ensure we're on the right page
                success = self._navigate_to_url(reel_url)
                if not success:
                    self.logger.warning(f"⚠️  Failed to navigate to reel {i}: {reel_url}")
                    continue
                
                # Extract caption using existing method (this is the working video extraction pipeline)
                reel_result = self.extract_video_caption(
                    reel_url, 
                    save_html=save_html, 
                    take_screenshot=take_screenshot
                )
                
                if reel_result:
                    # Extract post date if not already present
                    post_date = reel_result.get('post_date')
                    if not post_date:
                        post_date = self._extract_post_date()
                        if post_date:
                            print(f"📅 Extracted post date: {post_date}")
                        else:
                            print(f"⚠️  Could not extract post date for {shortcode}")
                    
                    # Add profile-specific metadata
                    reel_result.update({
                        "position_in_profile": position,
                        "extraction_timestamp": time.time(),
                        "profile_username": username,
                        "post_date": post_date  # Add the extracted post date
                    })
                    results.append(reel_result)
                    
                    # Log success/failure
                    if reel_result.get('success'):
                        caption_preview = reel_result.get('caption', '')[:50] + '...' if reel_result.get('caption') else 'No caption'
                        print(f"✅ Reel {i}/{len(reel_urls)} processed successfully: {caption_preview}")
                    else:
                        error_msg = reel_result.get('error', 'Unknown error')
                        print(f"❌ Reel {i}/{len(reel_urls)} failed: {error_msg}")
                
                # Add brief delay between reel processing
                if i < len(reel_urls):
                    print(f"⏱️  Brief delay before next reel...")
                    self._random_delay(
                        self.config.profile_scraping.reel_processing_delay_min,
                        self.config.profile_scraping.reel_processing_delay_max
                    )
            
            successful_extractions = sum(1 for r in results if r.get('success', False))
            failed_extractions = len(results) - successful_extractions
            
            self.logger.info(f"🎯 Profile scraping completed: {successful_extractions} successful, {failed_extractions} failed")
            
            return {
                "profile_url": profile_url,
                "profile_username": username,
                "scan_timestamp": scan_timestamp,
                "total_reels_discovered": total_discovered_before_skip,
                "total_reels_processed": len(results),
                "successful_extractions": successful_extractions,
                "failed_extractions": failed_extractions,
                "success": True,
                "reels": results
            }
            
        except Exception as e:
            self.logger.error(f"Error in profile reel scraping: {str(e)}")
            return {
                "profile_url": profile_url,
                "username": username if 'username' in locals() else "unknown",
                "success": False,
                "error": str(e),
                "total_reels_discovered": 0,
                "reels": []
            }
    
    def _extract_username_from_profile_url(self, profile_url: str) -> Optional[str]:
        """Extract username from Instagram profile URL."""
        try:
            # Remove trailing slash and extract last path component
            clean_url = profile_url.rstrip('/')
            username = clean_url.split('/')[-1]
            
            # Validate username format (basic check)
            if username and re.match(r'^[A-Za-z0-9_.]+$', username):
                return username
            
            return None
        except Exception:
            return None
    
    def _discover_reel_urls(self, max_reels: int) -> List[str]:
        """Discover reel URLs from the current profile page using infinite scroll."""
        discovered_urls = set()
        scroll_attempts = 0
        max_scroll_attempts = self.config.profile_scraping.max_scroll_attempts
        no_new_content_count = 0
        max_no_new_content = self.config.profile_scraping.no_new_content_threshold
        
        try:
            while len(discovered_urls) < max_reels and scroll_attempts < max_scroll_attempts:
                scroll_attempts += 1
                self.logger.debug(f"Scroll attempt {scroll_attempts}, found {len(discovered_urls)} reels so far")
                
                # Get current reel URLs from page
                current_urls = self._extract_reel_urls_from_page()
                new_urls_count = 0
                
                for url in current_urls:
                    if url not in discovered_urls:
                        discovered_urls.add(url)
                        new_urls_count += 1
                        if len(discovered_urls) >= max_reels:
                            break
                
                self.logger.debug(f"Found {new_urls_count} new reels on this scroll")
                
                # If we have enough reels, stop
                if len(discovered_urls) >= max_reels:
                    break
                
                # If no new content found, increment counter
                if new_urls_count == 0:
                    no_new_content_count += 1
                    if no_new_content_count >= max_no_new_content:
                        self.logger.info(f"No new content found after {max_no_new_content} scroll attempts, stopping")
                        break
                else:
                    no_new_content_count = 0  # Reset counter
                
                # Scroll down to load more content
                self._scroll_for_more_reels()
                
                # Wait for content to load
                self._random_delay(
                    self.config.profile_scraping.scroll_delay_min,
                    self.config.profile_scraping.scroll_delay_max
                )
            
            # Convert set to list and limit to max_reels
            result_urls = list(discovered_urls)[:max_reels]
            self.logger.info(f"Reel discovery completed: {len(result_urls)} URLs found (target: {max_reels})")
            
            return result_urls
            
        except Exception as e:
            self.logger.error(f"Error during reel discovery: {str(e)}")
            return list(discovered_urls)
    
    def _extract_reel_urls_from_page(self) -> List[str]:
        """Extract reel URLs from the current page DOM."""
        try:
            reel_urls = []
            
            # Multiple strategies to find reel links
            selectors = [
                # Standard reel links
                "a[href*='/reel/']",
                # Alternative selectors for different page layouts
                "article a[href*='/reel/']",
                "div[role='button'] a[href*='/reel/']",
                # Links in grid layout
                "div._ac7v a[href*='/reel/']",
                "div._aagw a[href*='/reel/']"
            ]
            
            for selector in selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    for element in elements:
                        href = element.get_attribute('href')
                        if href and '/reel/' in href:
                            # Clean up URL
                            clean_url = href.split('?')[0]  # Remove query parameters
                            if self.is_valid_instagram_url(clean_url):
                                reel_urls.append(clean_url)
                except Exception:
                    continue
            
            # Alternative: Extract from JavaScript variables (Instagram often stores data in window._sharedData)
            try:
                js_urls = self.driver.execute_script("""
                    let reelUrls = [];
                    const links = document.querySelectorAll('a[href*="/reel/"]');
                    links.forEach(link => {
                        if (link.href && link.href.includes('/reel/')) {
                            reelUrls.push(link.href.split('?')[0]);
                        }
                    });
                    return reelUrls;
                """)
                
                if js_urls:
                    for url in js_urls:
                        if self.is_valid_instagram_url(url):
                            reel_urls.append(url)
                            
            except Exception as e:
                self.logger.debug(f"JavaScript extraction failed: {str(e)}")
            
            # Remove duplicates while preserving order
            unique_urls = []
            seen = set()
            for url in reel_urls:
                if url not in seen:
                    seen.add(url)
                    unique_urls.append(url)
            
            self.logger.debug(f"Extracted {len(unique_urls)} reel URLs from current page view")
            return unique_urls
            
        except Exception as e:
            self.logger.error(f"Error extracting reel URLs from page: {str(e)}")
            return []
    
    def _click_first_reel_and_get_url(self) -> str:
        """Click on the first reel and get its URL."""
        try:
            print("🔍 Looking for first reel to click...")
            
            # First, try to close any blocking dialogs/modals
            print("🔍 Checking for blocking modals/dialogs...")
            try:
                # Try to find and close any "Not Now" or "Close" buttons
                close_buttons = self.driver.find_elements(By.XPATH, "//button[contains(text(), 'Not Now') or contains(text(), 'Close') or contains(text(), 'Dismiss')]")
                for btn in close_buttons:
                    if btn.is_displayed():
                        btn.click()
                        print("✅ Closed a blocking dialog")
                        self._random_delay(0.5, 1)
                        break
            except:
                pass
            
            # Check for and remove any overlay divs
            try:
                self.driver.execute_script("""
                    // Remove any full-screen overlays that might be blocking clicks
                    var overlays = document.querySelectorAll('div[style*="position: fixed"], div[style*="position: absolute"]');
                    overlays.forEach(function(el) {
                        if (el.style.zIndex > 1000 || el.classList.contains('overlay')) {
                            el.remove();
                            console.log('Removed potential blocking overlay');
                        }
                    });
                """)
            except:
                pass
            
            # Scroll to see reels
            print("📜 Scrolling to find reels...")
            self.driver.execute_script("window.scrollTo(0, 500);")
            self._random_delay(0.5, 0.5)
            
            # Wait for reels to be fully loaded and clickable
            print("⏳ Waiting for reels to be interactive...")
            
            # First wait for reel links to exist
            try:
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "a[href*='/reel/']"))
                )
            except TimeoutException:
                print("⚠️ No reel links found on page")
                return None
            
            # Wait for Instagram's JavaScript to fully load
            # The key indicator is when ALL reels have loaded (usually 24+ instead of initial 12)
            print("⏳ Waiting for Instagram JavaScript to initialize...")
            
            # Wait for the page to stabilize with all reels loaded
            reels_count = 0
            stable_count = 0
            max_wait = 10  # Maximum 10 seconds
            
            for i in range(max_wait):
                current_count = len(self.driver.find_elements(By.CSS_SELECTOR, "a[href*='/reel/']"))
                print(f"   Found {current_count} reels (attempt {i+1}/{max_wait})")
                
                # Check if count is stable (same for 2 checks in a row)
                if current_count == reels_count and current_count >= 24:
                    stable_count += 1
                    if stable_count >= 2:
                        print(f"✅ Page stabilized with {current_count} reels loaded")
                        break
                else:
                    stable_count = 0
                    reels_count = current_count
                
                self._random_delay(1, 1.5)
            
            # Extra wait after stabilization for click handlers to attach
            if reels_count >= 24:
                print("⏳ Waiting for click handlers to attach...")
                self._random_delay(2, 3)
            else:
                print(f"⚠️ Only {reels_count} reels loaded, may not be fully ready")
            
            # Try broader selectors first
            reel_selectors = [
                "a[href*='/reel/']",  # Any link with /reel/
                "article a",          # Any link in article
                "[role='link'][href*='/reel/']"  # Any role=link with reel
            ]
            
            print("🔍 Trying to find reel links...")
            for selector in reel_selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    print(f"🔍 Selector '{selector}' found {len(elements)} elements")
                    
                    # Find the element closest to x=652 (the reliable first reel position)
                    # This should be the first reel in the second column
                    target_x = 652
                    sorted_elements = sorted(elements, key=lambda e: (
                        abs(e.location['x'] - target_x),  # First priority: closest to x=652
                        e.location['y']  # Second priority: higher up on page (lower y value)
                    ))
                    
                    if sorted_elements:
                        first_element = sorted_elements[0]
                        print(f"📍 Selected reel at position x={first_element.location['x']}, y={first_element.location['y']}")
                        
                        # Only process the best candidate
                        elements_to_try = [first_element]
                    else:
                        elements_to_try = elements
                    
                    for element in elements_to_try:
                        href = element.get_attribute('href')
                        if href and '/reel/' in href:
                            clean_url = href.split('?')[0]
                            print(f"✅ Found reel URL: {clean_url}")
                            
                            # Debug: Check element properties before clicking
                            print(f"📊 Element details:")
                            print(f"   - Tag: {element.tag_name}")
                            print(f"   - Displayed: {element.is_displayed()}")
                            print(f"   - Enabled: {element.is_enabled()}")
                            print(f"   - Size: {element.size}")
                            print(f"   - Location: {element.location}")
                            
                            # Check if element is covered by another element
                            try:
                                element_at_location = self.driver.execute_script(
                                    "return document.elementFromPoint(arguments[0], arguments[1]);",
                                    element.location['x'] + element.size['width']/2,
                                    element.location['y'] + element.size['height']/2
                                )
                                if element_at_location:
                                    print(f"   - Element at click point: {element_at_location.tag_name if hasattr(element_at_location, 'tag_name') else 'unknown'}")
                            except:
                                pass
                            
                            print("👆 Attempting to click on the reel...")
                            
                            # If there's a div covering it, try to find the actual clickable element
                            # Instagram often has nested elements like: <a><div><div><img></div></div></a>
                            try:
                                # First, try to find an img or video thumbnail inside the link
                                clickable_element = element.find_element(By.TAG_NAME, "img")
                                print("   Found img element inside link, clicking that instead")
                            except:
                                clickable_element = element  # Fallback to the link itself
                            
                            # Try multiple click strategies
                            click_successful = False
                            
                            # Strategy 1: Click the image/thumbnail directly
                            try:
                                clickable_element.click()
                                print("✅ Clicked using regular click on inner element")
                                click_successful = True
                            except Exception as e:
                                print(f"❌ Regular click on inner element failed: {str(e)[:100]}")
                            
                            # Strategy 2: JavaScript click on the link
                            if not click_successful:
                                try:
                                    self.driver.execute_script("arguments[0].click();", element)
                                    print("✅ Clicked using JavaScript click on link")
                                    click_successful = True
                                except Exception as e:
                                    print(f"❌ JavaScript click failed: {str(e)[:100]}")
                            
                            # Strategy 3: Scroll element into center view and try again
                            if not click_successful:
                                try:
                                    print("📍 Scrolling element to center of viewport and retrying...")
                                    self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)
                                    self._random_delay(0.5, 1)
                                    element.click()
                                    print("✅ Clicked after scrolling to center")
                                    click_successful = True
                                except Exception as e:
                                    print(f"❌ Click after scroll failed: {str(e)[:100]}")
                            
                            if not click_successful:
                                print("❌ All click strategies failed - cannot open reel modal")
                            
                            # Wait for the reel modal to open
                            print("⏳ Waiting for reel to open...")
                            try:
                                # Wait for either URL change or video element to appear
                                WebDriverWait(self.driver, 5).until(
                                    lambda driver: (
                                        '/reel/' in driver.current_url or
                                        driver.find_elements(By.TAG_NAME, "video")
                                    )
                                )
                            except TimeoutException:
                                print("⚠️ Reel didn't open within timeout")
                                # Debug: Check what's on the page now
                                print("🔍 Debugging why reel didn't open:")
                                print(f"   - Current URL: {self.driver.current_url}")
                                print(f"   - Video elements found: {len(self.driver.find_elements(By.TAG_NAME, 'video'))}")
                                dialog_selector = '[role="dialog"]'
                                print(f"   - Dialog/Modal elements: {len(self.driver.find_elements(By.CSS_SELECTOR, dialog_selector))}")
                                # Check if we're still on the same page
                                current_reels = self.driver.find_elements(By.CSS_SELECTOR, "a[href*='/reel/']")
                                print(f"   - Reel links still visible: {len(current_reels)}")
                            
                            print(f"📍 Current URL after click: {self.driver.current_url}")
                            
                            # Verify modal opened
                            if '/reel/' in self.driver.current_url:
                                print("✅ Reel opened successfully")
                                return clean_url
                            else:
                                print("⚠️ Reel didn't open properly, but continuing anyway")
                                return clean_url
                            
                except Exception as e:
                    print(f"⚠️  Error with selector {selector}: {str(e)}")
                    continue
            
            # If no reels found, show page info
            print("❌ No reel links found. Checking page...")
            print(f"📍 Current URL: {self.driver.current_url}")
            
            # Check what's actually on the page
            all_links = self.driver.find_elements(By.CSS_SELECTOR, "a")
            print(f"🔗 Total links on page: {len(all_links)}")
            
            # Show first few link hrefs for debugging
            for i, link in enumerate(all_links[:5]):
                href = link.get_attribute('href')
                print(f"   Link {i+1}: {href}")
            
            return None
            
        except Exception as e:
            print(f"❌ Error finding reels: {str(e)}")
            return None
    
    def _navigate_to_next_reel_with_arrow(self) -> bool:
        """Navigate to the next reel using the right arrow button in reel viewer.
        
        Returns:
            bool: True if navigation was successful, False otherwise
        """
        try:
            self.logger.debug("🔄 Attempting to navigate to next reel via arrow...")
            
            # Multiple selectors for Instagram's next arrow button
            arrow_selectors = [
                "button[aria-label='Next']",
                "button[aria-label='Go to next']", 
                "svg[aria-label='Next']",
                ".x1i10hfl[role='button']",  # Instagram's common button class
                "[data-testid='carousel-next-button']",
                "button[aria-label='Go to next post']",
                "button[aria-label='Next Post']"
            ]
            
            # Get current URL to compare after navigation
            current_url_before = self.driver.current_url
            
            for selector in arrow_selectors:
                try:
                    # Find the arrow button
                    arrow_elements = self.driver.find_elements("css selector", selector)
                    
                    for element in arrow_elements:
                        # Check if element is visible and clickable
                        if element.is_displayed() and element.is_enabled():
                            self.logger.debug(f"Found arrow button with selector: {selector}")
                            
                            # Click the arrow
                            try:
                                element.click()
                                self.logger.debug("Arrow button clicked")
                            except Exception:
                                # Try JavaScript click if regular click fails
                                self.driver.execute_script("arguments[0].click();", element)
                                self.logger.debug("Arrow button clicked via JavaScript")
                            
                            # Wait for navigation to complete
                            self._random_delay(1, 2)
                            
                            # Check if URL changed (indicating successful navigation)
                            current_url_after = self.driver.current_url
                            
                            if current_url_after != current_url_before:
                                self.logger.info(f"✅ Successfully navigated to next reel: {current_url_after}")
                                return True
                            else:
                                self.logger.debug("URL did not change after arrow click")
                                # Continue trying other selectors
                                break
                                
                except Exception as e:
                    self.logger.debug(f"Error with arrow selector {selector}: {str(e)}")
                    continue
            
            # If no arrow worked, try keyboard navigation as fallback
            try:
                self.logger.debug("Trying keyboard arrow navigation as fallback...")
                ActionChains(self.driver).send_keys(Keys.ARROW_RIGHT).perform()
                self._random_delay(1, 2)
                
                current_url_after = self.driver.current_url
                if current_url_after != current_url_before:
                    self.logger.info(f"✅ Successfully navigated via keyboard: {current_url_after}")
                    return True
                    
            except Exception as e:
                self.logger.debug(f"Keyboard navigation failed: {str(e)}")
            
            self.logger.warning("❌ No navigation method worked - no arrow button found or end of reels reached")
            return False
            
        except Exception as e:
            self.logger.error(f"Error in arrow navigation: {str(e)}")
            return False
    
    def _scroll_for_more_reels(self) -> None:
        """Scroll down to trigger loading of more reels."""
        try:
            # Try multiple scrolling strategies
            
            # Strategy 1: Scroll to bottom of page
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            self._random_delay(1, 2)
            
            # Strategy 2: Scroll by viewport height
            self.driver.execute_script("window.scrollBy(0, window.innerHeight);")
            self._random_delay(1, 2)
            
            # Strategy 3: Find and scroll to last reel element
            try:
                reel_elements = self.driver.find_elements(By.CSS_SELECTOR, "a[href*='/reel/']")
                if reel_elements:
                    last_element = reel_elements[-1]
                    self.driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'end'});", last_element)
                    self._random_delay(1, 2)
            except Exception:
                pass
            
            # Strategy 4: Send Page Down key
            try:
                ActionChains(self.driver).send_keys(Keys.PAGE_DOWN).perform()
                self._random_delay(0.5, 1)
            except Exception:
                pass
                
        except Exception as e:
            self.logger.debug(f"Error during scrolling: {str(e)}")
    
    def _random_delay(self, min_delay: Optional[float] = None, max_delay: Optional[float] = None) -> None:
        """Add random delay to avoid detection."""
        min_delay = min_delay or self.config.anti_detection.min_delay
        max_delay = max_delay or self.config.anti_detection.max_delay
        delay = random.uniform(min_delay, max_delay)
        time.sleep(delay)
    
    def _apply_stealth_settings(self, driver: webdriver) -> None:
        """Apply stealth JavaScript to avoid detection."""
        try:
            # Remove webdriver property
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            # Override the `plugins` property to use a custom getter.
            driver.execute_script("Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]})")
            
            # Override the `languages` property to use a custom getter.
            driver.execute_script("Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']})")
            
            # Override the `permissions` property
            driver.execute_script("const originalQuery = window.navigator.permissions.query; window.navigator.permissions.query = (parameters) => (parameters.name === 'notifications' ? Promise.resolve({ state: Notification.permission }) : originalQuery(parameters));")
            
            self.logger.debug("Applied stealth settings to driver")
            
        except Exception as e:
            self.logger.warning(f"Failed to apply stealth settings: {str(e)}")
    
    def _get_random_user_agent(self) -> str:
        """Get a random user agent string."""
        try:
            return self.user_agent.random
        except Exception:
            # Fallback user agents
            fallback_agents = [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0"
            ]
            return random.choice(fallback_agents)
    
    def _test_instagram_access(self) -> bool:
        """Test if browser can access Instagram without being blocked."""
        try:
            self.logger.info("Testing Instagram access...")
            self.driver.get(self.config.instagram.base_url)
            
            # Wait for page to load
            self._random_delay(2, 4)
            
            # Check if we're blocked or redirected
            current_url = self.driver.current_url.lower()
            page_title = self.driver.title.lower()
            
            if "blocked" in page_title or "error" in page_title:
                self.logger.error("Instagram access blocked - detected by anti-bot measures")
                return False
            
            if "instagram.com" not in current_url:
                self.logger.error(f"Redirected away from Instagram to: {current_url}")
                return False
            
            # Check for common blocking indicators
            try:
                # Look for challenge or suspicious activity pages
                challenge_indicators = [
                    "challenge",
                    "suspicious",
                    "verify",
                    "unusual activity"
                ]
                
                page_source = self.driver.page_source.lower()
                for indicator in challenge_indicators:
                    if indicator in page_source:
                        self.logger.warning(f"Potential challenge detected: {indicator}")
                        break
                
            except Exception as e:
                self.logger.debug(f"Could not analyze page source: {str(e)}")
            
            self.logger.info("Instagram access test successful")
            return True
            
        except Exception as e:
            self.logger.error(f"Instagram access test failed: {str(e)}")
            return False
    
    def _navigate_to_url(self, url: str) -> bool:
        """Navigate to a URL with error handling and retries."""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                self.logger.debug(f"Navigating to {url} (attempt {attempt + 1}/{max_retries})")
                self.driver.get(url)
                
                # Wait for page to load
                self._random_delay(1, 3)
                
                # Check if navigation was successful
                current_url = self.driver.current_url
                if "instagram.com" not in current_url.lower():
                    raise WebDriverException(f"Redirected away from Instagram: {current_url}")
                
                # Handle common popups
                self._handle_popups()
                
                return True
                
            except TimeoutException:
                self.logger.warning(f"Timeout loading {url} on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    self._random_delay(2, 5)
                    continue
                    
            except Exception as e:
                self.logger.warning(f"Navigation error on attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries - 1:
                    self._random_delay(2, 5)
                    continue
        
        self.logger.error(f"Failed to navigate to {url} after {max_retries} attempts")
        return False
    
    def _handle_popups(self) -> None:
        """Handle common Instagram popups and modals."""
        popup_selectors = [
            self.config.instagram.selectors.close_button,
            "button[aria-label='Close']",
            "button[aria-label='Not Now']",
            "button:contains('Not Now')",
            "div[role='dialog'] button",
            "[data-testid='modal-close-button']"
        ]
        
        for selector in popup_selectors:
            try:
                popup = self.driver.find_element(By.CSS_SELECTOR, selector)
                if popup.is_displayed() and popup.is_enabled():
                    popup.click()
                    self.logger.debug(f"Closed popup with selector: {selector}")
                    self._random_delay(0.5, 1.5)
                    break
            except (NoSuchElementException, ElementNotInteractableException):
                continue
            except Exception as e:
                self.logger.debug(f"Error handling popup with {selector}: {str(e)}")
                continue
    
    @staticmethod
    def is_valid_instagram_url(url: str) -> bool:
        """Validate if URL is a valid Instagram post/reel/IGTV URL."""
        if not url or not isinstance(url, str):
            return False
        
        # Instagram URL patterns
        instagram_patterns = [
            r'https?://(?:www\.)?instagram\.com/p/[A-Za-z0-9_-]+/?',
            r'https?://(?:www\.)?instagram\.com/reel/[A-Za-z0-9_-]+/?',
            r'https?://(?:www\.)?instagram\.com/tv/[A-Za-z0-9_-]+/?',
            r'https?://(?:www\.)?instagram\.com/stories/[A-Za-z0-9_.]+/[0-9]+/?',
            # Profile-based URLs with username prefix
            r'https?://(?:www\.)?instagram\.com/[A-Za-z0-9_.]+/p/[A-Za-z0-9_-]+/?',
            r'https?://(?:www\.)?instagram\.com/[A-Za-z0-9_.]+/reel/[A-Za-z0-9_-]+/?',
            r'https?://(?:www\.)?instagram\.com/[A-Za-z0-9_.]+/tv/[A-Za-z0-9_-]+/?'
        ]
        
        for pattern in instagram_patterns:
            if re.match(pattern, url.strip()):
                return True
        
        return False
    
    @staticmethod
    def is_valid_instagram_profile_url(url: str) -> bool:
        """Validate if URL is a valid Instagram profile URL."""
        if not url or not isinstance(url, str):
            return False
        
        # Instagram profile URL patterns
        profile_patterns = [
            r'https?://(?:www\.)?instagram\.com/[A-Za-z0-9_.]+/?',
            r'https?://(?:www\.)?instagram\.com/[A-Za-z0-9_.]+$'
        ]
        
        url = url.strip()
        for pattern in profile_patterns:
            if re.match(pattern, url):
                # Ensure it's not a post, reel, TV, or story URL
                if not any(path in url for path in ['/p/', '/reel/', '/tv/', '/stories/', '/explore/', '/accounts/', '/help/', '/legal/']):
                    return True
        
        return False
    
    @staticmethod
    def extract_shortcode_from_url(url: str) -> Optional[str]:
        """Extract shortcode from Instagram URL."""
        if not url:
            return None
        
        # Pattern to match Instagram shortcodes
        patterns = [
            r'/p/([A-Za-z0-9_-]+)',
            r'/reel/([A-Za-z0-9_-]+)',
            r'/tv/([A-Za-z0-9_-]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return None
    
    @staticmethod
    def get_url_type(url: str) -> Optional[str]:
        """Determine the type of Instagram URL (post, reel, igtv, story)."""
        if not url:
            return None
        
        url_lower = url.lower()
        
        if '/p/' in url_lower:
            return 'post'
        elif '/reel/' in url_lower:
            return 'reel'
        elif '/tv/' in url_lower:
            return 'igtv'
        elif '/stories/' in url_lower:
            return 'story'
        
        return None
    
    def _extract_post_date(self) -> Optional[str]:
        """Extract Instagram post date from the current page.
        
        Attempts to extract the original posting date from Instagram's DOM.
        This method should be called when on a specific reel/post page.
        
        Returns:
            ISO format date string (YYYY-MM-DD) if found, None otherwise
        """
        try:
            # Method 1: Try to find date in time elements
            time_elements = self.driver.find_elements(By.TAG_NAME, "time")
            for time_elem in time_elements:
                datetime_attr = time_elem.get_attribute("datetime")
                if datetime_attr:
                    # Parse ISO datetime and extract date
                    from datetime import datetime
                    try:
                        dt = datetime.fromisoformat(datetime_attr.replace('Z', '+00:00'))
                        return dt.strftime('%Y-%m-%d')
                    except ValueError:
                        continue
            
            # Method 2: Look for structured data with date information
            script_elements = self.driver.find_elements(By.XPATH, "//script[@type='application/ld+json']")
            for script in script_elements:
                try:
                    import json
                    data = json.loads(script.get_attribute('innerHTML'))
                    if isinstance(data, dict):
                        # Look for datePublished or uploadDate
                        date_published = data.get('datePublished') or data.get('uploadDate')
                        if date_published:
                            from datetime import datetime
                            dt = datetime.fromisoformat(date_published.replace('Z', '+00:00'))
                            return dt.strftime('%Y-%m-%d')
                except (json.JSONDecodeError, ValueError, KeyError):
                    continue
            
            # Method 3: Try to extract from page source using regex
            page_source = self.driver.page_source
            import re
            
            # Look for various date patterns in the page source
            date_patterns = [
                r'"taken_at_timestamp":(\d+)',
                r'"date":"(\d{4}-\d{2}-\d{2})',
                r'datetime="([^"]+)"',
                r'"uploadDate":"([^"]+)"',
                r'"datePublished":"([^"]+)"'
            ]
            
            for pattern in date_patterns:
                matches = re.findall(pattern, page_source)
                if matches:
                    date_str = matches[0]
                    try:
                        if date_str.isdigit():
                            # Unix timestamp
                            from datetime import datetime
                            dt = datetime.fromtimestamp(int(date_str))
                            return dt.strftime('%Y-%m-%d')
                        else:
                            # ISO date string
                            from datetime import datetime
                            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                            return dt.strftime('%Y-%m-%d')
                    except (ValueError, OSError):
                        continue
            
            self.logger.warning("Could not extract post date from current page")
            return None
            
        except Exception as e:
            self.logger.error(f"Error extracting post date: {e}")
            return None
    
    def _is_pinned_reel(self, position: int) -> bool:
        """Check if a reel at the given position is likely pinned.
        
        Instagram typically shows pinned reels first, usually the first 1-3 reels.
        This is a heuristic and may not be 100% accurate.
        
        Args:
            position: 1-based position of the reel in the profile
            
        Returns:
            True if the reel is likely pinned, False otherwise
        """
        # Consider first 3 reels as potentially pinned
        return position <= 3
    
    
    
    def get_browser_info(self) -> Dict[str, Any]:
        """Get information about the current browser session."""
        if not self.driver:
            return {"status": "No active session"}
        
        try:
            capabilities = self.driver.capabilities
            info = {
                "browser_name": capabilities.get('browserName', 'unknown'),
                "browser_version": capabilities.get('browserVersion', 'unknown'),
                "platform": capabilities.get('platformName', 'unknown'),
                "driver_version": capabilities.get('chrome', {}).get('chromedriverVersion', 'unknown'),
                "headless": self.config.selenium.headless,
                "window_size": f"{self.driver.get_window_size()['width']}x{self.driver.get_window_size()['height']}",
                "video_transcription": True
            }
            
            if self.video_transcriber:
                transcriber_info = self.video_transcriber.get_system_info()
                info["video_transcriber_status"] = transcriber_info
            
            if self.debug_helper:
                info["debug_helper_status"] = "initialized"
            
            return info
        except Exception as e:
            return {"error": str(e)}
    
    def __enter__(self):
        """Context manager entry."""
        self.start_session()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close_session()