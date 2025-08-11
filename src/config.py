import yaml
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


@dataclass
class WindowSize:
    width: int
    height: int


@dataclass
class SeleniumConfig:
    browser: str
    headless: bool
    page_load_timeout: int
    implicit_wait: int
    window_size: Optional[WindowSize]
    user_data_dir: Optional[str]


@dataclass
class AntiDetectionConfig:
    min_delay: float
    max_delay: float
    scroll_pause_time: float
    user_agent_rotation: bool
    random_viewport: bool


@dataclass
class ProfileScrapingConfig:
    max_scroll_attempts: int
    scroll_delay_min: float
    scroll_delay_max: float
    no_new_content_threshold: int
    reel_processing_delay_min: float
    reel_processing_delay_max: float


@dataclass
class InstagramCredentials:
    username: str
    password: str
    auto_login: bool


@dataclass
class InstagramSelectors:
    username_input: str
    password_input: str
    login_button: str
    video_element: str
    video_source: str
    caption_container: str
    next_button: str
    close_button: str
    popup_close: str
    not_now_button: str
    turn_on_notifications: str
    login_wall: str
    signup_prompt: str
    app_download: str
    reels_tab_link: str
    reel_links: str
    profile_reels_grid: str
    reel_grid_items: str


@dataclass
class InstagramConfig:
    base_url: str
    login_url: str
    credentials: InstagramCredentials
    selectors: InstagramSelectors


@dataclass
class LoggingConfig:
    level: str
    format: str
    file: str
    max_file_size: int
    backup_count: int


@dataclass
class OutputConfig:
    format: str
    save_to_file: bool
    output_directory: str


class Config:
    """Configuration manager for the Instagram scraper."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize configuration from YAML file."""
        self.config_path = config_path
        self._config_data = self._load_config()
        self._parse_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_file = Path(self.config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        try:
            with open(config_file, 'r', encoding='utf-8') as file:
                config_data = yaml.safe_load(file)
            return config_data
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Error loading configuration file: {str(e)}")
    
    def _parse_config(self) -> None:
        """Parse configuration data into structured objects."""
        try:
            # Parse Selenium configuration
            selenium_data = self._config_data.get("selenium", {})
            window_size_data = selenium_data.get("window_size")
            window_size = None
            if window_size_data:
                window_size = WindowSize(
                    width=window_size_data.get("width", 1920),
                    height=window_size_data.get("height", 1080)
                )
            
            self.selenium = SeleniumConfig(
                browser=selenium_data.get("browser", "chrome"),
                headless=selenium_data.get("headless", True),
                page_load_timeout=selenium_data.get("page_load_timeout", 30),
                implicit_wait=selenium_data.get("implicit_wait", 10),
                window_size=window_size,
                user_data_dir=selenium_data.get("user_data_dir")
            )
            
            # Parse Anti-detection configuration
            anti_detection_data = self._config_data.get("anti_detection", {})
            self.anti_detection = AntiDetectionConfig(
                min_delay=anti_detection_data.get("min_delay", 2),
                max_delay=anti_detection_data.get("max_delay", 5),
                scroll_pause_time=anti_detection_data.get("scroll_pause_time", 2),
                user_agent_rotation=anti_detection_data.get("user_agent_rotation", True),
                random_viewport=anti_detection_data.get("random_viewport", False)
            )
            
            # Parse Profile scraping configuration
            profile_data = self._config_data.get("profile_scraping", {})
            self.profile_scraping = ProfileScrapingConfig(
                max_scroll_attempts=profile_data.get("max_scroll_attempts", 20),
                scroll_delay_min=profile_data.get("scroll_delay_min", 2),
                scroll_delay_max=profile_data.get("scroll_delay_max", 4),
                no_new_content_threshold=profile_data.get("no_new_content_threshold", 3),
                reel_processing_delay_min=profile_data.get("reel_processing_delay_min", 1),
                reel_processing_delay_max=profile_data.get("reel_processing_delay_max", 3)
            )
            
            # Parse Instagram configuration
            instagram_data = self._config_data.get("instagram", {})
            credentials_data = instagram_data.get("credentials", {})
            selectors_data = instagram_data.get("selectors", {})
            
            # Load credentials from environment variables (secure)
            env_username = os.getenv("INSTAGRAM_USERNAME")
            env_password = os.getenv("INSTAGRAM_PASSWORD")
            
            credentials = InstagramCredentials(
                username=env_username or credentials_data.get("username", ""),
                password=env_password or credentials_data.get("password", ""),
                auto_login=credentials_data.get("auto_login", False)
            )
            
            selectors = InstagramSelectors(
                username_input=selectors_data.get("username_input", "input[name='username']"),
                password_input=selectors_data.get("password_input", "input[name='password']"),
                login_button=selectors_data.get("login_button", "button[type='submit']"),
                video_element=selectors_data.get("video_element", "video"),
                video_source=selectors_data.get("video_source", "video source"),
                caption_container=selectors_data.get("caption_container", "article div[data-testid='post-content'] span"),
                next_button=selectors_data.get("next_button", "button[aria-label='Next']"),
                close_button=selectors_data.get("close_button", "button[aria-label='Close']"),
                popup_close=selectors_data.get("popup_close", "button[aria-label='Close']"),
                not_now_button=selectors_data.get("not_now_button", "button:contains('Not Now')"),
                turn_on_notifications=selectors_data.get("turn_on_notifications", "button:contains('Turn On')"),
                login_wall=selectors_data.get("login_wall", ".x9f619.x78zum5.x1q0g3np"),
                signup_prompt=selectors_data.get("signup_prompt", "a[href='/accounts/emailsignup/']"),
                app_download=selectors_data.get("app_download", "a[href*='apps.apple.com'], a[href*='play.google.com']"),
                reels_tab_link=selectors_data.get("reels_tab_link", "a[href*='/reels/']"),
                reel_links=selectors_data.get("reel_links", "a[href*='/reel/']"),
                profile_reels_grid=selectors_data.get("profile_reels_grid", "article"),
                reel_grid_items=selectors_data.get("reel_grid_items", "div._ac7v")
            )
            
            self.instagram = InstagramConfig(
                base_url=instagram_data.get("base_url", "https://www.instagram.com"),
                login_url=instagram_data.get("login_url", "https://www.instagram.com/accounts/login/"),
                credentials=credentials,
                selectors=selectors
            )
            
            # Parse Logging configuration
            logging_data = self._config_data.get("logging", {})
            self.logging = LoggingConfig(
                level=logging_data.get("level", "INFO"),
                format=logging_data.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
                file=logging_data.get("file", "logs/scraper.log"),
                max_file_size=logging_data.get("max_file_size", 10485760),
                backup_count=logging_data.get("backup_count", 5)
            )
            
            # Parse Output configuration
            output_data = self._config_data.get("output", {})
            self.output = OutputConfig(
                format=output_data.get("format", "json"),
                save_to_file=output_data.get("save_to_file", True),
                output_directory=output_data.get("output_directory", "output")
            )
            
        except Exception as e:
            raise ValueError(f"Error parsing configuration: {str(e)}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key."""
        return self._config_data.get(key, default)
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        self._config_data.update(updates)
        self._parse_config()
    
    def save_config(self, file_path: Optional[str] = None) -> None:
        """Save current configuration to file."""
        output_path = file_path or self.config_path
        
        try:
            with open(output_path, 'w', encoding='utf-8') as file:
                yaml.dump(self._config_data, file, default_flow_style=False, indent=2)
        except Exception as e:
            raise RuntimeError(f"Error saving configuration: {str(e)}")
    
    def validate(self) -> bool:
        """Validate configuration values."""
        try:
            if self.selenium.browser.lower() not in ["chrome", "firefox"]:
                raise ValueError(f"Unsupported browser: {self.selenium.browser}")
            
            if self.selenium.page_load_timeout <= 0:
                raise ValueError("Page load timeout must be positive")
            
            if self.anti_detection.min_delay < 0 or self.anti_detection.max_delay < 0:
                raise ValueError("Delays must be non-negative")
            
            if self.anti_detection.min_delay > self.anti_detection.max_delay:
                raise ValueError("Min delay cannot be greater than max delay")
            
            return True
            
        except Exception as e:
            raise ValueError(f"Configuration validation failed: {str(e)}")
    
    def __str__(self) -> str:
        """String representation of the configuration."""
        return f"Config(browser={self.selenium.browser}, headless={self.selenium.headless})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the configuration."""
        return self.__str__()