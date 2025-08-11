"""
Instagram Caption Scraper Package

A tool for extracting video captions from Instagram using Selenium WebDriver.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .scraper import InstagramScraper
from .config import Config
from .utils import setup_logging, save_data

__all__ = ["InstagramScraper", "Config", "setup_logging", "save_data"]