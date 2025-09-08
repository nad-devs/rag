#!/usr/bin/env python3
"""
Script to fetch dates from Instagram URLs in enhanced_processed files
Reuses existing scraper setup for simplicity
"""

import json
import os
import time
from pathlib import Path
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.common.exceptions import TimeoutException

def setup_driver():
    """Setup Chrome driver using existing scraper configuration"""
    import random
    from webdriver_manager.chrome import ChromeDriverManager
    from selenium.webdriver.chrome.service import Service as ChromeService
    
    options = ChromeOptions()
    
    # Run in headless mode to avoid display issues
    options.add_argument("--headless=new")
    
    # Copy exact setup from scraper
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation", "enable-logging"])
    options.add_experimental_option('useAutomationExtension', False)
    
    # Desktop user agent (same as scraper)
    chrome_version = random.choice(["119", "120", "121", "122"])
    desktop_user_agent = f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{chrome_version}.0.0.0 Safari/537.36"
    options.add_argument(f"--user-agent={desktop_user_agent}")
    
    # Window size for desktop
    options.add_argument("--window-size=1920,1080")
    
    # Don't use user-data-dir to avoid conflicts
    print(f"✅ Running Chrome in headless mode")
    
    # Setup driver with ChromeDriverManager
    driver_path = ChromeDriverManager().install()
    service = ChromeService(driver_path)
    driver = webdriver.Chrome(service=service, options=options)
    
    return driver

def extract_date_from_page(driver, url):
    """
    Extract date from Instagram page
    Looking for: <time class="x1p4m5qa" datetime="2024-11-27T13:04:29.000Z">
    """
    try:
        print(f"  🔍 Visiting: {url}")
        driver.get(url)
        
        # Wait for page to load
        time.sleep(3)  # Simple wait to ensure page loads
        
        # Find time element with datetime attribute
        try:
            # Try to find time element
            time_elements = driver.find_elements(By.TAG_NAME, "time")
            
            for time_elem in time_elements:
                datetime_attr = time_elem.get_attribute("datetime")
                if datetime_attr:
                    # Parse the datetime (format: 2024-11-27T13:04:29.000Z)
                    dt = datetime.fromisoformat(datetime_attr.replace('Z', '+00:00'))
                    date_str = dt.strftime("%Y-%m-%d")
                    print(f"    ✅ Found date: {date_str}")
                    return date_str
            
            print(f"    ⚠️ No time element with datetime found")
            return None
            
        except Exception as e:
            print(f"    ⚠️ Error finding time element: {e}")
            return None
            
    except Exception as e:
        print(f"    ❌ Error loading page: {e}")
        return None

def main():
    """Update enhanced files with dates from Instagram"""
    enhanced_dir = Path("/var/www/uploads/instagram_caption_scraper/rag_system/enhanced_processed")
    
    # Get all enhanced files
    enhanced_files = list(enhanced_dir.glob("enhanced_*.json"))
    print(f"📁 Found {len(enhanced_files)} enhanced files")
    
    # Setup driver
    print("🌐 Setting up Chrome driver...")
    driver = setup_driver()
    
    stats = {"updated": 0, "already_has": 0, "failed": 0}
    
    try:
        total_files = len(enhanced_files)
        for i, file_path in enumerate(enhanced_files, 1):  # Process ALL files
            print(f"\n[{i}/{total_files}] Processing: {file_path.name}")
            
            # Load file
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Check if already has date
            if data.get("post_date"):
                print(f"  ⏭️ Already has date: {data['post_date']}")
                stats["already_has"] += 1
                continue
            
            # Get URL
            url = data.get("source_metadata", {}).get("url")
            if not url:
                print(f"  ⚠️ No URL found")
                stats["failed"] += 1
                continue
            
            # Extract date
            date = extract_date_from_page(driver, url)
            
            if date:
                # Update file
                data["post_date"] = date
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                print(f"  💾 Saved date: {date}")
                stats["updated"] += 1
            else:
                stats["failed"] += 1
            
            # Delay to avoid rate limiting
            time.sleep(2)
            
    finally:
        driver.quit()
        print("\n🏁 Browser closed")
    
    # Print stats
    print("\n" + "="*50)
    print(f"📊 Results:")
    print(f"   ✅ Updated: {stats['updated']}")
    print(f"   ⏭️ Already had dates: {stats['already_has']}")
    print(f"   ❌ Failed: {stats['failed']}")

if __name__ == "__main__":
    print("🚀 Instagram Date Fetcher")
    print("=" * 50)
    print("This will process ALL enhanced files")
    
    response = input("\nContinue? (y/n): ")
    if response.lower() == 'y':
        main()
    else:
        print("Cancelled")