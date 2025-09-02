#!/usr/bin/env python3
"""
Daily Instagram Content Automation Orchestrator
===============================================

This script orchestrates the complete daily automation pipeline:
1. Detect new files that need enhancement
2. Enhance raw files with Claude 3.5 Sonnet  
3. Update vector database with new enhanced content
4. Report results and costs

Designed to run as a daily cron job or GitHub Action.
"""

import os
import sys
import json
import argparse
import subprocess
import time
import requests  # Add for health checks
from datetime import datetime
from pathlib import Path

# Add paths for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
sys.path.append(str(current_dir / 'rag_system'))

def import_with_fallback():
    """Import required modules with helpful error messages"""
    try:
        from rag_system.daily_file_detector import DailyFileDetector
        from rag_system.claude_enhancer import ProductionEnhancer
        from rag_system.core.hybrid_rag_system import HybridRAGSystem
        return DailyFileDetector, ProductionEnhancer, HybridRAGSystem
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running this from the project root directory")
        sys.exit(1)

class DailyOrchestrator:
    """Main orchestrator for daily automation pipeline"""
    
    def __init__(self, base_path: str = None):
        self.base_path = base_path or os.getcwd()
        self.start_time = datetime.now()
        
        # Target profiles for Instagram scraping
        self.target_profiles = [
            {
                "url": "https://www.instagram.com/edhonour/",
                "name": "edhonour",
                "max_reels": 10,  # Reduced to 10 reels for faster testing
                "skip_first": 3,  # Skip first 3 pinned posts to avoid confusion
                "priority": "high"
            }
            # Add more profiles here as needed
        ]
        
        # Import modules
        DailyFileDetector, ProductionEnhancer, HybridRAGSystem = import_with_fallback()
        
        # Initialize components
        self.file_detector = DailyFileDetector(self.base_path)
        self.enhancer = ProductionEnhancer()
        
        # GPT4 Extraction Engine will be initialized when needed (requires more setup)
        self.vector_engine = None
        
        print(f"🎯 Daily Orchestrator initialized")
        print(f"📂 Base path: {self.base_path}")
        print(f"🕐 Started at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def initialize_vector_engine(self):
        """Initialize GPT4 Extraction Engine when needed"""
        if self.vector_engine is None:
            try:
                print("🚀 Initializing GPT4 Extraction Engine...")
                # Import here to avoid import errors if not needed
                _, _, HybridRAGSystem = import_with_fallback()
                self.vector_engine = HybridRAGSystem(enhanced_data_dir='rag_system/enhanced_processed')
                print("✅ GPT4 Extraction Engine initialized")
            except Exception as e:
                print(f"❌ Failed to initialize vector engine: {e}")
                return False
        return True
    
    def step_0_scrape_instagram(self) -> dict:
        """Step 0: Scrape new Instagram content from target profiles"""
        print("\n" + "="*60)
        print("📱 STEP 0: SCRAPING INSTAGRAM CONTENT")
        print("="*60)
        
        result = {
            'success': False,
            'profiles_processed': 0,
            'new_reels_found': 0,
            'errors': [],
            'profiles_results': []
        }
        
        try:
            for profile in self.target_profiles:
                profile_name = profile['name']
                profile_url = profile['url']
                max_reels = profile['max_reels']
                
                print(f"🎯 Scraping {profile_name} (max {max_reels} reels)")
                
                profile_result = {
                    'profile': profile_name,
                    'success': False,
                    'new_reels': 0,
                    'error': None
                }
                
                try:
                    # Run the existing main.py scraper with incremental mode
                    # Use headless mode with automatic login using .env credentials
                    cmd = [
                        sys.executable, 'main.py',
                        '--profile-url', profile_url,
                        '--max-reels', str(max_reels),
                        '--skip', '3',  # Click right arrow 3 times to skip pinned posts
                        '--headless',     # Use headless mode for server environment
                        '--login'        # Use automatic login with credentials from .env
                    ]
                    
                    print(f"   📋 Running: {' '.join(cmd)}")
                    
                    # Run scraper with real-time output (no capture for live streaming)
                    print(f"   🔄 Starting scraper process with live output...")
                    print(f"   📋 Command: {' '.join(cmd)}")
                    print(f"   " + "="*60)
                    
                    # Use call instead of Popen for direct output streaming
                    return_code = subprocess.call(cmd, cwd=self.base_path)
                    
                    print(f"   " + "="*60)
                    print(f"   📊 Scraper process finished with return code: {return_code}")
                    
                    if return_code == 0:
                        print(f"   ✅ {profile_name}: Scraping completed successfully")
                        
                        # Since we're streaming output directly, we can't parse it for counts
                        # But the user will see all the details in real-time
                        # We'll check the output directory to see what was actually processed
                        try:
                            from pathlib import Path
                            import glob
                            
                            # Check for new files in output directory (no subdirectories now)
                            output_pattern = "output/*.json"
                            recent_files = []
                            
                            for file_path in glob.glob(output_pattern):
                                file_time = Path(file_path).stat().st_mtime
                                # Files created in the last 10 minutes
                                if time.time() - file_time < 600:  
                                    recent_files.append(file_path)
                            
                            profile_result['new_reels'] = len(recent_files)
                            print(f"   📊 Found {len(recent_files)} recently created files")
                            
                        except Exception as e:
                            print(f"   ⚠️ Could not count new files: {e}")
                            profile_result['new_reels'] = 1  # Assume success if process completed
                        
                        profile_result['success'] = True
                        
                    else:
                        error_msg = f"Scraper failed with exit code {return_code}"
                        profile_result['error'] = error_msg
                        print(f"   ❌ {profile_name}: {error_msg}")
                        
                except Exception as e:
                    error_msg = f"Scraping failed: {str(e)}"
                    profile_result['error'] = error_msg
                    print(f"   ❌ {profile_name}: {error_msg}")
                
                result['profiles_results'].append(profile_result)
                result['profiles_processed'] += 1
                
                if profile_result['success']:
                    result['new_reels_found'] += profile_result['new_reels']
                else:
                    result['errors'].append(f"{profile_name}: {profile_result['error']}")
            
            # Consider it successful if we processed profiles (even if some failed)
            result['success'] = result['profiles_processed'] > 0
            
        except Exception as e:
            result['errors'].append(f"Instagram scraping failed: {str(e)}")
            print(f"❌ Instagram scraping error: {e}")
        
        return result
    
    def step_1_detect_files(self) -> dict:
        """Step 1: Detect new files that need enhancement"""
        print("\n" + "="*60)
        print("📋 STEP 1: DETECTING NEW FILES")
        print("="*60)
        
        try:
            report = self.file_detector.run_detection(verbose=True)
            
            files_count = len(report.get('files_to_process', []))
            if files_count > 0:
                print(f"✅ Found {files_count} files ready for enhancement")
                return {'success': True, 'report': report}
            else:
                print("ℹ️ No new files found - pipeline complete")
                return {'success': True, 'report': report, 'early_exit': True}
                
        except Exception as e:
            print(f"❌ File detection failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def step_2_enhance_files(self, files_to_process: list) -> dict:
        """Step 2: Enhance detected files with Claude"""
        print("\n" + "="*60)
        print("🤖 STEP 2: ENHANCING FILES WITH CLAUDE")
        print("="*60)
        
        try:
            file_paths = [file_info['file_path'] for file_info in files_to_process]
            print(f"🔄 Enhancing {len(file_paths)} files...")
            
            results = self.enhancer.enhance_batch(file_paths, max_concurrent=3)
            
            if len(results['failed']) == 0:
                print(f"✅ All {len(results['successful'])} files enhanced successfully")
                return {'success': True, 'results': results}
            else:
                print(f"⚠️ Enhancement completed with {len(results['failed'])} failures")
                return {'success': False, 'results': results, 'partial': True}
                
        except Exception as e:
            print(f"❌ Enhancement failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def step_3_update_vectors(self, enhanced_files: list = None) -> dict:
        """Step 3: Update vector database with enhanced files using incremental updates"""
        print("\n" + "="*60)
        print("🗂️ STEP 3: UPDATING VECTOR DATABASE")
        print("="*60)
        
        try:
            if enhanced_files:
                # Process only the specific files that were just enhanced
                print(f"🔄 Processing {len(enhanced_files)} newly enhanced files...")
                
                # Import and use vector manager directly
                sys.path.append(str(Path(self.base_path) / 'rag_system'))
                from core.vector_manager import QdrantVectorManager
                
                vector_manager = QdrantVectorManager(qdrant_url="http://localhost:6333")
                vector_manager.setup_collections()
                
                # Process only the specific enhanced files  
                results = vector_manager.process_specific_files(enhanced_files)
                
                return {
                    'success': True,
                    'results': {
                        'total_added': results.get('total_added', 0),
                        'total_skipped': results.get('total_skipped', 0),
                        'failed': results.get('failed', [])
                    }
                }
            else:
                print("ℹ️ No specific files provided, using standard indexing...")
                result = subprocess.run(
                    [sys.executable, 'rebuild_vector_index.py'],
                    cwd=self.base_path,
                    capture_output=True,
                    text=True
                )
            
            if result.returncode == 0:
                print("✅ Vector indexing completed successfully")
                print("📋 Vector indexing output:")
                print(result.stdout)
                
                return {
                    'success': True,
                    'results': {
                        'stdout': result.stdout,
                        'stderr': result.stderr
                    }
                }
            else:
                print(f"❌ Vector indexing failed with return code {result.returncode}")
                print(f"Error output: {result.stderr}")
                
                return {
                    'success': False,
                    'error': f"Vector indexing failed: {result.stderr}"
                }
                
        except Exception as e:
            print(f"❌ Vector update failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def print_summary(self, scraping_result: dict, detection_result: dict, enhancement_result: dict, vector_result: dict):
        """Print simple summary for cron job monitoring"""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        # Simple success/fail status for cron monitoring
        enhancement_results = enhancement_result.get('results', {})
        
        new_reels = scraping_result.get('new_reels_found', 0)
        files_detected = len(detection_result.get('report', {}).get('files_to_process', []))
        files_enhanced = len(enhancement_results.get('successful', []))
        files_vectorized = vector_result.get('results', {}).get('total_added', 0)
        
        # Simple one-line summary for easy grep in logs
        print(f"[{end_time.strftime('%Y-%m-%d %H:%M:%S')}] COMPLETE - Reels: {new_reels}, Enhanced: {files_enhanced}, Vectorized: {files_vectorized}, Duration: {duration:.1f}s")
        
        # Return exit code for cron
        if detection_result.get('success', False):
            return 0
        else:
            return 1
    
    def run_preflight_checks(self) -> bool:
        """
        Run pre-flight checks to ensure all dependencies are ready
        Prevents wasting API calls on enhancement if indexing will fail
        """
        print("🔍 Running pre-flight checks...")
        
        checks_passed = 0
        total_checks = 3
        
        # 1. Check Qdrant vector database
        print("  📦 Checking Qdrant vector database...")
        try:
            # Try collections endpoint instead of health (which returns 404)
            response = requests.get("http://localhost:6333/collections", timeout=5)
            if response.status_code == 200:
                print("    ✅ Qdrant is responding")
                checks_passed += 1
            else:
                print(f"    ❌ Qdrant health check failed (status: {response.status_code})")
        except requests.exceptions.ConnectionError:
            print("    ❌ Cannot connect to Qdrant (not running?)")
            print("    💡 Run: ./start_automation.sh or docker-compose up -d")
        except requests.exceptions.Timeout:
            print("    ❌ Qdrant connection timeout")
        except Exception as e:
            print(f"    ❌ Qdrant check failed: {e}")
        
        # 2. Check vector collections exist
        print("  🗂️  Checking vector collections...")
        try:
            response = requests.get("http://localhost:6333/collections", timeout=5)
            if response.status_code == 200:
                collections = response.json()
                expected_collections = ["instagram_content_vectors", "instagram_qa_vectors", 
                                      "instagram_tech_vectors", "instagram_context_vectors"]
                existing_collections = [c['name'] for c in collections.get('result', {}).get('collections', [])]
                
                if all(col in existing_collections for col in expected_collections):
                    print("    ✅ All vector collections exist")
                    checks_passed += 1
                else:
                    missing = set(expected_collections) - set(existing_collections)
                    print(f"    ❌ Missing collections: {missing}")
                    print("    💡 Run: python initialize_qdrant.py")
            else:
                print(f"    ❌ Cannot fetch collections (status: {response.status_code})")
        except Exception as e:
            print(f"    ❌ Collections check failed: {e}")
        
        # 3. Check browser profile directory exists
        print("  🌐 Checking browser profile directory...")
        profile_dir = Path("./browser_profiles/chrome_automation")
        if profile_dir.exists():
            print("    ✅ Browser profile directory exists")
            checks_passed += 1
        else:
            print("    ❌ Browser profile directory missing")
            print("    💡 Creating directory...")
            try:
                profile_dir.mkdir(parents=True, exist_ok=True)
                print("    ✅ Browser profile directory created")
                checks_passed += 1
            except Exception as e:
                print(f"    ❌ Failed to create directory: {e}")
        
        # Summary
        print(f"\n📊 Pre-flight summary: {checks_passed}/{total_checks} checks passed")
        
        if checks_passed == total_checks:
            print("🚀 All systems ready! Proceeding with automation...")
            return True
        else:
            print("❌ Pre-flight checks failed. Fix issues before running automation.")
            print("💡 Quick fix: Run ./start_automation.sh to set up all dependencies")
            return False

    def run_full_pipeline(self, skip_scraping: bool = False) -> int:
        """Run the complete daily automation pipeline"""
        
        # Step -1: Pre-flight checks to prevent API waste
        print("🔍 Starting daily automation with dependency validation...")
        if not self.run_preflight_checks():
            print("❌ Automation halted due to dependency issues")
            return 1  # Return error code for cron
        
        # Step 0: Instagram Scraping (optional)
        if not skip_scraping:
            scraping_result = self.step_0_scrape_instagram()
            # Don't fail pipeline if scraping fails - we might still have files to process
            if not scraping_result['success']:
                print("⚠️ Instagram scraping had issues, but continuing with existing files...")
        else:
            scraping_result = {'success': True, 'skipped': True, 'new_reels_found': 0}
            print("⏭️ Skipping Instagram scraping")
        
        # Step 1: Detect files
        detection_result = self.step_1_detect_files()
        if not detection_result['success']:
            vector_result = {'success': False, 'skipped': True}
            enhancement_result = {'success': False, 'skipped': True}
            return self.print_summary(scraping_result, detection_result, enhancement_result, vector_result)
        
        # Early exit if no files found
        if detection_result.get('early_exit'):
            enhancement_result = {'success': True, 'skipped': True, 'results': {'successful': [], 'failed': [], 'total_cost': 0.0}}
            vector_result = {'success': True, 'skipped': True, 'results': {'total_added': 0, 'total_skipped': 0, 'failed': []}}
            return self.print_summary(scraping_result, detection_result, enhancement_result, vector_result)
        
        # Step 2: Enhance files
        files_to_process = detection_result['report']['files_to_process']
        enhancement_result = self.step_2_enhance_files(files_to_process)
        
        # Step 3: Update vectors with the specific files that were just enhanced
        enhanced_file_paths = []
        if enhancement_result.get('success') and 'results' in enhancement_result:
            # Get the paths of successfully enhanced files
            successful_results = enhancement_result['results'].get('successful', [])
            enhanced_file_paths = [result.get('output_file') for result in successful_results if result.get('output_file')]
        
        vector_result = self.step_3_update_vectors(enhanced_file_paths)
        
        # Print summary and return exit code
        return self.print_summary(scraping_result, detection_result, enhancement_result, vector_result)


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description='Daily Instagram Content Automation')
    parser.add_argument('--base-path', help='Base path to project directory')
    parser.add_argument('--skip-scraping', action='store_true', help='Skip Instagram scraping step')
    parser.add_argument('--detect-only', action='store_true', help='Only run file detection')
    parser.add_argument('--enhance-only', action='store_true', help='Only run enhancement (skip detection)')
    parser.add_argument('--vectors-only', action='store_true', help='Only update vectors (skip detection/enhancement)')
    parser.add_argument('--no-report', action='store_true', help='Skip saving detailed report')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without executing')
    
    args = parser.parse_args()
    
    try:
        orchestrator = DailyOrchestrator(args.base_path)
        
        if args.dry_run:
            print("🔍 DRY RUN - No changes will be made")
            detection_result = orchestrator.step_1_detect_files()
            return 0
        
        elif args.detect_only:
            detection_result = orchestrator.step_1_detect_files()
            return 0 if detection_result['success'] else 1
            
        elif args.enhance_only:
            # Skip detection, enhance any files found by detector
            detection_result = orchestrator.step_1_detect_files()
            if detection_result['success'] and not detection_result.get('early_exit'):
                enhancement_result = orchestrator.step_2_enhance_files(detection_result['report']['files_to_process'])
                return 0 if enhancement_result['success'] else 1
            else:
                print("No files to enhance")
                return 0
                
        elif args.vectors_only:
            vector_result = orchestrator.step_3_update_vectors()
            return 0 if vector_result['success'] else 1
            
        else:
            # Run full pipeline
            exit_code = orchestrator.run_full_pipeline(skip_scraping=args.skip_scraping)
            return exit_code
            
    except KeyboardInterrupt:
        print("\n⏹️ Daily automation interrupted by user")
        return 130
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)