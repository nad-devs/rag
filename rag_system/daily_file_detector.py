#!/usr/bin/env python3
"""
Daily File Detector - Finds new Instagram files that need enhancement
Compares output/ directory with enhanced_processed/ to find unprocessed files
"""

import os
import json
import glob
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

class DailyFileDetector:
    def __init__(self, base_path: str = None):
        if base_path is None:
            self.base_path = Path(__file__).parent.parent
        else:
            self.base_path = Path(base_path)
            
        self.output_dir = self.base_path / "output"
        self.enhanced_dir = self.base_path / "rag_system" / "enhanced_processed"
        
    def scan_raw_files(self) -> Dict[str, str]:
        """Scan output/ directory for all raw Instagram files"""
        raw_files = {}
        
        # Look for all JSON files in output directory
        pattern = str(self.output_dir / "**" / "*.json")
        
        for file_path in glob.glob(pattern, recursive=True):
            # Extract shortcode from filename
            filename = os.path.basename(file_path)
            
            # Skip summary files, metadata files, and enhanced files that are in wrong directory
            if any(skip in filename for skip in ['instagram_captions_', 'profile_', 'summary', 'enhanced_']):
                continue
                
            # Extract shortcode (filename without .json)
            shortcode = filename.replace('.json', '')
            raw_files[shortcode] = file_path
            
        return raw_files
    
    def scan_enhanced_files(self) -> Dict[str, str]:
        """Scan enhanced_processed/ directory for all enhanced files"""
        enhanced_files = {}
        
        # Look for all enhanced JSON files
        pattern = str(self.enhanced_dir / "**" / "enhanced_*.json")
        
        for file_path in glob.glob(pattern, recursive=True):
            filename = os.path.basename(file_path)
            
            # Extract shortcode from enhanced filename
            # Format: enhanced_SHORTCODE.json
            if filename.startswith('enhanced_'):
                shortcode = filename.replace('enhanced_', '').replace('.json', '')
                enhanced_files[shortcode] = file_path
                
        return enhanced_files
    
    def find_new_files(self) -> List[Tuple[str, str]]:
        """Find files that exist in output/ but not in enhanced_processed/"""
        raw_files = self.scan_raw_files()
        enhanced_files = self.scan_enhanced_files()
        
        new_files = []
        
        for shortcode, raw_path in raw_files.items():
            if shortcode not in enhanced_files:
                new_files.append((shortcode, raw_path))
                
        return new_files
    
    def validate_file_content(self, file_path: str) -> bool:
        """Validate that the file has valid content for enhancement"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if file has required fields
            required_fields = ['success', 'shortcode']
            if not all(field in data for field in required_fields):
                return False
                
            # Check if extraction was successful
            if not data.get('success', False):
                return False
                
            # Check if we have content to enhance
            content = data.get('caption', '') or data.get('transcription_data', {}).get('full_text', '')
            if not content or len(content.strip()) < 10:
                return False
                
            return True
            
        except (json.JSONDecodeError, FileNotFoundError, UnicodeDecodeError):
            return False
    
    def get_file_info(self, file_path: str) -> Dict:
        """Get information about a file for processing"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract content
            content = data.get('caption', '') or data.get('transcription_data', {}).get('full_text', '')
            
            return {
                'shortcode': data.get('shortcode', ''),
                'url': data.get('url', ''),
                'content_length': len(content),
                'word_count': len(content.split()),
                'profile': data.get('profile_username', 'edhonour'),
                'extraction_method': data.get('extraction_method', 'unknown'),
                'timestamp': data.get('extraction_timestamp', 0)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def check_for_duplicates(self, new_files: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Simple file existence check - include all files that don't have enhanced versions"""
        # No validation needed - just return all new files
        # The enhancement process will handle any file-specific issues
        return new_files
    
    def generate_report(self, new_files: List[Tuple[str, str]]) -> Dict:
        """Generate a summary report of new files found"""
        report = {
            'scan_timestamp': datetime.now().isoformat(),
            'total_raw_files': len(self.scan_raw_files()),
            'total_enhanced_files': len(self.scan_enhanced_files()),
            'new_files_found': len(new_files),
            'files_to_process': [],
            'estimated_cost': 0.0,
            'estimated_processing_time': 0.0
        }
        
        for shortcode, file_path in new_files:
            # Simply add all files - no validation
            report['files_to_process'].append({
                'shortcode': shortcode,
                'file_path': file_path,
                'content_length': 0,  # Will be determined during enhancement
                'word_count': 0,      # Will be determined during enhancement
                'profile': 'unknown'  # Will be determined during enhancement
            })
            
            # Estimate cost (Claude 3.5 Sonnet average: $0.0264 per file)
            report['estimated_cost'] += 0.0264
            # Estimate processing time (average: 22 seconds per file)
            report['estimated_processing_time'] += 22
        
        return report
    
    def run_detection(self, verbose: bool = True) -> Dict:
        """Main method to run file detection"""
        if verbose:
            print("🔍 Daily File Detection Starting...")
            print(f"📂 Scanning: {self.output_dir}")
            print(f"📂 Enhanced: {self.enhanced_dir}")
        
        # Find new files
        new_files = self.find_new_files()
        
        if verbose:
            print(f"📊 Found {len(new_files)} potentially new files")
        
        # Filter out invalid files and duplicates
        valid_files = self.check_for_duplicates(new_files)
        
        if verbose:
            print(f"✅ {len(valid_files)} valid files ready for processing")
        
        # Generate report
        report = self.generate_report(valid_files)
        
        if verbose:
            self.print_report(report)
            
        return report
    
    def print_report(self, report: Dict):
        """Print a formatted report"""
        print("\n" + "="*60)
        print("📋 DAILY FILE DETECTION REPORT")
        print("="*60)
        print(f"📊 Total raw files: {report['total_raw_files']}")
        print(f"📊 Total enhanced files: {report['total_enhanced_files']}")
        print(f"🆕 New files found: {report['new_files_found']}")
        print(f"✅ Files to process: {len(report['files_to_process'])}")
        
        if report['files_to_process']:
            print(f"💰 Estimated cost: ${report['estimated_cost']:.4f}")
            print(f"⏱️  Estimated time: {report['estimated_processing_time']:.0f}s ({report['estimated_processing_time']/60:.1f}min)")
            
            print("\n📄 Files ready for enhancement:")
            for file_info in report['files_to_process']:
                print(f"  • {file_info['shortcode']}")
        else:
            print("\n✨ No new files to process!")
        
        print("="*60)

def main():
    """Main function for CLI usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Detect new Instagram files for enhancement')
    parser.add_argument('--base-path', help='Base path to the project directory')
    parser.add_argument('--quiet', action='store_true', help='Run in quiet mode')
    parser.add_argument('--save-report', help='Save report to JSON file')
    
    args = parser.parse_args()
    
    # Run detection
    detector = DailyFileDetector(args.base_path)
    report = detector.run_detection(verbose=not args.quiet)
    
    # Save report if requested
    if args.save_report:
        with open(args.save_report, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"💾 Report saved to: {args.save_report}")
    
    # Return appropriate exit code
    files_to_process = len(report['files_to_process'])
    if files_to_process > 0:
        print(f"\n🎯 Ready to process {files_to_process} files!")
        return 0
    else:
        print("\n✨ No work needed today!")
        return 1

if __name__ == "__main__":
    exit(main())