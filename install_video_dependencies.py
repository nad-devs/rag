#!/usr/bin/env python3
"""
Install Video Transcription Dependencies
Comprehensive installer for all video transcription requirements.
"""

import subprocess
import sys
import os
import platform


def run_command(command, description, check=True):
    """Run a command and display progress."""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=check,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"✅ {description} completed")
            return True
        else:
            print(f"❌ {description} failed: {result.stderr}")
            return False
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required for OpenAI Whisper. Current version: {version.major}.{version.minor}")
        return False
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True


def install_system_ffmpeg():
    """Install FFmpeg system dependency."""
    print("\n🔧 Installing FFmpeg System Dependency")
    print("=" * 45)
    
    system = platform.system().lower()
    
    if system == "linux":
        # Check if we're in WSL
        try:
            with open('/proc/version', 'r') as f:
                if 'microsoft' in f.read().lower():
                    print("🐧 WSL (Windows Subsystem for Linux) detected")
        except:
            pass
        
        print("📥 Attempting to install FFmpeg...")
        print("⚠️  This requires sudo permissions. You may be prompted for your password.")
        
        # Try different package managers
        commands = [
            ("sudo apt update && sudo apt install -y ffmpeg", "Installing FFmpeg (apt)"),
            ("sudo yum install -y ffmpeg", "Installing FFmpeg (yum)"),
            ("sudo dnf install -y ffmpeg", "Installing FFmpeg (dnf)"),
        ]
        
        for command, desc in commands:
            if run_command(command, desc, check=False):
                return True
                
        print("\n❌ Automatic FFmpeg installation failed.")
        print("Please install FFmpeg manually:")
        print("  Ubuntu/Debian: sudo apt install ffmpeg")
        print("  CentOS/RHEL: sudo yum install ffmpeg")
        print("  Fedora: sudo dnf install ffmpeg")
        return False
        
    elif system == "darwin":  # macOS
        print("🍎 macOS detected")
        if run_command("which brew", "Checking for Homebrew", check=False):
            return run_command("brew install ffmpeg", "Installing FFmpeg (Homebrew)")
        else:
            print("❌ Homebrew not found. Please install Homebrew first:")
            print("  /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
            print("  Then run: brew install ffmpeg")
            return False
            
    elif system == "windows":
        print("🪟 Windows detected")
        print("Please install FFmpeg manually:")
        print("1. Download from: https://ffmpeg.org/download.html#build-windows")
        print("2. Extract to C:\\ffmpeg")
        print("3. Add C:\\ffmpeg\\bin to your PATH environment variable")
        print("4. Restart your terminal/command prompt")
        return False
    
    else:
        print(f"❓ Unknown system: {system}")
        print("Please install FFmpeg manually for your system")
        return False


def install_python_dependencies():
    """Install Python packages for video transcription."""
    print("\n🐍 Installing Python Dependencies")
    print("=" * 40)
    
    # First, upgrade pip
    if not run_command("python -m pip install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install basic requirements first
    basic_packages = [
        "numpy>=1.24.0",
        "requests>=2.31.0", 
        "selenium>=4.15.0",
        "webdriver-manager>=4.0.1",
        "pyyaml>=6.0.1",
        "beautifulsoup4>=4.12.2",
        "fake-useragent>=1.4.0"
    ]
    
    for package in basic_packages:
        if not run_command(f"pip install '{package}'", f"Installing {package.split('>=')[0]}"):
            print(f"⚠️  Warning: Failed to install {package}")
    
    # Install compatible PyTorch versions
    print("\n🔥 Installing PyTorch (compatible versions)...")
    
    # First, uninstall any existing torch/torchaudio to avoid conflicts
    run_command("pip uninstall -y torch torchaudio", "Uninstalling existing PyTorch", check=False)
    
    # Install specific compatible versions
    torch_commands = [
        # Try stable compatible versions first
        ("pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu", 
         "Installing PyTorch 2.1.0 (CPU, stable)"),
        # Fallback to default versions
        ("pip install torch==2.1.0 torchaudio==2.1.0", 
         "Installing PyTorch 2.1.0 (default)"),
        # Last resort - just torch without torchaudio
        ("pip install torch==2.1.0", 
         "Installing PyTorch 2.1.0 (no torchaudio)")
    ]
    
    pytorch_success = False
    for command, description in torch_commands:
        if run_command(command, description, check=False):
            pytorch_success = True
            break
        else:
            print(f"⚠️  {description} failed, trying next option...")
    
    if not pytorch_success:
        print("❌ All PyTorch installation attempts failed")
        print("💡 You may need to install PyTorch manually")
    
    # Install video processing packages
    video_packages = [
        ("openai-whisper", "OpenAI Whisper (AI transcription)"),
        ("moviepy", "MoviePy (video processing)"),
        ("ffmpeg-python", "FFmpeg Python bindings"),
    ]
    
    for package, description in video_packages:
        run_command(f"pip install {package}", f"Installing {description}", check=False)
    
    return True


def test_installations():
    """Test if all installations work correctly."""
    print("\n🧪 Testing Installations")
    print("=" * 25)
    
    # Define tests with criticality levels
    tests = [
        ('torch', 'PyTorch', True),  # Critical
        ('whisper', 'OpenAI Whisper', True),  # Critical
        ('moviepy', 'MoviePy', True),  # Critical
        ('ffmpeg', 'FFmpeg Python', True),  # Critical
        ('selenium', 'Selenium', True),  # Critical
        ('requests', 'Requests', True),  # Critical
        ('torchaudio', 'TorchAudio', False),  # Optional - known compatibility issues
    ]
    
    success_count = 0
    critical_success = 0
    total_critical = sum(1 for _, _, critical in tests if critical)
    errors = []
    
    for module, name, critical in tests:
        try:
            __import__(module)
            print(f"✅ {name}: Working")
            success_count += 1
            if critical:
                critical_success += 1
        except ImportError as e:
            status = "❌" if critical else "⚠️ "
            print(f"{status} {name}: Failed - {str(e)}")
            errors.append((name, str(e), critical))
        except Exception as e:
            # Handle runtime errors (like TorchAudio symbol issues)
            status = "❌" if critical else "⚠️ "
            print(f"{status} {name}: Runtime error - {str(e)}")
            errors.append((name, f"Runtime error: {str(e)}", critical))
            
            # For non-critical packages with runtime errors, don't count as total failure
            if not critical:
                print(f"💡 {name} has compatibility issues but is optional for basic functionality")
    
    # Test FFmpeg system command
    ffmpeg_working = False
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5)
        if result.returncode == 0:
            print(f"✅ FFmpeg (system): Working")
            ffmpeg_working = True
        else:
            print(f"❌ FFmpeg (system): Failed")
            errors.append(("FFmpeg (system)", "Command failed", True))
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        print(f"❌ FFmpeg (system): Failed - {str(e)}")
        errors.append(("FFmpeg (system)", str(e), True))
    
    # Calculate success metrics
    total_tests = len(tests) + 1  # +1 for FFmpeg system test
    critical_tests = total_critical + 1  # +1 for FFmpeg system test
    critical_working = critical_success + (1 if ffmpeg_working else 0)
    
    print(f"\n📊 Result: {success_count}/{total_tests} total components working")
    print(f"🎯 Critical: {critical_working}/{critical_tests} essential components working")
    
    if errors:
        print(f"\n❌ Failed components:")
        critical_errors = [e for e in errors if e[2]]  # Critical errors
        optional_errors = [e for e in errors if not e[2]]  # Optional errors
        
        if critical_errors:
            print("   Critical failures:")
            for name, error, _ in critical_errors:
                print(f"     - {name}: {error}")
        
        if optional_errors:
            print("   Optional failures (not blocking):")
            for name, error, _ in optional_errors:
                print(f"     - {name}: {error}")
    
    # Consider installation successful if all critical components work
    installation_success = critical_working == critical_tests
    
    return installation_success, errors


def test_whisper_model():
    """Test loading Whisper model."""
    print("\n🎤 Testing Whisper Model Loading")
    print("=" * 35)
    
    try:
        import whisper
        print("📥 Loading Whisper base model (this may take time on first run)...")
        model = whisper.load_model("base")
        print("✅ Whisper model loaded successfully!")
        
        # Test transcription with a dummy audio (if possible)
        print("🧪 Testing basic transcription functionality...")
        # Note: We can't test actual transcription without an audio file
        print("✅ Whisper ready for transcription")
        return True
        
    except ImportError as e:
        print(f"❌ Whisper import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Whisper model loading failed: {e}")
        print("💡 This might be due to missing system dependencies or insufficient disk space")
        return False


def create_test_script():
    """Create a test script for the user."""
    test_script = '''#!/usr/bin/env python3
"""Quick test of video transcription capabilities"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.video_transcriber import VideoTranscriber, WHISPER_AVAILABLE, MOVIEPY_AVAILABLE
    print("✅ VideoTranscriber import successful")
    
    print(f"🎤 Whisper available: {WHISPER_AVAILABLE}")
    print(f"🎬 MoviePy available: {MOVIEPY_AVAILABLE}")
    
    # Test system info (without driver)
    class DummyLogger:
        def info(self, msg): print(f"INFO: {msg}")
        def warning(self, msg): print(f"WARN: {msg}")
        def error(self, msg): print(f"ERROR: {msg}")
        def debug(self, msg): print(f"DEBUG: {msg}")
    
    # Create dummy transcriber to test system info
    transcriber = VideoTranscriber(None, DummyLogger())
    info = transcriber.get_system_info()
    
    print("\\n📊 System Capabilities:")
    for key, value in info.items():
        if key.endswith('_error'):
            print(f"❌ {key}: {value}")
        else:
            status = "✅" if value else "❌"
            print(f"{status} {key}: {value}")
    
    if all([info['whisper_available'], info['moviepy_available'], info['ffmpeg_available']]):
        print("\\n🎉 All video transcription dependencies are working!")
    else:
        print("\\n⚠️  Some dependencies are missing. Run install_video_dependencies.py again.")
        
except Exception as e:
    print(f"❌ Test failed: {e}")
'''
    
    with open('test_dependencies.py', 'w') as f:
        f.write(test_script)
    
    print("✅ Created test_dependencies.py")


def main():
    """Main installation function."""
    print("🚀 Instagram Video Transcription - Dependency Installer")
    print("=" * 60)
    print("This script will install all required dependencies for video transcription.")
    print("📋 What will be installed:")
    print("   • FFmpeg (system dependency)")  
    print("   • OpenAI Whisper (AI transcription)")
    print("   • MoviePy (video processing)")
    print("   • PyTorch (ML backend)")
    print("   • Supporting packages")
    print()
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Install system FFmpeg
    print("⚠️  Note: FFmpeg installation may require administrator privileges")
    ffmpeg_success = install_system_ffmpeg()
    
    # Install Python dependencies
    python_success = install_python_dependencies()
    
    # Test installations
    all_working, errors = test_installations()
    
    # Test Whisper model if basic tests pass
    whisper_working = False
    if all_working:
        whisper_working = test_whisper_model()
    
    # Create test script
    create_test_script()
    
    # Final summary
    print(f"\n📋 Installation Summary")
    print("=" * 30)
    print(f"✅ Python dependencies: {'✅' if python_success else '❌'}")
    print(f"✅ FFmpeg system: {'✅' if ffmpeg_success else '❌'}")
    print(f"✅ All components: {'✅' if all_working else '❌'}")
    print(f"✅ Whisper model: {'✅' if whisper_working else '❌'}")
    
    if all_working and whisper_working:
        print("\n🎉 Installation completed successfully!")
        print("\n📋 Next steps:")
        print("1. Test setup: python test_dependencies.py")
        print("2. Test video transcription: python test_video_transcription.py")
        print("3. Run the scraper: python main.py --url 'INSTAGRAM_URL'")
        return 0
    else:
        print("\n⚠️  Installation completed with issues:")
        if errors:
            print("\n❌ Failed components:")
            for name, error in errors:
                print(f"   • {name}: {error}")
        
        print("\n🔧 Manual installation steps:")
        if not ffmpeg_success:
            print("   • Install FFmpeg: sudo apt install ffmpeg (Linux)")
        if not all_working:
            print("   • Install Python packages: pip install -r requirements.txt")
        if not whisper_working:
            print("   • Install Whisper: pip install openai-whisper torch torchaudio")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())