# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an Instagram content scraper and RAG (Retrieval-Augmented Generation) system that:
1. Scrapes Instagram reels/videos and captions using Selenium
2. Enhances content with Claude AI for deeper analysis
3. Stores content in a Qdrant vector database (Docker container on port 6333)
4. Provides a web interface for searching enhanced content (Gunicorn on port 80)

## Key Architecture Components

### Main Entry Points
- `main.py` - CLI for Instagram scraping operations
- `daily_orchestrator.py` - Automated daily pipeline for content enhancement (runs as cron job)
- `rag_system/web_interface.py` - Web app for searching content (production on port 80 with Gunicorn)

### Core Subsystems

**Instagram Scraping (`src/`)**
- `src/scraper.py` - Main InstagramScraper class handling browser automation
- `src/config.py` - Configuration management
- `src/utils.py` - Utility functions for URL validation, file operations
- `src/video_transcriber.py` - Video transcription using Whisper

**RAG System (`rag_system/`)**
- `rag_system/core/hybrid_rag_system.py` - Main RAG system combining vector and keyword search
- `rag_system/core/vector_manager.py` - Qdrant vector database operations
- `rag_system/claude_enhancer.py` - Content enhancement with Claude API
- `rag_system/claude_rag_engine.py` - Claude-powered search and synthesis
- `rag_system/daily_file_detector.py` - Detects new files for processing

**Data Storage**
- `output/` - Raw scraped JSON files organized by profile
- `qdrant_storage/` - Vector database storage
- `browser_profiles/` - Chrome profiles for persistent login

## Common Development Commands

```bash
# Install dependencies
pip install -r requirements.txt
sudo pip install gunicorn  # For production server

# Docker Operations
docker-compose up -d        # Start Qdrant vector database
docker-compose down         # Stop Qdrant
docker ps                   # Check running containers

# Run Instagram scraper with browser handoff (most reliable)
python main.py --profile-url "https://www.instagram.com/username/" --max-reels 25 --browser-handoff

# Run with auto-login (requires .env credentials)
python main.py --profile-url "https://www.instagram.com/username/" --max-reels 25 --login

# Incremental update (only new content)
python main.py --profile-url "https://www.instagram.com/username/" --incremental --browser-handoff

# Validate and fill gaps in collection
python main.py --profile-url "https://www.instagram.com/username/" --validate-up-to 500 --browser-handoff

# Run daily orchestrator pipeline
python daily_orchestrator.py --dry-run  # Test mode
python daily_orchestrator.py           # Production mode

# Start web interface (Development)
python rag_system/web_interface.py --dev --port 5111  # Development mode with Flask

# Start web interface (Production)
sudo nohup gunicorn --bind 0.0.0.0:80 --workers 3 --timeout 300 rag_system.web_interface:app > gunicorn.out 2>&1 &

# Stop production server
sudo pkill -f gunicorn

# Restart production server
sudo pkill -f gunicorn
sudo nohup gunicorn --bind 0.0.0.0:80 --workers 3 --timeout 300 rag_system.web_interface:app > gunicorn.out 2>&1 &

# Check server status
ps aux | grep gunicorn
tail -f gunicorn.out        # View production logs

# Check port usage
sudo ss -tulpn | grep -E ':80|:6333'

# Run tests
pytest test_metadata_enhanced_system.py
python -m pytest

# Check Qdrant vector database
python check_docker_qdrant.py
```

## Environment Configuration

Required `.env` file:
```
INSTAGRAM_USERNAME=your_username
INSTAGRAM_PASSWORD=your_password
ANTHROPIC_API_KEY=sk-ant-api...
OPENAI_API_KEY=sk-proj-...  # Optional, for Whisper transcription
```

## Key Configuration Files

- `config.yaml` - Selenium settings, Instagram selectors, scraping parameters
- `requirements.txt` - Python dependencies (includes gunicorn for production)
- `docker-compose.yml` - Qdrant vector database setup
- `gunicorn.out` - Production server logs
- `nohup.out` - Legacy Flask server logs (deprecated)

## Important Implementation Notes

1. **Authentication Methods**: The scraper supports three login modes:
   - `--browser-handoff`: Opens browser for manual login (most reliable)
   - `--login`: Auto-login with credentials from .env
   - `--interactive-login`: Pauses for manual login in automated browser

2. **Rate Limiting**: Built-in delays and randomization in `config.yaml` to avoid detection

3. **Vector Database**: Qdrant runs on port 6333, stores embeddings using sentence-transformers

4. **Content Enhancement**: Claude API enhances raw content with themes, hooks, and insights

5. **Error Handling**: Extensive retry logic and error recovery in scraper.py

6. **Profile Scraping**: Supports incremental updates, gap detection, and batch processing

7. **Production Web Server**: Uses Gunicorn instead of Flask's development server
   - Runs 3 worker processes for concurrent request handling
   - 300-second timeout for long-running Claude API calls
   - Accessible at https://rag.skilliks.ai/ (port 80)

## Production Deployment Details

### Current Production Setup
- **URL**: https://rag.skilliks.ai/
- **Web Server**: Gunicorn with 3 workers on port 80
- **Database**: Qdrant vector database in Docker on port 6333
- **Process Management**: Running with nohup (systemd service planned)
- **Automation**: Daily orchestrator runs as cron job

### Server Architecture
- **Port 80**: Gunicorn serving the RAG web interface
- **Port 6333**: Qdrant vector database (Docker container)
- **Workers**: 3 Gunicorn workers for concurrent request handling
- **Timeout**: 300 seconds for long-running Claude API calls

### Recent Changes (Production Ready)
- Replaced Flask development server with Gunicorn
- Modified `web_interface.py` to support both dev and production modes
- Added `gunicorn==22.0.0` to requirements.txt
- Updated default port from 5000 to 80 in web_interface.py

## Testing Approach

- Unit tests in `test_metadata_enhanced_system.py`
- Test functions embedded in modules (run with `if __name__ == "__main__"`)
- Use `--debug-mode` flag for verbose logging during scraping