#!/bin/bash
# Test script to simulate cron environment

# Simulate cron's minimal PATH
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

echo "Testing yt-dlp availability in cron-like environment:"
echo "PATH=$PATH"
echo ""

# Test 1: Try to run yt-dlp with just the command name
echo "Test 1: Running 'yt-dlp --version':"
yt-dlp --version 2>&1 || echo "FAILED: yt-dlp not found in PATH"
echo ""

# Test 2: Try with full path
echo "Test 2: Running with full path '/home/arjun/.local/bin/yt-dlp --version':"
/home/arjun/.local/bin/yt-dlp --version 2>&1 || echo "FAILED: yt-dlp not found at full path"
echo ""

# Test 3: Check Python subprocess behavior
echo "Test 3: Python subprocess test:"
/usr/bin/python3 -c "
import subprocess
try:
    result = subprocess.run(['yt-dlp', '--version'], capture_output=True, text=True)
    print(f'SUCCESS: {result.stdout.strip()}')
except FileNotFoundError:
    print('FAILED: FileNotFoundError - yt-dlp not found')
"
