#!/usr/bin/env python3
"""
Check Docker Qdrant status and collections
"""

import subprocess
import json

def check_docker_qdrant():
    """Check if Qdrant Docker container is running"""
    print("🐳 DOCKER QDRANT STATUS CHECK")
    print("="*60)
    
    try:
        # Check if Docker is running
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Docker is not installed or not running")
            return False
        
        print(f"✅ Docker version: {result.stdout.strip()}")
        
        # Check for Qdrant containers
        result = subprocess.run(['docker', 'ps', '-a', '--filter', 'ancestor=qdrant/qdrant', '--format', 'json'], 
                              capture_output=True, text=True)
        
        if result.returncode != 0:
            print("❌ Error checking Docker containers")
            return False
        
        containers = []
        for line in result.stdout.strip().split('\n'):
            if line:
                try:
                    containers.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        
        if not containers:
            print("❌ No Qdrant containers found")
            print("💡 To start Qdrant:")
            print("   docker run -p 6333:6333 -d qdrant/qdrant")
            return False
        
        print(f"📊 Found {len(containers)} Qdrant container(s):")
        for container in containers:
            name = container.get('Names', 'unnamed')
            status = container.get('State', 'unknown')
            ports = container.get('Ports', 'no ports')
            created = container.get('CreatedAt', 'unknown')
            
            print(f"   Container: {name}")
            print(f"      Status: {status}")
            print(f"      Ports: {ports}")
            print(f"      Created: {created}")
            
            if status != 'running':
                container_id = container.get('ID', '')
                print(f"   ⚠️ Container is not running!")
                print(f"   💡 To start: docker start {container_id}")
        
        return True
        
    except FileNotFoundError:
        print("❌ Docker command not found. Is Docker installed?")
        return False
    except Exception as e:
        print(f"❌ Error checking Docker: {e}")
        return False

def check_qdrant_api():
    """Check if Qdrant API is accessible"""
    print("\n🌐 QDRANT API CHECK")
    print("="*40)
    
    try:
        import requests
        response = requests.get('http://localhost:6333/collections', timeout=5)
        
        if response.status_code == 200:
            collections = response.json()
            print("✅ Qdrant API is accessible")
            print(f"📊 Found {len(collections.get('result', {}).get('collections', []))} collections")
            
            for collection in collections.get('result', {}).get('collections', []):
                name = collection.get('name', 'unknown')
                print(f"   - {name}")
            
            return True
        else:
            print(f"❌ Qdrant API returned status {response.status_code}")
            return False
            
    except ImportError:
        print("⚠️ requests library not installed, trying with curl...")
        try:
            result = subprocess.run(['curl', '-s', 'http://localhost:6333/collections'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print("✅ Qdrant API is accessible (via curl)")
                print(f"Response: {result.stdout[:200]}...")
                return True
            else:
                print("❌ Qdrant API not accessible")
                return False
        except Exception as e:
            print(f"❌ Error checking API with curl: {e}")
            return False
    except Exception as e:
        print(f"❌ Error checking Qdrant API: {e}")
        return False

def main():
    print("🧪 DOCKER QDRANT DEBUG TOOL")
    print("="*60)
    
    # Check Docker and Qdrant status
    docker_ok = check_docker_qdrant()
    
    if docker_ok:
        api_ok = check_qdrant_api()
        
        if api_ok:
            print("\n✅ Qdrant is running and accessible!")
            print("💡 Now run: python debug_vector_indexing.py")
        else:
            print("\n❌ Qdrant container is running but API is not accessible")
            print("💡 Check if port 6333 is properly mapped")
    else:
        print("\n❌ Qdrant Docker container issues detected")
        print("💡 Start Qdrant with: docker run -p 6333:6333 -d qdrant/qdrant")

if __name__ == "__main__":
    main()