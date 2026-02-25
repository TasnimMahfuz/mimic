#!/usr/bin/env python3
"""
Test script to verify static file serving for visualization images.
"""

import requests
from pathlib import Path

def test_static_file_serving():
    """Test that static files are accessible via HTTP."""
    
    base_url = "http://localhost:8000"
    
    # Find an existing run directory with images
    outputs_dir = Path("outputs")
    run_dirs = list(outputs_dir.glob("run_*"))
    
    if not run_dirs:
        print("❌ No run directories found in outputs/")
        return False
    
    # Get the first run directory with PNG files
    test_run = None
    test_image = None
    
    for run_dir in run_dirs:
        png_files = list(run_dir.glob("*.png"))
        if png_files:
            test_run = run_dir.name
            test_image = png_files[0].name
            break
    
    if not test_run or not test_image:
        print("❌ No PNG files found in run directories")
        return False
    
    # Test the URL
    test_url = f"{base_url}/outputs/{test_run}/{test_image}"
    
    print(f"Testing static file serving...")
    print(f"  URL: {test_url}")
    print(f"  Local path: outputs/{test_run}/{test_image}")
    
    try:
        response = requests.get(test_url, timeout=5)
        
        if response.status_code == 200:
            print(f"✅ SUCCESS: Image accessible (status: {response.status_code})")
            print(f"   Content-Type: {response.headers.get('content-type')}")
            print(f"   Content-Length: {len(response.content)} bytes")
            return True
        else:
            print(f"❌ FAILED: Status code {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False
    
    except requests.exceptions.ConnectionError:
        print("❌ FAILED: Cannot connect to backend (is it running?)")
        print("   Start backend with: uvicorn app.main:app --reload")
        return False
    
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


if __name__ == "__main__":
    success = test_static_file_serving()
    exit(0 if success else 1)
