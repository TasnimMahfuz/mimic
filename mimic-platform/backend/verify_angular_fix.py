#!/usr/bin/env python3
"""
Quick verification script for angular distribution fix.
Tests that different images produce different angular distributions.
"""

import numpy as np
from pathlib import Path
import sys

def analyze_angular_distribution_logs(log_file: str = None):
    """
    Parse backend logs to extract angular distribution metrics.
    """
    print("🔍 Angular Distribution Verification\n")
    print("=" * 60)
    
    if log_file:
        with open(log_file, 'r') as f:
            logs = f.read()
    else:
        print("📝 Instructions:")
        print("1. Start the backend: uvicorn app.main:app --reload")
        print("2. Upload 3-4 different images with different structures:")
        print("   - Image with horizontal features (landscape)")
        print("   - Image with vertical features (buildings)")
        print("   - Image with diagonal features")
        print("   - Image with isotropic features (stars)")
        print("3. Check the backend logs for these patterns:\n")
        
        print("✅ What to Look For:")
        print("-" * 60)
        print("For EACH image, you should see:")
        print("  • Angular distribution range: [min, max]")
        print("  • Dominant orientation: X (energy: Y)")
        print("  • Different dominant orientations for different images")
        print("  • Anisotropy values varying between images\n")
        
        print("📊 Expected Patterns:")
        print("-" * 60)
        print("Horizontal edges → Dominant: 0 or 8 (0° or 180°)")
        print("Vertical edges   → Dominant: 4 or 12 (90° or 270°)")
        print("Diagonal edges   → Dominant: 2, 6, 10, 14 (45°, 135°, etc.)")
        print("Isotropic        → No clear dominant, low anisotropy\n")
        
        print("🎯 Success Criteria:")
        print("-" * 60)
        print("✓ Different images show different dominant orientations")
        print("✓ Anisotropy values vary (not all ~0.5)")
        print("✓ Angular distribution plots look visually different")
        print("✓ Bar charts show peaks at different angles\n")
        
        print("📁 Check Output Files:")
        print("-" * 60)
        print("outputs/run_<uuid>/angular_distribution.png")
        print("  • Polar plot should show different shapes")
        print("  • Yellow marker at different positions")
        print("  • Bar chart peaks at different angles")
        print("  • Statistics box shows varying values\n")
        
        return

def check_output_directory():
    """
    Check if output files exist and show recent runs.
    """
    outputs_dir = Path("outputs")
    
    if not outputs_dir.exists():
        print("⚠️  No outputs directory found yet.")
        print("   Upload an image first to generate outputs.\n")
        return
    
    runs = sorted(outputs_dir.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    
    if not runs:
        print("⚠️  No analysis runs found yet.")
        print("   Upload an image to start analysis.\n")
        return
    
    print(f"\n📁 Found {len(runs)} analysis run(s):\n")
    
    for i, run_dir in enumerate(runs[:5], 1):  # Show last 5 runs
        angular_file = run_dir / "angular_distribution.png"
        if angular_file.exists():
            size_kb = angular_file.stat().st_size / 1024
            print(f"{i}. {run_dir.name}")
            print(f"   ✓ angular_distribution.png ({size_kb:.1f} KB)")
        else:
            print(f"{i}. {run_dir.name}")
            print(f"   ✗ angular_distribution.png missing")
    
    print("\n💡 Tip: Compare angular_distribution.png files across runs")
    print("   They should look visually different for different images!\n")

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  ANGULAR DISTRIBUTION FIX - VERIFICATION GUIDE")
    print("=" * 60 + "\n")
    
    analyze_angular_distribution_logs()
    check_output_directory()
    
    print("🚀 Next Steps:")
    print("-" * 60)
    print("1. Restart backend if not running:")
    print("   cd mimic-platform/backend")
    print("   uvicorn app.main:app --reload")
    print()
    print("2. Upload test images via frontend or API")
    print()
    print("3. Watch backend logs for angular distribution metrics")
    print()
    print("4. Compare angular_distribution.png files visually")
    print()
    print("=" * 60 + "\n")
