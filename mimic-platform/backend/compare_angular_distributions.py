#!/usr/bin/env python3
"""
Compare angular distributions across multiple runs to verify they're different.
"""

import json
from pathlib import Path
import sys

def compare_runs():
    """Compare angular distribution metadata across runs."""
    outputs_dir = Path("outputs")
    
    if not outputs_dir.exists():
        print("❌ No outputs directory found")
        return
    
    runs = sorted(outputs_dir.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    
    if len(runs) < 2:
        print("⚠️  Need at least 2 runs to compare")
        return
    
    print("\n" + "=" * 80)
    print("  ANGULAR DISTRIBUTION COMPARISON")
    print("=" * 80 + "\n")
    
    print(f"📊 Analyzing {min(len(runs), 5)} most recent runs:\n")
    
    file_sizes = []
    
    for i, run_dir in enumerate(runs[:5], 1):
        angular_file = run_dir / "angular_distribution.png"
        metadata_file = run_dir / "metadata.json"
        
        if not angular_file.exists():
            continue
        
        size_kb = angular_file.stat().st_size / 1024
        file_sizes.append(size_kb)
        
        print(f"{i}. {run_dir.name}")
        print(f"   📁 Size: {size_kb:.1f} KB")
        
        if metadata_file.exists():
            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)
                    timestamp = metadata.get('timestamp', 'unknown')
                    print(f"   🕐 Time: {timestamp}")
            except:
                pass
        
        print()
    
    # Analyze size variation
    if len(file_sizes) >= 2:
        min_size = min(file_sizes)
        max_size = max(file_sizes)
        variation = ((max_size - min_size) / min_size) * 100
        
        print("=" * 80)
        print("📈 ANALYSIS:")
        print("=" * 80)
        print(f"File size range: {min_size:.1f} KB - {max_size:.1f} KB")
        print(f"Variation: {variation:.1f}%")
        print()
        
        if variation > 10:
            print("✅ GOOD: File sizes vary significantly (>10%)")
            print("   This suggests angular distributions are different!")
        elif variation > 5:
            print("⚠️  MODERATE: File sizes vary somewhat (5-10%)")
            print("   Distributions may be slightly different")
        else:
            print("❌ WARNING: File sizes very similar (<5%)")
            print("   Distributions might still be too similar")
        
        print()
        print("💡 To verify properly:")
        print("   1. Open the angular_distribution.png files visually")
        print("   2. Check if polar plots have different shapes")
        print("   3. Check if yellow markers are at different positions")
        print("   4. Check if bar chart peaks are at different angles")
        print("   5. Check if anisotropy values differ in statistics box")
        print()
    
    print("=" * 80 + "\n")

if __name__ == "__main__":
    compare_runs()
