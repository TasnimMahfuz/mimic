#!/usr/bin/env python3
"""
Test script to check if generated images have actual visible content.
"""

from PIL import Image
import numpy as np
from pathlib import Path

def analyze_image(path):
    """Analyze an image and return statistics."""
    try:
        img = np.array(Image.open(path))
        
        # Handle different image types
        if len(img.shape) == 3:
            # Color image - convert to grayscale for analysis
            img_gray = np.mean(img[:,:,:3], axis=2)
        else:
            img_gray = img
        
        stats = {
            'shape': img.shape,
            'min': float(img_gray.min()),
            'max': float(img_gray.max()),
            'mean': float(img_gray.mean()),
            'median': float(np.median(img_gray)),
            'std': float(np.std(img_gray)),
        }
        
        # Classify brightness
        if stats['mean'] < 30:
            stats['classification'] = '❌ VERY DARK (likely appears black)'
        elif stats['mean'] < 100:
            stats['classification'] = '⚠️  DARK (may be hard to see)'
        elif stats['mean'] > 200:
            stats['classification'] = '✅ BRIGHT (mostly white background - good for plots)'
        else:
            stats['classification'] = '✅ NORMAL (good visibility)'
        
        return stats
    except Exception as e:
        return {'error': str(e)}

def main():
    # Find latest run
    outputs_dir = Path('outputs')
    runs = sorted(outputs_dir.glob('run_*'), key=lambda p: p.stat().st_mtime, reverse=True)
    
    if not runs:
        print("❌ No analysis runs found")
        return
    
    latest_run = runs[0]
    print(f"\n{'='*80}")
    print(f"  IMAGE BRIGHTNESS ANALYSIS")
    print(f"{'='*80}\n")
    print(f"Latest run: {latest_run.name}\n")
    
    # Files to check
    files_to_check = {
        'Transform Comparison': [
            'raw.png',
            'normalized.png',
            'wavelet_edge.png',
            'curvelet_edge.png',
        ],
        'Spectral Analysis': [
            'radial_energy.png',
            'scale_energy.png',
            'coefficient_histogram.png',
            'wavelet_coefficients.png',
        ],
        'Directional Analysis': [
            'orientation_map.png',
            'directional_energy.png',
            'angular_distribution.png',
            'frequency_cone.png',
        ],
    }
    
    for category, files in files_to_check.items():
        print(f"\n{category}:")
        print("-" * 80)
        
        for filename in files:
            filepath = latest_run / filename
            if not filepath.exists():
                print(f"  ❌ {filename:35s} - FILE MISSING")
                continue
            
            stats = analyze_image(filepath)
            if 'error' in stats:
                print(f"  ❌ {filename:35s} - ERROR: {stats['error']}")
                continue
            
            size_kb = filepath.stat().st_size / 1024
            print(f"  {filename:35s} ({size_kb:6.1f} KB)")
            print(f"     Mean brightness: {stats['mean']:6.1f}/255  {stats['classification']}")
    
    print(f"\n{'='*80}")
    print("\n💡 INTERPRETATION:")
    print("-" * 80)
    print("• Plots (spectral/directional analysis): Should have mean > 200 (white background)")
    print("• Images (raw/normalized): Can be dark for astronomy (lots of black space)")
    print("• Edge maps: Should have some white pixels (edges)")
    print("\n⚠️  If plots show as 'VERY DARK', there's a problem with plot generation")
    print("⚠️  If images show as 'VERY DARK', that's normal for astronomy images")
    print("=" * 80 + "\n")

if __name__ == '__main__':
    main()
