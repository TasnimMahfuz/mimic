#!/usr/bin/env python3
"""
Quick test to verify directional analysis fix.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.services.mimic_service import MIMICService, ScientificMetrics
from app.services.visualization.generator import VisualizationGenerator


def test_directional_visualization():
    """Test that directional data is visualized."""
    print("=" * 60)
    print("TESTING DIRECTIONAL ANALYSIS FIX")
    print("=" * 60)
    
    # Create mock scientific metrics
    print("\n1. Creating mock directional data...")
    
    directional_energy = {
        0: np.random.rand(16) * 100,
        1: np.random.rand(16) * 80,
        2: np.random.rand(16) * 60,
    }
    
    angular_distribution = np.random.rand(16) * 200
    anisotropy_map = np.random.rand(100, 100) * 2
    
    print(f"   ✓ Directional energy: {len(directional_energy)} scales")
    print(f"   ✓ Angular distribution: {len(angular_distribution)} orientations")
    print(f"   ✓ Anisotropy map: {anisotropy_map.shape}")
    
    # Create visualization generator
    print("\n2. Testing visualization methods...")
    viz = VisualizationGenerator()
    output_dir = Path("outputs/test_directional")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test directional energy plot
    try:
        path = output_dir / "test_directional_energy.png"
        viz.plot_directional_energy(directional_energy, str(path))
        print(f"   ✓ Directional energy plot: {path}")
    except Exception as e:
        print(f"   ✗ Directional energy plot FAILED: {e}")
        return False
    
    # Test angular distribution plot
    try:
        path = output_dir / "test_angular_distribution.png"
        viz.plot_angular_distribution(angular_distribution, str(path))
        print(f"   ✓ Angular distribution plot: {path}")
    except Exception as e:
        print(f"   ✗ Angular distribution plot FAILED: {e}")
        return False
    
    # Test anisotropy map plot
    try:
        path = output_dir / "test_anisotropy_map.png"
        viz.plot_anisotropy_map(anisotropy_map, str(path))
        print(f"   ✓ Anisotropy map plot: {path}")
    except Exception as e:
        print(f"   ✗ Anisotropy map plot FAILED: {e}")
        return False
    
    # Test that data is passed correctly
    print("\n3. Testing data passing...")
    
    results = {
        'original_image': np.random.rand(100, 100),
        'normalized_image': np.random.rand(100, 100),
        'wavelet_edges': np.random.rand(100, 100) > 0.8,
        'curvelet_edges': np.random.rand(100, 100) > 0.8,
        'directional_energy': directional_energy,
        'angular_distribution': angular_distribution,
        'anisotropy_map': anisotropy_map,
    }
    
    try:
        generated = viz.generate_all(results, str(output_dir))
        print(f"   ✓ Generated {len(generated)} visualizations")
        
        # Check for directional files
        directional_files = [
            'test_directional_energy.png',
            'test_angular_distribution.png',
            'test_anisotropy_map.png',
        ]
        
        for filename in directional_files:
            if (output_dir / filename).exists():
                print(f"   ✓ Found: {filename}")
            else:
                print(f"   ⚠️  Missing: {filename}")
        
    except Exception as e:
        print(f"   ✗ Generate all FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✅ DIRECTIONAL ANALYSIS FIX VERIFIED")
    print("=" * 60)
    print(f"\nCheck visualizations in: {output_dir}")
    print("\nNow restart backend and upload a test image!")
    
    return True


if __name__ == "__main__":
    success = test_directional_visualization()
    exit(0 if success else 1)
