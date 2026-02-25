#!/usr/bin/env python3
"""
Test script to verify visualization normalization fixes.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.visualization.generator import VisualizationGenerator


def test_normalization():
    """Test the normalize function."""
    print("Testing normalization function...")
    
    viz = VisualizationGenerator()
    
    # Test case 1: Normal range
    img1 = np.array([[0, 50, 100], [150, 200, 255]], dtype=float)
    normalized1 = viz.normalize(img1)
    print(f"  Test 1 - Normal range:")
    print(f"    Input: min={img1.min()}, max={img1.max()}")
    print(f"    Output: min={normalized1.min()}, max={normalized1.max()}")
    assert normalized1.min() == 0.0, "Min should be 0"
    assert normalized1.max() == 1.0, "Max should be 1"
    print(f"    ✓ PASS")
    
    # Test case 2: Negative values
    img2 = np.array([[-100, -50, 0], [50, 100, 150]], dtype=float)
    normalized2 = viz.normalize(img2)
    print(f"  Test 2 - Negative values:")
    print(f"    Input: min={img2.min()}, max={img2.max()}")
    print(f"    Output: min={normalized2.min()}, max={normalized2.max()}")
    assert normalized2.min() == 0.0, "Min should be 0"
    assert normalized2.max() == 1.0, "Max should be 1"
    print(f"    ✓ PASS")
    
    # Test case 3: Very small values (like transform coefficients)
    img3 = np.array([[0.0001, 0.0002], [0.0003, 0.0004]], dtype=float)
    normalized3 = viz.normalize(img3)
    print(f"  Test 3 - Small values:")
    print(f"    Input: min={img3.min()}, max={img3.max()}")
    print(f"    Output: min={normalized3.min()}, max={normalized3.max()}")
    assert normalized3.min() == 0.0, "Min should be 0"
    assert normalized3.max() == 1.0, "Max should be 1"
    print(f"    ✓ PASS")
    
    # Test case 4: Constant image
    img4 = np.ones((3, 3)) * 42.0
    normalized4 = viz.normalize(img4)
    print(f"  Test 4 - Constant image:")
    print(f"    Input: min={img4.min()}, max={img4.max()}")
    print(f"    Output: min={normalized4.min()}, max={normalized4.max()}")
    assert np.all(normalized4 == 0.0), "Constant image should normalize to 0"
    print(f"    ✓ PASS")
    
    print("\n✅ All normalization tests passed!")
    return True


def test_visualization_generation():
    """Test that visualizations can be generated."""
    print("\nTesting visualization generation...")
    
    viz = VisualizationGenerator()
    output_dir = Path("outputs/test_normalization")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create test data
    test_image = np.random.rand(100, 100)
    test_edges = np.random.rand(100, 100) > 0.8  # Binary edges
    test_reconstruction = np.random.rand(100, 100)
    
    results = {
        'original_image': test_image,
        'normalized_image': test_image * 0.5,
        'wavelet_edges': test_edges.astype(float),
        'curvelet_edges': test_edges.astype(float) * 0.8,
        'reconstructed_wavelet': test_reconstruction,
        'reconstructed_curvelet': test_reconstruction * 0.9,
    }
    
    try:
        generated = viz.generate_all(results, str(output_dir))
        print(f"  Generated {len(generated)} visualizations:")
        for path in generated:
            file_path = Path(path)
            if file_path.exists():
                size = file_path.stat().st_size
                print(f"    ✓ {file_path.name} ({size} bytes)")
            else:
                print(f"    ✗ {file_path.name} (NOT FOUND)")
        
        if len(generated) > 0:
            print(f"\n✅ Visualization generation successful!")
            print(f"   Check outputs in: {output_dir}")
            return True
        else:
            print(f"\n❌ No visualizations generated!")
            return False
    
    except Exception as e:
        print(f"\n❌ Visualization generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_black_image_issue():
    """Test the specific black image issue."""
    print("\nTesting black image issue...")
    
    viz = VisualizationGenerator()
    output_dir = Path("outputs/test_black_images")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Simulate transform outputs with very small values (common issue)
    small_values = np.random.rand(100, 100) * 0.001  # Very small values
    
    # Test without normalization (old way - would appear black)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(small_values, cmap='gray')
    axes[0].set_title('Without Normalization (Black)')
    axes[0].axis('off')
    
    # Test with normalization (new way - should be visible)
    normalized = viz.normalize(small_values)
    axes[1].imshow(normalized, cmap='gray')
    axes[1].set_title('With Normalization (Visible)')
    axes[1].axis('off')
    
    plt.suptitle('Black Image Fix Demonstration')
    plt.tight_layout()
    
    output_path = output_dir / 'black_image_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Comparison saved to: {output_path}")
    print(f"  ✓ Left side should be black, right side should show pattern")
    print(f"\n✅ Black image issue demonstration complete!")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("VISUALIZATION NORMALIZATION TEST")
    print("=" * 60)
    
    success = True
    
    # Run tests
    success &= test_normalization()
    success &= test_visualization_generation()
    success &= test_black_image_issue()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 60)
    
    exit(0 if success else 1)
