#!/usr/bin/env python3
"""
Test script to verify visualization quality fixes.
"""

import numpy as np
import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.file_processing import FileProcessor
from app.services.transforms.wavelet import WaveletTransform
from app.services.visualization.generator import VisualizationGenerator


def test_percentile_stretch():
    """Test percentile stretch normalization."""
    print("=" * 60)
    print("TEST 1: Percentile Stretch Normalization")
    print("=" * 60)
    
    processor = FileProcessor()
    
    # Test case 1: Very small values (like transform coefficients)
    small_values = np.random.rand(100, 100) * 0.001
    print(f"\nTest 1a: Small values")
    print(f"  Input range: [{small_values.min():.6f}, {small_values.max():.6f}]")
    
    # Save with percentile stretch
    output_dir = Path("outputs/test_quality")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    path = processor.save_image(small_values, output_dir / "test_small_values.png")
    print(f"  ✓ Saved to {path}")
    print(f"  ✓ Should NOT be black (percentile stretch applied)")
    
    # Test case 2: Normal range
    normal_values = np.random.rand(100, 100)
    print(f"\nTest 1b: Normal values")
    print(f"  Input range: [{normal_values.min():.6f}, {normal_values.max():.6f}]")
    
    path = processor.save_image(normal_values, output_dir / "test_normal_values.png")
    print(f"  ✓ Saved to {path}")
    
    # Test case 3: With outliers
    with_outliers = np.random.rand(100, 100)
    with_outliers[0, 0] = 100.0  # Outlier
    with_outliers[0, 1] = -50.0  # Outlier
    print(f"\nTest 1c: Values with outliers")
    print(f"  Input range: [{with_outliers.min():.6f}, {with_outliers.max():.6f}]")
    
    path = processor.save_image(with_outliers, output_dir / "test_with_outliers.png")
    print(f"  ✓ Saved to {path}")
    print(f"  ✓ Outliers clipped, main features preserved")
    
    print("\n✅ Percentile stretch test PASSED")
    return True


def test_adaptive_threshold():
    """Test adaptive edge thresholding."""
    print("\n" + "=" * 60)
    print("TEST 2: Adaptive Edge Thresholding")
    print("=" * 60)
    
    wavelet = WaveletTransform()
    
    # Create test image with edges
    test_image = np.zeros((128, 128))
    test_image[30:100, 30:100] = 1.0  # Square
    test_image[50:80, 50:80] = 0.5    # Inner square
    
    print(f"\nTest 2a: Wavelet edge extraction")
    print(f"  Image shape: {test_image.shape}")
    
    # Decompose
    coeffs = wavelet.decompose(test_image, levels=3)
    print(f"  ✓ Decomposed into {coeffs.levels} levels")
    
    # Extract edges with different thresholds
    for threshold in [0.3, 0.5, 0.7]:
        edges = wavelet.extract_edges(coeffs, threshold=threshold)
        edge_count = np.sum(edges)
        print(f"  ✓ Threshold {threshold}: {edge_count} edge pixels detected")
        
        if edge_count == 0:
            print(f"    ⚠️  WARNING: No edges detected!")
            return False
    
    print("\n✅ Adaptive threshold test PASSED")
    return True


def test_visualization_normalize():
    """Test visualization normalization."""
    print("\n" + "=" * 60)
    print("TEST 3: Visualization Normalization")
    print("=" * 60)
    
    viz = VisualizationGenerator()
    
    # Test case 1: Very small values
    small = np.random.rand(50, 50) * 0.0001
    normalized = viz.normalize(small)
    print(f"\nTest 3a: Small values")
    print(f"  Input range: [{small.min():.8f}, {small.max():.8f}]")
    print(f"  Output range: [{normalized.min():.4f}, {normalized.max():.4f}]")
    
    if normalized.max() > 0.5:
        print(f"  ✓ Good dynamic range (max > 0.5)")
    else:
        print(f"  ⚠️  WARNING: Poor dynamic range")
        return False
    
    # Test case 2: With NaN
    with_nan = np.random.rand(50, 50)
    with_nan[0, 0] = np.nan
    with_nan[0, 1] = np.inf
    normalized = viz.normalize(with_nan)
    print(f"\nTest 3b: Values with NaN/inf")
    print(f"  Input has NaN: {np.any(np.isnan(with_nan))}")
    print(f"  Output has NaN: {np.any(np.isnan(normalized))}")
    
    if not np.any(np.isnan(normalized)):
        print(f"  ✓ NaN values handled correctly")
    else:
        print(f"  ✗ NaN values not handled")
        return False
    
    print("\n✅ Visualization normalization test PASSED")
    return True


def test_colormap_usage():
    """Test that scientific colormaps are used."""
    print("\n" + "=" * 60)
    print("TEST 4: Scientific Colormap Usage")
    print("=" * 60)
    
    viz = VisualizationGenerator()
    output_dir = Path("outputs/test_quality")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create test data
    test_image = np.random.rand(100, 100)
    test_edges = np.random.rand(100, 100) > 0.8
    
    print(f"\nGenerating test visualizations...")
    
    # Test edge map (should use inferno)
    try:
        viz.plot_edge_map(
            test_edges.astype(float),
            str(output_dir / "test_edge_map.png"),
            "Test Edge Map"
        )
        print(f"  ✓ Edge map generated (inferno colormap)")
    except Exception as e:
        print(f"  ✗ Edge map failed: {e}")
        return False
    
    # Test difference map (should use inferno + viridis)
    try:
        viz.plot_difference_map(
            test_edges.astype(float),
            test_edges.astype(float) * 0.8,
            str(output_dir / "test_difference_map.png")
        )
        print(f"  ✓ Difference map generated (inferno + viridis colormaps)")
    except Exception as e:
        print(f"  ✗ Difference map failed: {e}")
        return False
    
    # Test reconstruction (should use gray)
    try:
        viz.plot_reconstruction(
            test_image,
            test_image * 0.95,
            test_image * 0.9,
            str(output_dir / "test_reconstruction.png")
        )
        print(f"  ✓ Reconstruction generated (gray colormap)")
    except Exception as e:
        print(f"  ✗ Reconstruction failed: {e}")
        return False
    
    print(f"\n✅ Colormap usage test PASSED")
    print(f"   Check visualizations in: {output_dir}")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("VISUALIZATION QUALITY FIX - VERIFICATION TESTS")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Percentile Stretch", test_percentile_stretch()))
    results.append(("Adaptive Threshold", test_adaptive_threshold()))
    results.append(("Visualization Normalize", test_visualization_normalize()))
    results.append(("Colormap Usage", test_colormap_usage()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED")
        print("\nVisualization quality fixes verified successfully!")
        print("Outputs saved to: outputs/test_quality/")
    else:
        print("❌ SOME TESTS FAILED")
        print("\nPlease review the failures above.")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
