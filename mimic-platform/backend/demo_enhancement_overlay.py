"""
Demo script to verify enhancement overlay generation.

This script demonstrates the enhancement processing functionality
and generates enhancement_overlay.png visualization.

**Validates: Requirements 10.1, 10.2, 10.3, 10.4, 11.7, 13.10**
"""

import numpy as np
from pathlib import Path
import sys

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.mimic_service import MIMICService
from app.services.visualization.curvelet_visualizer import CurveletVisualizer


def create_test_image():
    """Create a test image with some structure."""
    print("Creating test image...")
    image = np.random.rand(256, 256) * 0.3
    
    # Add some edges and structures
    image[80:120, :] = 0.8  # Horizontal edge
    image[:, 80:120] = 0.2  # Vertical edge
    image[150:180, 150:180] = 0.9  # Bright square
    
    # Add some noise
    noise = np.random.randn(256, 256) * 0.05
    image = np.clip(image + noise, 0.0, 1.0)
    
    print(f"  ✓ Test image created: shape={image.shape}, range=[{image.min():.3f}, {image.max():.3f}]")
    return image


def main():
    """Run enhancement overlay demo."""
    print("=" * 60)
    print("Enhancement Overlay Generation Demo")
    print("=" * 60)
    print()
    
    # Create output directory
    output_dir = Path("outputs/demo_enhancement")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print()
    
    # Initialize services
    print("Initializing services...")
    mimic_service = MIMICService()
    visualizer = CurveletVisualizer()
    print("  ✓ Services initialized")
    print()
    
    # Create test image
    test_image = create_test_image()
    print()
    
    # Test 1: Apply enhancement with default parameters
    print("Test 1: Enhancement with default parameters")
    print("-" * 60)
    enhanced_default = mimic_service.apply_enhancement(
        test_image,
        enhancement_factor=2.0,
        smoothing=1.0
    )
    print(f"  ✓ Enhanced image: range=[{enhanced_default.min():.3f}, {enhanced_default.max():.3f}]")
    
    # Generate overlay
    output_path = output_dir / "enhancement_overlay_default.png"
    difference = visualizer.generate_enhancement_overlay(
        test_image,
        enhanced_default,
        str(output_path)
    )
    print(f"  ✓ Overlay saved: {output_path}")
    print(f"  ✓ Difference range: [{difference.min():.3f}, {difference.max():.3f}]")
    print()
    
    # Test 2: Enhancement with high contrast
    print("Test 2: Enhancement with high contrast (factor=3.5)")
    print("-" * 60)
    enhanced_high = mimic_service.apply_enhancement(
        test_image,
        enhancement_factor=3.5,
        smoothing=0.5
    )
    print(f"  ✓ Enhanced image: range=[{enhanced_high.min():.3f}, {enhanced_high.max():.3f}]")
    
    output_path = output_dir / "enhancement_overlay_high_contrast.png"
    difference = visualizer.generate_enhancement_overlay(
        test_image,
        enhanced_high,
        str(output_path)
    )
    print(f"  ✓ Overlay saved: {output_path}")
    print(f"  ✓ Difference range: [{difference.min():.3f}, {difference.max():.3f}]")
    print()
    
    # Test 3: Enhancement with heavy smoothing
    print("Test 3: Enhancement with heavy smoothing (smoothing=3.0)")
    print("-" * 60)
    enhanced_smooth = mimic_service.apply_enhancement(
        test_image,
        enhancement_factor=1.5,
        smoothing=3.0
    )
    print(f"  ✓ Enhanced image: range=[{enhanced_smooth.min():.3f}, {enhanced_smooth.max():.3f}]")
    
    output_path = output_dir / "enhancement_overlay_smooth.png"
    difference = visualizer.generate_enhancement_overlay(
        test_image,
        enhanced_smooth,
        str(output_path)
    )
    print(f"  ✓ Overlay saved: {output_path}")
    print(f"  ✓ Difference range: [{difference.min():.3f}, {difference.max():.3f}]")
    print()
    
    # Test 4: No enhancement (identity case)
    print("Test 4: No enhancement (factor=1.0, smoothing=0.0)")
    print("-" * 60)
    enhanced_none = mimic_service.apply_enhancement(
        test_image,
        enhancement_factor=1.0,
        smoothing=0.0
    )
    print(f"  ✓ Enhanced image: range=[{enhanced_none.min():.3f}, {enhanced_none.max():.3f}]")
    
    output_path = output_dir / "enhancement_overlay_none.png"
    difference = visualizer.generate_enhancement_overlay(
        test_image,
        enhanced_none,
        str(output_path)
    )
    print(f"  ✓ Overlay saved: {output_path}")
    print(f"  ✓ Difference range: [{difference.min():.3f}, {difference.max():.3f}]")
    print(f"  ✓ Difference should be near zero: {np.allclose(difference, 0.0, atol=1e-6)}")
    print()
    
    # Verify all files were created
    print("Verification")
    print("-" * 60)
    expected_files = [
        "enhancement_overlay_default.png",
        "enhancement_overlay_high_contrast.png",
        "enhancement_overlay_smooth.png",
        "enhancement_overlay_none.png"
    ]
    
    all_exist = True
    for filename in expected_files:
        filepath = output_dir / filename
        exists = filepath.exists()
        size = filepath.stat().st_size if exists else 0
        status = "✓" if exists and size > 0 else "✗"
        print(f"  {status} {filename}: {size:,} bytes")
        all_exist = all_exist and exists and size > 0
    
    print()
    print("=" * 60)
    if all_exist:
        print("✓ Demo completed successfully!")
        print(f"✓ All enhancement overlay visualizations generated in {output_dir}")
    else:
        print("✗ Demo completed with errors")
        print("✗ Some visualizations were not generated")
    print("=" * 60)
    
    return 0 if all_exist else 1


if __name__ == "__main__":
    sys.exit(main())
