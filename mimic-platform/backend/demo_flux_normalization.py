"""
Demonstration script for flux normalization functionality.

This script creates a sample astronomical image and demonstrates
the complete flux normalization pipeline with visualization.
"""
import numpy as np
from pathlib import Path
from app.services.mimic_service import MIMICService


def main():
    """Run flux normalization demonstration."""
    print("=" * 60)
    print("Flux Normalization Demonstration")
    print("=" * 60)
    
    # Initialize service
    service = MIMICService()
    
    # Create a synthetic astronomical image
    print("\n1. Creating synthetic astronomical image...")
    size = 256
    
    # Background with Poisson noise (simulating photon counts)
    background = np.random.poisson(10, (size, size)).astype(np.float64)
    
    # Add a bright point source (simulating a star or X-ray source)
    y, x = np.ogrid[:size, :size]
    center = size // 2
    r = np.sqrt((x - center)**2 + (y - center)**2)
    point_source = 2000 * np.exp(-r**2 / (2 * 15**2))
    
    # Add a diffuse extended source (simulating a galaxy cluster)
    extended_x, extended_y = size // 4, 3 * size // 4
    r_extended = np.sqrt((x - extended_x)**2 + (y - extended_y)**2)
    extended_source = 500 * np.exp(-r_extended**2 / (2 * 30**2))
    
    # Combine all components
    image = background + point_source + extended_source
    
    print(f"   Image shape: {image.shape}")
    print(f"   Original intensity range: [{image.min():.2f}, {image.max():.2f}]")
    print(f"   Mean intensity: {image.mean():.2f}")
    print(f"   Standard deviation: {image.std():.2f}")
    
    # Create output directory
    output_dir = Path("outputs/demo_normalization")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "normalized.png"
    
    # Perform flux normalization
    print("\n2. Performing flux normalization...")
    normalized, scale_params = service.normalize_flux(image, str(output_path))
    
    print(f"   Normalized range: [{normalized.min():.10f}, {normalized.max():.10f}]")
    print(f"   Scale parameters:")
    print(f"      - min: {scale_params['min']:.2f}")
    print(f"      - max: {scale_params['max']:.2f}")
    print(f"      - range: {scale_params['range']:.2f}")
    
    # Verify normalization properties
    print("\n3. Verifying normalization properties...")
    assert normalized.min() == 0.0, "Minimum should be exactly 0.0"
    assert normalized.max() == 1.0, "Maximum should be exactly 1.0"
    assert np.all(normalized >= 0.0), "All values should be >= 0.0"
    assert np.all(normalized <= 1.0), "All values should be <= 1.0"
    print("   ✓ All normalization properties verified")
    
    # Test denormalization
    print("\n4. Testing denormalization round-trip...")
    denormalized = service.denormalize_flux(normalized, scale_params)
    max_error = np.max(np.abs(denormalized - image))
    relative_error = max_error / scale_params['range']
    print(f"   Maximum absolute error: {max_error:.2e}")
    print(f"   Relative error: {relative_error:.2e}")
    assert np.allclose(denormalized, image, rtol=1e-10), "Round-trip should be accurate"
    print("   ✓ Denormalization round-trip successful")
    
    # Verify visualization files
    print("\n5. Verifying visualization files...")
    assert output_path.exists(), "normalized.png should exist"
    print(f"   ✓ Created: {output_path}")
    print(f"      Size: {output_path.stat().st_size / 1024:.1f} KB")
    
    raw_path = output_dir / "normalized_raw.png"
    assert raw_path.exists(), "normalized_raw.png should exist"
    print(f"   ✓ Created: {raw_path}")
    print(f"      Size: {raw_path.stat().st_size / 1024:.1f} KB")
    
    # Analyze feature preservation
    print("\n6. Analyzing feature preservation...")
    
    # Check point source
    point_source_value = normalized[center, center]
    print(f"   Point source (center) normalized intensity: {point_source_value:.4f}")
    assert point_source_value > 0.9, "Point source should be very bright"
    
    # Check extended source
    extended_value = normalized[extended_y, extended_x]
    print(f"   Extended source normalized intensity: {extended_value:.4f}")
    assert 0.1 < extended_value < 0.5, "Extended source should be moderate"
    
    # Check background
    background_value = normalized[10, 10]
    print(f"   Background normalized intensity: {background_value:.4f}")
    assert background_value < 0.1, "Background should be dark"
    
    print("   ✓ Spatial features preserved correctly")
    
    print("\n" + "=" * 60)
    print("Demonstration completed successfully!")
    print("=" * 60)
    print(f"\nVisualization files saved to: {output_dir}")
    print("You can view the normalized images to verify the output.")


if __name__ == "__main__":
    main()
