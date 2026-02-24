"""
Demo script to test wavelet visualization generation.

This script demonstrates the wavelet transform visualization capabilities
by processing a sample image and generating the required output files.
"""

import numpy as np
from pathlib import Path
from app.services.transforms.wavelet import WaveletTransform

def create_test_image():
    """Create a test image with clear features."""
    # Create a 256x256 image with geometric shapes
    image = np.zeros((256, 256))
    
    # Add a square
    image[50:150, 50:150] = 0.8
    
    # Add a circle
    y, x = np.ogrid[:256, :256]
    center_y, center_x = 180, 180
    radius = 40
    mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    image[mask] = 0.6
    
    # Add some noise
    image += np.random.normal(0, 0.05, image.shape)
    
    # Clip to [0, 1]
    image = np.clip(image, 0, 1)
    
    return image

def main():
    """Run the wavelet visualization demo."""
    print("Wavelet Visualization Demo")
    print("=" * 50)
    
    # Create output directory
    output_dir = Path("outputs/demo_wavelet_viz")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Create test image
    print("\n1. Creating test image...")
    image = create_test_image()
    print(f"   Image shape: {image.shape}")
    print(f"   Image range: [{image.min():.3f}, {image.max():.3f}]")
    
    # Initialize wavelet transform
    print("\n2. Initializing wavelet transform...")
    wavelet = WaveletTransform(wavelet='db4')
    print(f"   Wavelet: {wavelet.wavelet}")
    
    # Decompose image
    print("\n3. Decomposing image...")
    coeffs = wavelet.decompose(image, levels=3)
    print(f"   Levels: {coeffs.levels}")
    print(f"   Approximation shape: {coeffs.approximation.shape}")
    print(f"   Detail levels: {len(coeffs.details)}")
    
    # Extract edges
    print("\n4. Extracting edges...")
    edges = wavelet.extract_edges(coeffs, threshold=0.5)
    print(f"   Edge map shape: {edges.shape}")
    print(f"   Edge pixels detected: {np.sum(edges)}")
    
    # Generate visualizations
    print("\n5. Generating visualizations...")
    
    # Save edge visualization
    edge_path = output_dir / "wavelet_edge.png"
    wavelet.save_edge_visualization(edges, image, str(edge_path))
    print(f"   ✓ Saved: {edge_path}")
    
    # Save coefficient visualization
    coeff_path = output_dir / "wavelet_coefficients.png"
    wavelet.save_coefficient_visualization(coeffs, str(coeff_path))
    print(f"   ✓ Saved: {coeff_path}")
    
    # Compute additional metrics
    print("\n6. Computing metrics...")
    magnitudes = wavelet.get_coefficient_magnitudes(coeffs)
    energy = wavelet.compute_energy_per_scale(coeffs)
    
    print(f"   Coefficient magnitudes per level:")
    for i, mag in enumerate(magnitudes, 1):
        print(f"     Level {i}: shape={mag.shape}, mean={mag.mean():.4f}")
    
    print(f"   Energy per scale:")
    for level, e in energy.items():
        print(f"     Level {level}: {e:.2f}")
    
    print("\n" + "=" * 50)
    print("Demo complete! Check the output directory for visualizations.")
    print(f"Output directory: {output_dir.absolute()}")

if __name__ == "__main__":
    main()
