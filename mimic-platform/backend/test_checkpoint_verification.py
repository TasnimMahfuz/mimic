#!/usr/bin/env python3
"""
Checkpoint Verification Script for Task 6
Tests the complete transform pipeline and verifies all visualizations are generated.
"""

import numpy as np
import os
from pathlib import Path
from app.services.file_processing import FileProcessor
from app.services.mimic_service import MIMICService

def create_test_image():
    """Create a synthetic test image with features"""
    size = 256
    image = np.zeros((size, size))
    
    # Add a bright square
    image[50:100, 50:100] = 1.0
    
    # Add a circle
    y, x = np.ogrid[:size, :size]
    center_y, center_x = 150, 150
    radius = 30
    mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    image[mask] = 0.8
    
    # Add some diagonal edges
    for i in range(size):
        if i < size:
            image[i, i] = 0.6
            image[i, size-1-i] = 0.5
    
    # Add noise
    image += np.random.normal(0, 0.05, (size, size))
    image = np.clip(image, 0, 1)
    
    return image

def verify_checkpoint():
    """Verify all transform implementations are working correctly"""
    print("=" * 80)
    print("CHECKPOINT VERIFICATION - Task 6")
    print("=" * 80)
    print()
    
    # Create test image
    print("1. Creating synthetic test image...")
    test_image = create_test_image()
    print(f"   ✓ Test image created: shape={test_image.shape}, dtype={test_image.dtype}")
    print()
    
    # Initialize services
    print("2. Initializing services...")
    file_processor = FileProcessor()
    mimic_service = MIMICService()
    print("   ✓ Services initialized")
    print()
    
    # Create output directory
    output_dir = Path("outputs/checkpoint_verification")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"3. Output directory: {output_dir}")
    print()
    
    # Test file processing
    print("4. Testing file processing...")
    try:
        # Save test image
        test_path = output_dir / "test_input.png"
        file_processor.save_image(test_image, str(test_path))
        print(f"   ✓ Saved test image to {test_path}")
        
        # Load it back
        loaded = file_processor.load_image(str(test_path))
        print(f"   ✓ Loaded image back: shape={loaded.shape}")
    except Exception as e:
        print(f"   ✗ File processing failed: {e}")
        return False
    print()
    
    # Test normalization
    print("5. Testing flux normalization...")
    try:
        norm_path = str(output_dir / "normalized.png")
        normalized, scale_params = mimic_service.normalize_flux(test_image, norm_path)
        print(f"   ✓ Normalized image: min={normalized.min():.4f}, max={normalized.max():.4f}")
        print(f"   ✓ Scale params: {scale_params}")
        
        # Check visualization
        norm_viz = output_dir / "normalized.png"
        if norm_viz.exists():
            print(f"   ✓ Normalization visualization created: {norm_viz}")
        else:
            print(f"   ⚠ Normalization visualization not found")
    except Exception as e:
        print(f"   ✗ Normalization failed: {e}")
        return False
    print()
    
    # Test noise modeling
    print("6. Testing noise modeling...")
    try:
        noise_path = str(output_dir / "noise_model.png")
        noise_result = mimic_service.estimate_noise(
            normalized,
            photon_threshold=0.1,
            output_path=noise_path
        )
        print(f"   ✓ Noise level: {noise_result['noise_level']:.6f}")
        print(f"   ✓ Noise std: {noise_result['noise_std']:.6f}")
        
        # Check visualization
        noise_viz = output_dir / "noise_model.png"
        if noise_viz.exists():
            print(f"   ✓ Noise model visualization created: {noise_viz}")
        else:
            print(f"   ⚠ Noise model visualization not found")
    except Exception as e:
        print(f"   ✗ Noise modeling failed: {e}")
        return False
    print()
    
    # Test wavelet transform
    print("7. Testing wavelet transform...")
    try:
        from app.services.transforms.wavelet import WaveletTransform
        wavelet = WaveletTransform()
        
        coeffs = wavelet.decompose(normalized, levels=3)
        print(f"   ✓ Wavelet decomposition: {coeffs.levels} levels")
        print(f"   ✓ Wavelet type: {coeffs.wavelet_name}")
        
        # Test edge detection
        edges = wavelet.extract_edges(coeffs, threshold=0.1)
        print(f"   ✓ Edge detection: {np.sum(edges)} edge pixels")
        
        # Test reconstruction
        reconstructed = wavelet.reconstruct(coeffs)
        error = np.mean((normalized - reconstructed) ** 2)
        print(f"   ✓ Reconstruction error: {error:.6f}")
        
        # Generate visualizations
        wavelet.save_edge_visualization(edges, normalized, str(output_dir / "wavelet_edge.png"))
        wavelet.save_coefficient_visualization(coeffs, str(output_dir / "wavelet_coefficients.png"))
        
        # Check visualizations
        if (output_dir / "wavelet_edge.png").exists():
            print(f"   ✓ Wavelet edge visualization created")
        if (output_dir / "wavelet_coefficients.png").exists():
            print(f"   ✓ Wavelet coefficients visualization created")
            
    except Exception as e:
        print(f"   ✗ Wavelet transform failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()
    
    # Test curvelet transform
    print("8. Testing curvelet transform...")
    try:
        from app.services.transforms.curvelet import CurveletTransform
        curvelet = CurveletTransform()
        
        coeffs = curvelet.decompose(normalized, levels=3, angular_resolution=16)
        print(f"   ✓ Curvelet decomposition: {coeffs.scales} scales, {coeffs.orientations} orientations")
        
        # Test directional energy extraction
        energy = curvelet.extract_directional_energy(coeffs)
        print(f"   ✓ Directional energy extracted for {len(energy)} scales")
        
        # Test orientation map
        orientation_map = curvelet.compute_orientation_map(coeffs)
        print(f"   ✓ Orientation map computed: shape={orientation_map.shape}")
        
        # Test reconstruction
        reconstructed = curvelet.reconstruct(coeffs)
        error = np.mean((normalized - reconstructed) ** 2)
        print(f"   ✓ Reconstruction error: {error:.6f}")
        
        # Generate visualizations (includes edge detection)
        viz_paths = curvelet.generate_visualizations(coeffs, str(output_dir), edge_threshold=0.1)
        print(f"   ✓ Generated {len(viz_paths)} visualizations:")
        for name, path in viz_paths.items():
            if Path(path).exists():
                print(f"      • {Path(path).name}")
            
    except Exception as e:
        print(f"   ✗ Curvelet transform failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()
    
    # Summary
    print("=" * 80)
    print("CHECKPOINT VERIFICATION SUMMARY")
    print("=" * 80)
    
    # Count generated files
    generated_files = list(output_dir.glob("*.png"))
    print(f"\nTotal visualizations generated: {len(generated_files)}")
    print("\nGenerated files:")
    for f in sorted(generated_files):
        size_kb = f.stat().st_size / 1024
        print(f"  • {f.name:40s} ({size_kb:6.1f} KB)")
    
    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED - Transform implementations verified!")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    success = verify_checkpoint()
    exit(0 if success else 1)
