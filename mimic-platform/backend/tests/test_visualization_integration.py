"""
Integration tests for comprehensive visualization generation.

Tests the complete visualization pipeline with realistic data structures
from the MIMIC analysis pipeline.

**Validates: Requirements 11.1-11.16, 13.11**
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from app.services.visualization.generator import VisualizationGenerator
from app.services.transforms.wavelet import WaveletTransform
from app.services.transforms.curvelet import CurveletTransform


@pytest.fixture
def test_image():
    """Create a test image with features."""
    size = 128
    image = np.zeros((size, size))
    
    # Add some features
    image[40:60, 40:60] = 1.0  # Square
    image[70:90, :] = 0.5  # Horizontal bar
    
    # Add noise
    image += np.random.randn(size, size) * 0.1
    image = np.clip(image, 0, 1)
    
    return image


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def complete_results(test_image):
    """Create complete processing results with real transforms."""
    # Apply wavelet transform
    wavelet_transform = WaveletTransform()
    wavelet_coeffs = wavelet_transform.decompose(test_image, levels=3)
    wavelet_recon = wavelet_transform.reconstruct(wavelet_coeffs)
    wavelet_edges_raw = wavelet_transform.extract_edges(wavelet_coeffs, threshold=0.1)
    
    # Ensure wavelet edges match original image shape
    if wavelet_edges_raw.shape != test_image.shape:
        import cv2
        wavelet_edges = cv2.resize(
            wavelet_edges_raw.astype(np.float32),
            (test_image.shape[1], test_image.shape[0]),
            interpolation=cv2.INTER_NEAREST
        ).astype(np.uint8)
    else:
        wavelet_edges = wavelet_edges_raw
    
    # Apply curvelet transform
    curvelet_transform = CurveletTransform()
    curvelet_coeffs = curvelet_transform.decompose(test_image, levels=3, angular_resolution=8)
    curvelet_recon = curvelet_transform.reconstruct(curvelet_coeffs)
    curvelet_edges = (np.random.rand(128, 128) > 0.5).astype(np.uint8)
    
    # Compute metrics
    radial_energy = np.random.rand(50)
    
    # Collect all coefficients for histogram
    all_coeffs = []
    for scale_idx in range(curvelet_coeffs.scales):
        if scale_idx in curvelet_coeffs.coefficients:
            for angle_idx in curvelet_coeffs.coefficients[scale_idx]:
                coeff = curvelet_coeffs.coefficients[scale_idx][angle_idx]
                all_coeffs.append(coeff.flatten())
    
    coefficient_histogram = np.concatenate(all_coeffs) if all_coeffs else np.array([])
    
    # Scale energy
    scale_energy = {}
    for scale_idx in range(curvelet_coeffs.scales):
        if scale_idx in curvelet_coeffs.coefficients:
            energy = 0.0
            for angle_idx in curvelet_coeffs.coefficients[scale_idx]:
                coeff = curvelet_coeffs.coefficients[scale_idx][angle_idx]
                energy += np.sum(np.abs(coeff) ** 2)
            scale_energy[scale_idx] = energy
    
    # Frequency cone (FFT of image)
    frequency_cone = np.fft.fftshift(np.fft.fft2(test_image))
    
    # Reconstruction error
    reconstruction_error = np.abs(test_image - curvelet_recon)
    
    # Reconstruction error curve
    reconstruction_error_curve = {}
    for scale_idx in range(3):
        reconstruction_error_curve[scale_idx] = np.random.rand() * 0.1
    
    return {
        'original_image': test_image,
        'normalized_image': test_image / test_image.max(),
        'wavelet_coeffs': wavelet_coeffs,
        'wavelet_edges': wavelet_edges,
        'curvelet_edges': curvelet_edges,
        'reconstructed_wavelet': wavelet_recon,
        'reconstructed_curvelet': curvelet_recon,
        'radial_energy': radial_energy,
        'coefficient_histogram': coefficient_histogram,
        'scale_energy': scale_energy,
        'frequency_cone': frequency_cone,
        'reconstruction_error': reconstruction_error,
        'reconstruction_error_curve': reconstruction_error_curve
    }


class TestVisualizationIntegration:
    """Integration tests for complete visualization pipeline."""
    
    def test_generate_all_visualizations(self, complete_results, temp_output_dir):
        """
        Test generation of all required visualizations.
        
        **Validates: Requirements 11.1-11.16**
        """
        generator = VisualizationGenerator()
        
        # Generate all visualizations
        generated_files = generator.generate_all(complete_results, temp_output_dir)
        
        # Verify files were generated
        assert len(generated_files) > 0
        
        # Check that all files exist and are non-empty
        for file_path in generated_files:
            path = Path(file_path)
            assert path.exists(), f"File not found: {file_path}"
            assert path.stat().st_size > 0, f"File is empty: {file_path}"
        
        # Verify expected visualizations are present
        output_dir = Path(temp_output_dir)
        expected_files = [
            'raw.png',
            'normalized.png',
            'wavelet_coefficients.png',
            'wavelet_edge.png',
            'curvelet_edge.png',
            'reconstruction.png',
            'radial_energy.png',
            'coefficient_histogram.png',
            'scale_energy.png',
            'frequency_cone.png',
            'reconstruction_error.png',
            'reconstruction_error_curve.png',
            'difference_map.png',
            'edge_overlay.png'
        ]
        
        for filename in expected_files:
            file_path = output_dir / filename
            assert file_path.exists(), f"Expected file not found: {filename}"
    
    def test_visualization_with_minimal_data(self, test_image, temp_output_dir):
        """Test visualization generation with minimal data."""
        generator = VisualizationGenerator()
        
        # Minimal results
        minimal_results = {
            'original_image': test_image,
            'normalized_image': test_image / test_image.max()
        }
        
        # Should not crash
        generated_files = generator.generate_all(minimal_results, temp_output_dir)
        
        # Should generate at least the basic images
        assert len(generated_files) >= 2
    
    def test_visualization_file_sizes(self, complete_results, temp_output_dir):
        """Test that generated visualizations have reasonable file sizes."""
        generator = VisualizationGenerator()
        
        generated_files = generator.generate_all(complete_results, temp_output_dir)
        
        for file_path in generated_files:
            path = Path(file_path)
            size = path.stat().st_size
            
            # Files should be between 1KB and 10MB
            assert 1000 < size < 10_000_000, f"Unexpected file size for {file_path}: {size} bytes"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
