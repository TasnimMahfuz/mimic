"""
Unit tests for wavelet transform module.

Tests the WaveletTransform class for decomposition, reconstruction,
and edge detection functionality.

**Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 11.3, 13.5**
"""

import pytest
import numpy as np
from pathlib import Path
from app.services.transforms.wavelet import WaveletTransform, WaveletCoefficients


class TestWaveletTransform:
    """Test suite for WaveletTransform class."""
    
    @pytest.fixture
    def wavelet_transform(self):
        """Create a WaveletTransform instance for testing."""
        return WaveletTransform(wavelet='db4')
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample test image."""
        # Create a 128x128 image with some structure
        x = np.linspace(0, 4*np.pi, 128)
        y = np.linspace(0, 4*np.pi, 128)
        X, Y = np.meshgrid(x, y)
        image = np.sin(X) * np.cos(Y) + 0.5
        # Normalize to [0, 1]
        image = (image - image.min()) / (image.max() - image.min())
        return image
    
    def test_decompose_produces_correct_levels(self, wavelet_transform, sample_image):
        """
        Test that wavelet decomposition produces the correct number of levels.
        
        **Validates: Requirement 5.2** - Decompose into at least 3 scale levels
        """
        levels = 3
        coeffs = wavelet_transform.decompose(sample_image, levels=levels)
        
        assert coeffs.levels == levels
        assert len(coeffs.details) == levels
        assert coeffs.wavelet_name == 'db4'
    
    def test_decompose_extracts_coefficients_at_each_scale(
        self, wavelet_transform, sample_image
    ):
        """
        Test that coefficients are extracted at each scale level.
        
        **Validates: Requirement 5.3** - Extract wavelet coefficients at each scale
        """
        levels = 4
        coeffs = wavelet_transform.decompose(sample_image, levels=levels)
        
        # Check approximation coefficients exist
        assert coeffs.approximation is not None
        assert coeffs.approximation.ndim == 2
        
        # Check detail coefficients at each level
        for level_idx, (cH, cV, cD) in enumerate(coeffs.details):
            assert cH is not None and cH.ndim == 2
            assert cV is not None and cV.ndim == 2
            assert cD is not None and cD.ndim == 2
            
            # All coefficients should be valid arrays
            assert cH.size > 0
            assert cV.size > 0
            assert cD.size > 0
    
    def test_decompose_minimum_three_levels(self, wavelet_transform):
        """
        Test that decomposition produces at least 3 levels for valid images.
        
        **Validates: Requirement 5.2** - At least 3 scale levels
        """
        # Create image large enough for 3 levels
        image = np.random.rand(256, 256)
        
        coeffs = wavelet_transform.decompose(image, levels=3)
        
        assert coeffs.levels >= 3
        assert len(coeffs.details) >= 3
    
    def test_decompose_handles_small_images(self, wavelet_transform):
        """Test that decomposition handles images too small for requested levels."""
        # Create a small image
        small_image = np.random.rand(32, 32)
        
        # Request more levels than possible
        coeffs = wavelet_transform.decompose(small_image, levels=10)
        
        # Should automatically reduce to maximum possible levels
        assert coeffs.levels <= 10
        assert len(coeffs.details) == coeffs.levels
    
    def test_decompose_invalid_input(self, wavelet_transform):
        """Test that decompose raises error for invalid input."""
        # 3D image should raise error
        invalid_image = np.random.rand(64, 64, 3)
        
        with pytest.raises(ValueError, match="Image must be 2D"):
            wavelet_transform.decompose(invalid_image)
        
        # Invalid levels should raise error
        valid_image = np.random.rand(128, 128)
        
        with pytest.raises(ValueError, match="Levels must be at least 1"):
            wavelet_transform.decompose(valid_image, levels=0)
    
    def test_reconstruct_approximates_original(
        self, wavelet_transform, sample_image
    ):
        """
        Test that reconstruction approximates the original image.
        
        **Validates: Requirement 5.4** (implied) - Inverse transform capability
        """
        # Decompose
        coeffs = wavelet_transform.decompose(sample_image, levels=3)
        
        # Reconstruct
        reconstructed = wavelet_transform.reconstruct(coeffs)
        
        # Check shape matches (may differ slightly due to padding)
        assert reconstructed.shape[0] >= sample_image.shape[0]
        assert reconstructed.shape[1] >= sample_image.shape[1]
        
        # Crop to original size for comparison
        reconstructed_cropped = reconstructed[
            :sample_image.shape[0],
            :sample_image.shape[1]
        ]
        
        # Reconstruction should be very close to original
        mse = np.mean((sample_image - reconstructed_cropped) ** 2)
        assert mse < 1e-10, f"Reconstruction error too high: {mse}"
    
    def test_reconstruct_perfect_for_unmodified_coefficients(
        self, wavelet_transform
    ):
        """Test that reconstruction is perfect for unmodified coefficients."""
        image = np.random.rand(128, 128)
        
        coeffs = wavelet_transform.decompose(image, levels=3)
        reconstructed = wavelet_transform.reconstruct(coeffs)
        
        # Crop to original size
        reconstructed = reconstructed[:image.shape[0], :image.shape[1]]
        
        # Should be nearly identical
        assert np.allclose(image, reconstructed, atol=1e-10)
    
    def test_extract_edges_produces_binary_map(
        self, wavelet_transform, sample_image
    ):
        """
        Test that edge extraction produces a binary edge map.
        
        **Validates: Requirement 5.4** - Edge detection from coefficients
        """
        coeffs = wavelet_transform.decompose(sample_image, levels=3)
        edges = wavelet_transform.extract_edges(coeffs, threshold=0.5)
        
        # Check output is binary
        assert edges.dtype in [np.uint8, np.int32, np.int64]
        assert set(np.unique(edges)).issubset({0, 1})
        
        # Check output is 2D
        assert edges.ndim == 2
    
    def test_extract_edges_threshold_effect(self, wavelet_transform):
        """Test that higher thresholds produce fewer edges."""
        # Create image with clear edges
        image = np.zeros((128, 128))
        image[40:80, 40:80] = 1.0  # Square in center
        
        coeffs = wavelet_transform.decompose(image, levels=3)
        
        # Extract edges with different thresholds
        edges_low = wavelet_transform.extract_edges(coeffs, threshold=0.3)
        edges_high = wavelet_transform.extract_edges(coeffs, threshold=0.7)
        
        # Higher threshold should produce fewer edge pixels
        assert np.sum(edges_high) <= np.sum(edges_low)
    
    def test_extract_edges_invalid_threshold(self, wavelet_transform, sample_image):
        """Test that invalid threshold raises error."""
        coeffs = wavelet_transform.decompose(sample_image, levels=3)
        
        with pytest.raises(ValueError, match="Threshold must be in"):
            wavelet_transform.extract_edges(coeffs, threshold=1.5)
        
        with pytest.raises(ValueError, match="Threshold must be in"):
            wavelet_transform.extract_edges(coeffs, threshold=-0.1)
    
    def test_extract_edges_detects_square(self, wavelet_transform):
        """Test that edge detection finds edges of a square."""
        # Create image with clear square
        image = np.zeros((128, 128))
        image[30:90, 30:90] = 1.0
        
        coeffs = wavelet_transform.decompose(image, levels=3)
        edges = wavelet_transform.extract_edges(coeffs, threshold=0.4)
        
        # Should detect some edges
        assert np.sum(edges) > 0
        
        # Edge map should be 2D with correct dimensions
        assert edges.ndim == 2
        assert edges.shape == coeffs.details[0][0].shape
    
    def test_get_coefficient_magnitudes(self, wavelet_transform, sample_image):
        """Test extraction of coefficient magnitudes."""
        coeffs = wavelet_transform.decompose(sample_image, levels=3)
        magnitudes = wavelet_transform.get_coefficient_magnitudes(coeffs)
        
        # Should have one magnitude array per level
        assert len(magnitudes) == 3
        
        # Each magnitude should be 2D and non-negative
        for mag in magnitudes:
            assert mag.ndim == 2
            assert np.all(mag >= 0)
    
    def test_compute_energy_per_scale(self, wavelet_transform, sample_image):
        """Test computation of energy distribution across scales."""
        coeffs = wavelet_transform.decompose(sample_image, levels=3)
        energy = wavelet_transform.compute_energy_per_scale(coeffs)
        
        # Should have energy for approximation (0) and 3 detail levels
        assert len(energy) == 4
        assert 0 in energy  # Approximation
        assert 1 in energy  # Level 1 details
        assert 2 in energy  # Level 2 details
        assert 3 in energy  # Level 3 details
        
        # All energies should be non-negative
        for level, e in energy.items():
            assert e >= 0
    
    def test_wavelet_fallback(self):
        """Test that invalid wavelet falls back to db4."""
        wt = WaveletTransform(wavelet='invalid_wavelet')
        assert wt.wavelet == 'db4'
    
    def test_different_wavelet_families(self, sample_image):
        """Test that different wavelet families work correctly."""
        wavelets = ['haar', 'db4', 'sym4', 'coif1']
        
        for wavelet_name in wavelets:
            wt = WaveletTransform(wavelet=wavelet_name)
            coeffs = wt.decompose(sample_image, levels=3)
            
            assert coeffs.wavelet_name == wavelet_name
            assert coeffs.levels == 3
            
            # Reconstruction should work
            reconstructed = wt.reconstruct(coeffs)
            assert reconstructed.shape[0] >= sample_image.shape[0]
    
    def test_edge_detection_on_gradient_image(self, wavelet_transform):
        """Test edge detection on image with gradient."""
        # Create image with horizontal gradient
        image = np.linspace(0, 1, 128).reshape(1, -1)
        image = np.repeat(image, 128, axis=0)
        
        coeffs = wavelet_transform.decompose(image, levels=3)
        edges = wavelet_transform.extract_edges(coeffs, threshold=0.5)
        
        # Should detect some edges (though gradient is smooth)
        assert edges.shape == (coeffs.details[0][0].shape)
    
    def test_coefficient_structure(self, wavelet_transform, sample_image):
        """Test the structure of WaveletCoefficients dataclass."""
        coeffs = wavelet_transform.decompose(sample_image, levels=3)
        
        # Check dataclass attributes
        assert hasattr(coeffs, 'approximation')
        assert hasattr(coeffs, 'details')
        assert hasattr(coeffs, 'levels')
        assert hasattr(coeffs, 'wavelet_name')
        
        # Check types
        assert isinstance(coeffs.approximation, np.ndarray)
        assert isinstance(coeffs.details, list)
        assert isinstance(coeffs.levels, int)
        assert isinstance(coeffs.wavelet_name, str)
    
    def test_save_edge_visualization(self, wavelet_transform, sample_image, tmp_path):
        """
        Test that edge visualization is saved correctly.
        
        **Validates: Requirements 5.5, 11.3** - Save wavelet_edge.png
        """
        # Decompose and extract edges
        coeffs = wavelet_transform.decompose(sample_image, levels=3)
        edges = wavelet_transform.extract_edges(coeffs, threshold=0.5)
        
        # Save visualization
        output_path = tmp_path / "wavelet_edge.png"
        wavelet_transform.save_edge_visualization(edges, sample_image, str(output_path))
        
        # Check file was created
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    
    def test_save_coefficient_visualization(self, wavelet_transform, sample_image, tmp_path):
        """
        Test that coefficient visualization is saved correctly.
        
        **Validates: Requirements 5.6, 11.3** - Generate wavelet_coefficients.png
        """
        # Decompose
        coeffs = wavelet_transform.decompose(sample_image, levels=3)
        
        # Save visualization
        output_path = tmp_path / "wavelet_coefficients.png"
        wavelet_transform.save_coefficient_visualization(coeffs, str(output_path))
        
        # Check file was created
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    
    def test_visualizations_with_different_levels(self, wavelet_transform, tmp_path):
        """Test that visualizations work with different decomposition levels."""
        image = np.random.rand(256, 256)
        
        for levels in [3, 4, 5]:
            coeffs = wavelet_transform.decompose(image, levels=levels)
            edges = wavelet_transform.extract_edges(coeffs, threshold=0.5)
            
            # Test edge visualization
            edge_path = tmp_path / f"edges_{levels}.png"
            wavelet_transform.save_edge_visualization(edges, image, str(edge_path))
            assert edge_path.exists()
            
            # Test coefficient visualization
            coeff_path = tmp_path / f"coeffs_{levels}.png"
            wavelet_transform.save_coefficient_visualization(coeffs, str(coeff_path))
            assert coeff_path.exists()

